"""GPU 여러 장 메모리 검증 (GPU 없이) — FSDP · PP · TP로 나눴을 때 GPU 1장이 실제로 잡는 메모리.

    python scripts/trace_parallel.py run      # 전체 실험 → data/trace_parallel.csv (설정마다 별도 프로세스)
    python scripts/trace_parallel.py report   # reports/parallel_report.md (sim_3d 식과 비교)

방법
  - PyTorch 가짜 텐서(FakeTensorMode) + 가짜 분산 통신(fake process group)으로 rank 0 하나를 실행
    → 실제 학습 코드(HF Transformers, PyTorch FSDP2·DTensor TP)가 GPU 1장에 할당하는 메모리를 추적
  - 메모리 측정: PyTorch 공식 MemTracker / FSDPMemTracker
    (기존 fitcfg 추적기와 GPU 1장 기준 0.0% 일치 확인 — Colab T4 실측으로 검증된 추적기)
  - 옵티마이저는 foreach=False (foreach 임시 버퍼는 별도 항으로 다룸)

실험
  fsdp : FSDP2(층마다 fully_shard + 루트), 혼합 정밀도(fp32 보관 · bf16 계산) = sim_3d의 Full 16 B/param 가정과 동일
         LoRA는 bf16 고정 가중치 + bf16 어댑터
  pp   : 1F1B에서 stage s가 동시에 보관하는 micro-batch 수(pp−s)만큼 forward를 쌓은 뒤 backward
         (첫 stage = 임베딩 + 층, 마지막 stage = 층 + norm + lm_head + loss)
  tp   : DTensor로 q·k·v·gate·up 열 분할, o·down 행 분할 (Megatron 방식)
"""
from __future__ import annotations
import csv, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "data" / "trace_parallel.csv"
GiB = 1024 ** 3
FIELDS = ["kind", "model", "method", "wrap", "ws", "pp", "stage", "inflight", "n_layers", "seq", "mbs", "ckpt",
          "peak_gb", "breakdown", "ok", "note", "secs"]


def jobs():
    J = []
    for model, wraps in (("llama-3.2-1b", ("root",)), ("llama-3.1-8b", ("root", "split"))):
        for wrap in wraps:
            for method in ("full", "lora"):
                for ws in (2, 4, 8):
                    for ckpt in (1, 0):
                        J.append(dict(kind="fsdp", model=model, method=method, wrap=wrap, ws=ws, seq=2048, ckpt=ckpt))
    for method in ("lora", "full"):
        for stage, inflight in (("first", 4), ("mid", 3), ("last", 1)):
            for seq in (1024, 2048):
                for ckpt in (1, 0):
                    J.append(dict(kind="pp", model="llama-3.1-8b", method=method, pp=4, stage=stage,
                                  inflight=inflight, n_layers=8, seq=seq, ckpt=ckpt))
    for model in ("llama-3.2-1b", "llama-3.1-8b"):
        for ws in (2, 4):
            for ckpt in (1, 0):
                J.append(dict(kind="tp", model=model, method="full", ws=ws, seq=2048, ckpt=ckpt))
    return J


def key(d):
    return tuple(str(d.get(k, "")) for k in ("kind", "model", "method", "wrap", "ws", "stage", "seq", "ckpt"))


# ---------------------------------------------------------------------------
# 실행 (자식 프로세스)
# ---------------------------------------------------------------------------
def _hf_config(model_key):
    from fitcfg.models import get_model
    from fitcfg.trace import _hf_config as f
    return f(get_model(model_key))


def _snap_total(tracker):
    snap = tracker.get_tracker_snapshot("peak")
    total = sum(v["Total"] for v in snap.values())
    parts = {}
    for v in snap.values():
        for k, b in v.items():
            if k != "Total":
                name = getattr(k, "value", str(k))
                parts[name] = parts.get(name, 0) + b
    return total, {k: round(b / GiB, 3) for k, b in parts.items() if b}


def run_fsdp(d):
    import torch, torch.distributed as dist
    from torch.testing._internal.distributed.fake_pg import FakeStore
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker
    from transformers import AutoModelForCausalLM
    ws = d["ws"]
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=ws)
    mesh = init_device_mesh("cpu", (ws,))
    cfg = _hf_config(d["model"])
    with FakeTensorMode():
        if d["method"] == "full":
            m = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa", dtype=torch.float32)
            mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        else:
            from peft import LoraConfig, get_peft_model
            m = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa", dtype=torch.bfloat16)
            m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             lora_dropout=0.0, task_type="CAUSAL_LM"), autocast_adapter_dtype=False)
            mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
        base = m.get_base_model() if hasattr(m, "get_base_model") else m
        base.config.use_cache = False
        if d["ckpt"]:
            if d["method"] != "full":
                base.enable_input_require_grads()
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for layer in base.model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=mp)
        if d.get("wrap") == "split":
            fully_shard(base.model.embed_tokens, mesh=mesh, mp_policy=mp)
            fully_shard([base.model.norm, base.lm_head], mesh=mesh, mp_policy=mp)
        fully_shard(m, mesh=mesh, mp_policy=mp)
        m.train()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5, foreach=False)
        ids = torch.randint(0, cfg.vocab_size, (1, d["seq"]))
        mt = FSDPMemTracker(m, opt)
        mt.track_inputs((ids,))
        out = m(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        del out
        with mt:                         # 기울기 누적 2 micro-batch → 두 번째(기울기가 남아 있는 상태 = 실제 학습)를 측정
            for i in range(2):
                out = m(input_ids=ids, labels=ids)
                out.loss.backward()
                del out
                if i == 0:
                    mt.reset_mod_stats()
        return _snap_total(mt)


def run_pp(d):
    """1F1B stage 하나: 층 n개만 남긴 모델에 micro-batch를 inflight개 forward로 쌓고 backward.
    같은 층을 여러 번 호출하므로 MemTracker(모듈 단위) 대신 fitcfg 추적기(저장공간 단위)를 쓴다
    — GPU 1장 기준 MemTracker와 0.0% 일치 확인, PP에는 FSDP식 저장공간 크기 변경이 없음."""
    import gc, torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForCausalLM
    from fitcfg.trace import _make_tracker
    cfg = _hf_config(d["model"])
    first, last = d["stage"] == "first", d["stage"] == "last"
    tr = _make_tracker()()
    with FakeTensorMode(), tr:
        m = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa", dtype=torch.bfloat16)
        m.config.use_cache = False
        inner = m.model
        inner.layers = torch.nn.ModuleList(list(inner.layers)[: d["n_layers"]])
        if not first:
            inner.embed_tokens = None
        if not last:
            inner.norm = None
            m.lm_head = None
        gc.collect()
        if d["method"] == "lora":
            from peft import LoraConfig, get_peft_model
            for p in m.parameters():
                p.requires_grad_(False)
            pm = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                              lora_dropout=0.0, task_type="CAUSAL_LM"), autocast_adapter_dtype=False)
            params = [p for p in pm.parameters() if p.requires_grad]
        else:
            params = [p for p in m.parameters()]
        opt = torch.optim.AdamW(params, lr=1e-5, foreach=False)
        S, H = d["seq"], cfg.hidden_size
        pos = torch.arange(S).unsqueeze(0)
        ids = torch.randint(0, cfg.vocab_size, (1, S))

        def stage_fwd(x):
            h = inner.embed_tokens(x) if first else x
            if first and not h.requires_grad:          # LoRA: enable_input_require_grads와 같은 처리
                h.requires_grad_(True)
            pe = inner.rotary_emb(h, pos)
            for layer in inner.layers:
                fn = lambda hh, layer=layer: layer(hh, attention_mask=None, position_ids=pos, position_embeddings=pe)
                o = checkpoint(fn, h, use_reentrant=False) if d["ckpt"] else fn(h)
                h = o[0] if isinstance(o, tuple) else o
            if last:
                logits = m.lm_head(inner.norm(h))
                return m.loss_function(logits=logits, labels=ids, vocab_size=cfg.vocab_size)
            return h

        def one_step():
            outs = []
            for _ in range(d["inflight"]):
                x = ids if first else torch.randn(1, S, H, dtype=torch.bfloat16, requires_grad=True)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    outs.append(stage_fwd(x))
            for o in outs:
                if last:
                    o.backward()
                else:
                    o.backward(torch.ones_like(o))
            del outs

        one_step()                       # 옵티마이저 상태를 만들기 위한 1 step (측정 제외)
        opt.step()
        opt.zero_grad(set_to_none=True)
        gc.collect()
        static = tr.cur
        tr.reset_peak()
        # 측정: 1F1B 정상 구간 — 앞선 micro-batch의 기울기가 이미 있는 상태에서 forward를 inflight개 쌓고 backward
        warm = ids if first else torch.randn(1, S, H, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            o = stage_fwd(warm)
        o.backward() if last else o.backward(torch.ones_like(o))
        del o
        one_step()
        gc.collect()
        return tr.peak, {"static": round(static / GiB, 3)}


def run_tp(d):
    """층마다 Megatron식 TP (q·k·v·gate·up 열 분할, o·down 행 분할). 임베딩·lm_head는 복제."""
    import torch, torch.distributed as dist
    from torch.testing._internal.distributed.fake_pg import FakeStore
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
    from torch.distributed._tools.mem_tracker import MemTracker
    from transformers import AutoModelForCausalLM
    ws = d["ws"]
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=ws)
    mesh = init_device_mesh("cpu", (ws,))
    cfg = _hf_config(d["model"])
    with FakeTensorMode():
        m = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa", dtype=torch.bfloat16)
        m.config.use_cache = False
        plan = {"self_attn.q_proj": ColwiseParallel(), "self_attn.k_proj": ColwiseParallel(),
                "self_attn.v_proj": ColwiseParallel(), "self_attn.o_proj": RowwiseParallel(),
                "mlp.gate_proj": ColwiseParallel(), "mlp.up_proj": ColwiseParallel(), "mlp.down_proj": RowwiseParallel()}
        for layer in m.model.layers:
            parallelize_module(layer, mesh, plan)
        if d["ckpt"]:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-5, foreach=False)
        ids = torch.randint(0, cfg.vocab_size, (1, d["seq"]))
        out = m(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        del out
        mt = MemTracker()
        mt.track_external(m, opt)
        with mt:                         # 기울기 누적 2 micro-batch → 두 번째를 측정
            for i in range(2):
                out = m(input_ids=ids, labels=ids)
                out.loss.backward()
                del out
                if i == 0:
                    mt.reset_mod_stats()
        return _snap_total(mt)


def child(d):
    t0 = time.time()
    # 가짜 텐서에서는 Module.to()의 텐서 교체가 막혀 있어 fitcfg.trace와 같은 우회를 쓴다 (PEFT가 내부에서 .to() 호출)
    import torch
    from peft.tuners import tuners_utils
    from fitcfg.trace import _fake_safe_to
    torch.nn.Module.to = _fake_safe_to
    tuners_utils.BaseTunerLayer._move_adapter_to_device_of_base_layer = lambda self, *a, **k: None
    try:
        total, parts = {"fsdp": run_fsdp, "pp": run_pp, "tp": run_tp}[d["kind"]](d)
        d.update(peak_gb=round(total / GiB, 4), breakdown=json.dumps(parts), ok=1, note="")
    except Exception as e:  # noqa: BLE001
        d.update(ok=0, note=f"{type(e).__name__}: {str(e)[:200]}")
    d["secs"] = round(time.time() - t0, 1)
    print("RESULT " + json.dumps(d, ensure_ascii=False), flush=True)


def run_all(kinds):
    done = set()
    if OUT.exists():
        done = {key(r) for r in csv.DictReader(open(OUT, newline=""))}
    todo = [j for j in jobs() if j["kind"] in kinds and key(j) not in done]
    print(f"남은 {len(todo)}개", flush=True)
    new = not OUT.exists()
    for i, j in enumerate(todo, 1):
        p = subprocess.run([sys.executable, __file__, "child", json.dumps(j)], capture_output=True, text=True, timeout=1800)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        r = json.loads(line[-1][7:]) if line else dict(j, ok=0, note=("CRASH " + p.stderr[-200:]).replace("\n", " "))
        with open(OUT, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
                new = False
            w.writerow(r)
        print(f"[{i}/{len(todo)}] {r['kind']} {r['model']} {r['method']} ws{r.get('ws','')} {r.get('stage','')} "
              f"s{r['seq']} ckpt{r['ckpt']} → {r.get('peak_gb', r.get('note'))} ({r.get('secs','')}s)", flush=True)


# ---------------------------------------------------------------------------
# 비교 (sim_3d 식)
# ---------------------------------------------------------------------------
def predict(r) -> tuple[float, float]:
    """(검증 전 식, 고친 식) GiB — CUDA 기본 사용량 제외. 추적 설정과 다른 정밀도 가정은 맞춰서 비교.
    검증 전 식 = TP 비분할 0.3, checkpointing 최소값 없음, FSDP 임베딩/lm_head 반영 없음."""
    sys.path.insert(0, str(REPO / "scripts"))
    import fitcfg.parallel as S
    saved = (S.TP_UNSPLIT, S.CKPT_MIN)
    S.TP_UNSPLIT, S.CKPT_MIN = 0.3, 0.0
    try:
        old = _predict_one(r, S, wrap="ideal")
    finally:
        S.TP_UNSPLIT, S.CKPT_MIN = saved
    return old, _predict_one(r, S, wrap=r.get("wrap") or None)


def _predict_one(r, S, wrap):
    from fitcfg.models import get_model
    m = get_model(r["model"])
    kind, method = r["kind"], r["method"]
    S.SEQ = int(r["seq"])
    ckpt = r["ckpt"] == "1"
    if kind == "fsdp":
        c = S.Ctx(m, method, 1, 1, int(r["ws"]), 1, ckpt, True, 15.0, 32)
        return sum(S.stage_mem_parts(c, m.n_layers, 0, True, True, wrap=wrap).values())
    if kind == "tp":                    # 추적: 층만 TP, 임베딩·lm_head는 복제, bf16 순수 학습(8 B/param)
        c = S.Ctx(m, method, int(r["ws"]), 1, 1, 1, ckpt, False, 15.0, 32)
        p = S.stage_mem_parts(c, m.n_layers, 0, True, True, shard_vocab=False)
    else:                               # PP stage
        pp, inflight = int(r["pp"]), int(r["inflight"])
        c = S.Ctx(m, method, 1, pp, 1, 1, ckpt, False, 15.0, 32)
        p = S.stage_mem_parts(c, int(r["n_layers"]), pp - inflight, r["stage"] == "first", r["stage"] == "last")
    if method == "full":                # sim은 혼합정밀 16 B/param, 추적은 bf16 순수 8 B/param
        p["static"] *= 8 / 16
    return sum(p.values())


def report():
    rows = [r for r in csv.DictReader(open(OUT, newline=""))]
    ok = [r for r in rows if r["ok"] == "1"]
    L = ["# GPU 여러 장 메모리 검증 (가짜 분산 통신, GPU 없이)", "",
         f"- 실험 {len(rows)}개 중 성공 {len(ok)}개 · 측정: PyTorch MemTracker/FSDPMemTracker (PP는 fitcfg 추적기)",
         "- 비교 대상: `scripts/sim_3d.py`의 GPU당 메모리 식 (CUDA 기본 사용량 제외) · 역전파 시점 최대값 (옵티마이저 step 임시버퍼 제외)",
         "- FSDP 감싸는 방식: root = 층만 따로·임베딩/lm_head는 루트(기본 예제 방식) / split = 임베딩, norm+lm_head도 따로", "",
         "| 방식 | 모델 | 학습 | 분할 | 길이 | ckpt | 추적 (GiB) | 검증 전 식 | 오차 | 고친 식 | 오차 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    errs = {}
    for r in ok:
        old, new = predict(r)
        act = float(r["peak_gb"])
        e0, e1 = (old - act) / act * 100, (new - act) / act * 100
        g = (r["kind"], r["method"], r.get("wrap") or "")
        errs.setdefault(g, []).append((e0, e1))
        if r["kind"] == "fsdp":
            part = f"GPU {r['ws']}장 ({r['wrap']})"
        elif r["kind"] == "tp":
            part = f"TP {r['ws']}"
        else:
            part = f"PP4 {r['stage']} (micro-batch {r['inflight']}개 보관)"
        L.append(f"| {r['kind'].upper()} | {r['model']} | {r['method']} | {part} | {r['seq']} | {r['ckpt']} | {act:.2f} | "
                 f"{old:.2f} | {e0:+.1f}% | {new:.2f} | {e1:+.1f}% |")
    L += ["", "## 요약 (평균 절대 오차 / 과소예측 = OOM 위험)", "",
          "| 방식 | 학습 | 감싸기 | n | 검증 전 식 | 고친 식 | 고친 식 최대 | 고친 식 과소예측 |", "|---|---|---|---|---|---|---|---|"]
    for (k, meth, wrap), es in sorted(errs.items()):
        a0 = sum(abs(x) for x, _ in es) / len(es)
        a1 = sum(abs(y) for _, y in es) / len(es)
        mx = max(abs(y) for _, y in es)
        under = sum(y < 0 for _, y in es)
        L.append(f"| {k.upper()} | {meth} | {wrap or '-'} | {len(es)} | {a0:.1f}% | {a1:.1f}% | {mx:.1f}% | {under}/{len(es)} |")
    bad = [r for r in rows if r["ok"] != "1"]
    if bad:
        L += ["", "## 실패", ""] + [f"- {r['kind']} {r['model']} {r['method']} ws{r.get('ws','')}: {r['note']}" for r in bad[:10]]
    p = REPO / "reports" / "parallel_report.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    i = L.index("## 요약 (평균 절대 오차 / 과소예측 = OOM 위험)")
    print("\n".join(L[i:]))
    print("->", p)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "child":
        child(json.loads(sys.argv[2]))
    elif cmd == "run":
        run_all(sys.argv[2:] or ["fsdp", "pp", "tp"])
    elif cmd == "report":
        report()
