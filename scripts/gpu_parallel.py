"""GPU 여러 장 설정의 'GPU 1장 몫'을 실제 GPU 1장에서 측정 (MIG 슬라이스 1개로 가능).

    python scripts/gpu_parallel.py run               # data/gpu_parallel.csv (설정마다 별도 프로세스, 이어하기)
    python scripts/gpu_parallel.py run --kinds fsdp  # 일부만
    python scripts/gpu_parallel.py report            # reports/gpu_parallel_report.md (가짜 텐서 추적·sim_3d 식과 비교)

방법
  - trace_parallel.py(가짜 텐서, CPU)와 같은 설정·같은 코드 경로를 **실제 CUDA 텐서**로 실행한다.
  - 분산 통신은 가짜 process group(fake PG): rank 0만 실행하고 all-gather/reduce-scatter는 아무 일도 하지 않는다.
    → 메모리: 실제 CUDA 할당기 기준 rank 0의 peak (통신 버퍼 할당 포함, 단편화 포함)
    → 시간  : rank 0의 계산 시간 (통신 시간 = 0). sim_3d의 계산 시간 항을 검증하는 용도
  - 가중치는 meta 장치에서 만든 뒤 나눠서(to_empty) 올린다 → 전체 모델이 GPU 1장에 안 들어가도 측정 가능.
    값은 의미 없는 난수(학습 결과는 보지 않음).
  - MIG 슬라이스는 다른 슬라이스와 통신이 불가능하므로 진짜 멀티 GPU 측정은 아니다. 통신 시간은 여전히 가정.
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "data" / "gpu_parallel.csv"
GiB = 1024 ** 3
FIELDS = ["kind", "model", "method", "wrap", "ws", "pp", "stage", "inflight", "n_layers", "seq", "mbs", "ckpt",
          "peak_alloc_gb", "peak_reserved_gb", "static_gb", "step_ms", "tok_per_s", "gpu_name", "ok", "note", "secs"]


def jobs():
    """trace_parallel과 같은 설정 + 실제 GPU에서만 의미 있는 시간 측정용 추가 설정."""
    import trace_parallel as T
    J = T.jobs()
    # FSDP 8B는 ws 4/8 split만 추가로 4096 길이까지 (시간 = 토큰 수 비례 여부 확인)
    for method in ("full", "lora"):
        for ws in (4, 8):
            J.append(dict(kind="fsdp", model="llama-3.1-8b", method=method, wrap="split", ws=ws, seq=4096, ckpt=1))
    return J


def key(d):
    return tuple(str(d.get(k, "")) for k in ("kind", "model", "method", "wrap", "ws", "stage", "seq", "ckpt"))


# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------
def _hf_config(model_key, tiny=False):
    if tiny:   # CPU 테스트용 작은 모델
        from transformers import LlamaConfig
        return LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                           tie_word_embeddings=True)
    from fitcfg.models import get_model
    from fitcfg.trace import _hf_config as f
    return f(get_model(model_key))


def _materialize(m, cfg, dev):
    """meta 장치 모델 → 실제 장치. 값은 작은 난수, RoPE 버퍼는 다시 계산."""
    import torch
    m.to_empty(device=dev)
    with torch.no_grad():
        for p in m.parameters():
            t = p.to_local() if hasattr(p, "to_local") else p
            t.normal_(0, 0.02)
    inner = m.get_base_model().model if hasattr(m, "get_base_model") else m.model
    cls = type(inner.rotary_emb)
    try:                                   # transformers 버전에 따라 device 인자가 없을 수 있음
        rot = cls(config=cfg, device=dev)
    except TypeError:
        rot = cls(config=cfg)              # CPU에서 inv_freq 계산 후 이동
    inner.rotary_emb = rot.to(dev)


def _sync(dev):
    import torch
    if dev == "cuda":
        torch.cuda.synchronize()


def _mem(dev):
    import torch
    if dev != "cuda":
        return 0, 0, 0
    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved(), torch.cuda.memory_allocated()


def _reset(dev):
    import torch
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _lora(m):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                        lora_dropout=0.0, task_type="CAUSAL_LM"), autocast_adapter_dtype=False)


def _train_loop(m, opt, ids, dev, n_time=3):
    """warm-up 1 step(옵티마이저 상태 생성) → peak 초기화 → 기울기 누적 2 micro-batch(두 번째가 측정 대상) → 시간."""
    import torch
    out = m(input_ids=ids, labels=ids); out.loss.backward(); del out
    opt.step(); opt.zero_grad(set_to_none=True)
    _sync(dev)
    static = _mem(dev)[2]
    _reset(dev)
    for _ in range(2):
        out = m(input_ids=ids, labels=ids); out.loss.backward(); del out
    _sync(dev)
    pa, pr, _ = _mem(dev)
    opt.zero_grad(set_to_none=True)
    ts = []
    for _ in range(n_time):                  # 시간: micro-batch 1개의 forward+backward (통신 0)
        _sync(dev); t0 = time.perf_counter()
        out = m(input_ids=ids, labels=ids); out.loss.backward(); del out
        _sync(dev); ts.append(time.perf_counter() - t0)
    opt.zero_grad(set_to_none=True)
    return pa, pr, static, sorted(ts)[len(ts) // 2]


# ---------------------------------------------------------------------------
# 실험
# ---------------------------------------------------------------------------
def run_fsdp(d, dev, tiny):
    import torch, torch.distributed as dist
    from torch.testing._internal.distributed.fake_pg import FakeStore
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from transformers import AutoModelForCausalLM
    ws = d["ws"]
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=ws)
    mesh = init_device_mesh(dev, (ws,))
    cfg = _hf_config(d["model"], tiny)
    with torch.device("meta"):
        if d["method"] == "full":
            m = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa", dtype=torch.float32)
            mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        else:
            m = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa", dtype=torch.bfloat16)
            for p in m.parameters():
                p.requires_grad_(False)
            m = _lora(m).to(torch.bfloat16)   # FSDP2는 한 단위 안의 파라미터 dtype이 같아야 함 (어댑터 fp32 → bf16)
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
    _materialize(m, cfg, dev)
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5, foreach=False)
    ids = torch.randint(0, cfg.vocab_size, (1, d["seq"]), device=dev)
    return _train_loop(m, opt, ids, dev)


def run_tp(d, dev, tiny):
    import torch, torch.distributed as dist
    from torch.testing._internal.distributed.fake_pg import FakeStore
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
    from transformers import AutoModelForCausalLM
    ws = d["ws"]
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=ws)
    mesh = init_device_mesh(dev, (ws,))
    cfg = _hf_config(d["model"], tiny)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa", dtype=torch.bfloat16)
    m.config.use_cache = False
    plan = {"self_attn.q_proj": ColwiseParallel(), "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(), "self_attn.o_proj": RowwiseParallel(),
            "mlp.gate_proj": ColwiseParallel(), "mlp.up_proj": ColwiseParallel(), "mlp.down_proj": RowwiseParallel()}
    for layer in m.model.layers:
        parallelize_module(layer, mesh, plan)
    _materialize(m, cfg, dev)
    if getattr(cfg, "tie_word_embeddings", False):
        m.tie_weights()        # meta → to_empty 과정에서 공유가 풀려 임베딩·lm_head가 따로 잡히는 것 방지 (1B는 공유 모델)
    if d["ckpt"]:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-5, foreach=False)
    ids = torch.randint(0, cfg.vocab_size, (1, d["seq"]), device=dev)
    return _train_loop(m, opt, ids, dev)


def run_pp(d, dev, tiny):
    """1F1B stage 하나: 층 n개 + (첫 stage면 임베딩, 마지막이면 norm·lm_head·loss). inflight개 forward를 쌓고 backward."""
    import gc, torch
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForCausalLM
    cfg = _hf_config(d["model"], tiny)
    first, last = d["stage"] == "first", d["stage"] == "last"
    n_layers = min(d["n_layers"], cfg.num_hidden_layers)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa", dtype=torch.bfloat16)
    m.config.use_cache = False
    inner = m.model
    inner.layers = torch.nn.ModuleList(list(inner.layers)[:n_layers])
    if not first:
        inner.embed_tokens = None
    if not last:
        inner.norm = None
        m.lm_head = None
    if d["method"] == "lora":
        for p in m.parameters():
            p.requires_grad_(False)
        m = _lora(m)
    _materialize(m, cfg, dev)
    base = m.get_base_model() if hasattr(m, "get_base_model") else m
    inner = base.model
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5, foreach=False)
    S, H = d["seq"], cfg.hidden_size
    pos = torch.arange(S, device=dev).unsqueeze(0)
    ids = torch.randint(0, cfg.vocab_size, (1, S), device=dev)

    def stage_fwd(x):
        h = inner.embed_tokens(x) if first else x
        if first and not h.requires_grad:
            h.requires_grad_(True)
        pe = inner.rotary_emb(h, pos)
        for layer in inner.layers:
            fn = lambda hh, layer=layer: layer(hh, attention_mask=None, position_ids=pos, position_embeddings=pe)
            o = checkpoint(fn, h, use_reentrant=False) if d["ckpt"] else fn(h)
            h = o[0] if isinstance(o, tuple) else o
        if last:
            logits = base.lm_head(inner.norm(h))
            return base.loss_function(logits=logits, labels=ids, vocab_size=cfg.vocab_size)
        return h

    def inp():
        return ids if first else torch.randn(1, S, H, dtype=torch.bfloat16, device=dev, requires_grad=True)

    def bwd(o):
        o.backward() if last else o.backward(torch.ones_like(o))

    def one_step(k):
        outs = []
        for _ in range(k):
            with torch.autocast(dev, dtype=torch.bfloat16):
                outs.append(stage_fwd(inp()))
        for o in outs:
            bwd(o)
        del outs

    one_step(d["inflight"]); opt.step(); opt.zero_grad(set_to_none=True)
    gc.collect(); _sync(dev)
    static = _mem(dev)[2]
    _reset(dev)
    with torch.autocast(dev, dtype=torch.bfloat16):   # 1F1B 정상 구간: 기울기가 남아 있는 상태에서 inflight개 쌓기
        o = stage_fwd(inp())
    bwd(o); del o
    one_step(d["inflight"])
    _sync(dev)
    pa, pr, _ = _mem(dev)
    opt.zero_grad(set_to_none=True)
    ts = []
    for _ in range(3):                                  # 시간: micro-batch 1개 forward+backward (stage 계산 시간)
        _sync(dev); t0 = time.perf_counter()
        one_step(1)
        _sync(dev); ts.append(time.perf_counter() - t0)
    return pa, pr, static, sorted(ts)[1]


def child(d, dev, tiny):
    t0 = time.time()
    import torch
    try:
        pa, pr, static, st = {"fsdp": run_fsdp, "pp": run_pp, "tp": run_tp}[d["kind"]](d, dev, tiny)
        d.update(peak_alloc_gb=round(pa / GiB, 4), peak_reserved_gb=round(pr / GiB, 4), static_gb=round(static / GiB, 4),
                 step_ms=round(st * 1000, 1), tok_per_s=round(d["seq"] / st, 1), ok=1, note="")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in msg.lower() or "NVML_SUCCESS == r" in msg
        d.update(ok=0, note="OOM" if oom else f"{type(e).__name__}: {msg[:200]}")
    d["gpu_name"] = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    d["secs"] = round(time.time() - t0, 1)
    print("RESULT " + json.dumps(d, ensure_ascii=False), flush=True)


def run_all(kinds, dev, tiny, out):
    done = set()
    if out.exists():
        done = {key(r) for r in csv.DictReader(open(out, newline=""))}
    todo = [j for j in jobs() if j["kind"] in kinds and key(j) not in done]
    if tiny:
        todo = [dict(j, model="tiny") for j in todo[:: max(1, len(todo) // 6)]]
    print(f"남은 {len(todo)}개", flush=True)
    new = not out.exists()
    for i, j in enumerate(todo, 1):
        cmd = [sys.executable, __file__, "child", json.dumps(j), "--device", dev] + (["--tiny"] if tiny else [])
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            r = json.loads(line[-1][7:]) if line else dict(j, ok=0, note=("CRASH " + p.stderr[-300:]).replace("\n", " "))
        except subprocess.TimeoutExpired:
            r = dict(j, ok=0, note="TIMEOUT")
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader(); new = False
            w.writerow(r)
        print(f"[{i}/{len(todo)}] {r['kind']} {r['model']} {r['method']} ws{r.get('ws','')} {r.get('stage','')} "
              f"s{r['seq']} ckpt{r['ckpt']} → {r.get('peak_alloc_gb') or r.get('note')} GB, {r.get('step_ms','')} ms "
              f"({r.get('secs','')}s)", flush=True)


# ---------------------------------------------------------------------------
# 비교 리포트
# ---------------------------------------------------------------------------
def report(out):
    import trace_parallel as T
    g = [r for r in csv.DictReader(open(out, newline=""))]
    tr = {T.key(r): r for r in csv.DictReader(open(T.OUT, newline=""))} if T.OUT.exists() else {}
    L = ["# 실제 GPU 1장에서 측정한 병렬 설정의 rank 0 메모리·계산 시간\n",
         "가짜 분산 통신(fake PG) + 실제 CUDA 텐서. 메모리 = max_memory_allocated (CUDA 컨텍스트 제외), "
         "시간 = micro-batch 1개 forward+backward (통신 0).\n",
         "| 종류 | 모델 | 방식 | 나눔 | stage | seq | ckpt | GPU 실측 GB | 가짜텐서 추적 GB | 추적 오차 | sim 식 GB | 식 오차 | ms | tok/s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    errs_t, errs_s = [], []
    for r in g:
        if r["ok"] != "1":
            L.append(f"| {r['kind']} | {r['model']} | {r['method']} | {r.get('wrap') or ''} ws{r.get('ws') or ''} pp{r.get('pp') or ''} "
                     f"| {r.get('stage','')} | {r['seq']} | {r['ckpt']} | {r['note'][:30]} | | | | | | |")
            continue
        meas = float(r["peak_alloc_gb"])
        t = tr.get(key(r))
        tg = float(t["peak_gb"]) if t and t.get("ok") == "1" else None
        try:
            sg = T.predict(r)[1]
        except Exception:
            sg = None
        et = (tg - meas) / meas * 100 if tg else None
        es = (sg - meas) / meas * 100 if sg else None
        if et is not None: errs_t.append(abs(et))
        if es is not None: errs_s.append(abs(es))
        f = lambda v, p="{:.2f}": p.format(v) if v is not None else "-"
        L.append(f"| {r['kind']} | {r['model']} | {r['method']} | {r.get('wrap') or ''} ws{r.get('ws') or ''} pp{r.get('pp') or ''} "
                 f"| {r.get('stage','')} | {r['seq']} | {r['ckpt']} | {meas:.2f} | {f(tg)} | {f(et, '{:+.1f}%')} | {f(sg)} | "
                 f"{f(es, '{:+.1f}%')} | {r['step_ms']} | {r['tok_per_s']} |")
    mean = lambda a: sum(a) / len(a) if a else float("nan")
    L.insert(2, f"\n**요약**: 성공 {sum(r['ok']=='1' for r in g)}/{len(g)} · 가짜 텐서 추적 평균 |오차| {mean(errs_t):.1f}% "
                f"(n={len(errs_t)}) · sim_3d 식 평균 |오차| {mean(errs_s):.1f}% (n={len(errs_s)})\n")
    p = REPO / "reports" / "gpu_parallel_report.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(p)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "child":
        child(json.loads(sys.argv[2]), sys.argv[sys.argv.index("--device") + 1], "--tiny" in sys.argv)
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--kinds", nargs="+", default=["fsdp", "pp", "tp"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", action="store_true", help="CPU 동작 확인용 작은 모델")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.cmd == "run":
        run_all(a.kinds, a.device, a.tiny, Path(a.out))
    else:
        report(Path(a.out))
