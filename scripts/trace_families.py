"""처음 보는 모델 계열에 대한 일반화 검증 (GPU 없이, FakeTensor 추적).

기존 4개 모델(Llama-3.2-1B, Qwen2.5-3B/7B, Llama-3.1-8B)로만 맞춘 상수로
보정에 한 번도 쓰지 않은 계열(Mistral, Phi-3, Gemma-2, Gemma-3)을 예측해 오차를 본다.

    python scripts/trace_families.py            # 추적 → data/trace_families.csv (이어하기)
    python scripts/trace_families.py --report   # reports/family_report.md

구조 값은 각 모델의 공개 config.json 기준 (가중치 다운로드 없이 구조만 생성).
"""
from __future__ import annotations
import argparse, csv, gc, sys, time
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fitcfg.models import ModelSpec  # noqa: E402

OUT = REPO / "data" / "trace_families.csv"
GiB = 1024 ** 3

# 각 모델의 공개 config.json 값
FAMILIES = {
    "mistral-7b-v0.3": ("MistralConfig", dict(
        vocab_size=32768, hidden_size=4096, intermediate_size=14336, num_hidden_layers=32,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128, sliding_window=None,
        tie_word_embeddings=False, max_position_embeddings=32768),
        {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"]}),
    "phi-3-mini": ("Phi3Config", dict(
        vocab_size=32064, hidden_size=3072, intermediate_size=8192, num_hidden_layers=32,
        num_attention_heads=32, num_key_value_heads=32, tie_word_embeddings=False,
        max_position_embeddings=4096, sliding_window=None, pad_token_id=32000),
        {"attn": ["qkv_proj", "o_proj"]}),          # q·k·v가 한 행렬로 합쳐진 구조
    "gemma-2-2b": ("Gemma2Config", dict(
        vocab_size=256000, hidden_size=2304, intermediate_size=9216, num_hidden_layers=26,
        num_attention_heads=8, num_key_value_heads=4, head_dim=256, sliding_window=4096,
        final_logit_softcapping=30.0, attn_logit_softcapping=50.0, query_pre_attn_scalar=256,
        hidden_activation="gelu_pytorch_tanh", tie_word_embeddings=True, max_position_embeddings=8192),
        {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"]}),
    "gemma-3-1b": ("Gemma3TextConfig", dict(
        vocab_size=262144, hidden_size=1152, intermediate_size=6912, num_hidden_layers=26,
        num_attention_heads=4, num_key_value_heads=1, head_dim=256, sliding_window=512,
        final_logit_softcapping=None, query_pre_attn_scalar=256,
        hidden_activation="gelu_pytorch_tanh", tie_word_embeddings=True, max_position_embeddings=32768),
        {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"]}),
    "qwen3-4b": ("Qwen3Config", dict(
        vocab_size=151936, hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128, tie_word_embeddings=True,
        max_position_embeddings=32768, rope_theta=1000000),
        {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"]}),
    "olmo2-1b": ("Olmo2Config", dict(
        vocab_size=100352, hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
        num_attention_heads=16, num_key_value_heads=16, tie_word_embeddings=False,
        max_position_embeddings=4096, rope_theta=500000, pad_token_id=100277),
        {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"]}),
    "smollm2-1.7b": ("LlamaConfig", dict(
        vocab_size=49152, hidden_size=2048, intermediate_size=8192, num_hidden_layers=24,
        num_attention_heads=32, num_key_value_heads=32, tie_word_embeddings=True,
        max_position_embeddings=8192, rope_theta=130000),
        {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"]}),
    "granite-3.3-2b": ("GraniteConfig", dict(
        vocab_size=49152, hidden_size=2048, intermediate_size=8192, num_hidden_layers=40,
        num_attention_heads=32, num_key_value_heads=8, tie_word_embeddings=True, max_position_embeddings=131072,
        attention_multiplier=0.015625, embedding_multiplier=12.0, logits_scaling=8.0, residual_multiplier=0.22,
        rope_theta=10000000.0, pad_token_id=0),
        {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"]}),
}
METHODS = ["lora", "qlora", "full"]
BATCHES = [1, 4]
SEQS = [512, 1024, 2048, 4096]
FIELDS = ["model", "attn_impl", "method", "batch", "seq", "ckpt", "n_params", "mem_model_gb",
          "mem_steady_gb", "peak_alloc_gb", "ok", "note", "secs"]


def hf_config(key):
    import transformers
    cls, kw, _ = FAMILIES[key]
    return getattr(transformers, cls)(**kw)


def spec_for(key) -> ModelSpec:
    """config.json 값 + 실제 파라미터 수(meta 장치에서 세기) → fitcfg ModelSpec.
    = 사용자가 '새 모델 이름'을 넣었을 때 도구가 할 수 있는 것과 같은 정보."""
    import torch
    from transformers import AutoModelForCausalLM
    cfg = hf_config(key)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    n = sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
    return ModelSpec(name=key, n_params=n, n_layers=cfg.num_hidden_layers, hidden=cfg.hidden_size,
                     n_heads=cfg.num_attention_heads, n_kv_heads=cfg.num_key_value_heads,
                     vocab=cfg.vocab_size, intermediate=cfg.intermediate_size,
                     tied_embeddings=bool(cfg.tie_word_embeddings))


def trace_one(key, method, batch, seq, ckpt, attn_impl="sdpa", steps=2):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from peft.tuners import tuners_utils
    from transformers import AutoModelForCausalLM
    from fitcfg.trace import _make_tracker, _fake_safe_to, _recast

    torch_dtype = torch.bfloat16
    targets = FAMILIES[key][2]["attn"]
    orig_to = torch.nn.Module.to
    orig_move = tuners_utils.BaseTunerLayer._move_adapter_to_device_of_base_layer
    torch.nn.Module.to = _fake_safe_to
    tuners_utils.BaseTunerLayer._move_adapter_to_device_of_base_layer = lambda self, *a, **k: None
    fake, tracker = FakeTensorMode(), _make_tracker()()
    prev = torch.get_default_dtype()
    try:
        with fake, tracker:
            torch.set_default_dtype(torch_dtype)
            try:
                model = AutoModelForCausalLM.from_config(hf_config(key), attn_implementation=attn_impl)
            finally:
                torch.set_default_dtype(prev)
            model.config.use_cache = False
            if method == "qlora":
                lm_head = model.get_output_embeddings()
                _recast(model, lambda m, k: (not isinstance(m, torch.nn.Linear)) or m is lm_head, torch.float32)
                for p in model.parameters():
                    p.requires_grad_(False)
            if ckpt:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if method in ("lora", "qlora"):
                from peft import LoraConfig, get_peft_model
                if ckpt:
                    model.enable_input_require_grads()
                model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=targets,
                                                         lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
            model.train()
            gc.collect()
            mem_model = tracker.cur
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=1e-5, foreach=True)
            ids = torch.randint(0, model.config.vocab_size, (batch, seq))
            tracker.reset_peak()
            for _ in range(steps):
                with torch.autocast("cpu", dtype=torch_dtype):
                    out = model(input_ids=ids, labels=ids)
                out.loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                del out
                gc.collect()
            peak, steady = tracker.peak, tracker.cur
            del model, opt, params, ids
            gc.collect()
    finally:
        torch.nn.Module.to = orig_to
        tuners_utils.BaseTunerLayer._move_adapter_to_device_of_base_layer = orig_move
    return mem_model / GiB, steady / GiB, peak / GiB


def run(models, attn_impls, OUT=OUT):
    done = set()
    if OUT.exists():
        with open(OUT, newline="") as f:
            done = {(r["model"], r["attn_impl"], r["method"], r["batch"], r["seq"], r["ckpt"]) for r in csv.DictReader(f)}
    new = not OUT.exists()
    specs = {k: spec_for(k) for k in models}
    jobs = [(k, a, m, b, s, c) for k in models for a in attn_impls.get(k, ["sdpa"])
            for m, b, s, c in product(METHODS, BATCHES, SEQS, [1, 0])]
    todo = [j for j in jobs if tuple(map(str, j)) not in done]
    print(f"전체 {len(jobs)}개 / 남은 {len(todo)}개", flush=True)
    for i, (k, a, m, b, s, c) in enumerate(todo, 1):
        t0 = time.time()
        row = dict(model=k, attn_impl=a, method=m, batch=b, seq=s, ckpt=c, n_params=specs[k].n_params)
        try:
            mm, st, pk = trace_one(k, m, b, s, bool(c), a)
            row.update(mem_model_gb=round(mm, 4), mem_steady_gb=round(st, 4), peak_alloc_gb=round(pk, 4), ok=1, note="")
        except Exception as e:  # noqa: BLE001
            row.update(ok=0, note=f"{type(e).__name__}: {str(e)[:150]}")
        row["secs"] = round(time.time() - t0, 1)
        with open(OUT, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
                new = False
            w.writerow(row)
        print(f"[{i}/{len(todo)}] {k} {a} {m} b{b} s{s} ckpt{c} → {row.get('peak_alloc_gb', row['note'])} ({row['secs']}s)", flush=True)


def report():
    """기존 4개 모델 추적 데이터로만 상수를 맞추고 → 새 계열을 예측."""
    from functools import partial
    from fitcfg.calibration import load_measurements, fit, predict_gb, metrics, Measurement
    from fitcfg.memory import TrainConfig
    old, _ = load_measurements(REPO / "data" / "trace_measurements.csv")
    consts = partial(fit, target="alloc", lam=0.001, use_method_priors=False)(old)
    rows = [r for f in sorted(OUT.parent.glob("trace_families*.csv")) for r in csv.DictReader(open(f, newline=""))]
    specs = {k: spec_for(k) for k in {r["model"] for r in rows}}
    res = []   # (model, impl, method, ckpt, seq, pred, actual)
    for r in rows:
        if r["ok"] != "1":
            continue
        k = r["model"]
        cfg = TrainConfig(model=k, method=r["method"], batch_size=int(r["batch"]), seq_len=int(r["seq"]),
                          grad_checkpointing=r["ckpt"] == "1", dtype="bf16", flash_attn=r["attn_impl"] != "eager")
        m = Measurement(cfg=cfg, spec=specs[k], ok=True, oom=False, device_peak_gb=None,
                        peak_alloc_gb=float(r["peak_alloc_gb"]), mem_model_gb=float(r["mem_model_gb"]),
                        mem_steady_gb=float(r["mem_steady_gb"]), gpu_total_gb=None, gpu_name="trace-cpu",
                        run_id="fam")
        res.append((k, r["attn_impl"], r["method"], r["ckpt"], int(r["seq"]), predict_gb(m, consts), m.peak_alloc_gb))

    def tab(key, title):
        L = [f"### {title}", "", "| 구분 | n | MAPE | 최대 | 과소예측 비율 |", "|---|---|---|---|---|"]
        for v in sorted({key(x) for x in res}):
            mm = metrics((p, a) for *_, p, a in [x for x in res if key(x) == v])
            L.append(f"| {v} | {mm.n} | {mm.mape:.1f}% | {mm.max_ape:.1f}% | {mm.under_rate * 100:.0f}% |")
        return L + [""]

    L = ["# 처음 보는 모델 계열 일반화 검증 (FakeTensor, GPU 없이)", "",
         "- 상수: 기존 4개 모델(Llama-3.2-1B, Qwen2.5-3B/7B, Llama-3.1-8B) 추적 288행으로만 보정",
         "- 대상: 보정에 쓰지 않은 Mistral-7B-v0.3, Phi-3-mini, Gemma-2-2B, Gemma-3-1B",
         "- 비교 기준: 기존 4개 모델끼리 '모델 하나 빼고' 검증한 오차 = 1.1%", ""]
    L += tab(lambda x: f"{x[0]} ({x[1]})", "모델별")
    L += tab(lambda x: f"{x[0]} ({x[1]}) · {x[2]}", "모델 × 학습 방식")
    L += tab(lambda x: f"{x[0]} ({x[1]}) · ckpt{x[3]}", "모델 × checkpointing")
    L += tab(lambda x: f"{x[0]} ({x[1]}) · seq{x[4]:>4}", "모델 × 시퀀스 길이")
    p = REPO / "reports" / "family_report.md"
    p.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print("->", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--models", nargs="*", default=list(FAMILIES))
    a = ap.parse_args()
    if a.report:
        report()
    else:
        # Gemma-2는 HF가 학습 시 eager attention을 권장(softcapping) → 두 경로 모두 추적
        out = OUT if len(a.models) == len(FAMILIES) else OUT.with_name(f"trace_families_{a.models[0]}.csv")
        run(a.models, {"gemma-2-2b": ["sdpa", "eager"]}, out)
