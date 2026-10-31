"""GPU 없이 '텐서 수준 실측'을 만드는 메모리 추적기 (FakeTensor + TorchDispatchMode).

아이디어
    PyTorch의 FakeTensorMode는 실제 메모리·연산 없이 모양(shape)·dtype만 가진 가짜 텐서로
    모델을 그대로 실행한다. 여기에 모든 연산의 출력 저장공간(storage)을 가로채 기록하고,
    저장공간이 해제될 때(weakref 콜백) 빼 주면 → 학습 스텝 동안 '살아 있는 텐서 바이트'의
    최댓값 = torch.cuda.max_memory_allocated() 와 같은 의미의 값을 GPU 없이 얻는다.

    - 실제 HF 모델 코드(LlamaForCausalLM 등) + PEFT + torch.optim 을 그대로 실행하므로
      "공식이 가정한 것"이 아니라 "코드가 실제로 할당하는 것"을 잰다.
    - 8B 모델·seq 4096도 수 초면 추적된다 (연산을 실제로 하지 않음).

한계 (Colab 실측이 여전히 필요한 이유)
    - CUDA 컨텍스트, cuBLAS 워크스페이스, 할당자 캐시/단편화(reserved − allocated)는 안 잡힌다
      → cost model의 overhead 항은 실측으로만 보정.
    - bitsandbytes 4bit(QLoRA)는 CUDA 커널이라 추적 불가 → LoRA/Full만.
    - CPU 경로의 SDPA(flash-for-CPU)를 쓰므로, CUDA flash/mem-efficient 커널과 임시 버퍼가 약간 다를 수 있다.
"""
from __future__ import annotations
import weakref
from dataclasses import dataclass

GiB = 1024 ** 3


@dataclass
class TraceResult:
    mem_model_gb: float       # 모델(+LoRA) 준비 직후 살아 있는 바이트
    mem_steady_gb: float      # 학습 스텝 후 zero_grad 뒤 (weights + optimizer states)
    peak_alloc_gb: float      # 학습 구간 peak (max_memory_allocated 와 같은 의미)


def _make_tracker():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class LiveBytes(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.live: dict[int, int] = {}
            self.cur = 0
            self.peak = 0
            self._refs = []

        def _free(self, key):
            self.cur -= self.live.pop(key, 0)

        def _track(self, t):
            if not isinstance(t, torch.Tensor):
                return
            st = t.untyped_storage()
            key = id(st)
            if key in self.live:
                return
            n = st.nbytes()
            self.live[key] = n
            self.cur += n
            self.peak = max(self.peak, self.cur)
            weakref.finalize(st, self._free, key)

        def reset_peak(self):
            self.peak = self.cur

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in tree_flatten(out)[0]:
                self._track(t)
            return out

    return LiveBytes


def trace_step(model_key: str, method: str, batch: int, seq: int, ckpt: bool,
               optim: str = "adamw", lora_rank: int = 16, lora_targets: str = "attn",
               steps: int = 2, dtype: str = "bf16", n_layers: int | None = None,
               vocab: int | None = None) -> TraceResult:
    """fitcfg 모델 프리셋 구조로 HF 모델을 가짜 텐서로 만들어 학습 스텝을 추적한다.
    측정 하네스(scripts/measure.py)와 같은 루프: autocast + loss.backward + step + zero_grad.

    method="qlora" 는 **에뮬레이션**: bitsandbytes 4bit 커널은 CPU/가짜 텐서로 못 돌리므로,
    base Linear는 16bit로 두고 prepare_model_for_kbit_training 처럼 나머지(임베딩·norm·lm_head)를
    fp32로 올린 LoRA로 추적한다. 가중치 크기는 다르지만(→ 분석 시 mem_model을 빼고 봄)
    활성값·logits의 dtype 흐름은 QLoRA와 같다.
    n_layers / vocab: 항을 분리하기 위한 절제(ablation) 실험용 덮어쓰기."""
    import dataclasses
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from peft.tuners import tuners_utils
    from .models import get_model

    if optim == "adamw8bit":
        raise ValueError("8bit AdamW(bitsandbytes)는 추적할 수 없습니다.")
    spec = get_model(model_key)
    if n_layers or vocab:
        spec = dataclasses.replace(spec, n_layers=n_layers or spec.n_layers, vocab=vocab or spec.vocab)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    # fp16 + Full-FT 는 HF Trainer처럼 fp32 가중치 + autocast (측정 하네스와 동일)
    load_dtype = torch.float32 if (method == "full" and dtype == "fp16") else torch_dtype

    orig_to = torch.nn.Module.to
    orig_move = tuners_utils.BaseTunerLayer._move_adapter_to_device_of_base_layer
    torch.nn.Module.to = _fake_safe_to
    # PEFT는 어댑터를 base dtype으로 옮겼다가 다시 fp32로 올린다(autocast_adapter_dtype).
    # 최종 결과는 'fp32 어댑터'이므로 처음부터 fp32로 만들고 이동을 생략해도 메모리상 동일하다.
    tuners_utils.BaseTunerLayer._move_adapter_to_device_of_base_layer = lambda self, *a, **k: None
    try:
        return _run(FakeTensorMode(), _make_tracker()(), load_dtype, torch_dtype, spec, method,
                    batch, seq, ckpt, lora_rank, lora_targets, steps)
    finally:
        torch.nn.Module.to = orig_to
        tuners_utils.BaseTunerLayer._move_adapter_to_device_of_base_layer = orig_move


def _fake_safe_to(self, *args, **kwargs):
    """가짜 텐서는 Module.to()의 in-place 교체(swap_tensors)가 막혀 있다(FakeTensorMode가 weakref를 잡음).
    같은 장치로의 이동은 무시하고, dtype 변경은 파라미터를 새로 만들어 끼운다(공유/tied 가중치는 한 번만)."""
    import torch
    _, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
    if dtype is not None:
        _recast(self, lambda m, name: True, dtype)
    return self


def _recast(root, select, dtype):
    """select(module, param_name)이 참인 부동소수 파라미터를 dtype으로 새로 만들어 교체."""
    import torch
    memo = {}
    for m in root.modules():
        for k, p in list(m._parameters.items()):
            if p is None or not p.is_floating_point() or p.dtype == dtype or not select(m, k):
                continue
            if id(p) not in memo:
                memo[id(p)] = torch.nn.Parameter(p.detach().to(dtype), requires_grad=p.requires_grad)
            m._parameters[k] = memo[id(p)]
        for k, b in list(m._buffers.items()):
            if b is not None and b.is_floating_point() and b.dtype != dtype and select(m, k):
                m._buffers[k] = b.to(dtype)


def _run(fake, tracker, load_dtype, torch_dtype, spec, method, batch, seq, ckpt,
         lora_rank, lora_targets, steps):
    import gc
    import torch
    from transformers import AutoModelForCausalLM
    prev = torch.get_default_dtype()
    with fake, tracker:
        torch.set_default_dtype(load_dtype)
        try:
            model = AutoModelForCausalLM.from_config(_hf_config(spec), attn_implementation="sdpa")
        finally:
            torch.set_default_dtype(prev)
        model.config.use_cache = False
        if method == "qlora":
            # prepare_model_for_kbit_training 흉내: 양자화 대상(decoder Linear)이 아닌 파라미터를 fp32로
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
            mods = {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
            model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank,
                                                     target_modules=mods[lora_targets], lora_dropout=0.0,
                                                     bias="none", task_type="CAUSAL_LM"))
        model.train()
        gc.collect()
        mem_model = tracker.cur

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-5, foreach=True)
        ids = torch.randint(0, spec.vocab, (batch, seq))
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
    return TraceResult(mem_model_gb=round(mem_model / GiB, 4), mem_steady_gb=round(steady / GiB, 4),
                       peak_alloc_gb=round(peak / GiB, 4))


def _hf_config(spec):
    """fitcfg ModelSpec → HF config (가중치 다운로드 없이 구조만)."""
    from transformers import LlamaConfig, Qwen2Config
    common = dict(vocab_size=spec.vocab, hidden_size=spec.hidden, intermediate_size=spec.intermediate,
                  num_hidden_layers=spec.n_layers, num_attention_heads=spec.n_heads,
                  num_key_value_heads=spec.n_kv_heads, tie_word_embeddings=spec.tied_embeddings,
                  max_position_embeddings=32768, head_dim=spec.head_dim)
    if "qwen" in spec.name.lower():
        return Qwen2Config(**common)
    return LlamaConfig(**common)


# ---------------------------------------------------------------------------
# 처음 보는 모델을 설계도(config.json)로 직접 측정 — `fitcfg measure-cpu`
# ---------------------------------------------------------------------------
QLORA_ACT_RATIO = 1.3     # QLoRA 학습 중 메모리 ≈ LoRA × 1.3 (gfarm A100, 8개 계열 실측: ckpt 끔 1.29±0.06, 켬 1.27±0.15)


def _auto_targets(model) -> list[str]:
    """attention 투영 행렬 이름을 모델에서 찾는다 (q·k·v·o 또는 합본 qkv_proj)."""
    import torch
    names = set()
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and (".self_attn." in n or ".attention." in n or ".attn." in n):
            names.add(n.rsplit(".", 1)[-1])
    return sorted(names) or ["q_proj", "k_proj", "v_proj", "o_proj"]


def trace_hf(cfg: dict, method: str, batch: int, seq: int, ckpt: bool, lora_rank: int = 16,
             steps: int = 2, attn_impl: str = "sdpa") -> TraceResult:
    """HF 설계도 dict로 모델을 가짜 텐서로 만들어 LoRA/Full 학습 스텝을 추적 (bf16)."""
    import gc
    import torch
    import transformers
    from torch._subclasses.fake_tensor import FakeTensorMode
    from peft.tuners import tuners_utils
    from transformers import AutoModelForCausalLM

    if method not in ("lora", "full"):
        raise ValueError("trace_hf는 lora/full만 직접 추적합니다 (QLoRA는 measure_cpu가 규칙으로 환산).")
    c = {k: v for k, v in cfg.items() if k not in ("architectures", "auto_map", "torch_dtype", "dtype")}
    hf_cfg = transformers.AutoConfig.for_model(c.pop("model_type"), **c)
    orig_to = torch.nn.Module.to
    orig_move = tuners_utils.BaseTunerLayer._move_adapter_to_device_of_base_layer
    torch.nn.Module.to = _fake_safe_to
    tuners_utils.BaseTunerLayer._move_adapter_to_device_of_base_layer = lambda self, *a, **k: None
    tracker = _make_tracker()()
    prev = torch.get_default_dtype()
    try:
        with FakeTensorMode(), tracker:
            torch.set_default_dtype(torch.bfloat16)
            try:
                try:
                    model = AutoModelForCausalLM.from_config(hf_cfg, attn_implementation=attn_impl)
                except ValueError:
                    # transformers 5.21+는 Gemma-2의 sdpa를 막아 두었다(5.20까지 허용, gfarm 실측도 sdpa).
                    # eager로 추적하면 S×S 점수 행렬이 잡혀 20%+ 과대평가 → eager로 만든 뒤 sdpa로 바꿔 추적.
                    # softcap은 점수에 대한 원소별 연산이라 sdpa 메모리에는 영향이 없으므로 뺀다.
                    if hasattr(hf_cfg, "attn_logit_softcapping"):
                        hf_cfg.attn_logit_softcapping = None
                    model = AutoModelForCausalLM.from_config(hf_cfg, attn_implementation="eager")
                    if attn_impl == "sdpa":
                        for sub in [model, *model.modules()]:
                            sc = getattr(sub, "config", None)
                            if sc is not None and hasattr(sc, "_attn_implementation"):
                                sc._attn_implementation = "sdpa"
            finally:
                torch.set_default_dtype(prev)
            model.config.use_cache = False
            if ckpt:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if method == "lora":
                from peft import LoraConfig, get_peft_model
                for p in model.parameters():
                    p.requires_grad_(False)
                if ckpt:
                    model.enable_input_require_grads()
                model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank,
                                                         target_modules=_auto_targets(model), lora_dropout=0.0,
                                                         bias="none", task_type="CAUSAL_LM"))
            model.train()
            gc.collect()
            mem_model = tracker.cur
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, foreach=True)
            ids = torch.randint(0, hf_cfg.vocab_size, (batch, seq))
            tracker.reset_peak()
            for _ in range(steps):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    out = model(input_ids=ids, labels=ids)
                out.loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                del out
                gc.collect()
            peak, steady = tracker.peak, tracker.cur
    finally:
        torch.nn.Module.to = orig_to
        tuners_utils.BaseTunerLayer._move_adapter_to_device_of_base_layer = orig_move
    return TraceResult(mem_model_gb=round(mem_model / GiB, 4), mem_steady_gb=round(steady / GiB, 4),
                       peak_alloc_gb=round(peak / GiB, 4))


def measure_cpu(name: str, method: str, batch: int, seq: int, ckpt: bool, lora_rank: int = 16) -> dict:
    """처음 보는 모델의 GPU 메모리를 GPU 없이 측정. 실측 검증: LoRA·Full 1~2% (8개 계열),
    QLoRA는 'CPU LoRA 측정 × 1.3 + 4bit 가중치' 규칙으로 5% 안팎."""
    from .autospec import fetch_config, spec_from_config
    from .memory import TrainConfig, get_constants, GB
    cfg = fetch_config(name)
    traced = trace_hf(cfg, "full" if method == "full" else "lora", batch, seq, ckpt, lora_rank)
    out = {"method": method, "traced_as": "full" if method == "full" else "lora",
           "lora_model_gb": traced.mem_model_gb, "lora_peak_gb": traced.peak_alloc_gb}
    if method == "qlora":
        from .memory import estimate
        spec = spec_from_config(cfg, name)
        w_q = estimate(TrainConfig(model=name, method="qlora", lora_rank=lora_rank), spec).weights / GB
        act = traced.peak_alloc_gb - traced.mem_model_gb
        out.update(peak_alloc_gb=round(w_q + QLORA_ACT_RATIO * act, 3), qlora_weights_gb=round(w_q, 3),
                   rule=f"4bit 가중치 {w_q:.2f}GB + LoRA 학습 중 메모리 {act:.2f}GB × {QLORA_ACT_RATIO}")
    else:
        out["peak_alloc_gb"] = traced.peak_alloc_gb
    oh = get_constants(method).overhead_gb
    out["device_peak_gb"] = round(out["peak_alloc_gb"] + oh, 3)
    out["overhead_gb"] = oh
    return out
