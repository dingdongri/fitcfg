"""HF Trainer 플러그인 — 학습 코드에 한 줄 추가하면, 학습 시작 직전에 메모리를 예측해 OOM을 미리 알려 준다.

    from fitcfg.integrations.hf import FitcfgCallback
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      callbacks=[FitcfgCallback(seq_len=2048)])          # strict=True면 OOM 예상 시 학습을 멈춤
    trainer.train()

하는 일
  - 학습 시작 직전(on_train_begin): 모델·Trainer 설정에서 방법(Full/LoRA/QLoRA), micro-batch, checkpointing,
    GPU 수·ZeRO/FSDP, optimizer를 읽어 GPU당 peak 메모리를 예측 → 안전 한도를 넘으면 경고(+ 들어가는 추천 설정 제시),
    strict=True면 RuntimeError로 중단 (가중치만 올라간 상태라 몇 초 만에 끝남 — 몇 분 돌다 OOM 나는 것보다 낫다)
  - 학습 중: 앞 warmup 스텝을 뺀 실제 처리량(tok/s)을 잰다
  - 학습 끝(on_train_end): 예측 vs 실측(peak 메모리·처리량)을 fitcfg_run.json에 남긴다 → 이후 보정 데이터로 쓸 수 있음
GPU가 없거나 모르는 GPU면 예측을 건너뛰고 학습은 그대로 진행한다 (플러그인 때문에 학습이 깨지면 안 됨).
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    from transformers import TrainerCallback
except ImportError as e:  # pragma: no cover
    raise ImportError("fitcfg.integrations.hf는 transformers가 필요합니다: pip install transformers") from e

from ..autospec import classify, spec_from_config
from ..hardware import GPUSpec, custom_gpu, get_gpu, match_gpu
from ..memory import GB, TrainConfig, estimate, get_default_safety
from ..models import MODELS, ModelSpec
from ..throughput import estimate_throughput

_8BIT_OPTIMS = ("8bit", "bnb")
_ATTN_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "query_key_value", "c_attn", "dense", "out_proj"}


@dataclass
class Detected:
    """모델·Trainer 설정에서 읽어 낸 학습 조건."""
    model_name: str
    method: str                 # full / lora / qlora
    batch_size: int
    grad_accum: int
    seq_len: int
    grad_checkpointing: bool
    gpus: int
    zero_stage: int
    optim: str                  # adamw / adamw8bit
    lora_rank: int = 16
    lora_targets: str = "attn"
    dtype: str = "bf16"
    seq_len_guessed: bool = False


@dataclass
class Prediction:
    pred_mem_gb: float
    cap_gb: float
    gpu_mem_gb: float
    gpu_label: str
    fits: bool
    tokens_per_s: float | None
    structure: str
    margin: float
    breakdown: dict = field(default_factory=dict)
    suggestion: str | None = None


# ---------------------------------------------------------------------------
# 감지
# ---------------------------------------------------------------------------
def _model_config_dict(model) -> dict:
    base = model
    for attr in ("base_model", "model"):        # PeftModel → LoraModel → 원래 모델
        inner = getattr(base, attr, None)
        if inner is not None and hasattr(inner, "config"):
            base = inner
    cfg = getattr(base, "config", None) or getattr(model, "config", None)
    return cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg or {})


def _is_4bit(model, cfg: dict) -> bool:
    if getattr(model, "is_loaded_in_4bit", False):
        return True
    for m in (model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)):
        if m is not None and getattr(m, "is_loaded_in_4bit", False):
            return True
    q = cfg.get("quantization_config") or {}
    return bool(q.get("load_in_4bit") or q.get("_load_in_4bit"))


def _lora_info(model) -> tuple[bool, int, str]:
    pc = getattr(model, "peft_config", None)
    if not pc:
        return False, 16, "attn"
    c = pc.get("default") if isinstance(pc, dict) else pc
    if c is None and isinstance(pc, dict):
        c = next(iter(pc.values()))
    r = int(getattr(c, "r", 16) or 16)
    tm = getattr(c, "target_modules", None) or []
    if isinstance(tm, str):
        tgt = "all" if tm == "all-linear" else "attn"
    else:
        tgt = "attn" if set(tm) <= _ATTN_NAMES else "all"
    return True, r, tgt


def _zero_stage(args) -> int:
    if getattr(args, "fsdp", None):
        fs = args.fsdp
        txt = " ".join(str(x) for x in fs) if isinstance(fs, (list, tuple)) else str(fs)
        if "full_shard" in txt:
            return 3
        if "shard_grad_op" in txt:
            return 2
    ds = getattr(args, "deepspeed", None) or getattr(args, "hf_deepspeed_config", None)
    if ds:
        conf = ds
        if isinstance(ds, (str, Path)) and Path(ds).exists():
            conf = json.loads(Path(ds).read_text())
        elif hasattr(ds, "config"):
            conf = ds.config
        if isinstance(conf, dict):
            return int((conf.get("zero_optimization") or {}).get("stage", 0) or 0)
    return 0


def _ckpt_on(model) -> bool:
    """TrainingArguments 대신 model.gradient_checkpointing_enable()로 켠 경우도 잡는다."""
    m = model
    for _ in range(4):
        if getattr(m, "is_gradient_checkpointing", False):
            return True
        m = getattr(m, "base_model", None) or getattr(m, "model", None)
        if m is None:
            break
    return False


def detect(model, args, seq_len: int | None = None) -> Detected:
    cfg = _model_config_dict(model)
    is_lora, r, tgt = _lora_info(model)
    method = ("qlora" if _is_4bit(model, cfg) else "lora") if is_lora else "full"
    guessed = seq_len is None
    if guessed:   # 학습 데이터 길이는 Trainer가 모름 → 모델 최대 길이와 2048 중 작은 값 (경고로 알림)
        seq_len = min(int(cfg.get("max_position_embeddings") or 2048), 2048)
    optim = str(getattr(args, "optim", "adamw_torch"))
    return Detected(
        model_name=str(cfg.get("_name_or_path") or cfg.get("name_or_path") or cfg.get("model_type", "model")),
        method=method,
        batch_size=int(getattr(args, "per_device_train_batch_size", 8)),
        grad_accum=int(getattr(args, "gradient_accumulation_steps", 1)),
        seq_len=int(seq_len),
        grad_checkpointing=bool(getattr(args, "gradient_checkpointing", False)) or _ckpt_on(model),
        gpus=max(int(getattr(args, "world_size", 1) or 1), 1),
        zero_stage=_zero_stage(args),
        optim="adamw8bit" if any(k in optim.lower() for k in _8BIT_OPTIMS) else "adamw",
        lora_rank=r, lora_targets=tgt,
        dtype="fp16" if getattr(args, "fp16", False) else "bf16",
        seq_len_guessed=guessed,
    )


def _spec_and_structure(cfg: dict, name: str) -> tuple[ModelSpec, object]:
    from ..models import ALIASES
    key = ALIASES.get(name.lower().rstrip("/").split("/models--")[-1], name.lower())
    for k, m in MODELS.items():                  # 실측 보정한 모델(또는 같은 가중치의 공개 사본)이면 그 사양 그대로
        if m.name.lower() == name.lower() or k == key:
            return m, classify(cfg or {"model_type": "llama"}, verified=True)
    return spec_from_config(cfg, name), classify(cfg)


def _current_gpu() -> GPUSpec | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0)
        g = match_gpu(name)
        if g is None:
            total = torch.cuda.get_device_properties(0).total_memory / GB
            return custom_gpu(total)
        return g
    except Exception:
        return None


def predict(model, args, gpu: GPUSpec | None = None, seq_len: int | None = None,
            safety: float | None = None) -> tuple[Detected, Prediction | None]:
    """모델·TrainingArguments만으로 예측 (GPU 없으면 gpu 인자로 지정)."""
    d = detect(model, args, seq_len)
    g = gpu or _current_gpu()
    if g is None:
        return d, None
    cfg = _model_config_dict(model)
    spec, info = _spec_and_structure(cfg, d.model_name)
    margin = info.margin(d.method)
    if not g.bf16 and d.dtype == "bf16":
        d.dtype = "fp16"
    tc = TrainConfig(model=d.model_name, method=d.method, gpus=d.gpus, batch_size=d.batch_size, seq_len=d.seq_len,
                     grad_accum=d.grad_accum, grad_checkpointing=d.grad_checkpointing, optim=d.optim,
                     zero_stage=d.zero_stage, lora_rank=d.lora_rank, lora_targets=d.lora_targets,
                     dtype=d.dtype, flash_attn=g.flash)
    mem = estimate(tc, spec)
    s = safety if safety is not None else get_default_safety()
    pred = mem.total / GB * margin
    cap = g.mem_gb * s
    try:
        tps = estimate_throughput(tc, spec, g, d.grad_accum).tokens_per_s
    except Exception:
        tps = None
    p = Prediction(pred_mem_gb=round(pred, 2), cap_gb=round(cap, 2), gpu_mem_gb=g.mem_gb, gpu_label=g.label,
                   fits=pred <= cap, tokens_per_s=round(tps, 1) if tps else None, structure=info.kind,
                   margin=margin, breakdown=mem.as_gb())
    if not p.fits:
        p.suggestion = _suggest(d, spec, info, g, s)
    return d, p


def _suggest(d: Detected, spec: ModelSpec, info, g: GPUSpec, safety: float) -> str | None:
    """같은 global batch로 들어가는 1순위 설정 (recommend.plan 재사용)."""
    from .. import autospec
    from ..recommend import PlanRequest, plan
    key = next((k for k, m in MODELS.items() if m is spec), d.model_name)
    if key not in MODELS and key not in autospec._CACHE:
        autospec._CACHE[key] = (spec, info)          # 처음 보는 모델: 설계도로 만든 사양·구조 유형 그대로
    try:
        res = plan(PlanRequest(model=key, gpu=g, gpus=d.gpus, method=d.method, seq_len=d.seq_len,
                               global_batch=d.batch_size * d.grad_accum * d.gpus, lora_rank=d.lora_rank,
                               lora_targets=d.lora_targets, safety=safety), top_k=1)
    except Exception:
        return None
    if not res.best:
        return None
    b = res.best
    c = b.config
    hint = (f"per_device_train_batch_size={c.batch_size}, gradient_accumulation_steps={b.grad_accum}, "
            f"gradient_checkpointing={c.grad_checkpointing}")
    if c.optim == "adamw8bit" and d.optim != "adamw8bit":
        hint += ', optim="adamw_bnb_8bit"'
    return f"{b.reason} (예측 {b.pred_mem_gb:.1f}GB) → TrainingArguments({hint})"


# ---------------------------------------------------------------------------
# 콜백
# ---------------------------------------------------------------------------
class FitcfgCallback(TrainerCallback):
    """HF Trainer에 붙이는 fitcfg 플러그인.

    seq_len : 학습 시퀀스 길이 (패딩/잘라내기 길이). 모르면 생략 → 2048로 가정하고 경고
    strict  : True면 OOM 예상 시 학습 시작 전에 RuntimeError
    gpu     : GPU 프리셋 키(예: "rtx4090") 또는 GPUSpec. 생략하면 현재 GPU 자동 인식
    safety  : 안전계수 (기본: 보정값 0.895)
    log_path: 예측·실측 기록 파일 (None이면 기록 안 함). 상대경로면 output_dir 아래
    warmup_steps: 처리량 측정에서 뺄 앞 스텝 수
    """

    def __init__(self, seq_len: int | None = None, strict: bool = False, gpu=None, safety: float | None = None,
                 log_path: str | None = "fitcfg_run.json", warmup_steps: int = 3, calibration: str | None = "auto",
                 verbose: bool = True):
        self.seq_len = seq_len
        self.strict = strict
        self.gpu = get_gpu(gpu) if isinstance(gpu, str) else gpu
        self.safety = safety
        self.log_path = log_path
        self.warmup_steps = warmup_steps
        self.verbose = verbose
        self.detected: Detected | None = None
        self.prediction: Prediction | None = None
        self.measured: dict = {}
        self._t0 = None
        self._steps_timed = 0
        if calibration:
            self._load_calibration(calibration)

    @staticmethod
    def _load_calibration(calibration: str):
        from ..memory import load_calibration
        if calibration == "auto":
            from ..cli import DEFAULT_CALIB as p
        else:
            p = Path(calibration)
        if p.exists():
            load_calibration(p)

    def _say(self, msg: str):
        if self.verbose:
            print(f"[fitcfg] {msg}", flush=True)

    # -- 학습 시작 직전 ----------------------------------------------------
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        try:
            self.detected, self.prediction = predict(model, args, self.gpu, self.seq_len, self.safety)
        except Exception as e:      # 예측 실패가 학습을 막으면 안 됨
            self._say(f"예측을 건너뜁니다 ({type(e).__name__}: {e})")
            return
        d, p = self.detected, self.prediction
        if p is None:
            self._say("GPU를 찾지 못해 예측을 건너뜁니다 (FitcfgCallback(gpu='rtx4090')처럼 지정 가능).")
            return
        if getattr(state, "is_world_process_zero", True):
            self._say(f"{d.model_name} · {d.method} · micro-batch {d.batch_size} × 누적 {d.grad_accum} × GPU {d.gpus}"
                      f" · seq {d.seq_len}{' (가정)' if d.seq_len_guessed else ''} · checkpointing "
                      f"{'on' if d.grad_checkpointing else 'off'}" + (f" · ZeRO-{d.zero_stage}" if d.zero_stage else ""))
            self._say(f"예측 GPU당 peak {p.pred_mem_gb:.1f}GB / 안전 한도 {p.cap_gb:.1f}GB ({p.gpu_label})"
                      + (f" · 예상 처리량 {p.tokens_per_s:,.0f} tok/s" if p.tokens_per_s else "")
                      + (f" · 구조 {p.structure}, 여유 ×{p.margin:.2f}" if p.margin != 1.0 else ""))
            if d.seq_len_guessed:
                self._say("seq_len을 몰라 2048로 가정했습니다 → FitcfgCallback(seq_len=...)으로 지정하면 정확해집니다.")
        if not p.fits:
            msg = (f"OOM 예상: 예측 {p.pred_mem_gb:.1f}GB > 한도 {p.cap_gb:.1f}GB."
                   + (f" 들어가는 추천 설정: {p.suggestion}" if p.suggestion else
                      " 이 GPU 수로는 들어가는 설정이 없습니다 (QLoRA·GPU 추가·CPU offload 검토)."))
            if self.strict:
                self._write(args, status="stopped_predicted_oom")
                raise RuntimeError(f"[fitcfg] {msg}")
            self._say("⚠️ " + msg)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    # -- 처리량 측정 -------------------------------------------------------
    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step == self.warmup_steps:
            self._sync()
            self._t0 = time.perf_counter()
            self._steps_timed = 0
        elif self._t0 is not None and step > self.warmup_steps:
            self._steps_timed = step - self.warmup_steps

    @staticmethod
    def _sync():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    # -- 학습 끝 -----------------------------------------------------------
    def on_train_end(self, args, state, control, **kwargs):
        self._sync()
        m = {}
        if self._t0 is not None and self._steps_timed > 0 and self.detected:
            d = self.detected
            dt = time.perf_counter() - self._t0
            tokens = d.batch_size * d.seq_len * d.grad_accum * d.gpus * self._steps_timed
            m["tokens_per_s"] = round(tokens / dt, 1)
            m["steps_timed"] = self._steps_timed
        try:
            import torch
            if torch.cuda.is_available():
                m["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / GB, 3)
                m["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / GB, 3)
                # GPU 전체 사용량 = PyTorch 캐시 + CUDA 컨텍스트 등(non-torch). 실측 실험과 같은 방식으로 추정
                free, total = torch.cuda.mem_get_info()
                nontorch = max((total - free) - torch.cuda.memory_reserved(), 0) / GB
                m["nontorch_gb"] = round(nontorch, 3)
                m["device_peak_gb"] = round(m["peak_reserved_gb"] + nontorch, 3)
        except Exception:
            pass
        self.measured = m
        p = self.prediction
        if p and "device_peak_gb" in m:
            err = (p.pred_mem_gb - m["device_peak_gb"]) / m["device_peak_gb"] * 100
            self._say(f"실측 GPU peak {m['device_peak_gb']:.1f}GB vs 예측 {p.pred_mem_gb:.1f}GB ({err:+.1f}%)"
                      + (f" · 처리량 {m['tokens_per_s']:,.0f} tok/s (예측 {p.tokens_per_s:,.0f})"
                         if "tokens_per_s" in m and p.tokens_per_s else ""))
        self._write(args, status="finished")

    def _write(self, args, status: str):
        if not self.log_path:
            return
        path = Path(self.log_path)
        if not path.is_absolute():
            path = Path(getattr(args, "output_dir", ".") or ".") / path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "status": status,
                "detected": asdict(self.detected) if self.detected else None,
                "prediction": asdict(self.prediction) if self.prediction else None,
                "measured": self.measured,
                "note": "device_peak_gb = PyTorch peak reserved + CUDA 컨텍스트 등 (다른 프로세스가 같은 GPU를 쓰면 부정확)",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as e:
            self._say(f"기록 실패: {e}")
