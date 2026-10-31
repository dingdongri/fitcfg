"""파인튜닝 GPU 메모리 예측 엔진 (cost model, v0.2).

이 파일이 프로젝트의 심장이다. GPU 없이(순수 계산으로) 동작한다.

핵심 아이디어: 파인튜닝 1 스텝에서 GPU가 쓰는 peak 메모리를 아래 항으로 분해한다.

    peak = weights + optimizer + max( activations + logits ,  gradients + optimizer 임시 ) + overhead
           └─ 구조로 계산 ─┘        └── 역전파 중 peak ──┘   └── optimizer step 중 peak ─┘

- weights / gradients / optimizer : 파라미터 수 × dtype 바이트로 거의 정확히 계산된다.
- activations : B·S·L·(act_h·H + act_i·I)   (체크포인팅: B·S·L·ckpt_h·H + 1개 레이어 전체)
- logits      : logit_bytes · B · S · V   ← vocab이 128k~152k라 매우 큼
- optimizer 임시: opt_temp · 학습 파라미터 수 (AdamW foreach 중간 텐서; Full-FT에서 peak를 결정하기도 함)
- overhead    : CUDA 컨텍스트 + 할당자 캐시(reserved−allocated) 등 고정 비용

v0.1 → v0.2 (FakeTensor 추적으로 발견, 2026-09-25)
    - 활성값을 H 하나가 아니라 H와 I(MLP 폭)로 분리 → I/H 비율이 다른 모델(Qwen 5.4, Llama-8B 3.5)로 일반화
    - 체크포인팅 = 레이어 입력 L개 + 재계산 중인 1개 레이어 (비례 가정 폐기)
    - Full-FT peak는 합이 아니라 max(역전파 시점, optimizer step 시점)
    - v0.1 상수는 LoRA 활성값을 약 절반으로, logits를 약 40% 과소평가했었다

"상수로 근사" 하는 부분은 CostConstants 에 모여 있고, scripts/calibrate.py 가
실측 CSV로 이 값을 fit 해서 data/calibration.json 으로 저장한다.
    >>> from fitcfg import load_calibration
    >>> load_calibration("data/calibration.json")   # 이후 estimate()가 보정값 사용

예측 대상(= 실측의 어떤 숫자와 비교하나):
    device_peak = CUDA 컨텍스트 + torch.cuda.max_memory_reserved()
    → nvidia-smi 에 보이는 값에 가깝고, OOM 여부를 실제로 결정하는 숫자.

v0 → v0.1 변경점
    1) logits 항 추가 (Llama-3 1B에서는 사실상 가장 큰 항)
    2) QLoRA: 임베딩/lm_head는 양자화되지 않음 + peft가 fp32로 올림 → 분리 계산
    3) LoRA 어댑터: GQA(k/v 차원) 반영, peft 기본값대로 fp32
    4) Full-FT: fp32 master weight 유무(master_weights)를 구분
    5) 상수를 CostConstants 로 묶고 method별 보정값을 로드할 수 있게 함
    6) features(): 보정(calibrate.py)에 필요한 '기저(basis)' 값을 노출

멀티 GPU / DeepSpeed 항(DDP 버킷, ZeRO-3 live 레이어, offload 버킷)은 아직 실측이 없는
문헌 기반 가정이다 → A100/멀티 GPU를 쓸 수 있게 되면 측정해서 확인할 것.

알려진 한계
    - overhead(CUDA 컨텍스트, 할당자 캐시/단편화)는 GPU 실측으로만 보정 가능 (Colab 측정 필요).
    - QLoRA 활성값은 bitsandbytes 없이 에뮬레이션한 추적값 기반 → GPU 실측으로 확인 필요.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, fields, replace
from pathlib import Path
from typing import Literal
from .models import ModelSpec

GB = 1024 ** 3

# dtype별 파라미터당 바이트
BYTES = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "int8": 1.0, "nf4": 0.5, "int4": 0.5}


# ---------------------------------------------------------------------------
# 보정 대상 상수
# ---------------------------------------------------------------------------
@dataclass
class CostConstants:
    """보정(fit)하는 상수 묶음.

    기본값의 출처 (2026-09-25):
      - 활성값·logits·optimizer 임시 버퍼: **FakeTensor 추적(fitcfg/trace.py)으로 실제 HF 코드가
        할당하는 텐서를 1B~8B 규모에서 잰 값**으로 fit (scripts/trace_sweep.py → calibrate.py)
      - overhead: 문헌값. CUDA 컨텍스트·할당자 캐시는 GPU 실측으로만 보정 가능.
    """
    # 체크포인팅 없을 때 레이어·토큰당 저장 활성값 (bytes) = act_h·H + act_i·I
    act_h: float = 34.0
    act_i: float = 4.0
    # 체크포인팅 시 레이어·토큰당 저장되는 레이어 입력 (bytes per H), + 재계산 중인 1개 레이어는 act_h/act_i
    ckpt_h: float = 2.0
    # logits 관련 바이트 / (B·S·V 원소 1개): bf16 logits + fp32 upcast + log_softmax + 역전파 grad
    logit_bytes: float = 12.0
    # optimizer step 중 임시 버퍼 (학습 파라미터당 bytes). torch AdamW(foreach)의 중간 텐서
    opt_temp: float = 2.0
    # FlashAttention이 없는 GPU(T4 등)에서 저장되는 attention 행렬 바이트 / (B·heads·S² 원소 1개).
    # Flash GPU에서는 기저가 0이라 영향 없음. Colab T4 실측으로 발견·보정 (2026-10-01)
    attn_sq_bytes: float = 2.0
    # attention 폭(q_dim = heads × head_dim)이 hidden과 다른 모델(Qwen3-4B: 4096 vs 2560)의 추가 저장 활성값
    # (bytes / 토큰·레이어·차이 1). 보정 모델은 q_dim = hidden이라 0 → 보정값에 영향 없음.
    # FakeTensor 절제 실험(H 고정, head 수만 변경)으로 측정: 10.1 B (q 출력·RoPE·attention 출력 등 bf16 약 5개)
    act_q: float = 10.0
    # 최종 logits softcapping(Gemma-2: tanh(x/30)·30)이 추가로 저장하는 바이트 / (B·S·V 원소 1개)
    softcap_bytes: float = 2.0
    # 층마다 정규화가 2개 더 있는 구조(Gemma-2·3의 post-attention·post-FFN norm)가 저장하는 추가 활성값
    # (bytes / 토큰·레이어·hidden·추가 norm 1개). 절제 실험: Gemma-2 층 = 같은 크기 Llama 층 + 8.0 B/H (norm 2개)
    norm_bytes: float = 4.0
    # NF4 양자화 부가 메모리(파라미터당 바이트). double-quant 기준 ≈ 1/64 + 4/(64·256)
    nf4_extra: float = 0.016
    # QLoRA에서 양자화되지 않는 임베딩/lm_head의 파라미터당 바이트.
    # peft.prepare_model_for_kbit_training 이 bf16 파라미터를 fp32로 올리므로 4.0
    qlora_nonquant_bytes: float = 4.0
    # CUDA 컨텍스트 + 할당자 캐시 등 고정 오버헤드 (GB) — GPU 실측으로만 보정
    overhead_gb: float = 1.3

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CostConstants":
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in names})


# 식 설계 당시의 손 추정값 (비교용으로 보존)
GUESS_CONSTANTS = CostConstants()
DEFAULT_CONSTANTS = CostConstants()
# method별 기본값. FakeTensor 추적으로 fit 한 fitcfg/priors.json 에서 채운다 (없으면 DEFAULT_CONSTANTS).
DEFAULT_BY_METHOD: dict[str, CostConstants] = {}
PRIORS_INFO: dict = {}


def _load_priors() -> None:
    """fitcfg/priors.json (scripts/fit_trace_priors.py 가 생성) → 기본 상수.
    overhead_gb 는 '추적 상수항 + CUDA 컨텍스트 사전값' — GPU 실측 보정 전까지의 추정."""
    global DEFAULT_CONSTANTS
    p = Path(__file__).with_name("priors.json")
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    names = {f.name for f in fields(CostConstants)}
    for k, v in d.get("constants", {}).items():
        c = replace(GUESS_CONSTANTS, **{n: float(x) for n, x in v.items() if n in names})
        if k == "default":
            DEFAULT_CONSTANTS = c
        else:
            DEFAULT_BY_METHOD[k] = c
    PRIORS_INFO.update({k: d.get(k) for k in ("source", "created", "n_rows", "heldout_model_mape")})


_load_priors()

# 현재 활성화된 보정값: {"default": CostConstants, "qlora": ..., "lora": ..., "full": ...}
_ACTIVE: dict[str, CostConstants] = {}
# 보정 리포트가 권장한 안전계수 (없으면 기본 0.90)
DEFAULT_SAFETY = 0.90
_SAFETY: list[float] = []
_CALIB_INFO: dict = {}


def get_default_safety() -> float:
    return _SAFETY[0] if _SAFETY else DEFAULT_SAFETY


def calibration_info() -> dict:
    """현재 로드된 보정 정보 요약 (UI 표시용). 보정 전이면 빈 dict."""
    return dict(_CALIB_INFO)


def get_constants(method: str) -> CostConstants:
    """method별 보정값 → 없으면 공통 보정값 → 없으면 사전추정값."""
    return (_ACTIVE.get(method) or _ACTIVE.get("default") or DEFAULT_BY_METHOD.get(method)
            or DEFAULT_CONSTANTS)


def load_calibration(path: str | Path) -> dict[str, CostConstants]:
    """calibrate.py 가 만든 calibration.json 을 로드해서 활성화한다."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    _ACTIVE.clear()
    for method, d in data["constants"].items():
        _ACTIVE[method] = CostConstants.from_dict(d)
    from .throughput import load_tp_calibration      # 처리량 보정값도 같은 파일에 있음
    load_tp_calibration(path)
    _SAFETY.clear()
    if data.get("suggested_safety"):
        _SAFETY.append(float(data["suggested_safety"]))
    _CALIB_INFO.clear()
    mets = data.get("metrics", {})
    ho = mets.get("cv_model") or mets.get("cv_loo") or {}
    _CALIB_INFO.update(path=str(path), created=data.get("created"), n_rows=data.get("n_rows"),
                       gpu_filter=data.get("gpu_filter"), heldout_mape=ho.get("mape"),
                       synthetic="synthetic_truth" in data,
                       throughput_gpus=sorted(k for k in (data.get("throughput") or {}) if k != "default"))
    return dict(_ACTIVE)


def reset_calibration() -> None:
    """보정값을 버리고 사전추정값으로 되돌린다 (테스트용)."""
    _ACTIVE.clear()
    _SAFETY.clear()
    _CALIB_INFO.clear()
    from .throughput import reset_tp_calibration
    reset_tp_calibration()


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
Method = Literal["full", "lora", "qlora"]
Optim = Literal["adamw", "adamw8bit"]

# LoRA를 붙이는 대상 행렬
LORA_TARGETS = {
    "attn": ("q", "k", "v", "o"),
    "all": ("q", "k", "v", "o", "gate", "up", "down"),
}


@dataclass
class TrainConfig:
    model: str
    method: Method = "qlora"
    gpus: int = 1
    batch_size: int = 1              # per-GPU micro-batch
    seq_len: int = 2048
    grad_accum: int = 1              # 메모리엔 영향 적음(참고용)
    grad_checkpointing: bool = True
    optim: Optim = "adamw"
    zero_stage: int = 0              # 0/1/2/3 (FSDP full-shard ~ 3)
    cpu_offload_optim: bool = False
    cpu_offload_params: bool = False
    lora_rank: int = 16
    lora_targets: str = "attn"       # "attn"(q,k,v,o) | "all"(+gate,up,down)
    # Full-FT에서 fp32 master weight를 두는가.
    # None=자동: DeepSpeed ZeRO(>=1)면 True(혼합정밀), 아니면 False(순수 bf16 — 측정 하네스와 동일)
    master_weights: bool | None = None
    # 16비트 종류. bf16 미지원 GPU(T4 등)는 "fp16".
    # fp16 + Full-FT(ZeRO 없음)는 HF Trainer 방식대로 fp32 가중치 + fp16 autocast(AMP)로 계산한다
    # (fp16 가중치를 직접 학습하면 GradScaler가 동작하지 않음).
    dtype: str = "bf16"
    # FlashAttention 사용 가능 여부 (GPU에 따라 recommend가 설정). False면 attention 행렬 S² 항이 생긴다.
    flash_attn: bool = True


@dataclass
class MemoryBreakdown:
    weights: float = 0.0
    gradients: float = 0.0
    optimizer: float = 0.0
    activations: float = 0.0
    logits: float = 0.0
    overhead: float = 0.0
    # peak가 생기는 시점: "fwd_bwd"(활성값+logits가 최대) | "optim_step"(grad+optimizer 임시버퍼가 최대)
    phase: str = "fwd_bwd"
    other_phase_total: float = 0.0     # 다른 시점의 합 (참고용)

    @property
    def total(self) -> float:
        return (self.weights + self.gradients + self.optimizer
                + self.activations + self.logits + self.overhead)

    def as_gb(self) -> dict:
        d = {k: round(v / GB, 3) for k, v in asdict(self).items() if isinstance(v, float)}
        d["total"] = round(self.total / GB, 3)
        return d


@dataclass
class Features:
    """보정용 분해 (bytes). 구조로 계산되는 항(known_*)과, 보정 상수에 곱해질 기저(basis)."""
    known_weights: float          # 가중치 (샤딩/offload 반영)
    known_gradients: float        # 역전파가 끝났을 때의 grad
    known_optimizer: float        # optimizer 상태 (m, v, master)
    known_persistent: float       # 항상 잡혀 있는 추가 버퍼 (DDP gradient bucket 등)
    act_h_basis: float            # B·S·(L 또는 1)·H → × act_h
    act_i_basis: float            # B·S·(L 또는 1)·I → × act_i
    ckpt_basis: float             # 체크포인팅: B·S·L·H → × ckpt_h
    logit_basis: float            # B·S·V → × logit_bytes
    temp_basis: float             # GPU에서 optimizer step 하는 학습 파라미터 수 → × opt_temp
    ckpt: bool
    attn_sq_basis: float = 0.0
    act_q_basis: float = 0.0          # B·S·(활성 레이어)·(q_dim − hidden)
    softcap_basis: float = 0.0        # B·S·V (logits softcapping 모델만)
    norm_basis: float = 0.0           # B·S·(활성 레이어)·hidden·(추가 norm 수)    # Flash 없는 GPU: (L 또는 1)·B·heads·S² → × attn_sq_bytes

    @property
    def known(self) -> float:
        return self.known_weights + self.known_gradients + self.known_optimizer + self.known_persistent

    def fwd_bwd(self, c: "CostConstants") -> float:
        """활성값·logits가 최대인 시점 (overhead 제외)."""
        return (self.known_weights + self.known_optimizer + self.known_persistent
                + c.act_h * self.act_h_basis + c.act_i * self.act_i_basis
                + c.ckpt_h * self.ckpt_basis + c.logit_bytes * self.logit_basis
                + c.attn_sq_bytes * self.attn_sq_basis
                + c.act_q * self.act_q_basis + c.softcap_bytes * self.softcap_basis
                + c.norm_bytes * self.norm_basis)

    def optim_step(self, c: "CostConstants") -> float:
        """역전파 직후 optimizer step 시점: grad 전체 + optimizer 임시 버퍼 (overhead 제외)."""
        return (self.known_weights + self.known_optimizer + self.known_persistent
                + self.known_gradients + c.opt_temp * self.temp_basis)


# ---------------------------------------------------------------------------
# 계산
# ---------------------------------------------------------------------------
def adapter_params(m: ModelSpec, cfg: TrainConfig) -> int:
    """LoRA 어댑터 학습 파라미터 수. 행렬(in→out)마다 A(r×in) + B(out×r) = r·(in+out)."""
    H, I, KV, Q, r = m.hidden, m.intermediate, m.kv_dim, m.q_dim, cfg.lora_rank
    shapes = {"q": (H, Q), "k": (H, KV), "v": (H, KV), "o": (Q, H),
              "gate": (H, I), "up": (H, I), "down": (I, H)}
    per_layer = sum(r * (i + o) for t in LORA_TARGETS[cfg.lora_targets]
                    for (i, o) in [shapes[t]])
    return per_layer * m.n_layers


def _uses_master_weights(cfg: TrainConfig) -> bool:
    if cfg.master_weights is not None:
        return cfg.master_weights
    return cfg.zero_stage >= 1


def features(cfg: TrainConfig, model: ModelSpec,
             c: CostConstants | None = None) -> Features:
    """설정 하나를 '구조로 계산되는 부분'과 '보정 상수에 곱해질 기저'로 분해."""
    c = c or get_constants(cfg.method)
    P = model.n_params
    w = g = o = 0.0

    if cfg.method == "full" and cfg.dtype == "fp16" and not _uses_master_weights(cfg):
        # fp16 AMP: fp32 가중치(4) + fp32 grad(4) + fp32 Adam m,v(8) / 8bit(2)
        w = P * 4.0
        g = P * 4.0
        o = P * (2.0 if cfg.optim == "adamw8bit" else 8.0)
    elif cfg.method == "full":
        w = P * BYTES["bf16"]
        g = P * BYTES["bf16"]
        if _uses_master_weights(cfg):
            # 혼합정밀(DeepSpeed): fp32 master(4) + fp32 Adam m,v(8) / 8bit(2)
            o = P * (4.0 + (2.0 if cfg.optim == "adamw8bit" else 8.0))
        else:
            # 순수 bf16: Adam m,v 가 파라미터 dtype(bf16) → 2×2 / 8bit(2)
            o = P * (2.0 if cfg.optim == "adamw8bit" else 4.0)
    else:
        A = adapter_params(model, cfg)
        if cfg.method == "qlora":
            w = (model.linear_params * (BYTES["nf4"] + c.nf4_extra)
                 + model.embed_params * c.qlora_nonquant_bytes)
        else:  # lora: base를 bf16로 로드
            w = P * BYTES["bf16"]
        # peft 기본값(autocast_adapter_dtype=True): 어댑터는 fp32 → weight 4, grad 4
        w += A * 4.0
        g = A * 4.0
        o = A * (2.0 if cfg.optim == "adamw8bit" else 8.0)

    # --- ZeRO / FSDP 샤딩: N GPU로 나눔 ---
    N = max(cfg.gpus, 1)
    trainable = P if cfg.method == "full" else adapter_params(model, cfg)
    persistent = 0.0
    if N > 1 and cfg.zero_stage == 0:
        # PyTorch DDP는 all-reduce용 gradient bucket을 항상 따로 잡는다(gradient_as_bucket_view=False 기본)
        # (멀티 GPU 실측 전까지는 문헌 기반 가정)
        persistent = g
    if cfg.zero_stage >= 1:
        o /= N
        trainable /= N               # optimizer step도 샤드만큼만
    if cfg.zero_stage in (1, 2) and N > 1:
        # DeepSpeed ZeRO-1/2는 갱신된 파라미터를 모으는 all-gather 버퍼(기본 allgather_bucket_size 5e8 원소, 16bit)를 잡는다.
        # Kaggle T4 ×2 실측(Llama-3.2-1B Full ZeRO-2): 이 항 없이 −9%(13.05 vs 14.34GB) → 넣으면 −2.5%
        persistent += min(5e8, P) * BYTES["bf16"]
    if cfg.zero_stage >= 2:
        g /= N
    # ZeRO-3에서 forward/backward 중 all-gather로 잠시 모이는 파라미터 (현재+prefetch ≈ 2개 레이어)
    live_layers = 2 * (model.linear_params / model.n_layers) * BYTES["bf16"]
    if cfg.zero_stage >= 3:
        w = w / N + live_layers   # ZeRO-3 / FSDP full-shard 는 파라미터도 샤딩

    # --- CPU offload: GPU에서 제외 ---
    if cfg.cpu_offload_optim:
        o = 0.0
        trainable = 0                # optimizer step이 CPU에서 일어남
        if cfg.zero_stage >= 2 and cfg.method == "full":
            # ZeRO-Offload: 줄어든(reduce) gradient는 곧바로 CPU로 옮겨진다 → GPU엔 버킷만 남음
            g = min(g, live_layers)
    if cfg.cpu_offload_params:
        w = live_layers

    BS = cfg.batch_size * cfg.seq_len
    L, H, I = model.n_layers, model.hidden, model.intermediate
    n_act = 1 if cfg.grad_checkpointing else L      # 체크포인팅: 재계산 중인 1개 레이어만 전체 활성값
    return Features(
        known_weights=w, known_gradients=g, known_optimizer=o, known_persistent=persistent,
        act_h_basis=float(BS * n_act * H), act_i_basis=float(BS * n_act * I),
        ckpt_basis=float(BS * L * H) if cfg.grad_checkpointing else 0.0,
        logit_basis=float(BS * model.vocab), temp_basis=float(trainable),
        ckpt=cfg.grad_checkpointing,
        attn_sq_basis=0.0 if cfg.flash_attn else
        float(n_act * cfg.batch_size * model.n_heads * cfg.seq_len ** 2),
        act_q_basis=float(BS * n_act * (model.q_dim - H)),
        softcap_basis=float(BS * model.vocab) if model.logit_softcap else 0.0,
        norm_basis=float(BS * n_act * H * model.extra_norms),
    )


# DeepSpeed ZeRO-3(파라미터를 층마다 모았다 버림)는 할당기 캐시가 크게 쪼개진다.
# gfarm 실측(2026-10-31, Llama-3.2-1B Full · ZeRO-3 + 파라미터·optimizer offload): 할당 5.0GB인데 예약 8.8GB(×1.76).
# 실측 1건이라 보수적으로 '할당 예측 × 0.75'를 오버헤드에 더한다 (예측 10.4GB vs 실측 9.4GB).
ZERO3_FRAG = 0.75


def estimate(cfg: TrainConfig, model: ModelSpec,
             c: CostConstants | None = None) -> MemoryBreakdown:
    """설정 하나에 대한 per-GPU 예측 peak 메모리 분해 (bytes).

    peak = 가중치 + optimizer 상태 + max(활성값 + logits,  gradient + optimizer 임시버퍼) + overhead
    두 시점 중 큰 쪽이 peak. 반환되는 분해는 **peak 시점의 구성**이다(합계 = peak).
    """
    c = c or get_constants(cfg.method)
    f = features(cfg, model, c)
    fwd, step = f.fwd_bwd(c), f.optim_step(c)
    act = (c.act_h * f.act_h_basis + c.act_i * f.act_i_basis + c.ckpt_h * f.ckpt_basis
           + c.attn_sq_bytes * f.attn_sq_basis + c.act_q * f.act_q_basis + c.norm_bytes * f.norm_basis)
    overhead = c.overhead_gb * GB
    if cfg.zero_stage >= 3:
        overhead += ZERO3_FRAG * max(fwd, step)
    common = dict(weights=f.known_weights, overhead=overhead)
    if fwd >= step:
        return MemoryBreakdown(**common, gradients=f.known_persistent, optimizer=f.known_optimizer,
                               activations=act,
                               logits=c.logit_bytes * f.logit_basis + c.softcap_bytes * f.softcap_basis,
                               phase="fwd_bwd", other_phase_total=step + overhead)
    return MemoryBreakdown(**common, gradients=f.known_persistent + f.known_gradients,
                           optimizer=f.known_optimizer + c.opt_temp * f.temp_basis,
                           phase="optim_step", other_phase_total=fwd + overhead)


def cpu_ram_needed_gb(cfg: TrainConfig, model: ModelSpec) -> float:
    """CPU offload 시 필요한 CPU RAM(노드 전체, GB). offload가 없으면 0.
    DeepSpeed CPU Adam: fp32 master(4) + m,v(8) + fp32/bf16 grad 버퍼(4) ≈ 16 bytes/param,
    파라미터 offload 시 bf16 사본(2) 추가, 모델을 CPU로 읽어 들이는 bf16 사본(2), 프로세스 고정분 4GB.
    gfarm 실측(2026-10-31): Qwen2.5-3B ZeRO-2 offload가 컨테이너 CPU 한도 64GB를 넘어 강제 종료
    (이전 식 55GB로 과소) → pinned memory·임시 버퍼 여유 25%로 올림 (새 식 74GB)."""
    if not (cfg.cpu_offload_optim or cfg.cpu_offload_params):
        return 0.0
    trainable = model.n_params if cfg.method == "full" else adapter_params(model, cfg)
    b = trainable * 16.0 + (model.n_params * 2.0 if cfg.cpu_offload_params else 0.0) + model.n_params * 2.0
    return b * 1.25 / GB + 4.0


def fits(cfg: TrainConfig, model: ModelSpec, gpu_mem_gb: float,
         safety: float | None = None) -> bool:
    """예측 peak 메모리가 GPU 용량(안전마진 포함)에 들어가는지.
    safety 생략 시: 보정 리포트의 권장값(있으면) 또는 0.90."""
    safety = get_default_safety() if safety is None else safety
    return estimate(cfg, model).total <= gpu_mem_gb * GB * safety
