"""처리량(tokens/s)·학습시간 예측 (Phase 4, v1).

아이디어: 처리량 = 실제로 달성하는 FLOPS / 토큰당 학습 FLOPs
    tokens/s ≈ peak_TFLOPS × 효율(method) × 활용도(마이크로배치 토큰 수) / FLOPs_per_token

- FLOPs_per_token 은 모델 구조로 계산한다 (Linear 행렬곱 + attention + lm_head).
    full   : forward 1 + backward 2 (입력 grad + 가중치 grad)
    (Q)LoRA: forward 1 + backward 1 (동결된 가중치의 grad는 계산 안 함)
    checkpointing: 레이어 forward 1회 재계산 추가
- 효율(MFU 비슷한 값)과 활용도 곡선은 **상수**라서 실측 tokens_per_s 로 보정한다(calibration.py).
- 처리량 예측은 메모리보다 본질적으로 부정확하다 → 항상 **범위(low~high)** 로 제시한다.
  보정 전 범위는 ×/÷1.6, 보정 후에는 held-out 잔차로 좁힌다.

v2 (2026-10-30, A100 MIG 실측 100개 + 실제 학습 번들 17개로 확인):
  - 크기 효율 α: 행렬이 작은 모델(1B·3B)은 GPU를 덜 채운다 → 효율 × (H·I / Llama-8B의 H·I)^α
  - 재계산 비용 γ: checkpointing 재계산은 forward보다 싸다(실측 γ≈0.8)
  - optimizer step 시간: 학습 파라미터 × (읽고 쓰는 바이트) ÷ 메모리 대역폭 — Full 미세조정에서 크고,
    grad_accum이 클수록 묻힌다 (v1은 이를 빼서 1B Full을 17~22% 낮게 예측)
  - FlashAttention이 없는 GPU(T4): attention이 메모리 대역폭에 묶여 FLOPs보다 훨씬 느림 → attention 비용 × κ
  모델 하나를 빼고 맞춘 오차: A100 MIG 12.6% → 5.0%, 실제 학습 번들 6.9% → 4.0% (최대 21.6% → 7.1%)
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, asdict, fields, replace
from pathlib import Path

from .hardware import GPUSpec
from .memory import TrainConfig, adapter_params
from .models import ModelSpec


@dataclass
class ThroughputConstants:
    """실측 tokens/s 로 보정하는 상수. 기본값 = 문헌 기반 사전추정."""
    eff_full: float = 0.38        # 달성 FLOPS / peak (재계산 포함 하드웨어 FLOPs 기준)
    eff_lora: float = 0.33
    eff_qlora: float = 0.20       # 4bit 역양자화 오버헤드
    t_half: float = 1024.0        # hidden=4096 모델에서 활용도가 50%가 되는 마이크로스텝 토큰 수
    ddp_eff: float = 0.90         # 멀티 GPU 데이터 병렬 확장 효율 (PCIe 가정)
    zero3_eff: float = 0.80       # ZeRO-3/FSDP (파라미터 all-gather 통신 추가)
    cpu_adam_params_per_s: float = 5e8   # ZeRO-Offload CPU Adam 속도 (파라미터/초)
    band: float = 1.6             # 예측 범위 배수: [est/band, est×band]
    # --- v2 항 (기본값 = v1과 같은 동작) ---
    alpha: float = 0.0            # 크기 효율 지수: 효율 × (H·I / 4096·14336)^alpha
    recompute: float = 1.0        # checkpointing 재계산 비용 (forward 대비)
    attn_nonflash: float = 1.0    # FlashAttention 없는 GPU의 attention 비용 배수
    opt_bpp: float = 0.0          # optimizer step: 학습 파라미터 1개당 메모리 이동 바이트 (AdamW)
    opt8_bpp: float = 0.0         # 〃 8bit AdamW

    def eff(self, method: str) -> float:
        return {"full": self.eff_full, "lora": self.eff_lora, "qlora": self.eff_qlora}[method]

    def with_eff(self, method: str, value: float) -> "ThroughputConstants":
        return replace(self, **{f"eff_{method}": value})

    @classmethod
    def from_dict(cls, d: dict) -> "ThroughputConstants":
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_TP = ThroughputConstants()
_ACTIVE_TP: dict[str, ThroughputConstants] = {}     # {"default": ..., "t4": ...} (GPU key별)


def get_tp_constants(gpu_key: str) -> ThroughputConstants:
    return _ACTIVE_TP.get(gpu_key) or _ACTIVE_TP.get("default") or DEFAULT_TP


def is_calibrated(gpu_key: str) -> bool:
    return gpu_key in _ACTIVE_TP


def load_tp_calibration(path: str | Path) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    _ACTIVE_TP.clear()
    for k, d in (data.get("throughput") or {}).items():
        _ACTIVE_TP[k] = ThroughputConstants.from_dict(d)


def reset_tp_calibration() -> None:
    _ACTIVE_TP.clear()


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------
@dataclass
class FlopsPerToken:
    forward: float        # 1회 forward
    total: float          # 1 학습 스텝(재계산 포함)의 하드웨어 FLOPs / 토큰


def flops_per_token(cfg: TrainConfig, m: ModelSpec, c: ThroughputConstants | None = None) -> FlopsPerToken:
    """토큰 1개 학습의 FLOPs. c를 주면 attention(FlashAttention 없을 때)·재계산 비용 배수를 반영한
    '시간 환산 FLOPs'를 돌려준다 (c 없으면 순수 FLOPs)."""
    H, KV, Q, I, L, V, S = m.hidden, m.kv_dim, m.q_dim, m.intermediate, m.n_layers, m.vocab, cfg.seq_len
    lin_layer = 2 * (2 * H * Q + 2 * H * KV + 3 * H * I)      # Linear 행렬곱 (2·params): q·o, k·v, MLP
    attn_layer = 2 * S * Q                                     # causal QKᵀ + AV 평균
    if c is not None and not cfg.flash_attn:
        attn_layer *= c.attn_nonflash
    head = 2 * V * H
    fwd_layers = L * (lin_layer + attn_layer)
    fwd = fwd_layers + head
    if cfg.method == "full":
        total = 3 * fwd
    else:
        # 동결 가중치: 입력 grad만 (Linear ×1), attention 역전파는 가중치가 없어 ×2
        total = fwd + L * (lin_layer + 2 * attn_layer) + head
        total += 6 * adapter_params(m, cfg)                   # 어댑터 (무시 가능한 크기)
    if cfg.grad_checkpointing:
        total += fwd_layers * (c.recompute if c is not None else 1.0)   # 레이어 forward 재계산
    return FlopsPerToken(forward=fwd, total=total)


# ---------------------------------------------------------------------------
# 처리량
# ---------------------------------------------------------------------------
@dataclass
class ThroughputEstimate:
    tokens_per_s: float           # 전체(모든 GPU 합) 학습 처리량
    low: float
    high: float
    step_time_s: float            # optimizer step 1회 (grad_accum 포함)
    calibrated: bool

    def hours_for(self, total_tokens: float) -> tuple[float, float, float]:
        """(예상, 빠르면, 느리면) 시간."""
        h = lambda tps: total_tokens / tps / 3600  # noqa: E731
        return h(self.tokens_per_s), h(self.high), h(self.low)


def utilization(tokens: float, m: ModelSpec, c: ThroughputConstants) -> float:
    """마이크로스텝 토큰이 적으면 GPU가 덜 찬다. 작은 모델일수록 더 많은 토큰이 필요."""
    t_half = c.t_half * (4096 / m.hidden)
    return tokens / (tokens + t_half)


def size_efficiency(m: ModelSpec, c: ThroughputConstants) -> float:
    """행렬이 작은 모델은 같은 토큰 수에서도 GPU를 덜 채운다 (Llama-8B 기준 1)."""
    return ((m.hidden * m.intermediate) / (4096 * 14336)) ** c.alpha


def optimizer_seconds(cfg: TrainConfig, m: ModelSpec, gpu: GPUSpec, c: ThroughputConstants) -> float:
    """optimizer step 1회(GPU에서) 시간: 메모리 대역폭에 묶인 작업."""
    if cfg.cpu_offload_optim:
        return 0.0
    trainable = m.n_params if cfg.method == "full" else adapter_params(m, cfg)
    if cfg.zero_stage >= 1:
        trainable /= max(cfg.gpus, 1)                          # ZeRO: optimizer 상태를 나눠 가짐
    bpp = c.opt8_bpp if cfg.optim == "adamw8bit" else c.opt_bpp
    return trainable * bpp / (gpu.mem_bw_gbs * 1e9)


def estimate_throughput(cfg: TrainConfig, m: ModelSpec, gpu: GPUSpec,
                        grad_accum: int = 1, c: ThroughputConstants | None = None
                        ) -> ThroughputEstimate:
    c = c or get_tp_constants(gpu.key)
    N = max(cfg.gpus, 1)
    T = cfg.batch_size * cfg.seq_len                           # GPU당 마이크로스텝 토큰
    f = flops_per_token(cfg, m, c)
    achieved = gpu.tflops_16bit * 1e12 * c.eff(cfg.method) * utilization(T, m, c) * size_efficiency(m, c)
    if cfg.cpu_offload_params:
        achieved *= 0.35                                       # 파라미터를 매 스텝 CPU↔GPU 이동
    micro_s = f.total * T / achieved

    # optimizer step 추가 비용: ZeRO-Offload(CPU Adam + PCIe 전송)
    trainable = m.n_params if cfg.method == "full" else adapter_params(m, cfg)
    opt_s = optimizer_seconds(cfg, m, gpu, c)
    if cfg.cpu_offload_optim:
        opt_s = trainable / N * (1 / c.cpu_adam_params_per_s + 8 / 12e9)
    step_s = grad_accum * micro_s + opt_s

    scale = 1.0
    if N > 1:
        per_doubling = c.zero3_eff if cfg.zero_stage >= 3 else c.ddp_eff
        scale = per_doubling ** math.log2(N)
    tps = N * T * grad_accum / step_s * scale
    return ThroughputEstimate(tokens_per_s=tps, low=tps / c.band, high=tps * c.band,
                              step_time_s=step_s / scale, calibrated=is_calibrated(gpu.key))
