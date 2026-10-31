"""처리량 예측 v2 (크기 효율·재계산 비용·optimizer 시간·non-flash attention) — GPU 불필요."""
import os, sys
from dataclasses import replace
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fitcfg import TrainConfig, get_gpu, get_model, load_calibration, reset_calibration
from fitcfg.models import EVAL_MODELS
from fitcfg.throughput import DEFAULT_TP, estimate_throughput, flops_per_token, get_tp_constants

C = replace(DEFAULT_TP, alpha=0.2, recompute=0.8, opt_bpp=48.0, opt8_bpp=96.0, attn_nonflash=16.0)


def setup_function(_):
    reset_calibration()


def _cfg(**kw):
    base = dict(model="llama-3.2-1b", method="full", batch_size=2, seq_len=2048, grad_checkpointing=False)
    base.update(kw)
    return TrainConfig(**base)


def test_optimizer_time_is_amortized_by_grad_accum():
    m, g = get_model("llama-3.2-1b"), get_gpu("a100-mig3g40")
    t1 = estimate_throughput(_cfg(), m, g, 1, C).tokens_per_s
    t16 = estimate_throughput(_cfg(), m, g, 16, C).tokens_per_s
    assert t16 > t1 * 1.03                                  # 1B Full: optimizer step이 커서 누적이 크면 빨라짐
    lo1 = estimate_throughput(_cfg(method="lora"), m, g, 1, C).tokens_per_s
    lo16 = estimate_throughput(_cfg(method="lora"), m, g, 16, C).tokens_per_s
    assert lo16 / lo1 < 1.01                                # LoRA는 학습 파라미터가 작아 차이 없음
    t8bit = estimate_throughput(_cfg(optim="adamw8bit"), m, g, 1, C).tokens_per_s
    assert t8bit < t1                                       # 8bit AdamW step이 더 느림 (실측)


def test_size_efficiency_and_recompute():
    g = get_gpu("a100-mig3g40")
    small = estimate_throughput(_cfg(method="lora"), get_model("llama-3.2-1b"), g, 1, C)
    small0 = estimate_throughput(_cfg(method="lora"), get_model("llama-3.2-1b"), g, 1, replace(C, alpha=0.0))
    assert small.tokens_per_s < small0.tokens_per_s         # 작은 모델은 효율 감소
    big = get_model("llama-3.1-8b")
    a = estimate_throughput(_cfg(model="llama-3.1-8b", method="lora"), big, g, 1, C).tokens_per_s
    a0 = estimate_throughput(_cfg(model="llama-3.1-8b", method="lora"), big, g, 1, replace(C, alpha=0.0)).tokens_per_s
    assert abs(a / a0 - 1) < 1e-9                           # 기준 모델(Llama-8B)은 그대로
    f_on = flops_per_token(_cfg(grad_checkpointing=True), big, C).total
    f_off = flops_per_token(_cfg(grad_checkpointing=False), big, C).total
    f_on1 = flops_per_token(_cfg(grad_checkpointing=True), big, replace(C, recompute=1.0)).total
    assert f_off < f_on < f_on1


def test_nonflash_attention_cost_only_without_flash():
    m = get_model("llama-3.2-1b")
    with_flash = flops_per_token(_cfg(flash_attn=True), m, C).total
    no_flash = flops_per_token(_cfg(flash_attn=False), m, C).total
    assert no_flash > with_flash and flops_per_token(_cfg(flash_attn=True), m).total == with_flash


def test_q_dim_uses_real_head_dim():
    q = EVAL_MODELS["qwen3-4b"]
    assert q.head_dim == 128 and q.q_dim == 4096 and q.hidden == 2560
    assert get_model("llama-3.1-8b").q_dim == 4096


def test_calibration_has_default_for_unmeasured_gpus():
    load_calibration(os.path.join(ROOT, "data", "calibration.json"))
    d, mig = get_tp_constants("rtx4090"), get_tp_constants("a100-mig3g40")
    assert d.eff_lora == mig.eff_lora and d.band >= 1.5    # 측정 안 한 GPU: 같은 세대 실측 상수 + 넓은 범위
    assert mig.opt_bpp > 0 and 0 < mig.recompute <= 1
