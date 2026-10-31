"""5차 실측(2026-10-31)으로 고친 부분 — GPU 불필요."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fitcfg import (PlanRequest, TrainConfig, custom_gpu, estimate, get_model, load_calibration, plan,
                    reset_calibration)
from fitcfg.memory import GB, cpu_ram_needed_gb
from fitcfg.models import EVAL_MODELS

CAL = os.path.join(ROOT, "data", "calibration.json")


def setup_function(_):
    reset_calibration()
    load_calibration(CAL)


def test_zero3_offload_matches_measurement_conservatively():
    # 실측: Llama-3.2-1B Full, ZeRO-3 + 파라미터·optimizer offload, mb1·seq2048·ckpt → GPU peak 9.42GB
    cfg = TrainConfig(model="llama-3.2-1b", method="full", batch_size=1, seq_len=2048, grad_checkpointing=True,
                      zero_stage=3, cpu_offload_optim=True, cpu_offload_params=True, master_weights=True)
    p = estimate(cfg, get_model("llama-3.2-1b")).total / GB
    assert 9.42 <= p <= 9.42 * 1.2


def test_cpu_ram_estimate_rejects_case_that_was_killed():
    # 실측: Qwen2.5-3B ZeRO-2 offload가 CPU 한도 64GB 컨테이너에서 강제 종료
    cfg = TrainConfig(model="qwen2.5-3b", method="full", zero_stage=2, cpu_offload_optim=True)
    assert cpu_ram_needed_gb(cfg, get_model("qwen2.5-3b")) > 64
    r = plan(PlanRequest(model="qwen2.5-3b", gpu=custom_gpu(16), method="full", cpu_ram_gb=60))
    assert r.best is None


def test_long_sequence_and_structure_terms():
    # 긴 시퀀스 실측: Llama-3.1-8B QLoRA seq8192 ckpt → 28.39GB
    cfg = TrainConfig(model="llama-3.1-8b", method="qlora", batch_size=1, seq_len=8192, grad_checkpointing=True)
    assert abs(estimate(cfg, get_model("llama-3.1-8b")).total / GB / 28.39 - 1) < 0.05
    # Gemma-2 QLoRA b2 seq2048 ckpt off: 실측 37.91GB (구조 항 전 34.0 → 후 37.5)
    g = EVAL_MODELS["gemma-2-2b"]
    cfg = TrainConfig(model="gemma-2-2b", method="qlora", batch_size=2, seq_len=2048, grad_checkpointing=False)
    assert abs(estimate(cfg, g).total / GB / 37.91 - 1) < 0.03
