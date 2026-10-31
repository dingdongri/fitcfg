"""3D 병렬화 판단 (fitcfg.parallel) — GPU 불필요.  실행: python -m pytest -q tests/test_parallel.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fitcfg import PlanRequest, get_gpu, get_model, plan, reset_calibration
from fitcfg.parallel import decide, gpu_from_spec, default_link


def setup_function(_):
    reset_calibration()


def _gpus(*keys):
    specs = [get_gpu(k) for k in keys]
    return specs, [gpu_from_spec(s) for s in specs]


def test_mixed_gpus_recommend_pp_with_more_layers_on_faster_gpu():
    specs, gs = _gpus("rtx3090", "rtx3090", "rtx4090", "rtx4090")
    d = decide(get_model("llama-3.1-8b"), "lora", gs, default_link(specs), 2048, 32)
    assert d.verdict == "3d" and d.mixed and d.best.pp > 1
    assert sum(d.best.layers) == get_model("llama-3.1-8b").n_layers
    slow = [n for g, n in zip(d.best.stages, d.best.layers) if g == "rtx3090"]
    fast = [n for g, n in zip(d.best.stages, d.best.layers) if g == "rtx4090"]
    assert sum(fast) > sum(slow)                       # 빠른 GPU가 층을 더 맡는다
    assert d.gain_opt >= 1.10 and d.gain_pess >= 1.10


def test_nvlink_same_gpus_stay_data_parallel():
    _, gs = _gpus(*["a100-80"] * 8)
    for method in ("full", "lora", "qlora"):
        d = decide(get_model("llama-3.1-8b"), method, gs, "nvlink", 2048, 32)
        assert d.verdict == "dp", (method, d.reason)
        assert d.best_dp is not None and not d.best_dp.is_3d


def test_memory_forces_model_split():
    # 8B Full, A100-40 ×4: FSDP로도 GPU당 메모리 부족 → 모델을 나눠야 함 (3D가 필요한 경우 ①)
    _, gs = _gpus(*["a100-40"] * 4)
    d = decide(get_model("llama-3.1-8b"), "full", gs, "nvlink", 2048, 32)
    assert d.verdict == "3d" and d.best_dp is None and d.best.is_3d
    assert all(m <= 39.39 * 0.90 + 1e-6 for m in d.best.mem_gb)


def test_qlora_excludes_tensor_parallel():
    _, gs = _gpus(*["rtx4090"] * 4)
    d = decide(get_model("llama-3.1-8b"), "qlora", gs, "pcie-nop2p", 2048, 32)
    assert all(p.tp == 1 for p in d.top)


def test_plan_attaches_decision_and_mixed_gpu_request():
    mix = [get_gpu("rtx3090")] * 2 + [get_gpu("rtx4090")] * 2
    res = plan(PlanRequest(model="llama-3.1-8b", gpu=get_gpu("rtx4090"), method="lora", gpu_list=mix, global_batch=32))
    assert res.request.gpus == 4 and res.request.gpu.key == "rtx3090"   # 데이터 병렬은 느린 GPU 기준
    assert res.parallel is not None and res.parallel.verdict == "3d"
    single = plan(PlanRequest(model="llama-3.1-8b", gpu=get_gpu("rtx4090"), method="qlora"))
    assert single.parallel is None


def test_offload_note_when_gpu_memory_alone_is_not_enough():
    res = plan(PlanRequest(model="llama-3.1-8b", gpu=get_gpu("rtx4090"), gpus=4, method="full"))
    assert res.parallel.verdict == "none" and res.best is not None
    assert "offload" in res.best.strategy and any("CPU offload" in n for n in res.notes)
