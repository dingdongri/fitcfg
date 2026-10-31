"""fitcfg — 소규모 셋업 학습 설정 자동 추천기.

공개 API:
    from fitcfg import plan, PlanRequest, get_gpu, load_calibration
    result = plan(PlanRequest(model="llama-3.1-8b", gpu=get_gpu("rtx4090"), method="qlora"))
    print(result.best.reason, result.best.pred_mem_gb)
"""
from .models import ModelSpec, MODELS, get_model
from .hardware import GPUSpec, GPUS, get_gpu, custom_gpu, match_gpu
from .memory import (TrainConfig, MemoryBreakdown, CostConstants, estimate, fits,
                     features, load_calibration, reset_calibration, get_constants,
                     get_default_safety, calibration_info)
from .throughput import ThroughputEstimate, estimate_throughput, flops_per_token
from .recommend import (plan, PlanRequest, PlanResult, Recommendation, recommend,
                        naive_config)
from .parallel import ParallelDecision, ParallelPlan, decide, gpu_from_spec, LINKS
from .autospec import StructureInfo, structure_of, resolve, register

__all__ = [
    "ModelSpec", "MODELS", "get_model",
    "GPUSpec", "GPUS", "get_gpu", "custom_gpu", "match_gpu",
    "TrainConfig", "MemoryBreakdown", "CostConstants", "estimate", "fits",
    "features", "load_calibration", "reset_calibration", "get_constants",
    "get_default_safety", "calibration_info",
    "ThroughputEstimate", "estimate_throughput", "flops_per_token",
    "plan", "PlanRequest", "PlanResult", "Recommendation", "recommend", "naive_config",
    "ParallelDecision", "ParallelPlan", "decide", "gpu_from_spec", "LINKS",
    "StructureInfo", "structure_of", "resolve", "register",
]
