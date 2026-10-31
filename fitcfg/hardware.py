"""GPU 사양 프리셋.

메모리 예측(fits 판정)과 처리량 예측(peak FLOPS)에 쓴다.
- mem_gb        : PyTorch가 보는 사용 가능 메모리 (GiB, torch.cuda.get_device_properties().total_memory 기준)
                  → 광고 용량(예: "16GB")보다 조금 작다. 예측도 GiB 단위라 이 값과 비교해야 맞다.
- tflops_16bit  : 16비트(bf16, bf16 미지원이면 fp16) dense Tensor Core 연산 peak (sparsity 제외)
- bf16          : bf16 네이티브 지원 여부 (Ampere 이상). 미지원이면 fp16으로 학습
- match         : 측정 CSV의 gpu_name(torch.cuda.get_device_name) 매칭용 부분 문자열

출처: NVIDIA 제품 데이터시트의 dense 수치. 소비자용(RTX)은 fp32 누산 기준.
"""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    key: str
    label: str
    mem_gb: float
    tflops_16bit: float
    bf16: bool
    match: tuple[str, ...]
    note: str = ""
    # FlashAttention(SDPA flash 커널) 사용 가능 여부 = Ampere(sm80) 이상.
    # 불가능하면(T4, P100) GQA 모델의 SDPA가 math 커널로 떨어져 attention 행렬(B·heads·S²)을 저장한다.
    flash: bool = True
    # 메모리 대역폭(GB/s): optimizer step(가중치·상태를 읽고 쓰는 일)은 연산이 아니라 대역폭이 시간을 정한다
    mem_bw_gbs: float = 900.0


GPUS: dict[str, GPUSpec] = {g.key: g for g in [
    GPUSpec("t4", "NVIDIA T4 16GB (Colab/Kaggle 무료)", 14.74, 65.0, False, ("T4",),
            "bf16 미지원 → fp16", flash=False, mem_bw_gbs=320),
    GPUSpec("p100", "NVIDIA P100 16GB (Kaggle)", 15.89, 18.7, False, ("P100",),
            "Tensor Core 없음 → 느림", flash=False, mem_bw_gbs=732),
    GPUSpec("l4", "NVIDIA L4 24GB (Colab Pro)", 22.03, 121.0, True, ("L4",), mem_bw_gbs=300),
    GPUSpec("a10g", "NVIDIA A10G 24GB", 22.06, 70.0, True, ("A10G",), mem_bw_gbs=600),
    GPUSpec("rtx3090", "RTX 3090 24GB", 23.69, 71.0, True, ("3090",), mem_bw_gbs=936),
    GPUSpec("rtx4090", "RTX 4090 24GB", 23.64, 165.0, True, ("4090",), mem_bw_gbs=1008),
    GPUSpec("a100-40", "NVIDIA A100 40GB", 39.39, 312.0, True, ("A100-SXM4-40GB", "A100-PCIE-40GB", "A100 40"), mem_bw_gbs=1555),
    GPUSpec("a100-80", "NVIDIA A100 80GB", 79.15, 312.0, True, ("A100-SXM4-80GB", "A100 80GB", "A100-PCIE-80GB", "A100 80"), mem_bw_gbs=2039),
    GPUSpec("h100", "NVIDIA H100 80GB", 79.11, 989.0, True, ("H100",), mem_bw_gbs=3350),
    # MIG 슬라이스(A100 80GB를 7조각 중 3조각): 메모리 40GB, 연산 3/7. 측정값 total 39.2 GiB (gfarm, 2026-10-20)
    GPUSpec("a100-mig3g40", "A100 80GB MIG 3g.40gb", 39.2, round(312.0 * 3 / 7, 1), True, (),
            "MIG 슬라이스 → 다른 슬라이스와 NCCL 통신 불가(멀티 GPU 학습 불가)", mem_bw_gbs=1020),
]}


def get_gpu(key: str) -> GPUSpec:
    key = key.lower()
    if key not in GPUS:
        raise KeyError(f"unknown gpu '{key}'. available: {list(GPUS)}")
    return GPUS[key]


_MIG_RE = re.compile(r"MIG\s*(\d)g\.(\d+)gb", re.I)


def mig_spec(parent: GPUSpec, slices: int, mem_gb_label: int) -> GPUSpec:
    """MIG 슬라이스 사양: 연산은 부모의 slices/7, 메모리는 라벨 용량 × 0.98(40GB 라벨 → 39.2 GiB 실측 비율)."""
    if parent.key == "a100-80" and (slices, mem_gb_label) == (3, 40):
        return GPUS["a100-mig3g40"]
    return GPUSpec(f"{parent.key}-mig{slices}g{mem_gb_label}", f"{parent.label} MIG {slices}g.{mem_gb_label}gb",
                   round(mem_gb_label * 0.98, 2), round(parent.tflops_16bit * slices / 7, 1), parent.bf16, (),
                   "MIG 슬라이스 → 멀티 GPU 학습 불가", flash=parent.flash,
                   mem_bw_gbs=round(parent.mem_bw_gbs * min(1.0, mem_gb_label / max(parent.mem_gb, 1)), 1))


def match_gpu(gpu_name: str) -> GPUSpec | None:
    """측정 CSV의 gpu_name → 프리셋. 가장 구체적인(긴) 매칭을 우선.
    MIG 이름(예: 'NVIDIA A100-SXM4-80GB MIG 3g.40gb')은 부모 GPU의 80GB로 잘못 잡히지 않게 먼저 처리한다."""
    name = gpu_name or ""
    m = _MIG_RE.search(name)
    if m:
        parent = match_gpu(name[:m.start()]) or GPUS["a100-80"]
        return mig_spec(parent, int(m.group(1)), int(m.group(2)))
    hits = [(len(m), g) for g in GPUS.values() for m in g.match if m.lower() in name.lower()]
    if not hits:
        # 'A100'만 적힌 경우 등: 용량 정보가 없으면 80GB로 가정하지 않고 40GB로 (보수적)
        if "a100" in name.lower():
            return GPUS["a100-40"]
        return None
    return max(hits, key=lambda t: t[0])[1]


def custom_gpu(mem_gb: float, tflops_16bit: float = 100.0, bf16: bool = True) -> GPUSpec:
    """프리셋에 없는 GPU. 메모리만 알아도 메모리 예측/추천은 동작한다(처리량은 참고용)."""
    # bf16 지원(Ampere 이상) ≈ FlashAttention 지원으로 간주
    return GPUSpec("custom", f"사용자 지정 {mem_gb:g}GB", mem_gb, tflops_16bit, bf16, (), flash=bf16)
