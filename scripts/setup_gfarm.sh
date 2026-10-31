#!/usr/bin/env bash
# gfarm(AstraGo) MIG 워크로드 환경 준비.
#   이미지: xiilab/astrago:pytorch-23.07-cuda12.1 (torch 2.1, 드라이버 535 / CUDA 12.2)
#   사용법: cd /workspace/fitcfg && bash scripts/setup_gfarm.sh
# 컨테이너 디스크는 워크로드를 종료하면 사라진다 → 끝나면 상세정보의 "Commit Image"로 환경을 저장해 두면 재설치가 필요 없다.
set -e

# 컨테이너가 보는 CPU는 호스트 전체(256)지만 실제 할당은 8코어 → 스레드 과다 생성 방지
grep -q OMP_NUM_THREADS ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<'EOF'
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
export PIP_CONSTRAINT= PIP_CONFIG_FILE=/dev/null
EOF
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# NGC 컨테이너는 PIP_CONSTRAINT로 torch 버전을 고정해 둔다 → 해제
export PIP_CONSTRAINT=
# 이미지의 pip 설정 파일 5곳이 extra-index-url=pypi.ngc.nvidia.com(이 클러스터에서 DNS 해석 불가)을 넣어 둠
# → 설정 파일을 모두 무시하고 PyPI 기본값만 쓴다
export PIP_CONFIG_FILE=/dev/null
# venv는 이미지에 ensurepip가 없어 못 만든다 → 시스템 파이썬에 설치(컨테이너는 일회용이라 무방)

pip install -q -U pip

# 2) torch: 코드가 torch.distributed.fsdp.fully_shard(2.6+)를 쓴다.
#    드라이버 535(CUDA 12.2)에서 cu124 빌드는 CUDA minor-version 호환으로 동작해야 한다. 실패하면 2.5.1+cu121로 내린다.
pip install -q "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "[!] torch 2.6.0+cu124가 GPU를 못 잡음 → 2.5.1+cu121로 재설치"
  pip install -q --force-reinstall "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
fi

# 3) 나머지 (Colab 측정과 같은 계열)
pip install -q "transformers>=4.46" "peft>=0.12" accelerate bitsandbytes soxr datasets
# 이미지의 numpy 1.22·pandas 1.5는 transformers 5.x(HF Trainer)와 안 맞음 → 함께 올림 (numba는 numpy 2와 충돌, 사용 안 함)
pip install -q -U "numpy>=2" "pandas>=2.2" "scipy>=1.13" "scikit-learn>=1.5"
pip uninstall -y -q numba 2>/dev/null || true
# 이미지의 구버전 fsspec이면 datasets가 'maxdepth' 오류 → 범위를 직접 지정해 함께 맞춤 (pip 자동 해결은 오히려 버전을 내림)
pip install -q -U "datasets>=4.0" "huggingface_hub>=1.31,<3" "fsspec[http]>=2024.6" "pyarrow>=15"
# numba·구버전 numpy에 묶인 이미지 패키지(음성 librosa, RAPIDS cudf·cupy)는 transformers 불러오기를 깨뜨림 → 제거 (실험에 불필요)
pip uninstall -y -q librosa resampy cudf dask-cudf dask-cuda cupy-cuda12x cuml cugraph raft-dask pylibraft rmm 2>/dev/null || true
# NGC 이미지의 torch 2.1 전용 패키지는 새 torch와 충돌(torchvision import 실패 → transformers까지 깨짐) → 제거
pip uninstall -y -q torchvision torchtext torchdata torch-tensorrt torchaudio 2>/dev/null || true

# 4) 확인
python - <<'EOF'
import torch, transformers, peft
from transformers import Trainer   # HF Trainer까지 불러와져야 생성된 학습 스크립트가 동작
from datasets import load_dataset; load_dataset("tatsu-lab/alpaca", split="train[:10]")   # 데이터셋 다운로드 경로 확인
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
f, t = torch.cuda.mem_get_info(); print(f"free/total GiB {f/2**30:.2f} / {t/2**30:.2f}")
print("transformers", transformers.__version__, "peft", peft.__version__)
try:
    import bitsandbytes as bnb; print("bitsandbytes", bnb.__version__)
except Exception as e:
    print("bitsandbytes 불가:", e)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); import time; t0 = time.time()
for _ in range(50): y = x @ x
torch.cuda.synchronize(); dt = time.time() - t0
print(f"bf16 matmul {2*4096**3*50/dt/1e12:.1f} TFLOPS (MIG 3g 이론 peak ≈ 133.7)")
from torch.nn.attention import sdpa_kernel, SDPBackend
q = torch.randn(1, 32, 2048, 128, device="cuda", dtype=torch.bfloat16)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
print("SDPA flash 커널 OK")
from fitcfg.hardware import match_gpu
g = match_gpu(torch.cuda.get_device_name(0)); print("fitcfg 인식:", g.key, g.mem_gb, "GiB", g.tflops_16bit, "TFLOPS")
EOF
echo "=== 준비 완료. 새 터미널에서도 같은 설정이 적용된다 ==="
