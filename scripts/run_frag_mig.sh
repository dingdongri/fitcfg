#!/usr/bin/env bash
# QLoRA 할당기 단편화(reserved−allocated) 대응 시험: expandable_segments로 같은 8B 설정을 다시 측정 (별도 CSV)
#   사용법: cd /workspace/fitcfg && nohup bash scripts/run_frag_mig.sh > /workspace/run_frag.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PIP_CONFIG_FILE=/dev/null OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
while pgrep -f "scripts/sweep.py" > /dev/null || pgrep -f "scripts/gpu_parallel.py" > /dev/null; do
  echo "$(date '+%H:%M:%S') 다른 측정이 진행 중 → 60초 대기"; sleep 60
done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/sweep.py --preset mig-frag --tag gfarm-mig3g-expseg \
  --out data/measurements_mig_expseg.csv --skip-over 0 \
  --hf-override llama-3.1-8b=unsloth/Meta-Llama-3.1-8B
echo "=== 완료 → data/measurements_mig_expseg.csv ==="
