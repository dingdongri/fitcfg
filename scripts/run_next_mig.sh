#!/usr/bin/env bash
# 2차 대기열: (1) 병렬 rank 0 실측 재실행(1차는 transformers 5.20 RoPE 인자 오류로 전부 실패) (2) QLoRA 단편화 시험
#   사용법: cd /workspace/fitcfg && nohup bash scripts/run_next_mig.sh > /workspace/run_next.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PIP_CONFIG_FILE=/dev/null OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
step() { echo; echo "################ $(date '+%H:%M:%S') $*"; }
while pgrep -f "scripts/sweep.py" > /dev/null || pgrep -f "scripts/gpu_parallel.py" > /dev/null; do
  echo "$(date '+%H:%M:%S') 다른 측정이 진행 중 → 60초 대기"; sleep 60
done

step "1/3 병렬 설정의 GPU 1장 몫 (FSDP · TP · PP) — 실패 기록 지우고 다시"
[ -f data/gpu_parallel.csv ] && mv data/gpu_parallel.csv data/gpu_parallel_failed_v1.csv
python scripts/gpu_parallel.py run
python scripts/gpu_parallel.py report || true

step "2/3 QLoRA 단편화 시험 (expandable_segments)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/sweep.py --preset mig-frag --tag gfarm-mig3g-expseg \
  --out data/measurements_mig_expseg.csv --skip-over 0 --hf-override llama-3.1-8b=unsloth/Meta-Llama-3.1-8B

step "3/3 결과 정리"
python scripts/summarize_mig.py > reports/mig_summary.txt 2>&1 || true
tar czf /workspace/results_mig_v2.tar.gz data/measurements_mig.csv data/measurements_mig_expseg.csv data/gpu_parallel.csv \
    reports/gpu_parallel_report.md reports/mig_summary.txt 2>/dev/null
ls -la /workspace/results_mig_v2.tar.gz
echo "=== 전체 완료 $(date '+%H:%M:%S') → 파일 목록에서 workspace/results_mig_v2.tar.gz 다운로드 ==="
