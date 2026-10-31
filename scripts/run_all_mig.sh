#!/usr/bin/env bash
# gfarm MIG 3g.40gb 1장으로 할 수 있는 실험 전체를 차례대로 실행 (한 번에 하나씩 → GPU 측정이 섞이지 않음).
#   사용법: cd /workspace/fitcfg && nohup bash scripts/run_all_mig.sh > /workspace/run_all.log 2>&1 &
#   진행 확인: tail -n 30 /workspace/run_all.log
# 이미 끝난 설정은 CSV를 보고 건너뛰므로(이어하기) 중간에 끊겨도 다시 실행하면 된다.
set -u
cd "$(dirname "$0")/.."
export PIP_CONFIG_FILE=/dev/null OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
LLAMA="llama-3.1-8b=unsloth/Meta-Llama-3.1-8B llama-3.2-1b=unsloth/Llama-3.2-1B"
CSV=data/measurements_mig.csv
step() { echo; echo "################ $(date '+%H:%M:%S') $*"; }

# 0) 이미 돌고 있는 측정(mig-small 등)이 있으면 끝날 때까지 기다린다
while pgrep -f "scripts/sweep.py" > /dev/null || pgrep -f "scripts/measure.py" > /dev/null; do
  echo "$(date '+%H:%M:%S') 다른 측정이 진행 중 → 60초 대기"; sleep 60
done

step "1/5 소형 모델 재측정 (mig-small, 남은 것만)"
python scripts/sweep.py --preset mig-small --tag gfarm-mig3g --out $CSV --hf-override $LLAMA

step "2/5 8B 확장 (긴 시퀀스 · rank · 대상 행렬 · 8bit 옵티마이저)"
python scripts/sweep.py --preset mig-8b-ext --tag gfarm-mig3g --out $CSV --hf-override $LLAMA

step "3/5 처음 보는 계열 실측 (Gemma-3 1B, Gemma-2 2B, Phi-3 mini, Mistral 7B)"
python scripts/sweep.py --preset mig-family --tag gfarm-mig3g-family --out $CSV --hf-override $LLAMA

step "4/5 병렬 설정의 GPU 1장 몫 (FSDP · TP · PP, 가짜 통신 + 실제 GPU)"
python scripts/gpu_parallel.py run
python scripts/gpu_parallel.py report || true

step "5/5 결과 정리"
python scripts/summarize_mig.py > reports/mig_summary.txt 2>&1 || true
tar czf /workspace/results_mig.tar.gz data/measurements_mig.csv data/gpu_parallel.csv reports/gpu_parallel_report.md \
    reports/mig_summary.txt 2>/dev/null
ls -la /workspace/results_mig.tar.gz
echo "=== 전체 완료 $(date '+%H:%M:%S') → 파일 목록에서 workspace/results_mig.tar.gz 다운로드 ==="
