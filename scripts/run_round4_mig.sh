#!/usr/bin/env bash
# 4차 대기열 (2026-10-29): ① 추천 설정 실제 학습 검증 ② 처음 보는 모델 QLoRA 규칙
#   사용법: cd /workspace/fitcfg && nohup bash scripts/run_round4_mig.sh > /workspace/round4.log 2>&1 &
#   진행 확인: tail -n 20 /workspace/round4.log
set -u
cd "$(dirname "$0")/.."
export PIP_CONFIG_FILE=/dev/null PIP_CONSTRAINT= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
step() { echo; echo "################ $(date '+%H:%M:%S') $*"; }

# 0) 환경: 워크로드가 새로 만들어졌으면(torch 2.1) 설치부터
if ! python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.6') else 1)" 2>/dev/null; then
  step "0/3 환경 설치 (torch 2.6)"
  bash scripts/setup_gfarm.sh
fi
python -c "import datasets" 2>/dev/null || pip install -q datasets
while pgrep -f "scripts/sweep.py" > /dev/null || pgrep -f "scripts/run_bundle.py" > /dev/null; do
  echo "$(date '+%H:%M:%S') 다른 측정이 진행 중 → 60초 대기"; sleep 60
done

step "1/3 추천 설정으로 실제 학습 (6개 시나리오 × 기본값 + 추천 1~3위)"
bash scripts/e2e_mig.sh

step "2/3 처음 보는 모델 4종 QLoRA·LoRA (64개 설정)"
python scripts/sweep.py --preset mig-qlora-new --tag gfarm-mig3g-new --out data/measurements_mig_new.csv

step "3/3 결과 정리"
tar czf /workspace/results_mig_v4.tar.gz data/bundle_runs_mig.jsonl data/measurements_mig_new.csv reports/e2e_report.md \
    2>/dev/null
ls -la /workspace/results_mig_v4.tar.gz
echo "=== 전체 완료 $(date '+%H:%M:%S') → 파일 목록에서 workspace/results_mig_v4.tar.gz 다운로드 ==="
