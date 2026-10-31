#!/usr/bin/env bash
# 추천 설정으로 '실제 학습' 검증 (gfarm MIG 3g.40gb 1장)
#   시나리오마다 HF Trainer 기본값(naive) + fitcfg 추천 1~3위 번들을 만들어 실제 데이터(alpaca)로 학습
#   → 기본값 OOM 여부 · 추천 성공 여부 · 예측 vs 실측 peak · 추천 순위 = 실제 속도 순위인가
#   사용법: cd /workspace/fitcfg && nohup bash scripts/e2e_mig.sh > /workspace/e2e.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PIP_CONFIG_FILE=/dev/null OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
LOG=data/bundle_runs_mig.jsonl
STEPS=${STEPS:-10}             # 앞 3스텝은 속도 측정에서 제외(템플릿 WARMUP=3)
DATA=(--dataset tatsu-lab/alpaca --text-field text --dataset-split "train[:10000]")

# 승인 필요 모델 → 같은 가중치 공개 사본
mirror() { case "$1" in
  llama-3.1-8b) echo unsloth/Meta-Llama-3.1-8B ;;
  llama-3.2-1b) echo unsloth/Llama-3.2-1B ;;
  mistral-7b-v0.3) echo unsloth/mistral-7b-v0.3 ;;
  *) echo "" ;; esac; }

python -c "import datasets" 2>/dev/null || pip install -q datasets

# 이름  모델  방식  seq
SCENARIOS=(
  "q8b_2048    llama-3.1-8b    qlora 2048"
  "l8b_2048    llama-3.1-8b    lora  2048"
  "q7b_4096    qwen2.5-7b      qlora 4096"
  "l7b_2048    qwen2.5-7b      lora  2048"
  "lm7b_4096   mistral-7b-v0.3 lora  4096"
  "f1b_2048    llama-3.2-1b    full  2048"
)

for sc in "${SCENARIOS[@]}"; do
  read -r name model method seq <<< "$sc"
  export FITCFG_MODEL_OVERRIDE="$(mirror "$model")"
  [ -z "$FITCFG_MODEL_OVERRIDE" ] && unset FITCFG_MODEL_OVERRIDE
  echo; echo "################ $(date '+%H:%M:%S') $name ($model $method seq $seq)"
  python -m fitcfg plan --model "$model" --method "$method" --seq "$seq" --gpu a100-mig3g40 --top 3 \
      --out "runs/${name}_naive" --naive "${DATA[@]}" | tail -n 12
  for k in 1 2 3; do
    python -m fitcfg plan --model "$model" --method "$method" --seq "$seq" --gpu a100-mig3g40 --top 3 \
        --out "runs/${name}_rec$k" --pick "$k" "${DATA[@]}" > /dev/null 2>&1 || true
  done
  for b in naive rec1 rec2 rec3; do
    d="runs/${name}_$b"
    [ -f "$d/launch.sh" ] || { echo "  ($b 없음: 추천 후보가 $b보다 적음)"; continue; }
    if grep -q "\"bundle\": \"${name}_$b\"" "$LOG" 2>/dev/null; then echo "  ($b 이미 완료 → 건너뜀)"; continue; fi
    lab=$([ "$b" = naive ] && echo naive || echo "recommended#${b#rec}")
    python scripts/run_bundle.py "$d" --label "$lab" --max-steps "$STEPS" --log "$LOG" --timeout 2400 | tail -n 3
  done
done
unset FITCFG_MODEL_OVERRIDE
python scripts/e2e_report.py --log "$LOG" || true
echo "=== 실제 학습 검증 완료 $(date '+%H:%M:%S') ==="
