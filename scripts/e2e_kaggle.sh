#!/usr/bin/env bash
# GPU 2장 실측 (Kaggle 무료 T4 ×2, PCIe): DDP · ZeRO-2 · ZeRO-3 · ZeRO-2+offload 추천 번들을 실제 학습
#   → GPU당 peak 메모리 예측 · 추천 순위 = 실제 속도 순위인가 · GPU 1장 대비 확장 효율(ddp_eff 가정 0.9) 확인
#   사용법 (Kaggle 노트북): !bash scripts/e2e_kaggle.sh   → data/bundle_runs_kaggle.jsonl
set -u
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
LOG=data/bundle_runs_kaggle.jsonl
STEPS=${STEPS:-12}
DATA=(--dataset tatsu-lab/alpaca --text-field text --dataset-split "train[:10000]")
mirror() { case "$1" in llama-3.2-1b) echo unsloth/Llama-3.2-1B ;; *) echo "" ;; esac; }

# 이름  모델  방식  seq  GPU수  만들 번들(naive / 추천 순위)
SCENARIOS=(
  "l1b_2g    llama-3.2-1b lora  2048 2  naive rec1 rec2 rec3"
  "l1b_1g    llama-3.2-1b lora  2048 1  rec1"
  "q3b_2g    qwen2.5-3b   qlora 2048 2  naive rec1"
  "q3b_1g    qwen2.5-3b   qlora 2048 1  rec1"
  "f1b_2g    llama-3.2-1b full  1024 2  naive rec1 rec2 rec3"
)
for sc in "${SCENARIOS[@]}"; do
  read -r name model method seq ngpu picks <<< "$sc"
  export FITCFG_MODEL_OVERRIDE="$(mirror "$model")"; [ -z "$FITCFG_MODEL_OVERRIDE" ] && unset FITCFG_MODEL_OVERRIDE
  echo; echo "################ $(date '+%H:%M:%S') $name ($model $method seq $seq · T4 ×$ngpu)"
  for b in $picks; do
    d="runs/${name}_$b"
    if [ "$b" = naive ]; then extra=(--naive); else extra=(--pick "${b#rec}"); fi
    python -m fitcfg plan --model "$model" --method "$method" --seq "$seq" --gpu t4 --gpus "$ngpu" --global-batch 16 \
        --top 3 --cpu-ram 28 --out "$d" "${extra[@]}" "${DATA[@]}" 2>/dev/null | grep -E "^#${b#rec} |✅" | head -n 2 || true
    [ -f "$d/launch.sh" ] || { echo "  ($b 없음)"; continue; }
    if grep -q "\"bundle\": \"${name}_$b\"" "$LOG" 2>/dev/null; then echo "  ($b 이미 완료)"; continue; fi
    lab=$([ "$b" = naive ] && echo naive || echo "recommended#${b#rec}")
    python scripts/run_bundle.py "$d" --label "$lab" --max-steps "$STEPS" --log "$LOG" --timeout 2400 | tail -n 3
  done
done
unset FITCFG_MODEL_OVERRIDE
python scripts/e2e_report.py --log "$LOG" --out reports/e2e_kaggle_report.md 2>/dev/null || python scripts/e2e_report.py --log "$LOG" || true
echo "=== 완료 $(date '+%H:%M:%S') → data/bundle_runs_kaggle.jsonl ==="
