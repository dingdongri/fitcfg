#!/usr/bin/env bash
# 5차 대기열 (2026-10-31): ① 플러그인 실제 GPU 확인 ② CPU offload(ZeRO) 번들 ③ 긴 시퀀스
#   사용법: cd /workspace/fitcfg6 && nohup bash scripts/run_round5_mig.sh > /workspace/round5.log 2>&1 &
#   진행 확인: tail -n 20 /workspace/round5.log
set -u
cd "$(dirname "$0")/.."
export PIP_CONFIG_FILE=/dev/null PIP_CONSTRAINT= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
step() { echo; echo "################ $(date '+%H:%M:%S') $*"; }
while pgrep -f "scripts/sweep.py" > /dev/null || pgrep -f "scripts/run_bundle.py" > /dev/null; do
  echo "$(date '+%H:%M:%S') 다른 측정이 진행 중 → 60초 대기"; sleep 60
done

step "1/3 HF Trainer 플러그인 (실제 모델 · 사용자 코드 그대로)"
PL=data/plugin_runs_mig.jsonl
run() { tag=$1; shift
  if grep -q "\"tag\": \"$tag\"" "$PL" 2>/dev/null; then echo "  ($tag 이미 완료)"; return; fi
  timeout 1800 python scripts/plugin_gpu_check.py --tag "$tag" --log "$PL" "$@" 2>&1 | grep -E "^\[fitcfg\]|^\[plugin-check\]|Error" | tail -n 8; }
run q8b_b2_ck   --model unsloth/Meta-Llama-3.1-8B --method qlora --batch 2 --seq 2048 --ckpt
run l7b_b2_ck   --model Qwen/Qwen2.5-7B           --method lora  --batch 2 --seq 2048 --ckpt
run f1b_b4      --model unsloth/Llama-3.2-1B      --method full  --batch 4 --seq 2048
run new_qwen3   --model Qwen/Qwen3-4B-Base        --method lora  --batch 2 --seq 2048 --ckpt
run new_gemma2  --model unsloth/gemma-2-2b        --method qlora --batch 2 --seq 2048
run strict_oom  --model unsloth/Meta-Llama-3.1-8B --method lora  --batch 8 --seq 2048 --strict
run nostrict_oom --model unsloth/Meta-Llama-3.1-8B --method lora --batch 8 --seq 2048 --steps 3

step "2/3 CPU offload 번들 (GPU 16GB로 가정해 추천받은 ZeRO-2 + optimizer offload를 실제 실행)"
python -c "import deepspeed" 2>/dev/null || DS_BUILD_OPS=0 pip install -q "deepspeed>=0.14"
export DS_SKIP_CUDA_CHECK=1           # 이미지의 nvcc 12.1 ↔ torch cu124 (CPU Adam 빌드는 C++만 필요)
LOG=data/bundle_runs_mig.jsonl
DATA=(--dataset tatsu-lab/alpaca --text-field text --dataset-split "train[:10000]")
# 이름 모델 HF 가정GPU(GB) — offload가 추천되도록 작은 GPU를 가정 (메모리 예측은 GPU 크기와 무관하게 검증 가능)
for sc in "off1b llama-3.2-1b unsloth/Llama-3.2-1B 8" "off3b qwen2.5-3b Qwen/Qwen2.5-3B 16"; do
  read -r name model hf gmem <<< "$sc"
  export FITCFG_MODEL_OVERRIDE="$hf"
  for k in 1 2; do
    d="runs/${name}_rec$k"
    python -m fitcfg plan --model "$model" --method full --seq 2048 --gpu-mem "$gmem" --cpu-ram 60 --top 3 \
        --out "$d" --pick "$k" "${DATA[@]}" 2>/dev/null | grep -E "^#$k|✅" | head -n 2 || true
    [ -f "$d/launch.sh" ] || { echo "  (rec$k 없음)"; continue; }
    if grep -q "\"bundle\": \"${name}_rec$k\"" "$LOG" 2>/dev/null; then echo "  (이미 완료)"; continue; fi
    python scripts/run_bundle.py "$d" --label "offload#$k" --max-steps 10 --log "$LOG" --timeout 2400 | tail -n 3
  done
done
unset FITCFG_MODEL_OVERRIDE

step "3/3 긴 시퀀스 실측 (seq 4096~16384)"
python scripts/sweep.py --preset mig-long --tag gfarm-mig3g-long --out data/measurements_mig_long.csv

tar czf /workspace/results_mig_v5.tar.gz data/plugin_runs_mig.jsonl data/bundle_runs_mig.jsonl \
    data/measurements_mig_long.csv 2>/dev/null
ls -la /workspace/results_mig_v5.tar.gz
echo "=== 전체 완료 $(date '+%H:%M:%S') ==="
