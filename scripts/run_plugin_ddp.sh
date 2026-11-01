#!/usr/bin/env bash
# 플러그인 GPU 2장 확인 (Kaggle T4 ×2): 같은 설정을 GPU 2장(torchrun)·1장으로 → 예측 vs 실측, strict 중단
set -u
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
L=data/plugin_runs_ddp.jsonl
two() { tag=$1; shift; echo "### $tag (GPU 2장)"; timeout 1800 torchrun --standalone --nproc_per_node 2 scripts/plugin_ddp_check.py --tag "$tag" --log $L "$@" 2>&1 | grep -E "^\[fitcfg\]|^PLUGIN_RESULT|Error" | cut -c1-400; }
one() { tag=$1; shift; echo "### $tag (GPU 1장)"; CUDA_VISIBLE_DEVICES=0 timeout 1800 python scripts/plugin_ddp_check.py --tag "$tag" --log $L "$@" 2>&1 | grep -E "^\[fitcfg\]|^PLUGIN_RESULT|Error" | cut -c1-400; }
two l1b_lora_2g  --model unsloth/Llama-3.2-1B --method lora  --batch 2 --seq 2048 --ckpt
one l1b_lora_1g  --model unsloth/Llama-3.2-1B --method lora  --batch 2 --seq 2048 --ckpt
two q3b_qlora_2g --model Qwen/Qwen2.5-3B      --method qlora --batch 1 --seq 2048 --ckpt
two l1b_strict_2g --model unsloth/Llama-3.2-1B --method lora --batch 8 --seq 2048 --strict
echo "=== plugin_runs_ddp.jsonl ==="; cat $L
