"""HF Trainer 플러그인 예시 — GPU 없이도 돌아가는 아주 작은 모델로 흐름만 확인.

    pip install "fitcfg[hf]"      # 또는 저장소에서: pip install -e ".[hf]"
    python examples/hf_plugin_example.py

실제로는 내 학습 코드의 Trainer에 callbacks=[FitcfgCallback(seq_len=...)]만 추가하면 된다.
GPU가 있으면 그 GPU로, 없으면 아래처럼 gpu="rtx4090" 등으로 '이 GPU에서 돌리면'을 예측할 수 있다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments
from fitcfg.integrations.hf import FitcfgCallback, predict

SEQ = 64


class RandomText(torch.utils.data.Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, i):
        ids = torch.randint(0, 512, (SEQ,))
        return {"input_ids": ids, "labels": ids.clone()}


model = LlamaForCausalLM(LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2))
args = TrainingArguments(output_dir="outputs/plugin_demo", per_device_train_batch_size=4, gradient_accumulation_steps=2,
                         max_steps=8, use_cpu=not torch.cuda.is_available(), report_to=[], save_strategy="no",
                         logging_strategy="no", disable_tqdm=True)

# 1) 학습 없이 예측만 (TrainingArguments와 모델만 있으면 됨)
from fitcfg import get_gpu
gpu = None if torch.cuda.is_available() else get_gpu("rtx4090")      # GPU가 없으면 '4090에서 돌리면'을 예측
d, p = predict(model, args, gpu=gpu, seq_len=SEQ)
print(f"예측만: {d.method} · micro-batch {d.batch_size} → {p.pred_mem_gb:.2f}GB / 한도 {p.cap_gb:.1f}GB ({p.gpu_label})")

# 2) Trainer에 붙여서 학습 (시작 직전 예측 → 학습 → 끝나면 outputs/plugin_demo/fitcfg_run.json)
cb = FitcfgCallback(seq_len=SEQ, gpu=gpu, warmup_steps=2)
Trainer(model=model, args=args, train_dataset=RandomText(), callbacks=[cb]).train()
print("기록:", Path(args.output_dir) / "fitcfg_run.json")
