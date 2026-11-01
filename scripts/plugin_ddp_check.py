"""HF Trainer 플러그인을 GPU 여러 장(DDP, torchrun)에서 확인 — Kaggle T4 ×2 등.

    torchrun --standalone --nproc_per_node 2 scripts/plugin_ddp_check.py --model unsloth/Llama-3.2-1B --method lora --batch 2 --ckpt
    CUDA_VISIBLE_DEVICES=0 python scripts/plugin_ddp_check.py ...          # 같은 설정 GPU 1장 비교

사용자 코드 그대로: 모델 로드(각 프로세스가 자기 GPU) → (Q)LoRA → TrainingArguments → Trainer(callbacks=[FitcfgCallback]).
rank 0이 결과를 'PLUGIN_RESULT {json}' 한 줄로 출력하고 --log 파일에 추가한다.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", choices=["qlora", "lora", "full"], required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--log", default=str(REPO / "data" / "plugin_runs_ddp.jsonl"))
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    from fitcfg.integrations.hf import FitcfgCallback

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    # torch.cuda.is_bf16_supported()는 T4 같은 구형 GPU에서도 '에뮬레이션 가능'이라 True → 약 3배 느려짐 (Kaggle 실측)
    bf16 = torch.cuda.get_device_capability(rank)[0] >= 8
    dt = torch.bfloat16 if bf16 else torch.float16
    t0 = time.time()
    rec = {"tag": a.tag, "model": a.model, "method": a.method, "batch": a.batch, "accum": a.accum, "seq": a.seq,
           "ckpt": a.ckpt, "strict": a.strict, "world": world, "gpu_name": torch.cuda.get_device_name(rank)}
    cb = FitcfgCallback(seq_len=a.seq, strict=a.strict, warmup_steps=3, verbose=(rank == 0))
    try:
        # fp16 GPU에서 Full은 fp32 가중치 + AMP (HF Trainer 방식)
        load_dt = torch.float32 if (a.method == "full" and not bf16) else dt
        kw = dict(attn_implementation="sdpa", dtype=load_dt, device_map={"": rank})
        if a.method == "qlora":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
        model.config.use_cache = False
        if a.method == "qlora":
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        if a.method in ("lora", "qlora"):
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
                                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                                     lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))

        class Rand(torch.utils.data.Dataset):
            def __len__(self):
                return 10_000

            def __getitem__(self, i):
                g = torch.Generator().manual_seed(i)
                ids = torch.randint(0, model.config.vocab_size, (a.seq,), generator=g)
                return {"input_ids": ids, "labels": ids.clone()}

        args = TrainingArguments(output_dir=f"/tmp/plugin_{a.tag}", per_device_train_batch_size=a.batch,
                                 gradient_accumulation_steps=a.accum, gradient_checkpointing=a.ckpt,
                                 gradient_checkpointing_kwargs={"use_reentrant": False},
                                 max_steps=a.steps, bf16=bf16, fp16=not bf16, learning_rate=1e-4, report_to=[],
                                 logging_strategy="no", save_strategy="no", disable_tqdm=True,
                                 ddp_find_unused_parameters=False if world > 1 else None, dataloader_num_workers=0)
        t1 = time.time()
        Trainer(model=model, args=args, train_dataset=Rand(), callbacks=[cb]).train()
        rec.update(result="finished", train_s=round(time.time() - t1, 1))
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("[fitcfg]"):
            rec.update(result="stopped_by_plugin", message=msg[:300])
        elif "out of memory" in msg.lower():
            rec.update(result="oom", message=msg[:200])
        else:
            rec.update(result="error", message=msg[:400])
    except Exception as e:  # noqa: BLE001
        rec.update(result="oom" if "OutOfMemory" in type(e).__name__ else "error",
                   message=f"{type(e).__name__}: {str(e)[:300]}")
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["detected"] = asdict(cb.detected) if cb.detected else None
    rec["prediction"] = asdict(cb.prediction) if cb.prediction else None
    rec["measured"] = cb.measured
    if rank == 0:
        line = json.dumps(rec, ensure_ascii=False)
        print("PLUGIN_RESULT " + line, flush=True)
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    if rec["result"] in ("stopped_by_plugin", "oom", "error"):
        sys.exit(3)


if __name__ == "__main__":
    main()
