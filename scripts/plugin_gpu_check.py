"""HF Trainer 플러그인(FitcfgCallback)을 실제 GPU·실제 모델에서 확인 (GPU 필요).

    python scripts/plugin_gpu_check.py --model unsloth/Meta-Llama-3.1-8B --method qlora --batch 2 --seq 2048 --ckpt
    python scripts/plugin_gpu_check.py ... --strict        # OOM 예상이면 학습 시작 전에 멈추는지

사용자가 하는 그대로: 모델 로드 → (Q)LoRA → TrainingArguments → Trainer(callbacks=[FitcfgCallback(seq_len)]).
checkpointing은 TrainingArguments(gradient_checkpointing=True)로 켠다 (플러그인이 그 값을 읽는지 확인).
결과 한 줄 → data/plugin_runs_mig.jsonl: 플러그인이 감지한 조건 · 예측 · 실측(플러그인 기록) · 결과(성공/OOM/중단) · 걸린 시간
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # 생성 train.py와 같은 설정
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
    ap.add_argument("--tag", default="")
    ap.add_argument("--log", default=str(REPO / "data" / "plugin_runs_mig.jsonl"))
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    from fitcfg.integrations.hf import FitcfgCallback

    t0 = time.time()
    rec = {"tag": a.tag, "model": a.model, "method": a.method, "batch": a.batch, "accum": a.accum, "seq": a.seq,
           "ckpt": a.ckpt, "strict": a.strict, "gpu_name": torch.cuda.get_device_name(0)}
    out_dir = REPO / "runs" / "plugin" / f"{a.tag or 'run'}"
    cb = FitcfgCallback(seq_len=a.seq, strict=a.strict, warmup_steps=3)
    try:
        kw = dict(attn_implementation="sdpa", dtype=torch.bfloat16, device_map={"": 0})
        if a.method == "qlora":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16,
                                                           bnb_4bit_use_double_quant=True)
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
        rec["load_s"] = round(time.time() - t0, 1)

        class Rand(torch.utils.data.Dataset):
            def __len__(self):
                return 10_000

            def __getitem__(self, i):
                ids = torch.randint(0, model.config.vocab_size, (a.seq,))
                return {"input_ids": ids, "labels": ids.clone()}

        args = TrainingArguments(output_dir=str(out_dir), per_device_train_batch_size=a.batch,
                                 gradient_accumulation_steps=a.accum, gradient_checkpointing=a.ckpt,
                                 gradient_checkpointing_kwargs={"use_reentrant": False},
                                 max_steps=a.steps, bf16=True, learning_rate=1e-4, report_to=[],
                                 logging_strategy="no", save_strategy="no", disable_tqdm=True,
                                 dataloader_num_workers=0)
        t1 = time.time()
        Trainer(model=model, args=args, train_dataset=Rand(), callbacks=[cb]).train()
        rec.update(result="finished", train_s=round(time.time() - t1, 1))
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("[fitcfg]"):
            rec.update(result="stopped_by_plugin", message=msg[:300])
        elif "out of memory" in msg.lower() or "NVML_SUCCESS" in msg:
            rec.update(result="oom", message=msg[:200])
        else:
            rec.update(result="error", message=msg[:300], tb=traceback.format_exc()[-800:])
    except Exception as e:  # noqa: BLE001
        is_oom = "OutOfMemory" in type(e).__name__
        rec.update(result="oom" if is_oom else "error", message=f"{type(e).__name__}: {str(e)[:300]}",
                   tb=traceback.format_exc()[-800:])
    rec["wall_s"] = round(time.time() - t0, 1)
    from dataclasses import asdict
    rec["detected"] = asdict(cb.detected) if cb.detected else None
    rec["prediction"] = asdict(cb.prediction) if cb.prediction else None
    rec["measured"] = cb.measured
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    with open(a.log, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    p, m = rec["prediction"] or {}, rec["measured"] or {}
    print(f"[plugin-check] {a.tag}: {rec['result']} · 예측 {p.get('pred_mem_gb')}GB · 실측 {m.get('device_peak_gb')}GB · "
          f"감지 {(rec['detected'] or {}).get('method')}/ckpt={(rec['detected'] or {}).get('grad_checkpointing')} "
          f"· {rec['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
