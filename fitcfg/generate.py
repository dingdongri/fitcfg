"""실행 가능한 학습 번들 생성 (Phase 5).

추천 설정 하나 → 폴더 하나:
    train.py             HF Trainer 기반 학습 스크립트 (설정은 맨 위 CONFIG에 박혀 있음)
    ds_config.json       DeepSpeed 설정 (ZeRO 전략일 때만)
    launch.sh            실행 명령 (단일 GPU: python / 멀티·DeepSpeed: torchrun)
    requirements.txt     필요한 패키지
    README.md            예측값·실행법·검증법
    fitcfg_plan.json     추천 입력/예측 전체 (재현·추적용)

설계 결정
    - 실행기는 torchrun 하나로 통일하고, DDP/DeepSpeed 선택은 train.py의 TrainingArguments가 한다.
      (accelerate yaml을 따로 두면 설정 출처가 둘이 되어 예측과 실제가 어긋나기 쉬움)
    - 데이터는 seq_len 블록으로 packing → 모든 스텝의 토큰 수가 일정 → 메모리 예측의 전제와 일치.
"""
from __future__ import annotations
import io
import json
import pprint
import zipfile
from pathlib import Path

from .models import get_model
from .recommend import PlanRequest, Recommendation

TEMPLATE = Path(__file__).with_name("templates") / "train.py.tmpl"

LORA_MODULES = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}
DEFAULT_LR = {"full": 1e-5, "lora": 2e-4, "qlora": 2e-4}


def _optim_name(rec: Recommendation) -> str:
    """HF TrainingArguments.optim 값. ZeRO일 때는 ds_config의 optimizer가 우선한다."""
    if rec.config.method == "full" and rec.config.optim == "adamw8bit":
        return "adamw_bnb_8bit"
    return "adamw_torch"


def build_config(rec: Recommendation, req: PlanRequest, dataset: str = "data.jsonl",
                 text_field: str = "text", epochs: float | None = None,
                 learning_rate: float | None = None, dataset_split: str = "train") -> dict:
    cfg, spec = rec.config, get_model(req.model)
    use_ds = cfg.zero_stage >= 1 or cfg.cpu_offload_optim or cfg.cpu_offload_params
    tp = rec.throughput
    return {
        "model_key": req.model,
        "model_name": spec.name,
        "method": cfg.method,
        "dataset": dataset,
        "text_field": text_field,
        "dataset_split": dataset_split,
        "seq_len": cfg.seq_len,
        "micro_batch": cfg.batch_size,
        "grad_accum": rec.grad_accum,
        "gpus": cfg.gpus,
        "gradient_checkpointing": cfg.grad_checkpointing,
        "dtype": rec.dtype,
        "optim": _optim_name(rec),
        "optim_label": cfg.optim,
        "zero_stage": cfg.zero_stage,
        "cpu_offload_optim": cfg.cpu_offload_optim,
        "cpu_offload_params": cfg.cpu_offload_params,
        "lora_targets": cfg.lora_targets,
        "deepspeed": "ds_config.json" if use_ds else None,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": 2 * cfg.lora_rank,
        "lora_target_modules": LORA_MODULES[cfg.lora_targets],
        "learning_rate": learning_rate or DEFAULT_LR[cfg.method],
        "epochs": epochs if epochs is not None else req.epochs,
        "output_dir": "outputs",
        "save_model": True,
        "prediction": {
            "peak_gb": rec.pred_mem_gb,
            "gpu_mem_gb": req.gpu.mem_gb,
            "tokens_per_s": round(tp.tokens_per_s, 1),
            "tokens_per_s_low": round(tp.low, 1),
            "tokens_per_s_high": round(tp.high, 1),
        },
    }


def ds_config(rec: Recommendation) -> dict | None:
    """DeepSpeed 설정. HF Trainer가 "auto" 값을 TrainingArguments로 채운다."""
    cfg = rec.config
    if not (cfg.zero_stage >= 1 or cfg.cpu_offload_optim or cfg.cpu_offload_params):
        return None
    zero = {
        "stage": cfg.zero_stage,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
    }
    if cfg.cpu_offload_optim:
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    if cfg.zero_stage >= 3:
        zero.update({
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_gather_16bit_weights_on_model_save": True,
        })
        if cfg.cpu_offload_params:
            zero["offload_param"] = {"device": "cpu", "pin_memory": True}
    return {
        "bf16": {"enabled": "auto"},
        "fp16": {"enabled": "auto", "loss_scale": 0, "initial_scale_power": 16},
        # offload 시 DeepSpeed의 CPU Adam(빠름)을 쓰도록 optimizer를 명시
        "optimizer": {"type": "AdamW", "params": {"lr": "auto", "betas": "auto", "eps": "auto",
                                                   "weight_decay": "auto"}},
        # scheduler는 비워둠 → TrainingArguments의 cosine 스케줄러를 그대로 사용
        "zero_optimization": zero,
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    }


def launch_sh(rec: Recommendation) -> str:
    cfg = rec.config
    use_ds = ds_config(rec) is not None
    lines = ["#!/usr/bin/env bash", "# fitcfg가 생성한 실행 스크립트", "set -euo pipefail",
             'cd "$(dirname "$0")"', ""]
    if cfg.gpus == 1:
        # GPU가 여러 장 보이는 머신에서 GPU 1장 설정을 그냥 실행하면 HF Trainer가 nn.DataParallel로 모든 GPU를 쓴다
        # (Kaggle T4×2 실측: 메모리 +50%, 속도 1/4) → 예측과 같은 'GPU 1장'으로 고정. 이미 지정했으면 그대로.
        lines.append('export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"')
    if cfg.gpus == 1 and not use_ds:
        lines.append("python train.py \"$@\"")
    else:
        lines.append(f"torchrun --standalone --nproc_per_node {cfg.gpus} train.py \"$@\"")
    return "\n".join(lines) + "\n"


def requirements(rec: Recommendation) -> str:
    reqs = ["torch>=2.3", "transformers>=4.46", "datasets>=2.20", "accelerate>=0.33"]
    if rec.config.method in ("lora", "qlora"):
        reqs.append("peft>=0.12")
    if rec.config.method == "qlora" or _optim_name(rec) == "adamw_bnb_8bit":
        reqs.append("bitsandbytes>=0.43")
    if ds_config(rec):
        reqs.append("deepspeed>=0.14")
    return "\n".join(reqs) + "\n"


def readme(rec: Recommendation, req: PlanRequest, conf: dict) -> str:
    m, tp = rec.memory.as_gb(), rec.throughput
    hours = ""
    if rec.hours:
        h, fast, slow = rec.hours
        hours = f"| 예상 학습 시간 | **{h:.1f}시간** (범위 {fast:.1f}~{slow:.1f}) |\n"
    warn = "".join(f"- ⚠️ {w}\n" for w in rec.warnings)
    cmd = "bash launch.sh"
    return f"""# fitcfg 학습 번들 — {req.model} / {rec.config.method}

## 추천 설정
{rec.reason}

| 항목 | 값 |
|---|---|
| GPU | {req.gpu.label} × {rec.config.gpus} |
| micro-batch × GPU × 누적 | {rec.config.batch_size} × {rec.config.gpus} × {rec.grad_accum} = **{rec.effective_batch}** 시퀀스/스텝 |
| seq_len | {rec.config.seq_len} |
| 정밀도 | {rec.dtype} |
| **예측 peak 메모리 (GPU당)** | **{rec.pred_mem_gb:.2f} GB** / 허용 {rec.cap_gb:.2f} GB (GPU {req.gpu.mem_gb:.1f} GB) |
| 예측 처리량 | {tp.tokens_per_s:,.0f} tok/s (범위 {tp.low:,.0f}~{tp.high:,.0f}{', 보정됨' if tp.calibrated else ', 보정 전'}) |
{hours}
메모리 분해 (GB): weights {m['weights']} · gradients {m['gradients']} · optimizer {m['optimizer']} ·
activations {m['activations']} · logits {m['logits']} · overhead {m['overhead']}

{warn}
## 실행
1. 패키지: `pip install -r requirements.txt`
2. 데이터: `{conf['dataset']}` — 한 줄에 `{{"{conf['text_field']}": "..."}}` 형식의 jsonl
   (또는 `train.py`의 `CONFIG["dataset"]`에 HF 데이터셋 이름). 토큰을 이어붙여 {rec.config.seq_len}토큰 블록으로 학습합니다.
3. 학습: `{cmd}`
   - 빠른 확인: `FITCFG_MAX_STEPS=20 {cmd}` (20스텝만 돌리고 메모리·속도 리포트)
{"   - Llama 계열은 HF gated 모델: `huggingface-cli login` 필요" if "llama" in req.model else ""}

## 예측이 맞았는지 확인
학습이 끝나면 `outputs/fitcfg_run.json`에 **실측 peak 메모리·처리량**이 저장되고, 콘솔에 예측과 비교가 출력됩니다.
이 파일을 모아 fitcfg의 보정 데이터로 쓰면 예측이 더 정확해집니다.

> 설정(`micro_batch`, `seq_len`, `gradient_checkpointing`, `deepspeed`)을 바꾸면 메모리 예측이 달라집니다. fitcfg로 다시 확인하세요.
"""


def build_bundle(rec: Recommendation, req: PlanRequest, dataset: str = "data.jsonl",
                 text_field: str = "text", learning_rate: float | None = None,
                 dataset_split: str = "train") -> dict[str, str]:
    """{파일명: 내용} 번들."""
    conf = build_config(rec, req, dataset=dataset, text_field=text_field, learning_rate=learning_rate,
                        dataset_split=dataset_split)
    train = TEMPLATE.read_text(encoding="utf-8").replace(
        "__CONFIG__", pprint.pformat(conf, sort_dicts=False, width=100))
    files = {
        "train.py": train,
        "launch.sh": launch_sh(rec),
        "requirements.txt": requirements(rec),
        "README.md": readme(rec, req, conf),
        "fitcfg_plan.json": json.dumps(plan_record(rec, req), indent=2, ensure_ascii=False),
    }
    ds = ds_config(rec)
    if ds:
        files["ds_config.json"] = json.dumps(ds, indent=2)
    return files


def plan_record(rec: Recommendation, req: PlanRequest) -> dict:
    return {
        "request": {"model": req.model, "gpu": req.gpu.key, "gpu_mem_gb": req.gpu.mem_gb, "gpus": req.gpus,
                    "method": req.method, "seq_len": req.seq_len, "global_batch": req.global_batch,
                    "lora_rank": req.lora_rank, "lora_targets": req.lora_targets},
        "recommendation": {"strategy": rec.strategy, "micro_batch": rec.config.batch_size,
                           "grad_accum": rec.grad_accum, "gradient_checkpointing": rec.config.grad_checkpointing,
                           "optim": rec.config.optim, "zero_stage": rec.config.zero_stage,
                           "cpu_offload_optim": rec.config.cpu_offload_optim,
                           "cpu_offload_params": rec.config.cpu_offload_params, "dtype": rec.dtype},
        "prediction": {"memory_gb": rec.memory.as_gb(), "cap_gb": rec.cap_gb,
                       "tokens_per_s": rec.throughput.tokens_per_s, "tokens_per_s_low": rec.throughput.low,
                       "tokens_per_s_high": rec.throughput.high, "hours": rec.hours,
                       "throughput_calibrated": rec.throughput.calibrated},
        "warnings": rec.warnings,
    }


def write_bundle(files: dict[str, str], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out / name).write_text(content, encoding="utf-8")
    (out / "launch.sh").chmod(0o755)
    return out


def zip_bundle(files: dict[str, str], folder: str = "fitcfg_run") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            info = zipfile.ZipInfo(f"{folder}/{name}")
            info.external_attr = (0o755 if name.endswith(".sh") else 0o644) << 16
            z.writestr(info, content)
    return buf.getvalue()
