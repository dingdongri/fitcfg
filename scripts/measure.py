"""Phase 2 측정 하네스 — ★이 파일은 GPU에서 실행한다 (Colab/Kaggle/학교 서버).

설정 하나를 받아 짧은 파인튜닝을 돌리고, 메모리(구성요소별)와 tok/s를 측정해
CSV(기본 data/measurements.csv)에 한 줄 append 한다. 이 CSV가 cost model 보정(Phase 3)의 근거.

    python scripts/measure.py --model llama-3.2-1b --method qlora \
        --batch 4 --seq 1024 --steps 15 --ckpt

여러 설정을 쓸어담을 때는 scripts/sweep.py 를 쓴다 (설정마다 별도 프로세스 → OOM 후 오염 없음).

기록하는 메모리 (전부 GB)
    mem_model_gb     모델+LoRA 준비 직후 allocated        → weights 항 검증용
    mem_steady_gb    학습 후 zero_grad 뒤 allocated        → weights + optimizer 항 검증용
    peak_alloc_gb    학습 구간 max_memory_allocated        → PyTorch 텐서 peak
    peak_reserved_gb 학습 구간 max_memory_reserved         → 할당자 캐시 포함
    other_procs_gb   시작 시점 다른 프로세스의 GPU 사용량      → 아래 계산에서 제외
    nontorch_gb      (GPU 사용량) − reserved − other      → CUDA 컨텍스트, cuBLAS 워크스페이스 등
    device_peak_gb   nontorch_gb + peak_reserved_gb        → ★보정 타깃 (OOM을 실제로 결정하는 값)

주의
    - T4(Compute 7.5)는 bf16 미지원 → --dtype auto 이면 fp16 사용.
      학습 루프는 HF Trainer와 같게: autocast + (fp16이면) GradScaler, fp16 Full-FT는 fp32 가중치(AMP).
    - 데이터는 랜덤 토큰(메모리/속도 측정 목적).
    - Llama-3.2-1B는 HF gated 모델. 권한이 없으면 --hf-name unsloth/Llama-3.2-1B (동일 가중치 미러).
"""
from __future__ import annotations
import argparse, csv, sys, time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fitcfg.models import get_model  # noqa: E402

DEFAULT_CSV = REPO / "data" / "measurements.csv"
GiB = 1024 ** 3

FIELDS = [
    # 식별/설정
    "run_id", "timestamp", "tag", "model", "hf_name", "method", "batch", "seq", "ckpt",
    "optim", "lora_rank", "lora_targets", "gpus", "zero", "dtype", "steps",
    # 메모리
    "mem_model_gb", "mem_steady_gb", "peak_alloc_gb", "peak_reserved_gb",
    "other_procs_gb", "nontorch_gb", "device_peak_gb", "load_peak_alloc_gb", "gpu_total_gb",
    # 속도
    "tokens_per_s", "step_time_s",
    # 환경/결과
    "gpu_name", "torch", "transformers", "peft", "bitsandbytes", "ok", "note",
]

LORA_TARGET_MODULES = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


# 계열마다 다른 투영 행렬 이름 (Phi-3: q·k·v 합본, gate·up 합본)
LORA_TARGET_OVERRIDE = {
    "phi-3-mini": {"attn": ["qkv_proj", "o_proj"], "all": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]},
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="fitcfg 측정 하네스 (GPU 필요)")
    p.add_argument("--model", required=True, help="fitcfg 모델 키 (예: llama-3.2-1b)")
    p.add_argument("--hf-name", default=None, help="HF repo 이름 덮어쓰기 (gated 우회용 미러 등)")
    p.add_argument("--method", default="qlora", choices=["full", "lora", "qlora"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3, help="처리량 계산에서 제외할 앞쪽 스텝 수")
    p.add_argument("--ckpt", action="store_true", help="gradient checkpointing on")
    p.add_argument("--optim", default="adamw", choices=["adamw", "adamw8bit"])
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-targets", default="attn", choices=list(LORA_TARGET_MODULES))
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16"])
    p.add_argument("--out", default=str(DEFAULT_CSV), help="결과 CSV 경로")
    p.add_argument("--tag", default="", help="자유 메모 (예: colab-t4)")
    p.add_argument("--run-id", default=None, help="sweep.py가 부여하는 실행 ID")
    return p.parse_args(argv)


def append_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != FIELDS:
            raise SystemExit(f"ERROR: {path} 의 헤더가 현재 스키마와 다릅니다. "
                             f"다른 --out 경로를 쓰거나 파일을 옮겨주세요.")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def gpu_used_bytes() -> int:
    """nvidia-smi 수준의 사용량 (다른 프로세스 포함)."""
    import torch
    free, total = torch.cuda.mem_get_info()
    return total - free


def nvsmi_used_bytes() -> int:
    """torch(CUDA 컨텍스트)를 만들기 전에 nvidia-smi로 현재 GPU 사용량 조회. 실패 시 0."""
    import subprocess
    try:
        # MIG 슬라이스: memory.used가 부모 GPU 전체(다른 사용자 슬라이스 포함)라 의미 없음 → 0
        if "MIG" in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout:
            return 0
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                              "-i", "0"], capture_output=True, text=True, check=True).stdout
        return int(float(out.strip().splitlines()[0]) * 1024 ** 2)
    except Exception:
        return 0


def pick_dtype(name: str):
    import torch
    if name == "bf16":
        return torch.bfloat16, "bf16"
    if name == "fp16":
        return torch.float16, "fp16"
    major, _ = torch.cuda.get_device_capability(0)
    return (torch.bfloat16, "bf16") if major >= 8 else (torch.float16, "fp16")


def build_model(args, hf_name: str, dtype, device_map, compute_dtype=None):
    """모델 로드 + (Q)LoRA 부착 + gradient checkpointing. 반환: 학습 준비된 모델."""
    import transformers
    from packaging import version
    from transformers import AutoModelForCausalLM

    kw = dict(device_map=device_map, attn_implementation="sdpa")
    # transformers 4.56+ 는 torch_dtype → dtype 으로 이름이 바뀜
    if version.parse(transformers.__version__) >= version.parse("4.56"):
        kw["dtype"] = dtype
    else:
        kw["torch_dtype"] = dtype

    if args.method == "qlora":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype or dtype, bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(hf_name, **kw)
    model.config.use_cache = False          # 학습 시 KV cache 불필요
    ckpt_kwargs = {"use_reentrant": False}

    if args.method == "qlora":
        from peft import prepare_model_for_kbit_training
        # 주의: 이 함수가 비양자화 파라미터(임베딩, lm_head, norm)를 fp32로 올린다 → memory.py 반영
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.ckpt,
            gradient_checkpointing_kwargs=ckpt_kwargs)
    elif args.ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=ckpt_kwargs)
        if args.method == "lora":
            model.enable_input_require_grads()   # 동결된 입력에서도 grad가 흐르도록

    if args.method in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model
        lora = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank,
                          target_modules=LORA_TARGET_OVERRIDE.get(args.model, LORA_TARGET_MODULES)[args.lora_targets],
                          lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
    model.train()
    return model


def make_optimizer(model, name: str):
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=1e-5)
    return torch.optim.AdamW(params, lr=1e-5)


def _versions() -> dict:
    out = {}
    for mod in ("torch", "transformers", "peft", "bitsandbytes"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = ""
    return out


def main(argv=None):
    args = parse_args(argv)
    spec = get_model(args.model)
    hf_name = args.hf_name or spec.name
    out_path = Path(args.out)

    other = nvsmi_used_bytes()   # ★ CUDA 초기화 전에 측정 (노트북 커널 등 다른 프로세스 몫)
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU가 필요합니다. Colab(런타임 유형: T4 GPU)이나 학교 서버에서 실행하세요.")
        sys.exit(1)

    row = {k: "" for k in FIELDS}
    row.update(
        run_id=args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        timestamp=datetime.now().isoformat(timespec="seconds"), tag=args.tag,
        model=args.model, hf_name=hf_name, method=args.method, batch=args.batch,
        seq=args.seq, ckpt=int(args.ckpt), optim=args.optim, lora_rank=args.lora_rank,
        lora_targets=args.lora_targets, gpus=1, zero=0, steps=args.steps,
        gpu_name=torch.cuda.get_device_name(0),
        gpu_total_gb=round(torch.cuda.get_device_properties(0).total_memory / GiB, 3),
        ok=0, **_versions(),
    )

    try:
        dtype, dtype_name = pick_dtype(args.dtype)
        # HF Trainer(fp16=True/bf16=True)와 같은 방식으로 측정한다 → 생성되는 config와 일치
        #  - forward는 autocast(16bit)
        #  - fp16이면 GradScaler (학습 파라미터가 fp32여야 함)
        #  - fp16 + Full-FT: fp16 가중치를 직접 학습할 수 없으므로 fp32로 로드 (AMP)
        amp_fp32_weights = args.method == "full" and dtype_name == "fp16"
        load_dtype = torch.float32 if amp_fp32_weights else dtype
        row["dtype"] = "fp16-amp" if amp_fp32_weights else dtype_name
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype_name == "fp16"))

        # ---- 1) 모델 준비 ----
        torch.cuda.reset_peak_memory_stats()
        model = build_model(args, hf_name, load_dtype, device_map={"": 0}, compute_dtype=dtype)
        torch.cuda.synchronize()
        row["load_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / GiB, 4)
        row["mem_model_gb"] = round(torch.cuda.memory_allocated() / GiB, 4)

        opt = make_optimizer(model, args.optim)
        ids = torch.randint(0, spec.vocab, (args.batch, args.seq), device="cuda")

        # ---- 2) 학습 구간: 여기서부터 peak 측정 ----
        torch.cuda.reset_peak_memory_stats()
        times = []
        for step in range(args.steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=dtype):
                out = model(input_ids=ids, labels=ids)
            scaler.scale(out.loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            del out
            torch.cuda.synchronize()
            if step >= args.warmup:
                times.append(time.perf_counter() - t0)

        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        nontorch = gpu_used_bytes() - torch.cuda.memory_reserved() - other
        step_time = sum(times) / max(len(times), 1)
        row.update(
            mem_steady_gb=round(torch.cuda.memory_allocated() / GiB, 4),
            peak_alloc_gb=round(peak_alloc / GiB, 4),
            peak_reserved_gb=round(peak_reserved / GiB, 4),
            other_procs_gb=round(other / GiB, 4),
            nontorch_gb=round(nontorch / GiB, 4),
            device_peak_gb=round((nontorch + peak_reserved) / GiB, 4),
            step_time_s=round(step_time, 4),
            tokens_per_s=round(args.batch * args.seq / step_time, 1) if times else "",
            ok=1,
        )
        print(f"OK   device_peak={row['device_peak_gb']:.2f}GB "
              f"(alloc {row['peak_alloc_gb']:.2f} / reserved {row['peak_reserved_gb']:.2f}) "
              f"{row['tokens_per_s']} tok/s  step={step_time*1000:.0f}ms")
    except Exception as e:  # noqa: BLE001 — 어떤 실패든 CSV에 남긴다
        msg = str(e)
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in msg.lower()
        # MIG 슬라이스: 메모리 할당 실패 시 PyTorch가 OOM 메시지에 프로세스별 사용량을 넣으려고 NVML을 부르는데,
        # MIG에서는 NVML 권한이 없어 OutOfMemoryError 대신 'NVML_SUCCESS == r INTERNAL ASSERT'로 죽는다 → OOM으로 분류
        nvml_oom = "NVML_SUCCESS == r" in msg and "CUDACachingAllocator" in msg
        try:  # 실패 직전까지의 최대 사용량도 남긴다 (OOM이면 GPU 용량 근처여야 정상)
            row.update(peak_alloc_gb=round(torch.cuda.max_memory_allocated() / GiB, 4),
                       peak_reserved_gb=round(torch.cuda.max_memory_reserved() / GiB, 4))
        except Exception:
            pass
        row.update(ok=0, note="OOM" if is_oom else ("OOM (NVML assert, MIG)" if nvml_oom
                                                    else f"{type(e).__name__}: {msg[:150]}"))
        print(f"FAIL {row['note']}")
    finally:
        append_row(out_path, row)
        print(f"-> appended to {out_path}")


if __name__ == "__main__":
    main()
