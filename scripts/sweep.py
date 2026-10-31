"""Phase 2 측정 스윕 — 여러 설정을 measure.py 로 차례대로 측정한다 (GPU에서 실행).

    python scripts/sweep.py --preset colab-core --dry-run     # 목록 + 현재 cost model 예측 메모리만 출력
    python scripts/sweep.py --preset colab-core               # 실제 측정
    python scripts/sweep.py --preset colab-core colab-qwen3b --tag colab-t4

설계 포인트
    - 설정마다 **별도 프로세스**로 실행: OOM이 나도 다음 측정이 오염되지 않는다.
    - **이어하기**: CSV에 이미 있는 (설정, GPU) 조합은 건너뛴다. Colab이 끊겨도 다시 실행하면 이어짐.
    - 이 프로세스는 torch를 import하지 않는다 (GPU 컨텍스트를 잡아 측정을 오염시키지 않도록).
    - 예측 메모리가 GPU 용량의 --skip-over 배를 넘는 설정은 건너뛴다(뻔한 OOM에 시간 낭비 X).
      단, 1.0~skip-over 사이는 일부러 돌려서 **OOM 데이터**도 모은다 (OOM 회피율 평가용).
"""
from __future__ import annotations
import argparse, csv, subprocess, sys, time
from datetime import datetime
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fitcfg import TrainConfig, estimate, get_model, match_gpu  # noqa: E402

DEFAULT_CSV = REPO / "data" / "measurements.csv"
GiB = 1024 ** 3


def grid(model, method, batch, seq, ckpt=(True, False), optim=("adamw",),
         rank=(16,), targets=("attn",)):
    """곱집합으로 설정 dict 목록을 만든다."""
    return [dict(model=model, method=method, batch=b, seq=s, ckpt=c, optim=o,
                 lora_rank=r, lora_targets=t)
            for b, s, c, o, r, t in product(batch, seq, ckpt, optim, rank, targets)]


# ---------------------------------------------------------------------------
# 프리셋 (T4 15GB 기준으로 설계. L4 24GB / A100에서도 그대로 사용 가능)
# ---------------------------------------------------------------------------
PRESETS: dict[str, list[dict]] = {
    # 동작 확인용 1개
    "smoke": grid("llama-3.2-1b", "qlora", [1], [512], ckpt=[True]),

    # ★ 핵심: Llama-3.2-1B, 3가지 method × batch/seq/ckpt  (~44개)
    "colab-core": (
        grid("llama-3.2-1b", "qlora", [1, 2, 4, 8], [512, 1024, 2048])
        + grid("llama-3.2-1b", "lora", [1, 2, 4], [1024, 2048])
        + grid("llama-3.2-1b", "full", [1, 2, 4], [1024], optim=["adamw8bit"])
        + grid("llama-3.2-1b", "full", [1, 2], [1024], ckpt=[True])
    ),

    # 2번째 모델: activation 항과 logits 항을 분리하려면 L·H/V 비율이 다른 모델이 필요 (~22개)
    "colab-qwen3b": (
        grid("qwen2.5-3b", "qlora", [1, 2, 4], [512, 1024, 2048])
        + grid("qwen2.5-3b", "lora", [1, 2], [1024])
    ),

    # LoRA rank / 대상 행렬 변화 (어댑터 항 검증, 6개)
    "colab-lora-shape": grid("llama-3.2-1b", "qlora", [4], [1024], ckpt=[True],
                             rank=[8, 16, 64], targets=["attn", "all"]),

    # --- gfarm MIG 3g.40gb (A100, bf16, FlashAttention 커널) ---
    # ① 7B/8B: T4에서는 못 하던 본 타깃 크기. QLoRA/LoRA × batch × seq × ckpt (Full은 8B×16B/param=128GB라 제외)
    "mig-8b": (
        grid("qwen2.5-7b", "qlora", [1, 2, 4], [1024, 2048])
        + grid("qwen2.5-7b", "lora", [1, 2], [1024, 2048])
        + grid("llama-3.1-8b", "qlora", [1, 2, 4], [1024, 2048])
        + grid("llama-3.1-8b", "lora", [1, 2], [1024, 2048])
    ),
    # ② 소형 모델 재측정: 같은 설정을 T4(fp16·flash 없음)와 A100-MIG(bf16·flash)에서 비교 → GPU 종류 일반화 확인
    "mig-small": (
        grid("llama-3.2-1b", "qlora", [1, 4, 8], [1024, 2048])
        + grid("llama-3.2-1b", "lora", [1, 4], [1024, 2048])
        + grid("llama-3.2-1b", "full", [1, 4], [1024], optim=["adamw", "adamw8bit"])
        + grid("qwen2.5-3b", "qlora", [1, 4], [1024, 2048])
        + grid("qwen2.5-3b", "full", [1, 2], [1024], optim=["adamw8bit"])
    ),
    # ③ 8B 확장: 긴 시퀀스(4k·8k), LoRA rank·대상 행렬, 8bit 옵티마이저
    "mig-8b-ext": (
        grid("llama-3.1-8b", "qlora", [1], [4096, 8192], ckpt=[True])
        + grid("llama-3.1-8b", "lora", [1], [4096, 8192], ckpt=[True])
        + grid("qwen2.5-7b", "qlora", [1], [4096, 8192], ckpt=[True])
        + grid("llama-3.1-8b", "qlora", [2], [1024], ckpt=[True], rank=[8, 64], targets=["attn", "all"])
        + grid("llama-3.1-8b", "lora", [2], [1024], ckpt=[True], rank=[8, 64], targets=["attn", "all"])
        + grid("llama-3.1-8b", "lora", [2], [2048], ckpt=[True], optim=["adamw8bit"])
        + grid("llama-3.1-8b", "qlora", [2], [2048], ckpt=[True], optim=["adamw8bit"])
    ),
    # ⑤ 할당기 단편화 대응 시험: 같은 설정을 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 로 다시 (별도 CSV)
    "mig-frag": (
        grid("llama-3.1-8b", "qlora", [1, 2, 4], [1024, 2048], ckpt=[True])
        + grid("qwen2.5-7b", "qlora", [1, 2, 4], [1024, 2048], ckpt=[True])
        + grid("llama-3.1-8b", "lora", [1, 2], [1024, 2048], ckpt=[True])
        + grid("llama-3.1-8b", "qlora", [1, 2], [1024], ckpt=[False])
    ),
    # ⑥ QLoRA 일반화 규칙 검증 (2026-10-29): 처음 보는 4개 계열 × QLoRA·LoRA 같은 설정 → 'QLoRA 활성값 ≈ LoRA × k' 규칙 확인
    "mig-qlora-new": [d for m in ("smollm2-1.7b", "olmo2-1b", "granite-3.3-2b", "qwen3-4b")
                      for d in grid(m, "qlora", [1, 4], [1024, 2048]) + grid(m, "lora", [1, 4], [1024, 2048])],
    # ⑤ 긴 시퀀스 (2026-10-31): 4096 이상 실측이 6개뿐 → 긴 문서 학습(가장 OOM이 잦은 구간) 확인
    "mig-long": (
        grid("llama-3.1-8b", "qlora", [1], [4096, 8192, 16384], ckpt=[True])
        + grid("llama-3.1-8b", "qlora", [1], [4096, 8192], ckpt=[False])
        + grid("llama-3.1-8b", "lora", [1], [4096, 8192, 16384], ckpt=[True])
        + grid("qwen2.5-7b", "qlora", [1], [8192, 16384], ckpt=[True])
        + grid("qwen2.5-3b", "qlora", [1, 2], [8192], ckpt=[True])
        + grid("qwen2.5-3b", "lora", [1], [8192, 16384], ckpt=[True])
        + grid("llama-3.2-1b", "lora", [1], [8192, 16384], ckpt=[True, False])
        + grid("llama-3.2-1b", "full", [1], [8192], ckpt=[True])
    ),
    # ④ 처음 보는 계열 실측 (보정에 안 씀 = 일반화 시험). FakeTensor 추적 결과(family_report)와도 대조
    "mig-family": [d for m in ("gemma-3-1b", "gemma-2-2b", "phi-3-mini", "mistral-7b-v0.3")
                   for d in grid(m, "qlora", [1, 4], [1024, 2048]) + grid(m, "lora", [1, 4], [1024, 2048])],
}


# 승인(gated)이 필요한 원본 대신 쓰는 공개 사본 (같은 가중치). --hf-override 가 우선.
PUBLIC_MIRRORS = {
    "llama-3.1-8b": "unsloth/Meta-Llama-3.1-8B",
    "llama-3.2-1b": "unsloth/Llama-3.2-1B",
    "mistral-7b-v0.3": "unsloth/mistral-7b-v0.3",
}


def config_key(d: dict, gpu_name: str) -> tuple:
    return (d["model"], d["method"], int(d["batch"]), int(d["seq"]), int(bool(int(d["ckpt"]))),
            d["optim"], int(d["lora_rank"]), d["lora_targets"], gpu_name)


def done_keys(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {config_key(r, r["gpu_name"]) for r in csv.DictReader(f)}


def gpu_info() -> tuple[str, float]:
    """torch 없이 nvidia-smi로 GPU 이름/용량(GB) 조회.
    MIG 슬라이스는 --query-gpu가 부모 GPU 이름만 주고 용량은 권한 오류가 나므로 `nvidia-smi -L`로 프로필을 읽어
    torch.cuda.get_device_name()과 같은 'NVIDIA A100-SXM4-80GB MIG 3g.40gb' 형태로 맞춘다."""
    import re
    try:
        lines = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                               check=True).stdout.splitlines()
        m = next((re.search(r"MIG\s+(\d+g\.\d+gb)", l) for l in lines if "MIG" in l), None)
        if m:
            parent = re.search(r"GPU 0: (.+?) \(UUID", lines[0]).group(1).strip()
            name = f"{parent} MIG {m.group(1)}"
            g = match_gpu(name)
            return name, (g.mem_gb if g else 0.0)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
        name, mib = [x.strip() for x in out.split(",")]
        # torch.cuda.get_device_name 과 같은 문자열이 나오도록 그대로 사용
        return name, float(mib) / 1024
    except Exception:
        return "", 0.0


def predict_gb(d: dict, gpu_name: str = "") -> float:
    g = match_gpu(gpu_name)
    cfg = TrainConfig(model=d["model"], method=d["method"], batch_size=d["batch"],
                      seq_len=d["seq"], grad_checkpointing=d["ckpt"], optim=d["optim"],
                      lora_rank=d["lora_rank"], lora_targets=d["lora_targets"],
                      dtype="fp16" if (g and not g.bf16) else "bf16",
                      flash_attn=g.flash if g else True)
    return estimate(cfg, get_model(d["model"])).total / GiB


def run_one(d: dict, args, run_id: str) -> int:
    cmd = [sys.executable, str(REPO / "scripts" / "measure.py"),
           "--model", d["model"], "--method", d["method"],
           "--batch", str(d["batch"]), "--seq", str(d["seq"]),
           "--optim", d["optim"], "--lora-rank", str(d["lora_rank"]),
           "--lora-targets", d["lora_targets"], "--steps", str(args.steps),
           "--dtype", args.dtype, "--out", str(args.out), "--run-id", run_id]
    if d["ckpt"]:
        cmd.append("--ckpt")
    if args.tag:
        cmd += ["--tag", args.tag]
    hf = args.hf_override.get(d["model"]) or PUBLIC_MIRRORS.get(d["model"])
    if hf:
        cmd += ["--hf-name", hf]
    try:
        return subprocess.run(cmd, timeout=args.timeout).returncode
    except subprocess.TimeoutExpired:
        return -999


def record_crash(args, d: dict, run_id: str, rc: int, gpu_name: str, gpu_gb: float):
    """measure.py 가 행을 남기지 못하고 죽은 경우(커널 크래시, 타임아웃) 실패 행을 대신 기록."""
    from measure import FIELDS, append_row  # scripts/ 가 sys.path 에 있음
    if Path(args.out).exists():
        with open(args.out, newline="") as f:
            if any(r["run_id"] == run_id for r in csv.DictReader(f)):
                return
    row = {k: "" for k in FIELDS}
    row.update(run_id=run_id, timestamp=datetime.now().isoformat(timespec="seconds"),
               tag=args.tag, model=d["model"], method=d["method"], batch=d["batch"],
               seq=d["seq"], ckpt=int(d["ckpt"]), optim=d["optim"], lora_rank=d["lora_rank"],
               lora_targets=d["lora_targets"], gpus=1, zero=0, steps=args.steps,
               gpu_name=gpu_name, gpu_total_gb=round(gpu_gb, 3), ok=0,
               note="TIMEOUT" if rc == -999 else f"CRASH rc={rc}")
    append_row(Path(args.out), row)


def parse_hf_override(items: list[str]) -> dict:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        if not v:
            raise SystemExit(f"--hf-override 형식은 모델키=repo 입니다: {it}")
        out[k] = v
    return out


def main():
    p = argparse.ArgumentParser(description="fitcfg 측정 스윕")
    p.add_argument("--preset", nargs="+", default=["colab-core"], choices=list(PRESETS))
    p.add_argument("--out", default=str(DEFAULT_CSV))
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16"])
    p.add_argument("--tag", default="")
    p.add_argument("--timeout", type=int, default=1200, help="설정 1개당 최대 초")
    p.add_argument("--skip-over", type=float, default=1.5,
                   help="예측 메모리 > GPU용량 × 이 값 이면 건너뜀 (0이면 끔)")
    p.add_argument("--hf-override", nargs="*", default=[],
                   help="모델키=HF repo (예: llama-3.2-1b=unsloth/Llama-3.2-1B)")
    p.add_argument("--dry-run", action="store_true", help="실행하지 않고 목록만 출력")
    args = p.parse_args()
    args.hf_override = parse_hf_override(args.hf_override)
    args.out = Path(args.out)
    sys.path.insert(0, str(REPO / "scripts"))

    gpu_name, gpu_gb = gpu_info()
    configs, seen = [], set()
    for d in (d for name in args.preset for d in PRESETS[name]):   # 프리셋 간 중복 제거
        if (k := config_key(d, "")) not in seen:
            seen.add(k)
            configs.append(d)
    # 같은 모델끼리 묶고, 예측 메모리가 작은 것부터 (모델 캐시 재사용 + 앞쪽 성공 데이터 확보)
    configs.sort(key=lambda d: (d["model"], predict_gb(d, gpu_name)))

    done = done_keys(args.out)
    todo, skipped_done, skipped_big = [], 0, 0
    for d in configs:
        if config_key(d, gpu_name) in done:
            skipped_done += 1
        elif args.skip_over and gpu_gb and predict_gb(d, gpu_name) > gpu_gb * args.skip_over:
            skipped_big += 1
        else:
            todo.append(d)

    print(f"GPU: {gpu_name or '(감지 안 됨)'} {gpu_gb:.1f}GB | 전체 {len(configs)}개, "
          f"완료됨 {skipped_done}개, 예측상 과대 {skipped_big}개 건너뜀 → 실행 {len(todo)}개")
    for i, d in enumerate(todo, 1):
        pred = predict_gb(d, gpu_name)
        flag = "  (예측상 OOM 가능)" if gpu_gb and pred > gpu_gb else ""
        print(f"  [{i:>3}] {d['model']:<13} {d['method']:<5} b={d['batch']:<2} s={d['seq']:<4} "
              f"ckpt={int(d['ckpt'])} {d['optim']:<9} r={d['lora_rank']:<2} {d['lora_targets']:<4} "
              f"pred={pred:5.1f}GB{flag}")
    if args.dry_run or not todo:
        return
    if not gpu_name:
        raise SystemExit("ERROR: nvidia-smi로 GPU를 찾지 못했습니다. GPU 런타임에서 실행하세요.")

    t_start = time.time()
    for i, d in enumerate(todo, 1):
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        print(f"\n=== [{i}/{len(todo)}] {d}", flush=True)
        rc = run_one(d, args, run_id)
        if rc != 0:
            record_crash(args, d, run_id, rc, gpu_name, gpu_gb)
        elapsed = time.time() - t_start
        eta = elapsed / i * (len(todo) - i)
        print(f"    경과 {elapsed/60:.1f}분, 남은 예상 {eta/60:.1f}분", flush=True)
    print(f"\n완료. 결과: {args.out}")


if __name__ == "__main__":
    main()
