"""합성(가짜) 측정 데이터 생성기 — GPU 없이 calibrate 파이프라인을 검증/시연하기 위한 용도.

'진짜 상수'(TRUTH)를 사전값과 다르게 숨겨두고, sweep 프리셋 설정들에 대해
memory.py 식 + 노이즈 + 모델에 없는 작은 효과로 '실측'을 만든다.
보정이 제대로 동작하면 → 보정 후 MAPE가 노이즈 수준까지 내려가고 상수가 TRUTH 근처로 간다.

⚠️ 리포트/발표에 절대 실측처럼 쓰지 말 것. 파이프라인 테스트 전용.
"""
from __future__ import annotations
import csv, random, sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from fitcfg.memory import DEFAULT_CONSTANTS, TrainConfig, estimate, features, GB  # noqa: E402
from fitcfg.models import get_model  # noqa: E402
from fitcfg.hardware import get_gpu  # noqa: E402
from fitcfg.throughput import DEFAULT_TP, estimate_throughput  # noqa: E402

# 숨겨진 '진짜' 상수 (사전값과 일부러 다르게)
TRUTH = {
    "qlora": replace(DEFAULT_CONSTANTS, act_h=40.0, act_i=5.0, ckpt_h=2.5, logit_bytes=14.0,
                     nf4_extra=0.03, qlora_nonquant_bytes=4.0, overhead_gb=0.85),
    "lora": replace(DEFAULT_CONSTANTS, act_h=36.0, act_i=4.5, ckpt_h=2.4, logit_bytes=14.0,
                    overhead_gb=0.85),
    "full": replace(DEFAULT_CONSTANTS, act_h=36.0, act_i=4.5, ckpt_h=2.4, logit_bytes=12.0,
                    opt_temp=2.5, overhead_gb=0.85),
}
GPU_NAME, GPU_TOTAL_GB = "Tesla T4 (synthetic)", 14.74
# 숨겨진 '진짜' 처리량 상수 (T4는 사전값보다 느리다고 가정)
TRUTH_TP = replace(DEFAULT_TP, eff_full=0.26, eff_lora=0.21, eff_qlora=0.12, t_half=2048.0)


def write_synthetic_csv(path: str | Path, noise: float = 0.02, seed: int = 0) -> dict:
    from measure import FIELDS
    from sweep import PRESETS
    rng = random.Random(seed)
    configs = PRESETS["colab-core"] + PRESETS["colab-qwen3b"] + PRESETS["colab-lora-shape"]
    rows = []
    for i, d in enumerate(configs):
        cfg = TrainConfig(model=d["model"], method=d["method"], batch_size=d["batch"],
                          seq_len=d["seq"], grad_checkpointing=d["ckpt"], optim=d["optim"],
                          lora_rank=d["lora_rank"], lora_targets=d["lora_targets"], dtype="fp16",
                          flash_attn=False)
        spec, c = get_model(d["model"]), TRUTH[d["method"]]
        f = features(cfg, spec, c)
        total = estimate(cfg, spec, c).total / GB
        # 모델에 없는 효과: 할당자 단편화 ≈ 전체의 0~4% + 곱셈 노이즈
        actual = total * (1 + rng.uniform(0.0, 0.04)) * (1 + rng.gauss(0, noise))
        ok = actual <= GPU_TOTAL_GB
        r = {k: "" for k in FIELDS}
        r.update(run_id=f"syn{i:03d}", model=d["model"], hf_name=spec.name, method=d["method"],
                 batch=d["batch"], seq=d["seq"], ckpt=int(d["ckpt"]), optim=d["optim"],
                 lora_rank=d["lora_rank"], lora_targets=d["lora_targets"], gpus=1, zero=0,
                 dtype="fp16-amp" if d["method"] == "full" else "fp16", steps=15, gpu_name=GPU_NAME, gpu_total_gb=GPU_TOTAL_GB,
                 tag="synthetic", ok=int(ok), note="" if ok else "OOM")
        if ok:
            r.update(mem_model_gb=round(f.known_weights / GB * (1 + rng.gauss(0, 0.005)), 4),
                     mem_steady_gb=round((f.known_weights + f.known_optimizer) / GB, 4),
                     device_peak_gb=round(actual, 4),
                     peak_alloc_gb=round(actual - c.overhead_gb - 0.3, 4))
            tps = estimate_throughput(cfg, spec, get_gpu("t4"), 1, TRUTH_TP).tokens_per_s
            tps *= 1 + rng.gauss(0, 0.05)
            r.update(tokens_per_s=round(tps, 1), step_time_s=round(cfg.batch_size * cfg.seq_len / tps, 4))
        rows.append(r)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return {"memory": {k: v.to_dict() for k, v in TRUTH.items()}, "throughput_t4": TRUTH_TP.to_dict()}


if __name__ == "__main__":
    out = REPO / "data" / "synthetic_measurements.csv"
    write_synthetic_csv(out)
    print(f"-> {out}")
