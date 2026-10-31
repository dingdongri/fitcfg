"""생성된 학습 번들을 실제로 실행하고, 결과(성공/OOM, 실측 메모리·처리량)를 기록한다 (GPU에서 실행).

    python scripts/run_bundle.py runs/rec  --label recommended --max-steps 30
    python scripts/run_bundle.py runs/naive --label naive       --max-steps 30

- 번들의 launch.sh 를 FITCFG_MAX_STEPS 로 짧게 돌린다.
- 성공: 번들이 남긴 outputs/fitcfg_run.json (실측 peak·tok/s) 을 기록
- 실패: 출력에서 OOM 여부를 판정하고, fitcfg_plan.json 의 설정과 예측을 기록
- 결과는 data/bundle_runs.jsonl 에 한 줄씩 쌓인다 →
      calibrate.py --runs data/bundle_runs.jsonl   (보정 데이터로 추가)
      evaluate.py  --runs data/bundle_runs.jsonl   (Phase 7: 추천 vs 기본값 실제 OOM 비교)
"""
from __future__ import annotations
import argparse, json, os, subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def gpu_info() -> tuple[str, float | None]:
    """GPU 이름·용량. MIG 슬라이스는 --query-gpu가 권한 오류 → sweep.gpu_info와 같은 방식으로 이름을 맞춘다."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from sweep import gpu_info as _g
        name, gb = _g()
        return name, (round(gb, 3) if gb else None)
    except Exception:
        return "", None


def record_from_plan(bundle: Path) -> dict:
    plan = json.loads((bundle / "fitcfg_plan.json").read_text(encoding="utf-8"))
    rq, rc, pr = plan["request"], plan["recommendation"], plan["prediction"]
    return {
        "model_key": rq["model"], "method": rq["method"], "seq_len": rq["seq_len"], "gpus": rq["gpus"],
        "lora_rank": rq["lora_rank"], "lora_targets": rq["lora_targets"],
        "micro_batch": rc["micro_batch"], "grad_accum": rc["grad_accum"],
        "gradient_checkpointing": rc["gradient_checkpointing"], "optim_label": rc["optim"],
        "zero_stage": rc["zero_stage"], "cpu_offload_optim": rc["cpu_offload_optim"],
        "cpu_offload_params": rc["cpu_offload_params"], "dtype": rc["dtype"],
        "predicted_peak_gb": pr["memory_gb"]["total"], "predicted_tokens_per_s": round(pr["tokens_per_s"], 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle")
    p.add_argument("--label", default="recommended", help="recommended | naive | 자유")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--log", default=str(REPO / "data" / "bundle_runs.jsonl"))
    p.add_argument("--timeout", type=int, default=3600)
    a = p.parse_args()
    bundle = Path(a.bundle).resolve()
    run_json = bundle / "outputs" / "fitcfg_run.json"
    if run_json.exists():
        run_json.unlink()                       # 이전 실행 결과와 섞이지 않게

    env = {**os.environ, "FITCFG_MAX_STEPS": str(a.max_steps)}
    print(f"▶ {bundle.name}: bash launch.sh (FITCFG_MAX_STEPS={a.max_steps})", flush=True)
    try:
        proc = subprocess.run(["bash", str(bundle / "launch.sh")], cwd=bundle, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=a.timeout)
        out, rc = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or "") if isinstance(e.stdout, str) else "", -999
    tail = "\n".join(out.strip().splitlines()[-15:])
    print(tail)

    rec = record_from_plan(bundle)
    gpu_name, gpu_total = gpu_info()
    rec.update(label=a.label, bundle=bundle.name, timestamp=datetime.now().isoformat(timespec="seconds"),
               gpu_name=gpu_name, gpu_total_gb=gpu_total, max_steps=a.max_steps)
    if rc == 0 and run_json.exists():
        measured = json.loads(run_json.read_text(encoding="utf-8"))
        rec.update({k: measured[k] for k in ("device_peak_gb", "peak_alloc_gb", "tokens_per_s", "gpu_name",
                                             "transformers", "torch") if k in measured})
        rec.update(ok=1, note="")
    else:
        # MIG 슬라이스에서는 OOM이 'NVML_SUCCESS == r INTERNAL ASSERT'로 나타남 (measure.py와 같은 처리)
        oom = ("out of memory" in out.lower() or "outofmemoryerror" in out.lower()
               or ("NVML_SUCCESS == r" in out and "CUDACachingAllocator" in out))
        rec.update(ok=0, note="OOM" if oom else ("TIMEOUT" if rc == -999 else f"ERROR rc={rc}: "
                                                   + (out.strip().splitlines() or [""])[-1][:150]))
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    with open(a.log, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    verdict = "✅ 성공" if rec["ok"] else f"❌ {rec['note']}"
    extra = ""
    if rec["ok"] and rec.get("device_peak_gb"):
        err = (rec["predicted_peak_gb"] - rec["device_peak_gb"]) / rec["device_peak_gb"] * 100
        extra = f" | peak 실측 {rec['device_peak_gb']:.2f}GB vs 예측 {rec['predicted_peak_gb']:.2f}GB ({err:+.1f}%)"
    print(f"{verdict} [{a.label}]{extra}\n-> {a.log}")


if __name__ == "__main__":
    main()
