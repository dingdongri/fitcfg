"""GPU 없이 '텐서 수준 실측' 데이터셋 만들기 (fitcfg/trace.py 사용).

    python scripts/trace_sweep.py                 # data/trace_measurements.csv (이어하기 지원)
    python scripts/trace_sweep.py --ablation      # 항 분리용 절제 실험 → data/trace_ablation.csv

결과 CSV는 measure.py와 같은 스키마 (gpu_name="trace-cpu", device_peak_gb는 비움).
→ calibrate.py --csv data/trace_measurements.csv --target alloc 으로 텐서 항(활성값·logits)을 보정할 수 있다.
"""
from __future__ import annotations
import argparse, csv, sys, time
from datetime import datetime
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from measure import FIELDS  # noqa: E402

GPU_NAME = "trace-cpu"
MODELS = ["llama-3.2-1b", "qwen2.5-3b", "qwen2.5-7b", "llama-3.1-8b"]
METHODS = ["lora", "qlora", "full"]
BATCHES = [1, 2, 4]
SEQS = [512, 1024, 2048, 4096]


def key(r):
    return (r["model"], r["method"], int(r["batch"]), int(r["seq"]), int(r["ckpt"]), r.get("note", ""))


def run(out: Path, jobs: list[dict]):
    from fitcfg.trace import trace_step
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {key(r) for r in csv.DictReader(f)}
    todo = [j for j in jobs if key({**j, "ckpt": int(j["ckpt"])}) not in done]
    print(f"전체 {len(jobs)}개, 완료 {len(jobs) - len(todo)}개 → 실행 {len(todo)}개", flush=True)
    new = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, j in enumerate(todo, 1):
        row = {k: "" for k in FIELDS}
        row.update(run_id=f"trace-{datetime.now():%Y%m%d%H%M%S%f}", timestamp=datetime.now().isoformat(timespec="seconds"),
                   tag="trace-emul" if j["method"] == "qlora" else "trace", model=j["model"],
                   method=j["method"], batch=j["batch"], seq=j["seq"], ckpt=int(j["ckpt"]), optim="adamw",
                   lora_rank=16, lora_targets="attn", gpus=1, zero=0, dtype="bf16", steps=2,
                   gpu_name=GPU_NAME, note=j.get("note", ""))
        try:
            r = trace_step(j["model"], j["method"], j["batch"], j["seq"], j["ckpt"],
                           n_layers=j.get("n_layers"), vocab=j.get("vocab"))
            row.update(mem_model_gb=r.mem_model_gb, mem_steady_gb=r.mem_steady_gb,
                       peak_alloc_gb=r.peak_alloc_gb, ok=1)
        except Exception as e:  # noqa: BLE001
            row.update(ok=0, note=f"{j.get('note', '')} {type(e).__name__}: {str(e)[:120]}".strip())
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
                new = False
            w.writerow(row)
        el = time.time() - t0
        print(f"[{i}/{len(todo)}] {j['model']} {j['method']} b{j['batch']} s{j['seq']} ckpt{int(j['ckpt'])} "
              f"{j.get('note', '')} → peak {row.get('peak_alloc_gb')}GB  (남은 {el / i * (len(todo) - i) / 60:.1f}분)",
              flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--models", nargs="*", default=MODELS)
    a = p.parse_args()
    if a.ablation:
        # 같은 모델에서 레이어 수·vocab만 바꿔 '레이어당 활성값'과 'logits' 항을 분리
        jobs = [dict(model="llama-3.2-1b", method=m, batch=2, seq=1024, ckpt=c, n_layers=L, vocab=V,
                     note=f"L={L},V={V}")
                for m, c, L, V in product(["lora", "full"], [True, False], [4, 8, 16], [32000, 128256])]
        run(Path(a.out or REPO / "data" / "trace_ablation.csv"), jobs)
    else:
        jobs = [dict(model=mk, method=m, batch=b, seq=s, ckpt=c)
                for mk, m, b, s, c in product(a.models, METHODS, BATCHES, SEQS, [True, False])]
        run(Path(a.out or REPO / "data" / "trace_measurements.csv"), jobs)


if __name__ == "__main__":
    main()
