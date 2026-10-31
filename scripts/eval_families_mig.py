"""처음 보는 계열(보정에 안 쓴 EVAL_MODELS)의 GPU 실측 vs 예측 — 진짜 held-out 평가.
    python scripts/eval_families_mig.py [--cal data/calibration.json] [--csv data/measurements_mig.csv]
"""
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from fitcfg import load_calibration, match_gpu  # noqa: E402
from fitcfg.models import EVAL_MODELS, MODELS  # noqa: E402
import sweep  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cal", default=str(REPO / "data" / "calibration.json"))
ap.add_argument("--csv", default=str(REPO / "data" / "measurements_mig.csv"))
ap.add_argument("--safety", type=float, default=0.907)
ap.add_argument("--md", default="")
ap.add_argument("--scope", default="eval", choices=["eval", "fit"], help="eval=처음 보는 계열, fit=보정 대상 모델")
a = ap.parse_args()
if a.cal != "none":
    load_calibration(a.cal)
POOL = EVAL_MODELS if a.scope == "eval" else MODELS
rows = [r for r in csv.DictReader(open(a.csv)) if r["model"] in POOL]
by = defaultdict(list); oom = defaultdict(lambda: [0, 0, 0, 0])   # 정답, 놓침, 놓침(safety), 오경보(safety)
lines = []
for r in rows:
    d = dict(model=r["model"], method=r["method"], batch=int(r["batch"]), seq=int(r["seq"]), ckpt=r["ckpt"] == "1",
             optim=r["optim"], lora_rank=int(r["lora_rank"]), lora_targets=r["lora_targets"])
    pred = sweep.predict_gb(d, r["gpu_name"]); cap = match_gpu(r["gpu_name"]).mem_gb
    is_oom = r["note"].upper().startswith("OOM"); ok = r["ok"] == "1"
    k = (r["model"], r["method"])
    o = oom[r["model"]]
    if ok or is_oom:
        o[0] += (pred > cap) == is_oom
        o[1] += is_oom and pred <= cap
        o[2] += is_oom and pred <= cap * a.safety
        o[3] += ok and pred > cap * a.safety
    if ok:
        meas = float(r["device_peak_gb"]); e = (pred - meas) / meas * 100
        by[k].append(e)
        lines.append(f"| {r['model']} | {r['method']} | {d['batch']} | {d['seq']} | {int(d['ckpt'])} | {meas:.2f} | {pred:.2f} | {e:+.1f}% |")
out = ["| 모델 | 방식 | n | 평균 |오차| | 범위 | 과소예측 |", "|---|---|---|---|---|---|"]
allv = []
for (m, meth), es in sorted(by.items()):
    allv += es
    out.append(f"| {m} | {meth} | {len(es)} | {sum(map(abs, es))/len(es):.1f}% | {min(es):+.1f} ~ {max(es):+.1f}% | {sum(e < 0 for e in es)}/{len(es)} |")
out.append(f"| **전체** | | {len(allv)} | **{sum(map(abs, allv))/len(allv):.1f}%** | {min(allv):+.1f} ~ {max(allv):+.1f}% | {sum(e < 0 for e in allv)}/{len(allv)} |")
out += ["", "| 모델 | OOM 판정 정답 | 놓친 OOM | 놓친 OOM (safety) | 오경보 (safety) |", "|---|---|---|---|---|"]
for m, (c, mi, ms, fa) in oom.items():
    n = sum(1 for r in rows if r["model"] == m and (r["ok"] == "1" or r["note"].upper().startswith("OOM")))
    out.append(f"| {m} | {c}/{n} | {mi} | {ms} | {fa} |")
txt = "\n".join(out)
print(txt)
if a.md:
    Path(a.md).write_text(txt + "\n\n| 모델 | 방식 | b | seq | ckpt | 실측 | 예측 | 오차 |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(lines) + "\n")
