import csv, sys
sys.path.insert(0, "scripts")
import sweep
P = "data/measurements_mig.csv"
rows = list(csv.DictReader(open(P)))
for r in rows:
    if "NVML_SUCCESS == r" in (r["note"] or ""):
        r["note"] = "OOM (NVML assert, MIG)"
w = csv.DictWriter(open(P, "w", newline=""), fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
from fitcfg import match_gpu
print(f"{'model':<13}{'meth':<6}{'b':>2}{'seq':>5}{'ck':>3} {'pred':>6} {'meas':>6} {'err%':>6} {'tok/s':>7}  result")
for r in sorted(rows, key=lambda r: (r["model"], r["method"], int(r["batch"]), int(r["seq"]), r["ckpt"])):
    d = dict(model=r["model"], method=r["method"], batch=int(r["batch"]), seq=int(r["seq"]), ckpt=r["ckpt"] == "1",
             optim=r["optim"], lora_rank=int(r["lora_rank"]), lora_targets=r["lora_targets"])
    pred = sweep.predict_gb(d, r["gpu_name"])
    meas = float(r["device_peak_gb"]) if r["ok"] == "1" else None
    err = f"{(pred - meas) / meas * 100:+.1f}" if meas else ""
    if r["ok"] == "1":
        res = "OK" + ("  ← 예측은 OOM" if pred > match_gpu(r["gpu_name"]).mem_gb else "")
    else:
        res = r["note"][:40] + ("" if pred > match_gpu(r["gpu_name"]).mem_gb else "  ← 예측은 통과(놓침)")
    print(f"{r['model']:<13}{r['method']:<6}{r['batch']:>2}{r['seq']:>5}{r['ckpt']:>3} {pred:6.1f} "
          f"{(f'{meas:6.2f}' if meas else '     -')} {err:>6} {r['tokens_per_s'] or '-':>7}  {res}")
