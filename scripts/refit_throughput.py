"""처리량 상수만 다시 맞춘다 (메모리 보정값은 그대로) + 검증 리포트.

    python scripts/refit_throughput.py                       # data/calibration.json의 측정 파일로 다시 fit
    python scripts/refit_throughput.py --runs data/bundle_runs_mig.jsonl   # 검증에 쓸 실제 학습 번들

검증 세 가지 (모두 GPU별)
  - in-sample      : 맞춘 데이터 그대로
  - 모델 빼고 맞추기: 모델 하나를 빼고 fit → 그 모델 예측 (처음 보는 모델 일반화)
  - 실제 학습 번들  : 측정 하네스가 아닌, 도구가 만든 train.py를 HF Trainer로 돌린 결과 (fit에 안 씀)
결과: data/calibration.json의 throughput·throughput_metrics·throughput_eval 갱신, 패키지 사본(fitcfg/calibrated.json),
      reports/throughput_report.md
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fitcfg.calibration import (fit_throughput, load_measurements, load_runs, metrics, throughput_metrics,  # noqa: E402
                                _tp_pairs)
from fitcfg.hardware import match_gpu  # noqa: E402
from fitcfg.throughput import DEFAULT_TP  # noqa: E402


def _m(pairs):
    m = metrics(pairs)
    return {"n": m.n, "mape": round(m.mape, 2), "max_ape": round(m.max_ape, 2),
            "bias": round(sum(p / a - 1 for p, a in pairs) / len(pairs) * 100, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=str(ROOT / "data" / "calibration.json"))
    ap.add_argument("--runs", nargs="*", default=[str(ROOT / "data" / "bundle_runs_mig.jsonl")])
    ap.add_argument("--report", default=str(ROOT / "reports" / "throughput_report.md"))
    a = ap.parse_args()

    cal_path = Path(a.calib)
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    old_tp = cal.get("throughput", {})
    meas = []
    for p in cal["source_csv"]:
        meas += load_measurements(ROOT / p)[0]
    rows_ok = [m for m in meas if m.ok and m.tokens_per_s and match_gpu(m.gpu_name)]
    runs = []
    for p in a.runs:
        if Path(p).exists():
            runs += [m for m in load_runs(p)[0] if m.ok and m.tokens_per_s and match_gpu(m.gpu_name)]

    tp = fit_throughput(meas)
    ev = {}
    from fitcfg.throughput import ThroughputConstants
    for key, c in tp.items():
        if key == "default":
            continue
        rows = [m for m in rows_ok if match_gpu(m.gpu_name).key == key]
        gpu = match_gpu(rows[0].gpu_name)
        old = ThroughputConstants.from_dict(old_tp[key]) if key in old_tp else DEFAULT_TP
        e = {"in_sample_v1": _m(_tp_pairs(rows, gpu, old)), "in_sample_v2": _m(_tp_pairs(rows, gpu, c))}
        lomo_v1, lomo_v2 = [], []
        for name in sorted({m.spec.name for m in rows}):
            train = [m for m in rows if m.spec.name != name]
            test = [m for m in rows if m.spec.name == name]
            if len({m.spec.name for m in train}) < 1:
                continue
            c2 = fit_throughput(train).get(key)
            if c2 is None:
                continue
            lomo_v2 += _tp_pairs(test, gpu, c2)
            # v1 방식(α·γ·optimizer 항 없이 t_half·eff만) — 같은 격자에서 v2 항을 기본값에 고정
            from fitcfg import calibration as C
            saved = dict(C.TP_GRID)
            try:
                C.TP_GRID.update(alpha=[0.0], recompute=[1.0], attn_nonflash=[1.0], opt_bpp=[0.0], opt8_bpp=[0.0])
                c1 = fit_throughput(train).get(key)
            finally:
                C.TP_GRID.clear(); C.TP_GRID.update(saved)
            lomo_v1 += _tp_pairs(test, gpu, c1)
        if lomo_v2:
            e["model_holdout_v1"], e["model_holdout_v2"] = _m(lomo_v1), _m(lomo_v2)
        rr = [m for m in runs if match_gpu(m.gpu_name).key == key]
        if rr:
            e["bundles_v1"], e["bundles_v2"] = _m(_tp_pairs(rr, gpu, old)), _m(_tp_pairs(rr, gpu, c))
        e["n_models"] = len({m.spec.name for m in rows})
        ev[key] = e

    tpm = throughput_metrics(meas, tp)
    cal["throughput"] = {k: v.to_dict() for k, v in tp.items()}
    cal["throughput_metrics"] = {k: {kk: vv.__dict__ for kk, vv in v.items()} for k, v in tpm.items()}
    cal["throughput_eval"] = ev
    cal["throughput_version"] = 2
    cal["throughput_refit"] = datetime.now().isoformat(timespec="seconds")
    cal_path.write_text(json.dumps(cal, indent=2, ensure_ascii=False), encoding="utf-8")
    if cal_path.resolve() == (ROOT / "data" / "calibration.json").resolve():
        shutil.copy(cal_path, ROOT / "fitcfg" / "calibrated.json")

    # --- 리포트 ---
    L = ["# 처리량 예측 v2 (2026-10-30) — 단일 GPU 실측으로 식 개선", "",
         "v1 식: `처리량 = peak × 효율(방법) × 활용도(토큰 수) ÷ 토큰당 FLOPs`. v2에서 실측 잔차가 보인 세 가지를 추가:", "",
         "| 추가 항 | 근거 (v1 잔차) | 값 (A100 MIG) |", "|---|---|---|"]
    c = tp.get("a100-mig3g40")
    if c:
        L += [f"| 크기 효율 α: 효율 × (H·I ÷ Llama-8B)^α | 1B·3B는 과대, 7B·8B는 과소 예측 | α = {c.alpha:g} |",
              f"| 재계산 비용 γ (forward 대비) | checkpointing 켬 과소 / 끔 과대 | γ = {c.recompute:g} |",
              f"| optimizer step = 학습 파라미터 × 바이트 ÷ 대역폭 | 1B Full: 측정(누적 1) 과대·번들(누적 8~16) 과소 | AdamW {c.opt_bpp:g} B/param, 8bit {c.opt8_bpp:g} B/param |"]
    t4 = tp.get("t4")
    if t4:
        L += [f"| FlashAttention 없는 GPU의 attention 비용 κ | T4에서 seq가 길수록 과대 예측 | κ = {t4.attn_nonflash:g} (T4) |"]
    L += ["", "## 오차 (평균 절대 오차 %, 괄호는 최대)", "",
          "| GPU | 모델 수 | in-sample v1 → v2 | 모델 빼고 맞추기 v1 → v2 | 실제 학습 번들 v1 → v2 |", "|---|---|---|---|---|"]
    for k, e in ev.items():
        def f(a, b):
            return f"{e[a]['mape']:.1f} ({e[a]['max_ape']:.0f}) → **{e[b]['mape']:.1f}** ({e[b]['max_ape']:.0f})" if a in e else "—"
        L.append(f"| {k} | {e['n_models']} | {f('in_sample_v1', 'in_sample_v2')} | "
                 f"{f('model_holdout_v1', 'model_holdout_v2')} | {f('bundles_v1', 'bundles_v2')} |")
    L += ["", "- 실제 학습 번들(HF Trainer + 실제 데이터, 누적 4~16)은 fit에 쓰지 않은 검증 데이터",
          "- T4는 모델이 2개뿐이라 '모델 빼고 맞추기'가 곧 '한 모델로 다른 모델 맞추기' → 참고용",
          "- 측정 안 한 GPU(RTX 3090·4090 등)는 A100 MIG 상수를 쓰고 범위를 ±50%로 넓게 표시 (`default`)", "",
          "## 상수", "", "```json", json.dumps({k: {kk: round(v, 4) for kk, v in c.to_dict().items()} for k, c in tp.items()},
                                               indent=1, ensure_ascii=False), "```"]
    Path(a.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
