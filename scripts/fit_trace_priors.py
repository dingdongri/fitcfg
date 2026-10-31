"""FakeTensor 추적 데이터로 cost model의 '텐서 항' 사전값을 fit 하고 검증 리포트를 만든다 (GPU 불필요).

    python scripts/trace_sweep.py            # 1) 추적 데이터 생성 (data/trace_measurements.csv)
    python scripts/fit_trace_priors.py       # 2) fit → fitcfg/priors.json + reports/trace_report.md

priors.json 은 memory.py 가 import 시 읽어서 method별 기본 상수(DEFAULT_BY_METHOD)로 쓴다.
→ GPU 실측 보정(calibrate.py)은 이 값에서 출발해 overhead(CUDA 컨텍스트·단편화)만 주로 맞추면 된다.

리포트에는 설계 선택의 근거가 되는 절제(ablation) 비교를 함께 싣는다.
    v0.1 식(문헌 상수) / v0.2 식 + 사전 추정값 / v0.2 식 fit / v0.2에서 I(MLP 폭) 항 제거
검증은 '모델 하나를 통째로 빼고 fit → 그 모델 예측' (새 모델로 일반화되는가).
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fitcfg.calibration import (PEAK_FIELDS, cross_validate, fit, load_measurements, metrics,  # noqa: E402
                                predict_gb)
from fitcfg.memory import GB, GUESS_CONSTANTS, features  # noqa: E402

PRIORS_PATH = REPO / "fitcfg" / "priors.json"
# CUDA 컨텍스트(~0.3-0.5GB) + cuBLAS 워크스페이스 + 캐시 할당자 여유. GPU 실측(calibrate.py)으로 대체될 값.
CUDA_OVERHEAD_PRIOR_GB = 1.1


def v01_predict_gb(m) -> float:
    """비교용: v0.1 식(단일 act_factor, 체크포인팅 비례, 합산)과 문헌 상수. overhead 제외(alloc)."""
    f = features(m.cfg, m.spec)
    BS, L, H, V = m.cfg.batch_size * m.cfg.seq_len, m.spec.n_layers, m.spec.hidden, m.spec.vocab
    act = 12.0 * 2 * L * BS * H * (0.12 if m.cfg.grad_checkpointing else 1.0)
    w = m.mem_model_gb * GB if m.mem_model_gb else f.known_weights
    return (w + f.known_gradients + f.known_optimizer + act + 8.0 * BS * V) / GB


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", default=[str(REPO / "data" / "trace_measurements.csv")])
    p.add_argument("--lam", type=float, default=0.001)
    p.add_argument("--no-write", action="store_true", help="priors.json을 덮어쓰지 않음")
    a = p.parse_args()

    meas = []
    for c in a.csv:
        meas += load_measurements(c)[0]
    rows = [m for m in meas if m.ok and m.peak_alloc_gb]
    print(f"추적 데이터 {len(rows)}행 (모델 {len({m.cfg.model for m in rows})}개)")

    fit_fn = partial(fit, target="alloc", lam=a.lam, use_method_priors=False)
    no_i = partial(fit, target="alloc", lam=a.lam, use_method_priors=False, fixed={"act_i": 0.0})
    consts = fit_fn(meas)

    def pairs(pred):
        return [(pred(m), m.peak_alloc_gb) for m in rows]

    guess = {"default": GUESS_CONSTANTS}
    results = {
        "v0.1 식 + 문헌 상수": metrics(pairs(v01_predict_gb)),
        "v0.2 식 + 사전 추정값(fit 전)": metrics(pairs(lambda m: predict_gb(m, guess))),
        "v0.2 식 fit (in-sample)": metrics(pairs(lambda m: predict_gb(m, consts))),
    }
    cv_rows = {}
    for name, fn in [("v0.2 식 fit · LOO", (fit_fn, "loo")), ("v0.2 식 fit · 모델 제외 ★", (fit_fn, "model")),
                     ("v0.2에서 I항 제거 · 모델 제외", (no_i, "model"))]:
        res = cross_validate(meas, fn[1], "alloc", fn[0])
        cv_rows[name] = res
        results[name] = metrics((pr, m.peak_alloc_gb) for m, pr in res)

    # 모델별 / method별 held-out 오차 (모델 제외)
    ho = cv_rows["v0.2 식 fit · 모델 제외 ★"]

    def group_table(key):
        vals = sorted({key(m) for m, _ in ho})
        out = []
        for v in vals:
            mm = metrics((pr, m.peak_alloc_gb) for m, pr in ho if key(m) == v)
            out.append(f"| {v} | {mm.n} | {mm.mape:.1f}% | {mm.max_ape:.1f}% | {mm.under_rate * 100:.0f}% |")
        return "\n".join(out)

    phase_share = {}
    for m in rows:
        c = consts.get(m.cfg.method) or consts["default"]
        f = features(m.cfg, m.spec, c)
        ph = "optim_step" if f.optim_step(c) > f.fwd_bwd(c) else "fwd_bwd"
        phase_share.setdefault(m.cfg.method, []).append(ph == "optim_step")

    L = ["# FakeTensor 추적 기반 cost model 검증 (GPU 없이)", "",
         f"- 생성: {datetime.now():%Y-%m-%d %H:%M} · 데이터: {', '.join(Path(c).name for c in a.csv)} · "
         f"{len(rows)}행 = 모델 {sorted({m.cfg.model for m in rows})} × LoRA/QLoRA(에뮬)/Full × batch × seq × ckpt",
         "- 타깃: 학습 구간 텐서 peak (`max_memory_allocated`와 같은 의미). CUDA 컨텍스트·할당자 캐시는 제외 → GPU 실측으로 보정",
         "- 추적기 자체 검증: 알려진 할당 패턴(할당·view·in-place·해제)에서 cur/peak 정확히 일치 (tests/test_trace.py)", "",
         "## 1. 정확도 비교 (MAPE, 텐서 peak)", "",
         "| 모델 식 / 상수 | n | MAPE | median | max | 과소예측 비율 | 과소예측 p95 |", "|---|---|---|---|---|---|---|"]
    for k, v in results.items():
        L.append(f"| {k} | {v.row()} |")
    L += ["", "**읽는 법**: '모델 제외'는 그 모델의 데이터를 전혀 보지 않고 다른 모델로만 fit 한 뒤 예측한 값 — "
              "새 모델로 일반화되는지를 본다. I항 제거 비교는 활성값을 MLP 폭(I)으로 분리한 설계의 근거.", "",
          "## 2. 모델 제외 검증 — 모델별", "", "| 모델 | n | MAPE | max | 과소예측 |", "|---|---|---|---|---|",
          group_table(lambda m: m.cfg.model), "",
          "## 3. 모델 제외 검증 — method별", "", "| method | n | MAPE | max | 과소예측 |", "|---|---|---|---|---|",
          group_table(lambda m: m.cfg.method), "",
          "## 4. fit 된 상수 (→ fitcfg/priors.json)", "",
          "| 상수 | " + " | ".join(consts) + " |", "|---|" + "---|" * len(consts)]
    for fld in PEAK_FIELDS:
        L.append(f"| {fld} | " + " | ".join(f"{getattr(c, fld):.4g}" for c in consts.values()) + " |")
    L += ["", "## 5. peak 시점", ""]
    for meth, v in sorted(phase_share.items()):
        L.append(f"- {meth}: optimizer step 시점이 peak인 설정 {sum(v)}/{len(v)}")
    L += ["", "## 6. 한계", "",
          "- QLoRA는 bitsandbytes 없이 에뮬레이션(16bit Linear + fp32 norm/임베딩/lm_head) → 4bit 역양자화 임시버퍼는 미포함",
          "- CPU의 SDPA(flash-for-CPU) 경로 기준. CUDA flash/efficient 커널의 임시 버퍼와 약간 다를 수 있음",
          "- overhead(CUDA 컨텍스트, 캐시 할당자 단편화)는 이 방법으로 알 수 없음 → Colab 실측 필요", ""]
    rep = REPO / "reports" / "trace_report.md"
    rep.parent.mkdir(exist_ok=True)
    rep.write_text("\n".join(L), encoding="utf-8")

    if not a.no_write:
        pri = {"source": "FakeTensor trace (scripts/trace_sweep.py → scripts/fit_trace_priors.py)",
               "created": datetime.now().isoformat(timespec="seconds"), "n_rows": len(rows),
               "heldout_model_mape": results["v0.2 식 fit · 모델 제외 ★"].mape,
               # overhead_gb = 추적에서 나온 상수항(고정 크기 텐서) + CUDA 컨텍스트·캐시 사전값(GPU 실측 전)
               "cuda_overhead_prior_gb": CUDA_OVERHEAD_PRIOR_GB,
               "constants": {k: {**{f: round(getattr(c, f), 4) for f in PEAK_FIELDS if f != "overhead_gb"},
                                 "overhead_gb": round(c.overhead_gb + CUDA_OVERHEAD_PRIOR_GB, 3)}
                             for k, c in consts.items()}}
        PRIORS_PATH.write_text(json.dumps(pri, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"-> {PRIORS_PATH}")
    for k, v in results.items():
        print(f"{k:<32} {v.row()}")
    print(f"-> {rep}")


if __name__ == "__main__":
    main()
