"""Phase 3 — 실측으로 cost model 상수를 보정하고 예측 정확도를 리포트한다. (GPU 불필요)

    python scripts/calibrate.py                                  # data/measurements.csv 사용
    python scripts/calibrate.py --csv data/colab_t4.csv --gpu T4 # 특정 GPU 행만
    python scripts/calibrate.py --synthetic                      # 가짜 데이터로 파이프라인 시연

출력
    data/calibration.json          보정된 상수 (fitcfg.load_calibration 으로 로드)
    reports/calibration_report.md  보정 전/후 정확도, held-out MAPE, OOM 판정, 행별 오차
    reports/calibration_plot.png   예측 vs 실측 산점도 (matplotlib 있을 때만)

정확도는 세 가지로 본다.
    in-sample   : 전체로 fit 하고 같은 데이터로 평가 (낙관적, 참고용)
    LOO         : 한 행씩 빼고 fit → 뺀 행 예측 (새 설정에 대한 일반화)
    model-out   : 모델 하나를 통째로 빼고 fit → 그 모델 예측 (새 모델에 대한 일반화, ★가장 강한 주장)
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from fitcfg.calibration import (  # noqa: E402
    load_measurements, load_runs, fit, predict_gb, metrics, cross_validate, suggest_safety,
    oom_check, prior_constants, fit_throughput, throughput_metrics,
)
from fitcfg.memory import CostConstants, features, GB  # noqa: E402

CONST_FIELDS = ["act_h", "act_i", "ckpt_h", "logit_bytes", "opt_temp", "attn_sq_bytes", "nf4_extra",
                "qlora_nonquant_bytes", "overhead_gb"]


def parse_args():
    p = argparse.ArgumentParser(description="fitcfg cost model 보정 (Phase 3)")
    p.add_argument("--csv", nargs="+", default=[str(REPO / "data" / "measurements.csv")],
                   help="측정 CSV (여러 개 가능: 실측 + 추적 등)")
    p.add_argument("--runs", nargs="*", default=[],
                   help="run_bundle.py 결과 jsonl (실제 학습 번들 실행 결과도 보정 데이터로 사용)")
    p.add_argument("--target", default="device", choices=["device", "alloc"],
                   help="device=CUDA컨텍스트+reserved peak(OOM 기준, 기본) | alloc=max_memory_allocated")
    p.add_argument("--gpu", default="", help="gpu_name에 이 문자열이 포함된 행만 사용 (예: T4)")
    p.add_argument("--lam", type=float, default=0.01, help="사전값 쪽 ridge 강도")
    p.add_argument("--min-group", type=int, default=6, help="method별 보정에 필요한 최소 행 수")
    p.add_argument("--no-group", action="store_true", help="method별 보정 끄기 (공통 상수 하나만)")
    p.add_argument("--out", default=str(REPO / "data" / "calibration.json"))
    p.add_argument("--report", default=str(REPO / "reports" / "calibration_report.md"))
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--synthetic", action="store_true",
                   help="가짜 측정 CSV를 만들어 파이프라인을 시연 (data/synthetic_measurements.csv)")
    return p.parse_args()


def _rel(path) -> str:
    p = Path(path).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(path)


def fmt_consts(c: CostConstants) -> dict:
    return {k: round(getattr(c, k), 4) for k in CONST_FIELDS}


def main():
    args = parse_args()
    if args.synthetic:
        from synthetic import write_synthetic_csv
        args.csv = [str(REPO / "data" / "synthetic_measurements.csv")]
        truth = write_synthetic_csv(args.csv[0])
        args.out = str(REPO / "data" / "calibration_synthetic.json")
        args.report = str(REPO / "reports" / "calibration_report_synthetic.md")
        print(f"[synthetic] 가짜 측정 생성 → {args.csv}")

    meas, warns = [], []
    for path in args.csv:
        m, w = load_measurements(path)
        meas += m
        warns += w
    for path in args.runs:
        m, w = load_runs(path)
        meas += m
        warns += w
    for w in warns:
        print("WARN", w)
    if args.gpu:
        meas = [m for m in meas if args.gpu.lower() in m.gpu_name.lower()]
    ok_rows = [m for m in meas if m.ok and m.target(args.target)]
    n_oom = sum(m.oom for m in meas)
    print(f"측정 {len(meas)}행: 성공 {len(ok_rows)}, OOM {n_oom}, "
          f"기타 실패 {len(meas) - len(ok_rows) - n_oom}")
    if len(ok_rows) < 3:
        raise SystemExit("보정하려면 성공한 측정이 최소 3개 필요합니다.")

    fit_fn = partial(fit, target=args.target, lam=args.lam,
                     group_by_method=not args.no_group, min_group=args.min_group)
    prior = {"default": prior_constants(args.target)}
    consts = fit_fn(meas)

    # --- 정확도 ---
    before = [(predict_gb(m, prior), m.target(args.target)) for m in ok_rows]
    after = [(predict_gb(m, consts), m.target(args.target)) for m in ok_rows]
    cv, cv_rows = {}, {}
    for split in ("loo", "model"):
        res = cross_validate(meas, split, args.target, fit_fn)
        if res is not None:
            cv[split] = [(p, m.target(args.target)) for m, p in res]
            cv_rows[split] = [(m.cfg.method, p, m.target(args.target)) for m, p in res]
    safety = suggest_safety(cv.get("model") or cv.get("loo") or after)
    oom_prior, oom_after = oom_check(meas, prior), oom_check(meas, consts)
    oom_safe = oom_check(meas, consts, safety)

    tp = fit_throughput(meas)
    tpm = throughput_metrics(meas, tp)
    m_before, m_after = metrics(before), metrics(after)
    m_cv = {k: metrics(v) for k, v in cv.items()}

    # --- 저장: calibration.json ---
    out = {
        "version": 1,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_csv": [_rel(x) for x in args.csv] + [_rel(x) for x in args.runs],
        "sources": {s: sum(m.source == s for m in ok_rows) for s in ("measure", "trace", "bundle")},
        "target": args.target, "gpu_filter": args.gpu, "lambda": args.lam,
        "n_rows": len(ok_rows), "n_oom": n_oom,
        "constants": {k: fmt_consts(v) for k, v in consts.items()},
        "metrics": {
            "in_sample_before": m_before.__dict__, "in_sample_after": m_after.__dict__,
            **{f"cv_{k}": v.__dict__ for k, v in m_cv.items()},
        },
        "suggested_safety": safety,
        "throughput": {k: v.to_dict() for k, v in tp.items()},
        "throughput_metrics": {k: {kk: vv.__dict__ for kk, vv in v.items()} for k, v in tpm.items()},
    }
    if args.synthetic:
        out["synthetic_truth"] = truth
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- 리포트 ---
    plot_path = None
    if not args.no_plot:
        meths = [m.cfg.method for m in ok_rows]
        series = [("before (prior constants)", [(k, p, a) for k, (p, a) in zip(meths, before)]),
                  ("after (in-sample)", [(k, p, a) for k, (p, a) in zip(meths, after)])]
        ho = "model" if "model" in cv_rows else "loo"
        series.append((f"after (held-out: {ho})", cv_rows[ho]))
        plot_path = make_plot(Path(args.report).with_name(
            Path(args.report).stem.replace("report", "plot") + ".png"), series)
    write_report(Path(args.report), args, ok_rows, prior, consts, before, after, cv,
                 m_before, m_after, m_cv, safety, oom_prior, oom_after, oom_safe, meas, plot_path,
                 tp, tpm)

    # --- 콘솔 요약 ---
    print("\n              n | MAPE  | median | max   | under | p95 under")
    print(f"보정 전(in)  {m_before.row()}")
    print(f"보정 후(in)  {m_after.row()}")
    for k, v in m_cv.items():
        print(f"held-out {k:<5}{v.row()}")
    print(f"\n권장 safety = {safety}  (fits(..., safety={safety}))")
    print(f"OOM 판정 정확도: 보정 전 {oom_prior.correct}/{oom_prior.n}, "
          f"보정 후 {oom_after.correct}/{oom_after.n}, 보정+safety {oom_safe.correct}/{oom_safe.n}")
    for k, v in tpm.items():
        print(f"처리량[{k}] MAPE: 보정 전 {v['before'].mape:.1f}% → 보정 후 {v['after'].mape:.1f}% "
              f"(범위 ×/÷{tp[k].band})")
    print(f"\n-> {args.out}\n-> {args.report}" + (f"\n-> {plot_path}" if plot_path else ""))


def make_plot(path: Path, series: list[tuple[str, list[tuple[str, float, float]]]]):
    """series: [(제목, [(method, 예측, 실측), ...]), ...] → 패널별 예측 vs 실측 산점도."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    colors = {"qlora": "#2a6fdb", "lora": "#e07a1f", "full": "#3a9d5d"}
    hi = max(max(p, a) for _, pts in series for _, p, a in pts) * 1.08
    fig, axes = plt.subplots(1, len(series), figsize=(4.6 * len(series), 4.4), sharey=True)
    for ax, (title, pts) in zip(axes, series):
        ax.plot([0, hi], [0, hi], color="#999", lw=1, ls="--")
        for meth in sorted({m for m, _, _ in pts}):
            sub = [(p, a) for m, p, a in pts if m == meth]
            ax.scatter([a for _, a in sub], [p for p, _ in sub], s=22, label=meth,
                       color=colors.get(meth, "#666"))
        mape = sum(abs(p - a) / a for _, p, a in pts) / len(pts) * 100
        ax.set_title(f"{title}\nMAPE {mape:.1f}%", fontsize=10)
        ax.set_xlabel("measured peak (GB)")
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("predicted peak (GB)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_report(path: Path, args, rows, prior, consts, before, after, cv, m_before, m_after,
                 m_cv, safety, oom_prior, oom_after, oom_safe, meas, plot_path, tp=None, tpm=None):
    L = []
    L.append("# cost model 보정 리포트 (Phase 3)\n")
    csv_disp = ", ".join(_rel(x) for x in args.csv + args.runs)
    L.append(f"- 생성: {datetime.now():%Y-%m-%d %H:%M} · 데이터: `{csv_disp}`"
             + (" · ⚠️ **합성 데이터(파이프라인 시연용)**" if args.synthetic else ""))
    gpus = sorted({m.gpu_name for m in rows})
    models = sorted({m.cfg.model for m in rows})
    L.append(f"- 타깃: `{args.target}` · GPU: {', '.join(gpus) or '-'} · 모델: {', '.join(models)}")
    L.append(f"- 성공 {len(rows)}행 · OOM {sum(m.oom for m in meas)}행 · λ={args.lam}\n")

    L.append("## 1. 예측 정확도\n")
    L.append("| 평가 | n | MAPE | median APE | max APE | 과소예측 비율 | 과소예측 p95 |")
    L.append("|---|---|---|---|---|---|---|")
    L.append(f"| 보정 전 (문헌 상수) | {m_before.row()} |")
    L.append(f"| 보정 후 in-sample | {m_after.row()} |")
    names = {"loo": "보정 후 held-out: LOO", "model": "보정 후 held-out: 모델 제외 ★"}
    for k, v in m_cv.items():
        L.append(f"| {names[k]} | {v.row()} |")
    if "model" not in m_cv:
        L.append("\n> 모델 제외 검증은 2개 이상 모델을 측정해야 가능합니다 (예: `colab-qwen3b` 프리셋).")
    L.append(f"\n**권장 안전계수 safety = {safety}** — held-out 과소예측 p95를 덮는 값. "
             f"`fits(cfg, model, gpu_mem_gb, safety={safety})`\n")
    if plot_path:
        L.append(f"![pred vs actual]({plot_path.name})\n")

    L.append("## 2. OOM 판정 (예측 > GPU 용량 → OOM)\n")
    L.append("| 상수 | 정답 | 놓친 OOM (위험) | 오경보 (보수적) |")
    L.append("|---|---|---|---|")
    for name, c in [("보정 전", oom_prior), ("보정 후", oom_after), (f"보정 후 + safety {safety}", oom_safe)]:
        L.append(f"| {name} | {c.correct}/{c.n} | {c.missed_oom} | {c.false_alarm} |")
    L.append("")

    L.append("## 3. 처리량(tokens/s) 보정\n")
    if tp:
        L.append("| GPU | n | MAPE 보정 전 | MAPE 보정 후 | eff full / lora / qlora | t_half | 범위 |")
        L.append("|---|---|---|---|---|---|---|")
        for k, c in tp.items():
            b, a = tpm[k]["before"], tpm[k]["after"]
            L.append(f"| {k} | {a.n} | {b.mape:.1f}% | {a.mape:.1f}% | "
                     f"{c.eff_full:.3f} / {c.eff_lora:.3f} / {c.eff_qlora:.3f} | {c.t_half:.0f} | ×/÷{c.band} |")
        L.append("\n> 처리량 보정은 측정한 GPU에만 적용됩니다. 다른 GPU는 사전값(범위 ×/÷1.6)을 씁니다.\n")
    else:
        L.append("처리량 데이터가 있는 GPU가 없습니다 (GPU당 성공 행 3개 이상 필요).\n")

    L.append("## 4. 보정된 메모리 상수\n")
    keys = ["prior"] + list(consts)
    L.append("| 상수 | " + " | ".join(keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    all_c = {"prior": prior["default"], **consts}
    for f in CONST_FIELDS:
        L.append(f"| {f} | " + " | ".join(f"{getattr(all_c[k], f):.4g}" for k in keys) + " |")
    L.append("\n> method별 열은 해당 method 행이 `--min-group` 이상일 때만 생깁니다. "
             "그 외 method는 `default`를 씁니다.\n")

    L.append("## 5. 구성요소 검증 (측정값으로 항별 확인)\n")
    L.append("| 설정 | weights 예측 | mem_model 실측 | 오차 | weights+optim 예측 | mem_steady 실측 | 오차 |")
    L.append("|---|---|---|---|---|---|---|")
    seen = set()
    for m in rows:
        k = (m.cfg.model, m.cfg.method, m.cfg.optim, m.cfg.lora_rank, m.cfg.lora_targets)
        if k in seen or m.mem_model_gb is None:
            continue
        seen.add(k)
        c = consts.get(m.cfg.method) or consts["default"]
        f = features(m.cfg, m.spec, c)
        pw, ps = f.known_weights / GB, (f.known_weights + f.known_optimizer) / GB
        e1 = (pw - m.mem_model_gb) / m.mem_model_gb * 100
        s = m.mem_steady_gb
        e2 = f"{(ps - s) / s * 100:+.1f}%" if s else "-"
        L.append(f"| {m.cfg.model}/{m.cfg.method} {m.cfg.optim} r{m.cfg.lora_rank}{m.cfg.lora_targets} "
                 f"| {pw:.2f} | {m.mem_model_gb:.2f} | {e1:+.1f}% | {ps:.2f} | {s if s else '-'} | {e2} |")
    L.append("")

    # 잔차 진단: seq_len/batch/method별 평균 부호 오차 — 특정 축으로 체계적으로 틀리면 식에 항이 빠진 것
    # (예: T4에서 SDPA가 math 커널로 떨어지면 seq² 항이 생겨 긴 seq에서만 과소예측이 커진다)
    L.append("## 5-1. 잔차 진단 (보정 후 평균 부호 오차: +면 과대예측, −면 과소예측)\n")
    for name, key in [("seq_len", lambda m: m.cfg.seq_len), ("micro-batch", lambda m: m.cfg.batch_size),
                      ("method", lambda m: m.cfg.method),
                      ("checkpointing", lambda m: "on" if m.cfg.grad_checkpointing else "off")]:
        groups = {}
        for m, (pa, a) in zip(rows, after):
            groups.setdefault(key(m), []).append((pa - a) / a * 100)
        cells = " · ".join(f"{k}: {sum(v) / len(v):+.1f}% (n={len(v)})" for k, v in sorted(groups.items(), key=lambda t: (isinstance(t[0], str), t[0])))
        L.append(f"- **{name}** — {cells}")
    L.append("")

    L.append("## 6. 행별 오차 (보정 후, in-sample 기준 |오차| 큰 순)\n")
    L.append("| 설정 | 실측 GB | 보정 전 | 보정 후 | 오차(후) |")
    L.append("|---|---|---|---|---|")
    table = sorted(zip(rows, before, after), key=lambda t: -abs(t[2][0] - t[2][1]) / t[2][1])
    for m, (pb, a), (pa, _) in table:
        L.append(f"| {m.label} | {a:.2f} | {pb:.2f} | {pa:.2f} | {(pa - a) / a * 100:+.1f}% |")
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
