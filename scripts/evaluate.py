"""Phase 7 준비 — 시나리오 벤치: 'HF Trainer 기본값' vs 'fitcfg 추천' (GPU 불필요).

    python scripts/evaluate.py                 # 기본 시나리오 격자
    python scripts/evaluate.py --calib data/calibration.json

지표 (모두 cost model 예측 기반 시뮬레이션 — 실측 검증은 calibrate.py의 OOM 판정으로 따로 한다)
    1) 기본값 OOM 비율      : 기본값(micro 8, ckpt off, AdamW)이 메모리를 넘는 시나리오 비율
    2) 구제율               : 기본값이 OOM인 시나리오 중 fitcfg가 들어가는 설정을 찾은 비율
    3) 시행착오 횟수         : 사람이 흔히 하는 수동 탐색(배치 절반 → checkpointing → 8bit → offload)으로
                              처음 들어가는 설정을 찾기까지의 실패 횟수 (fitcfg는 0회)
    4) 처리량 이득           : 수동 탐색으로 찾은 첫 설정 대비 fitcfg 추천의 예측 처리량 배수

출력: reports/evaluation.md, reports/evaluation.csv
"""
from __future__ import annotations
import argparse, csv, statistics, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fitcfg import GPUS, PlanRequest, load_calibration, plan  # noqa: E402
from fitcfg.recommend import _evaluate, _make_config, naive_config  # noqa: E402
from fitcfg.models import get_model  # noqa: E402

MODELS = ["llama-3.2-1b", "qwen2.5-3b", "qwen2.5-7b", "llama-3.1-8b"]
METHODS = ["qlora", "lora", "full"]
GPU_KEYS = ["t4", "l4", "rtx4090", "a100-40", "a100-80"]
GPU_COUNTS = [1, 2]
SEQS = [1024, 2048, 4096]
MINUTES_PER_FAILED_TRY = 4.0      # 모델 로드 + OOM까지 걸리는 시간(보수적 가정)


def manual_search(req: PlanRequest):
    """사람의 전형적인 수동 탐색 순서를 흉내 낸다. 반환: (실패 횟수, 처음 들어간 설정 or None)."""
    spec = get_model(req.model)
    safety = 0.90 if req.safety is None else req.safety
    strategy = "single" if req.gpus == 1 else "ddp"
    tries = []
    for ckpt in (False, True):                       # 1) 배치를 절반씩 → 2) checkpointing 켜고 다시
        for mb in (8, 4, 2, 1):
            tries.append((strategy, mb, ckpt, "adamw"))
    if req.method == "full":                          # 3) 8bit optimizer  4) DeepSpeed offload
        tries += [(strategy, mb, True, "adamw8bit") for mb in (4, 2, 1)]
        if req.allow_offload:
            tries += [("zero-offload", mb, True, "adamw") for mb in (4, 2, 1)]
            tries += [("zero3-offload", mb, True, "adamw") for mb in (4, 2, 1)]
    fails = 0
    for st, mb, ck, op in tries:
        r = _evaluate(req, spec, _make_config(req, st, mb, ck, op), st, safety)
        if r.fits:
            return fails, r
        fails += 1
    return fails, None


def runs_section(paths: list[str]) -> str:
    """실제로 GPU에서 번들을 돌린 결과: 추천 설정 vs 기본값 설정의 OOM, 메모리 예측 오차."""
    import json
    recs = [json.loads(l) for p in paths for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not recs:
        return ""
    lines = ["## 실제 실행 검증 (GPU에서 생성 번들 실행)", "",
             "| 라벨 | 모델/방식 | 설정 | GPU | 결과 | 예측 peak | 실측 peak | 오차 | 처리량 실측/예측 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        cfg = (f"b{r['micro_batch']}×{r['grad_accum']} s{r['seq_len']} "
               f"ckpt{'on' if r['gradient_checkpointing'] else 'off'} ZeRO{r['zero_stage']}")
        res = "✅ 성공" if r.get("ok") else f"❌ {r.get('note', '')}"
        act = r.get("device_peak_gb")
        err = f"{(r['predicted_peak_gb'] - act) / act * 100:+.1f}%" if act else "-"
        tps = (f"{r['tokens_per_s']:,.0f} / {r['predicted_tokens_per_s']:,.0f}" if r.get("tokens_per_s") else "-")
        lines.append(f"| {r.get('label')} | {r['model_key']}/{r['method']} | {cfg} | {r.get('gpu_name', '')} | {res} | "
                     f"{r['predicted_peak_gb']:.2f} | {act if act else '-'} | {err} | {tps} |")
    by = {}
    for r in recs:
        by.setdefault(r.get("label"), []).append(r)
    lines.append("")
    for lab, rs in by.items():
        n_oom = sum(1 for r in rs if (r.get("note") or "").startswith("OOM"))
        lines.append(f"- **{lab}**: {len(rs)}회 실행 중 OOM {n_oom}회")
    errs = [abs(r["predicted_peak_gb"] - r["device_peak_gb"]) / r["device_peak_gb"] * 100
            for r in recs if r.get("ok") and r.get("device_peak_gb")]
    if errs:
        lines.append(f"- 성공한 실행의 메모리 예측 오차(MAPE): **{statistics.mean(errs):.1f}%** (n={len(errs)})")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calib", default=None)
    p.add_argument("--out", default=str(REPO / "reports" / "evaluation.md"))
    p.add_argument("--runs", nargs="*", default=[],
                   help="run_bundle.py 결과 jsonl → '실제 실행 검증' 절 추가 (추천 vs 기본값 실제 OOM)")
    args = p.parse_args()
    calib_note = "문헌 사전값(보정 전)"
    if args.calib:
        load_calibration(args.calib)
        calib_note = f"보정값 `{args.calib}`"

    rows = []
    for m in MODELS:
        for meth in METHODS:
            for g in GPU_KEYS:
                for n in GPU_COUNTS:
                    for s in SEQS:
                        req = PlanRequest(model=m, gpu=GPUS[g], gpus=n, method=meth, seq_len=s,
                                          global_batch=16, cpu_ram_gb=128)
                        res = plan(req, top_k=1)
                        naive = naive_config(req)
                        fails, manual = manual_search(req)
                        best = res.best
                        rows.append({
                            "model": m, "method": meth, "gpu": g, "gpus": n, "seq": s,
                            "naive_gb": naive.pred_mem_gb, "naive_oom": int(not naive.fits),
                            "fitcfg_found": int(best is not None),
                            "fitcfg_gb": best.pred_mem_gb if best else "",
                            "fitcfg_tps": round(best.throughput.tokens_per_s) if best else "",
                            "manual_fails": fails,
                            "manual_found": int(manual is not None),
                            "manual_tps": round(manual.throughput.tokens_per_s) if manual else "",
                            "fitcfg_strategy": best.strategy if best else "",
                        })

    n = len(rows)
    naive_oom = [r for r in rows if r["naive_oom"]]
    rescued = [r for r in naive_oom if r["fitcfg_found"]]
    both = [r for r in rows if r["fitcfg_found"] and r["manual_found"]]
    gains = [r["fitcfg_tps"] / r["manual_tps"] for r in both]
    fails = [r["manual_fails"] for r in rows if r["manual_found"]]
    only_fitcfg = [r for r in rows if r["fitcfg_found"] and not r["manual_found"]]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def by(key):
        lines = [f"| {key} | 시나리오 | 기본값 OOM | fitcfg 찾음 | 평균 시행착오 | 처리량 이득(중앙값) |",
                 "|---|---|---|---|---|---|"]
        for v in sorted({r[key] for r in rows}, key=str):
            sub = [r for r in rows if r[key] == v]
            sb = [r["fitcfg_tps"] / r["manual_tps"] for r in sub if r["fitcfg_found"] and r["manual_found"]]
            sf = [r["manual_fails"] for r in sub if r["manual_found"]]
            lines.append(f"| {v} | {len(sub)} | {sum(r['naive_oom'] for r in sub) / len(sub):.0%} | "
                         f"{sum(r['fitcfg_found'] for r in sub) / len(sub):.0%} | "
                         f"{statistics.mean(sf) if sf else 0:.1f}회 | "
                         f"{statistics.median(sb) if sb else float('nan'):.2f}× |")
        return "\n".join(lines)

    md = f"""# 시나리오 벤치: HF Trainer 기본값 vs fitcfg (예측 기반 시뮬레이션)

- 상수: {calib_note} · 시나리오 {n}개 = 모델 {len(MODELS)} × method {len(METHODS)} × GPU {len(GPU_KEYS)} × GPU 수 {GPU_COUNTS} × seq {SEQS}
- 공통: global batch 16, CPU RAM 128GB, 안전계수 0.90
- ⚠️ 모두 cost model **예측**으로 판정한 값이다. 예측 자체의 정확도는 `calibrate.py`(실측 MAPE, OOM 판정)로 따로 보인다.

## 요약
| 지표 | 값 |
|---|---|
| 기본값이 OOM인 시나리오 | **{len(naive_oom)}/{n} ({len(naive_oom) / n:.0%})** |
| 그중 fitcfg가 들어가는 설정을 찾음 (구제율) | **{len(rescued)}/{len(naive_oom)} ({len(rescued) / max(len(naive_oom), 1):.0%})** |
| 수동 탐색이 첫 성공까지 겪는 OOM 실패 | 평균 **{statistics.mean(fails):.1f}회** (≈ GPU {statistics.mean(fails) * MINUTES_PER_FAILED_TRY:.0f}분 낭비, 1회 {MINUTES_PER_FAILED_TRY:g}분 가정) → fitcfg 0회 |
| 수동 탐색은 못 찾고 fitcfg만 찾음 | {len(only_fitcfg)}개 |
| fitcfg 추천 vs 수동 탐색 첫 성공 설정의 처리량 | 중앙값 **{statistics.median(gains):.2f}×** (최대 {max(gains):.2f}×) |

수동 탐색 순서: micro-batch 8→4→2→1 → checkpointing 켜고 8→1 → (Full) 8bit AdamW → ZeRO-Offload → ZeRO-3 offload.

## method별
{by("method")}

## GPU별
{by("gpu")}

## 모델별
{by("model")}
"""
    if args.runs:
        md += "\n" + runs_section(args.runs)
    out.write_text(md, encoding="utf-8")
    print(md.split("## method별")[0])
    print(f"-> {out}\n-> {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
