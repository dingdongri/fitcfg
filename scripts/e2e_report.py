"""추천 설정 실제 학습 결과 정리 → reports/e2e_report.md

    python scripts/e2e_report.py [--log data/bundle_runs_mig.jsonl]

시나리오마다: HF 기본값 결과 · 추천 1~3위의 성공 여부 · 예측 vs 실측 peak · 예측 vs 실측 속도 · 순위 일치
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--log", default=str(REPO / "data" / "bundle_runs_mig.jsonl"))
ap.add_argument("--out", default=str(REPO / "reports" / "e2e_report.md"))
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.log, encoding="utf-8") if l.strip()]
by = defaultdict(dict)
for r in rows:                                   # 같은 번들을 여러 번 돌렸으면 마지막 결과
    sc, _, b = r["bundle"].rpartition("_")
    by[sc][b] = r


def cfg(r):
    return (f"micro {r['micro_batch']} × 누적 {r['grad_accum']} · ckpt {'on' if r['gradient_checkpointing'] else 'off'}"
            + (f" · {r['optim_label']}" if r.get("optim_label") and r["optim_label"] != "adamw" else ""))


L = ["# 추천 설정으로 실제 학습 (gfarm A100 MIG 3g.40gb)\n",
     "도구가 만든 학습 스크립트 그대로, 실제 데이터(alpaca, 길이 맞춰 이어붙임) · 짧은 학습(앞 3스텝은 속도 측정 제외)\n"]
n_sc = naive_oom = rec_ok = rec_n = top1_fastest = 0
order_ok = order_n = 0
peak_err, tps_err = [], []
for sc, d in sorted(by.items()):
    first = next(iter(d.values()))
    n_sc += 1
    L.append(f"\n## {sc} — {first['model_key']} · {first['method']} · seq {first['seq_len']}\n")
    L.append("| 설정 | 순위 | 결과 | peak 예측 | peak 실측 | 오차 | 속도 예측 | 속도 실측 | 오차 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    recs = []
    for b in ["naive", "rec1", "rec2", "rec3"]:
        if b not in d:
            continue
        r = d[b]
        ok = r.get("ok") == 1
        pk, pt = r.get("predicted_peak_gb"), r.get("predicted_tokens_per_s")
        mk, mt = r.get("device_peak_gb"), r.get("tokens_per_s")
        ek = f"{(pk - mk) / mk * 100:+.1f}%" if ok and mk else ""
        et = f"{(pt - mt) / mt * 100:+.1f}%" if ok and mt else ""
        if b == "naive":
            naive_oom += r.get("note") == "OOM"
        else:
            rec_n += 1
            rec_ok += ok
            if ok and mk:
                peak_err.append((pk - mk) / mk * 100)
            if ok and mt:
                tps_err.append((pt - mt) / mt * 100)
                recs.append((int(b[3:]), pt, mt))
        res = "✅ 성공" if ok else f"❌ {r.get('note', '')[:40]}"
        L.append(f"| {cfg(r)} | {'기본값' if b == 'naive' else b[3:] + '위'} | {res} | {pk:.1f} | "
                 f"{f'{mk:.2f}' if mk else '-'} | {ek} | {pt:,.0f} | {f'{mt:,.0f}' if mt else '-'} | {et} |")
    if len(recs) >= 2:
        fastest = max(recs, key=lambda x: x[2])[0]
        top1_fastest += fastest == 1
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                (_, pi, mi), (_, pj, mj) = recs[i], recs[j]
                if abs(pi - pj) / max(pi, pj) < 0.03:      # 예측상 거의 같은 쌍은 순위 평가에서 제외
                    continue
                order_n += 1
                order_ok += (pi > pj) == (mi > mj)
        L.append(f"\n실측 가장 빠른 설정: {fastest}위 추천" + (" ✅" if fastest == 1 else " ⚠️"))

mean = lambda v: sum(map(abs, v)) / len(v) if v else float("nan")
S = ["\n## 요약\n", "| 항목 | 결과 |", "|---|---|",
     f"| 시나리오 | {n_sc}개 |",
     f"| HF 기본값 → OOM | {naive_oom}/{n_sc} |",
     f"| 추천 설정 학습 성공 | **{rec_ok}/{rec_n}** |",
     f"| peak 메모리 예측 오차 (추천 설정) | 평균 {mean(peak_err):.1f}% (범위 {min(peak_err, default=0):+.1f} ~ {max(peak_err, default=0):+.1f}%) |",
     f"| 속도 예측 오차 | 평균 {mean(tps_err):.1f}% (범위 {min(tps_err, default=0):+.1f} ~ {max(tps_err, default=0):+.1f}%) |",
     f"| 추천 1위가 실제로 가장 빠름 | {top1_fastest}/{sum(1 for d in by.values() if sum(1 for b in d if b != 'naive' and d[b].get('ok') == 1) >= 2)} |",
     f"| 두 추천의 빠르기 순서가 맞음 (예측 차이 3% 이상인 쌍) | {order_ok}/{order_n} |"]
L[2:2] = S
Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(S))
print("->", a.out)
