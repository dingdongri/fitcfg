"""5차 실측 리포트: ① 플러그인 실제 GPU ② CPU offload 번들 ③ 긴 시퀀스  → reports/round5_report.md
    python scripts/round5_report.py
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from fitcfg import load_calibration  # noqa: E402

load_calibration(REPO / "data" / "calibration.json")


def pct(p, a):
    return (p - a) / a * 100


L = ["# 5차 실측 (2026-10-31, gfarm A100 MIG 3g.40gb) — 플러그인 실제 GPU · CPU offload · 긴 시퀀스", "",
     "## 요약", "",
     "| 확인한 것 | 결과 |", "|---|---|",
     "| HF Trainer 플러그인을 실제 모델에 붙이면 | 학습 방식·checkpointing 자동 감지 7/7 · 보정 모델 메모리 1.6~2.9% · "
     "strict 모드는 OOM 설정을 학습 전 7초 만에 멈추고 들어가는 설정 제안 |",
     "| 긴 시퀀스 (4K~16K) | 메모리 평균 2.1% · OOM 판정 18/18 · 처리량 3.3% (둘 다 보정에 안 쓴 데이터) |",
     "| CPU offload (DeepSpeed) | 문제 3개 발견 → 수정: MIG NVML 오류 우회 · ZeRO-3 단편화 여유 · CPU RAM 식 |",
     "| 처음 보는 구조 식 보완 (GPU 없이 절제 실험) | Gemma-2 LoRA 9.3→1.5% · QLoRA 15.5→7.7% · Qwen3 4.6→2.5% "
     "(Phi-3·Gemma 묶음 6.8→4.1%, 새 4계열 4.4→4.0%) |", "",
     "### GPU 없이 고친 처음 보는 구조의 메모리 식 (FakeTensor 절제 실험 → 상수, 실측은 검증에만 사용)",
     "- attention 폭(q_dim ≠ hidden, Qwen3-4B 4096 vs 2560): 토큰·층당 10 B × (q_dim − hidden) — head 수만 바꾼 추적에서 10.1 B",
     "- 층당 추가 norm(Gemma-2·3 post-norm 2개): 토큰·층·hidden당 4 B × 2 — 같은 크기 Llama 층 대비 +8.0 B/H",
     "- logits softcapping(Gemma-2): logits 원소당 +2 B",
     "- 보정한 모델(Llama·Qwen2.5·Mistral)은 세 항이 모두 0 → 기존 보정 그대로", ""]

# ① 플러그인
pl = REPO / "data" / "plugin_runs_mig.jsonl"
if pl.exists():
    rows = [json.loads(l) for l in pl.read_text(encoding="utf-8").splitlines() if l.strip()]
    L += ["## 1. HF Trainer 플러그인을 실제 모델·GPU에 붙여서 (사용자 코드 그대로, `scripts/plugin_gpu_check.py`)", "",
          "| 시나리오 | 모델 | 감지한 방식·ckpt | 결과 | 예측 GB | 실측 GB | 오차 | 예측 tok/s | 실측 tok/s |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d, p, m = r.get("detected") or {}, r.get("prediction") or {}, r.get("measured") or {}
        mp, pp = m.get("device_peak_gb"), p.get("pred_mem_gb")
        err = f"{pct(pp, mp):+.1f}%" if mp and pp else "—"
        res = {"finished": "학습 완료", "stopped_by_plugin": f"**학습 전 중단** ({r.get('wall_s')}s)",
               "oom": "OOM", "error": "오류"}.get(r["result"], r["result"])
        if r["result"] == "finished" and p and not p.get("fits"):
            res += " (OOM 경고 후 진행)"
        L.append(f"| {r['tag']} | {r['model'].split('/')[-1]} {r['method']} b{r['batch']} | "
                 f"{d.get('method')}·{'on' if d.get('grad_checkpointing') else 'off'} | {res} | {pp} | {mp or '—'} | {err} | "
                 f"{p.get('tokens_per_s') or '—'} | {m.get('tokens_per_s') or '—'} |")
    import sweep as _sw
    extra = []
    for r in rows:
        key = {"Qwen/Qwen3-4B-Base": "qwen3-4b", "unsloth/gemma-2-2b": "gemma-2-2b"}.get(r["model"])
        if key and r["result"] == "finished":
            d = dict(model=key, method=r["method"], batch=r["batch"], seq=r["seq"], ckpt=r["ckpt"], optim="adamw",
                     lora_rank=16, lora_targets="attn")
            p = _sw.predict_gb(d, r["gpu_name"]); m = r["measured"]["device_peak_gb"]
            extra.append(f"{r['tag']}: 새 구조 항 반영 예측(여유 계수 전) {p:.2f}GB vs 실측 {m:.2f}GB ({pct(p, m):+.1f}%)")
    if extra:
        L += ["", "- 처음 보는 모델은 서버에서 이전 식으로 예측(여유 계수 포함). 이번에 고친 식으로 다시 계산하면:"]
        L += [f"  - {e}" for e in extra]
    fin = [r for r in rows if r["result"] == "finished" and (r.get("measured") or {}).get("device_peak_gb")]
    if fin:
        e = [abs(pct(r["prediction"]["pred_mem_gb"], r["measured"]["device_peak_gb"])) for r in fin]
        ok_det = sum((r.get("detected") or {}).get("method") == r["method"] and
                     bool((r.get("detected") or {}).get("grad_checkpointing")) == bool(r["ckpt"]) for r in rows)
        L += ["", f"- 학습 방식·checkpointing 자동 감지 {ok_det}/{len(rows)} · 메모리 예측 평균 {sum(e)/len(e):.1f}% (최대 {max(e):.1f}%)"]
    L.append("")

# ② offload 번들
bl = REPO / "data" / "bundle_runs_mig.jsonl"
if bl.exists():
    rows = [json.loads(l) for l in bl.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if str(r.get("bundle", "")).startswith("off")]
    if rows:
        L += ["## 2. CPU offload 추천 번들 실제 실행 (DeepSpeed, 작은 GPU를 가정해 추천받은 설정)", "",
              "| 번들 | 모델 | 설정 | 결과 | 예측 GB | 실측 GB | 오차 | tok/s 예측/실측 |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            mp, pp = r.get("device_peak_gb"), r.get("predicted_peak_gb")
            strat = r.get("strategy") or ("ZeRO-3+param offload" if r.get("cpu_offload_params") else "ZeRO-2+optim offload")
            L.append(f"| {r['bundle']} | {r['model_key']} | mb{r['micro_batch']}·누적{r['grad_accum']}·ckpt "
                     f"{'on' if r['gradient_checkpointing'] else 'off'} · {strat} | {'성공' if r['ok'] else r.get('note')} | "
                     f"{pp} | {mp or '—'} | {f'{pct(pp, mp):+.1f}%' if mp else '—'} | "
                     f"{r.get('predicted_tokens_per_s')}/{r.get('tokens_per_s') or '—'} |")
        L.append("")

off = REPO / "data" / "bundle_runs_offload.jsonl"
if off.exists():
    from fitcfg import TrainConfig, estimate, get_model
    from fitcfg.memory import GB, cpu_ram_needed_gb
    L += ["**재실행 (DeepSpeed NVML 오류 우회 후)** — 할당/예약/비-PyTorch 메모리를 나눠 기록", "",
          "| 번들 | 결과 | 할당 GB | 예약 GB | 비-PyTorch GB | GPU peak GB | 이전 예측 | 새 예측 | CPU RAM 새 예측 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for l in off.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        tc = TrainConfig(model=r["model_key"], method="full", batch_size=r["micro_batch"], seq_len=r["seq_len"],
                         grad_checkpointing=r["gradient_checkpointing"], zero_stage=r["zero_stage"],
                         cpu_offload_optim=r["cpu_offload_optim"], cpu_offload_params=r["cpu_offload_params"],
                         master_weights=True)
        spec = get_model(r["model_key"])
        new = estimate(tc, spec).total / GB
        L.append(f"| {r['bundle']} | {'성공' if r['ok'] else r['note']} | {r.get('peak_alloc_gb', '—')} | "
                 f"{r.get('peak_reserved_gb', '—')} | {r.get('nontorch_gb', '—')} | {r.get('device_peak_gb', '—')} | "
                 f"{r['predicted_peak_gb']} | {new:.2f} | {cpu_ram_needed_gb(tc, spec):.0f} GB |")
    L += ["", "- 발견 ① gfarm(MIG)에서는 DeepSpeed가 NVML 권한 오류로 시작도 못 함 → 생성 `train.py`에 우회 코드 추가",
          "- 발견 ② ZeRO-3 + 파라미터 offload: 할당 예측은 맞지만(5.26 vs 5.00GB) 할당기 예약이 1.76배 → "
          "ZeRO-3에 단편화 여유(할당 × 0.75) 추가, 추천 시 경고",
          "- 발견 ③ 3B ZeRO-2 offload: CPU 메모리 한도 64GB 초과로 컨테이너가 재시작됨 → CPU RAM 식 보수적으로 수정 "
          "(이제 60GB 노드에서는 이 설정을 추천하지 않음)", ""]

# ③ 긴 시퀀스
lc = REPO / "data" / "measurements_mig_long.csv"
if lc.exists():
    import sweep
    rows = list(csv.DictReader(open(lc)))
    L += ["## 3. 긴 시퀀스 (4096~16384)", "", "| 모델 | 방식 | batch | seq | ckpt | 결과 | 예측 GB | 실측 GB | 오차 | OOM 판정 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    errs, judge = [], [0, 0, 0]
    for r in rows:
        d = dict(model=r["model"], method=r["method"], batch=int(r["batch"]), seq=int(r["seq"]), ckpt=r["ckpt"] == "1",
                 optim=r["optim"], lora_rank=int(r["lora_rank"]), lora_targets=r["lora_targets"])
        pred = sweep.predict_gb(d, r["gpu_name"])
        cap = float(r["gpu_total_gb"] or 39.25)
        oom = r["note"].upper().startswith("OOM")
        meas = float(r["device_peak_gb"]) if r["device_peak_gb"] else None
        pred_oom = pred > cap * 0.895
        j = "정답" if pred_oom == oom else ("놓침" if oom else "오경보")
        judge[0 if j == "정답" else 1 if j == "놓침" else 2] += 1
        if meas:
            errs.append(pct(pred, meas))
        L.append(f"| {r['model']} | {r['method']} | {r['batch']} | {r['seq']} | {'on' if r['ckpt']=='1' else 'off'} | "
                 f"{'OOM' if oom else '성공'} | {pred:.2f} | {meas or '—'} | {f'{pct(pred, meas):+.1f}%' if meas else '—'} | {j} |")
    if errs:
        L += ["", f"- 성공 {len(errs)}개 예측 평균 {sum(map(abs, errs))/len(errs):.1f}% (범위 {min(errs):+.1f} ~ {max(errs):+.1f}%) · "
                  f"OOM 판정(안전계수 0.895) 정답 {judge[0]} · 놓침 {judge[1]} · 오경보 {judge[2]}"]
if lc.exists():
    from fitcfg.calibration import load_measurements
    from fitcfg.hardware import match_gpu
    from fitcfg.throughput import estimate_throughput
    ms = [m for m in load_measurements(lc)[0] if m.ok and m.tokens_per_s]
    te = [estimate_throughput(m.cfg, m.spec, match_gpu(m.gpu_name), 1).tokens_per_s / m.tokens_per_s - 1 for m in ms]
    if te:
        L.append(f"- 처리량 예측 v2 (긴 시퀀스는 보정에 안 씀): 평균 {sum(map(abs, te))/len(te)*100:.1f}% "
                 f"(범위 {min(te)*100:+.1f} ~ {max(te)*100:+.1f}%)")
out = REPO / "reports" / "round5_report.md"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
