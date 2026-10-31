"""소규모 GPU 환경에서 3D 병렬화(TP×PP×DP)가 DP·FSDP 대비 이득인지 — GPU 없이 시뮬레이션.

    python scripts/sim_3d.py            # → reports/sim_3d_report.md

탐색 구조는 OMNIA 인턴 과제(search.py)와 같은 방식:
  축 열거(tp, pp, dp, mbs) → GPU 배치(TP 묶음은 같은 종류) → 동적 계획법으로 층 나누기(병목 최소, 메모리 제약)
  → 1F1B 시간 = Σ stage 시간 + (M−1)·병목 stage 시간 + DP/FSDP 통신 + optimizer update
여기에 키트에 없던 것을 추가:
  - FSDP(ZeRO-3): 가중치·기울기·옵티마이저를 dp로 나눠 보관, 대신 micro-batch마다 가중치 all-gather
  - 파인튜닝 방식별 메모리/연산: Full / LoRA / QLoRA (fitcfg의 FakeTensor 보정 상수 사용)
  - checkpointing on/off
통신과 계산의 겹침(overlap)은 확신할 수 없어서 두 가정을 모두 계산한다:
  pess = DP·FSDP 통신이 전혀 안 겹침,  opt = 계산 시간만큼 숨겨짐
TP 통신(층마다 all-reduce)과 PP의 forward 전송은 두 가정 모두 '안 겹침'(키트 규약과 같음).

한계: 전부 계산 기반(실측 아님). 대역폭·효율 값은 공개 수치 기반 가정. 결론은 '방향'으로만 읽을 것.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fitcfg.models import ModelSpec, get_model  # noqa: E402

import fitcfg.parallel as core  # noqa: E402  (계산 모델은 패키지로 옮김 — 이 스크립트는 하드웨어 목록·리포트만)
from fitcfg.parallel import (GPU, RTX3090, RTX4090, A100_40, A100_80, search, label)  # noqa: E402,F401

LLAMA_70B = ModelSpec(name="meta-llama/Llama-3.1-70B", n_params=70_550_000_000, n_layers=80, hidden=8192,
                      n_heads=64, n_kv_heads=8, vocab=128256, intermediate=28672, tied_embeddings=False)

# (이름, GPU 목록, GPU 간 통신 대역폭 GB/s: all-reduce 실효값)
#  4090은 GPU끼리 직접 통신(P2P)이 막혀 있어 PCIe·CPU 메모리 경유 → 실효 15GB/s로 가정
#  A100 SXM은 NVLink → 실효 230GB/s
HARDWARE = [
    ("4090 ×4 (PCIe)", [RTX4090] * 4, 15.0),
    ("4090 ×8 (PCIe)", [RTX4090] * 8, 15.0),
    ("A100-40 ×4 (NVLink)", [A100_40] * 4, 230.0),
    ("A100-80 ×8 (NVLink)", [A100_80] * 8, 230.0),
    ("3090 ×2 + 4090 ×2 (PCIe, 혼합)", [RTX3090] * 2 + [RTX4090] * 2, 15.0),
]
MODELS = [("Llama-8B", get_model("llama-3.1-8b")), ("Llama-70B", LLAMA_70B)]
METHODS = ["full", "lora", "qlora"]
SEQ, GLOBAL_BATCH, SAFETY, OVERHEAD_GB = core.SEQ, core.GLOBAL_BATCH, core.SAFETY, core.OVERHEAD_GB


def main():
    rows = []
    for hname, gpus, bw in HARDWARE:
        for mname, m in MODELS:
            for method in METHODS:
                out = {}
                for ov in (False, True):
                    res = search(m, method, gpus, bw, ov)
                    dpfam = [r for r in res if r[1]["tp"] == 1 and r[1]["pp"] == 1]
                    best = res[0] if res else None
                    bdp = dpfam[0] if dpfam else None
                    out[ov] = (best, bdp)
                rows.append((hname, mname, method, out))
                b, d = out[True]
                print(f"{hname:28} {mname:9} {method:5}  best={label(b[1]) if b else '불가':40} "
                      f"dp={label(d[1]) if d else '불가'}", flush=True)

    def ratio(best, bdp):
        if best is None:
            return "—"
        if bdp is None:
            return "DP·FSDP 불가 → 3D만 가능"
        r = bdp[0] / best[0]
        return "1.00× (DP·FSDP가 최적)" if r < 1.005 else f"{r:.2f}×"

    L = ["# 소규모 GPU에서 3D 병렬화는 이득인가 (계산 기반 시뮬레이션)", "",
         f"- 시퀀스 {SEQ}, global batch {GLOBAL_BATCH}, 메모리 여유 {SAFETY}, GPU당 기본 사용량 {OVERHEAD_GB}GB",
         "- 'DP·FSDP 최선' = TP·PP 없이 데이터 병렬 계열만 쓴 최선, '전체 최선' = TP·PP 포함 탐색",
         "- 이득 = DP·FSDP 최선 시간 ÷ 전체 최선 시간 (1.00×면 3D 병렬화 불필요)",
         "- 통신-계산 겹침 가정 두 가지: 비관(안 겹침) / 낙관(계산 시간만큼 숨겨짐)", "",
         "| 하드웨어 | 모델 | 방식 | 전체 최선 (낙관) | DP·FSDP 최선 (낙관) | 이득 (낙관) | 이득 (비관) |",
         "|---|---|---|---|---|---|---|"]
    for hname, mname, method, out in rows:
        b, d = out[True]
        bp, dpp = out[False]
        L.append(f"| {hname} | {mname} | {method} | {label(b[1]) if b else '불가'} | "
                 f"{label(d[1]) if d else '불가'} | {ratio(b, d)} | {ratio(bp, dpp)} |")
    p = REPO / "reports" / globals().get("_REPORT_NAME", "sim_3d_report.md")
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("->", p)


if __name__ == "__main__":
    if "--measured" in sys.argv:           # 실측으로 맞춘 계산 상수로 실행 → reports/sim_3d_report_recal.md
        core.EFF.update(core.MEASURED["EFF"])
        core.T_HALF_BASE = core.MEASURED["T_HALF_BASE"]
        _REPORT_NAME = "sim_3d_report_recal.md"
    main()
