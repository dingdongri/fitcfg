"""명령줄 인터페이스.

    python -m fitcfg plan --model llama-3.1-8b --gpu rtx4090 --method qlora
    python -m fitcfg plan --model llama-3.2-1b --gpu t4 --method lora --out runs/lora_t4   # 번들 생성
    python -m fitcfg gpus        # GPU 프리셋 목록
    python -m fitcfg models      # 모델 프리셋 목록
    python -m fitcfg measure-cpu --model Qwen/Qwen3-4B --method qlora --batch 1 --seq 2048   # GPU 없이 직접 측정

data/calibration.json 이 있으면 자동으로 보정값을 사용한다 (--calib 으로 다른 파일 지정, --no-calib 으로 끔).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from . import (GPUS, MODELS, PlanRequest, custom_gpu, get_gpu, load_calibration, plan,
               calibration_info)
from .generate import build_bundle, write_bundle

REPO = Path(__file__).resolve().parents[1]
# 저장소에서 실행하면 data/calibration.json(최신 보정), pip 설치본이면 패키지에 든 사본
DEFAULT_CALIB = (REPO / "data" / "calibration.json") if (REPO / "data" / "calibration.json").exists() \
    else Path(__file__).resolve().parent / "calibrated.json"


def maybe_load_calibration(path: str | None, disabled: bool = False) -> str:
    if disabled:
        return "GPU 실측 보정 끔 (FakeTensor 추적 기반 상수, overhead는 추정)"
    p = Path(path) if path else DEFAULT_CALIB
    if p.exists():
        load_calibration(p)
        info = calibration_info()
        mape = info.get("heldout_mape")
        tag = " ⚠️합성 데이터" if info.get("synthetic") else ""
        return (f"보정값 사용: {p.name} (측정 {info.get('n_rows')}행"
                + (f", held-out MAPE {mape:.1f}%" if mape is not None else "") + f"){tag}")
    return "GPU 실측 보정 전 (FakeTensor 추적 기반 상수, overhead는 추정) — data/calibration.json 없음"


def cmd_plan(a) -> int:
    status = maybe_load_calibration(a.calib, a.no_calib)
    gpu = custom_gpu(a.gpu_mem) if a.gpu_mem else get_gpu(a.gpu)
    gpu_list = None
    if a.gpu_mix:                     # 예: rtx3090:2,rtx4090:2
        gpu_list = []
        for part in a.gpu_mix.split(","):
            k, _, n = part.strip().partition(":")
            gpu_list += [get_gpu(k)] * int(n or 1)
    req = PlanRequest(model=a.model, gpu=gpu, gpus=a.gpus, method=a.method, seq_len=a.seq,
                      gpu_list=gpu_list, link=a.link,
                      global_batch=a.global_batch, lora_rank=a.lora_rank, lora_targets=a.lora_targets,
                      allow_offload=not a.no_offload, cpu_ram_gb=a.cpu_ram, safety=a.safety,
                      dataset_tokens=a.data_tokens, epochs=a.epochs)
    res = plan(req, top_k=a.top)
    print(f"[{status}]")
    print(f"{a.model} · {a.method} · {gpu.label} ×{a.gpus} · seq {a.seq} · global batch {a.global_batch}")
    print(f"후보 {res.n_candidates}개 중 {res.n_feasible}개가 메모리에 들어감 (허용 = GPU × {res.safety})")
    for n in res.notes:
        print(f"ℹ️ {n}")
    print()
    if not res.best and not (a.out and a.naive):
        print("❌ 들어가는 설정이 없습니다.")
        for d in res.diagnosis:
            print("  ", d.replace("**", ""))
        return 1
    if not res.best:
        print("❌ 들어가는 설정이 없습니다.")
    for i, r in enumerate(res.recommendations, 1):
        tp = r.throughput
        tags = f" [{', '.join(r.tags)}]" if r.tags else ""
        print(f"#{i}{tags}  {r.reason}")
        print(f"    메모리 {r.pred_mem_gb:.1f}/{r.cap_gb:.1f}GB · 처리량 {tp.tokens_per_s:,.0f} tok/s "
              f"({tp.low:,.0f}~{tp.high:,.0f})"
              + (f" · 시간 {r.hours[0]:.1f}h ({r.hours[1]:.1f}~{r.hours[2]:.1f})" if r.hours else ""))
        for w in r.warnings:
            print(f"    ⚠️ {w}")
    if res.parallel:
        pd = res.parallel
        head = {"3d": "✅ 3D 병렬화 추천", "dp": "➖ 3D 병렬화 불필요", "none": "❌ GPU 메모리만으로는 불가"}[pd.verdict]
        print(f"\n[GPU {pd.n_gpus}장 나누기 판단] {head}")
        print(f"  {pd.reason}")
        if pd.best and pd.best.is_3d:
            stages = " / ".join(f"{g}:{n}층({m:.1f}GB)" for g, n, m in zip(pd.best.stages, pd.best.layers, pd.best.mem_gb))
            print(f"  3D 최선: {pd.best.label} · stage별 GPU:층(예측 메모리) {stages}")
        if pd.best_dp:
            print(f"  DP·FSDP 최선: {pd.best_dp.label}")
        # 3D 판단은 '상대 비교'만 보여준다 (절대 속도는 위 추천의 처리량 예측을 따름 — 두 모델의 효율 가정이 다름)
        for nt in pd.notes:
            print(f"  ℹ️ {nt}")
    n = res.naive
    verdict = "OOM 예상" if not n.fits else "들어감"
    print(f"\n비교) HF Trainer 기본값({n.reason}): 예측 {n.pred_mem_gb:.1f}GB → {verdict}")
    if a.out:
        pick = res.naive if a.naive else res.recommendations[a.pick - 1]
        out = write_bundle(build_bundle(pick, req, dataset=a.dataset, text_field=a.text_field,
                                        dataset_split=a.dataset_split), a.out)
        which = "기본값(naive)" if a.naive else f"#{a.pick}"
        print(f"\n✅ {which} 설정으로 학습 번들 생성: {out}/  (실행: bash {out}/launch.sh)")
        if res.parallel and res.parallel.verdict == "3d" and res.parallel.best:
            import json, dataclasses
            pp = dataclasses.asdict(res.parallel.best)
            (out / "parallel_plan.json").write_text(json.dumps(
                {"verdict": res.parallel.verdict, "reason": res.parallel.reason, "plan": pp,
                 "link": res.parallel.link}, ensure_ascii=False, indent=2), encoding="utf-8")
            print("   + parallel_plan.json: 3D 나누기 계획 (TP·PP 실행 스크립트 자동 생성은 아직 미지원 — "
                  "번들은 데이터 병렬 최선 설정으로 생성)")
    return 0


def cmd_measure_cpu(a) -> int:
    """처음 보는 모델: CPU에서 가짜 텐서로 학습 1~2스텝을 흉내 내 GPU 메모리를 측정하고 예측식과 비교."""
    status = maybe_load_calibration(a.calib, a.no_calib)
    from .autospec import fetch_config, classify, spec_from_config
    from .memory import TrainConfig, estimate, GB
    from .models import MODELS
    try:
        from .trace import measure_cpu
    except ImportError as e:
        print(f"❌ CPU 측정에는 torch·transformers·peft가 필요합니다: {e}")
        return 1
    name = a.model
    if name.lower() in MODELS:
        print("ℹ️ 등록 모델은 이미 실측 보정되어 있어 CPU 측정이 필요 없습니다. HF 이름이나 config.json 경로를 넣으세요.")
        return 1
    cfg = fetch_config(name)
    info = classify(cfg)
    if not info.supported:
        print(f"❌ {info.note}")
        return 1
    spec = spec_from_config(cfg, name)
    print(f"[{status}]")
    print(f"{name} ({info.model_type}, {spec.n_params/1e9:.2f}B, {info.kind}) · {a.method} · batch {a.batch} · "
          f"seq {a.seq} · checkpointing {'켬' if a.ckpt else '끔'}")
    print("CPU에서 가짜 텐서로 학습 2스텝 추적 중… (실제 계산·가중치 다운로드 없음, 수십 초~몇 분)")
    r = measure_cpu(name, a.method, a.batch, a.seq, a.ckpt, a.lora_rank)
    tc = TrainConfig(model=name, method=a.method, batch_size=a.batch, seq_len=a.seq,
                     grad_checkpointing=a.ckpt, lora_rank=a.lora_rank)
    f = estimate(tc, spec).total / GB
    m = info.margin(a.method)
    print(f"\n  CPU 측정 (GPU에서 예상 peak) : {r['device_peak_gb']:.2f} GB"
          + (f"   ← {r['rule']}" if "rule" in r else ""))
    print(f"  예측식                      : {f:.2f} GB" + (f"  (여유 ×{m:.2f} 포함 {f*m:.2f} GB)" if m != 1.0 else ""))
    diff = (f - r["device_peak_gb"]) / r["device_peak_gb"] * 100
    print(f"  차이                        : {diff:+.1f}%  (예측식 기준)")
    if a.method == "qlora":
        print("  ℹ️ QLoRA(4bit)는 CPU로 직접 흉내 낼 수 없어 'LoRA 측정 × 1.3 + 4bit 가중치' 규칙으로 환산 (실측 8계열 5% 안팎)")
    else:
        print("  ℹ️ CPU 측정 정확도: 실측 8개 계열에서 LoRA·Full 1~2%")
    if a.gpu_mem:
        cap = a.gpu_mem * (a.safety or 0.895)
        ok = r["device_peak_gb"] <= cap
        print(f"\n  GPU {a.gpu_mem:.0f}GB (안전 한도 {cap:.1f}GB) → {'✅ 들어감' if ok else '❌ OOM 예상'}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fitcfg", description="소규모 셋업 파인튜닝 설정 추천기")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("plan", help="설정 추천 (+ --out 으로 학습 번들 생성)")
    q.add_argument("--model", required=True,
                   help=f"등록 모델({', '.join(MODELS)}) 또는 처음 보는 모델의 HF 이름(org/model)·config.json 경로")
    q.add_argument("--method", default="qlora", choices=["qlora", "lora", "full"])
    q.add_argument("--gpu", default="t4", choices=list(GPUS))
    q.add_argument("--gpu-mem", type=float, default=None, help="프리셋 대신 GPU 메모리(GB) 직접 지정")
    q.add_argument("--gpus", type=int, default=1)
    q.add_argument("--gpu-mix", default=None, help="종류가 섞인 GPU (예: rtx3090:2,rtx4090:2) — --gpu/--gpus 대신")
    q.add_argument("--link", default=None, choices=["nvlink", "pcie", "pcie-nop2p", "network"],
                   help="GPU 간 연결 (기본: GPU 종류로 추정)")
    q.add_argument("--seq", type=int, default=2048)
    q.add_argument("--global-batch", type=int, default=16)
    q.add_argument("--lora-rank", type=int, default=16)
    q.add_argument("--lora-targets", default="attn", choices=["attn", "all"])
    q.add_argument("--no-offload", action="store_true")
    q.add_argument("--cpu-ram", type=float, default=None, help="노드 CPU RAM(GB): offload 가능 여부 판단")
    q.add_argument("--safety", type=float, default=None)
    q.add_argument("--data-tokens", type=float, default=None, help="학습 데이터 총 토큰 수 (시간 예측)")
    q.add_argument("--epochs", type=float, default=1.0)
    q.add_argument("--top", type=int, default=5)
    q.add_argument("--out", default=None, help="학습 번들을 만들 폴더")
    q.add_argument("--pick", type=int, default=1, help="번들로 만들 추천 순위")
    q.add_argument("--naive", action="store_true",
                   help="추천 대신 HF Trainer 기본값 설정으로 번들 생성 (OOM 비교 실험용)")
    q.add_argument("--dataset", default="data.jsonl")
    q.add_argument("--text-field", default="text")
    q.add_argument("--dataset-split", default="train", help='HF 데이터셋 split (예: "train[:2000]")')
    q.add_argument("--calib", default=None)
    q.add_argument("--no-calib", action="store_true")
    c = sub.add_parser("measure-cpu", help="처음 보는 모델의 메모리를 GPU 없이 CPU에서 직접 측정")
    c.add_argument("--model", required=True, help="HF 이름(org/model)·config.json 경로·모델 폴더")
    c.add_argument("--method", default="lora", choices=["qlora", "lora", "full"])
    c.add_argument("--batch", type=int, default=1)
    c.add_argument("--seq", type=int, default=2048)
    c.add_argument("--ckpt", action="store_true", help="gradient checkpointing 켬")
    c.add_argument("--lora-rank", type=int, default=16)
    c.add_argument("--gpu-mem", type=float, default=None, help="GPU 메모리(GB)를 주면 들어가는지 판정")
    c.add_argument("--safety", type=float, default=None)
    c.add_argument("--calib", default=None)
    c.add_argument("--no-calib", action="store_true")
    sub.add_parser("gpus", help="GPU 프리셋 목록")
    sub.add_parser("models", help="모델 프리셋 목록")
    a = p.parse_args(argv)
    if a.cmd == "plan":
        return cmd_plan(a)
    if a.cmd == "measure-cpu":
        return cmd_measure_cpu(a)
    if a.cmd == "gpus":
        for g in GPUS.values():
            print(f"{g.key:<9} {g.mem_gb:6.1f}GB  {g.tflops_16bit:6.0f} TFLOPS  bf16={'O' if g.bf16 else 'X'}  {g.label}")
    if a.cmd == "models":
        for k, m in MODELS.items():
            print(f"{k:<14} {m.n_params/1e9:5.2f}B  L={m.n_layers} H={m.hidden} V={m.vocab:,}  {m.name}")
        print("\n그 밖의 모델은 HF 이름(org/model)이나 config.json 경로를 --model에 넣으면 설계도만 읽어 예측합니다.")
        print("  · Llama형(Qwen3·Phi-3·OLMo-2·Granite 등): 그대로 예측 (처음 보는 8개 계열 실측 오차 2~6%)")
        print("  · Gemma 계열·검증 안 된 구조: 여유를 더 두고 예측 + `fitcfg measure-cpu`로 직접 확인 권장")
        print("  · MoE(전문가 혼합): 지원 안 함")
    return 0


if __name__ == "__main__":
    sys.exit(main())
