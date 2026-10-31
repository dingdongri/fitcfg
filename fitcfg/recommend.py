"""설정 탐색 & 추천 (Phase 4, v1).

    from fitcfg import plan, PlanRequest, get_gpu
    result = plan(PlanRequest(model="llama-3.1-8b", gpu=get_gpu("rtx4090"), method="qlora"))
    best = result.best            # 가장 빠른(예측 처리량 최대) 설정
    result.naive                  # 'HF Trainer 기본값' 으로 돌렸다면? (OOM 회피 비교용)

탐색 공간 (완전탐색, 수백 개):
    micro-batch × gradient checkpointing × optimizer(8bit) × 분산 전략(DDP/ZeRO/offload)
    gradient accumulation 은 '목표 global batch'를 맞추도록 자동 계산 → 학습 결과(수렴)는 동일,
    메모리·속도만 달라지는 설정들 중에서 고른다. ← 이게 추천의 핵심 전제.

전략별 제약 (실제 스택에서 흔히 되는 조합만):
    - QLoRA : 단일 GPU / DDP (4bit 가중치는 ZeRO-3 샤딩이 까다로워 스코프 밖)
    - LoRA  : 단일 GPU / DDP / ZeRO-3 (bf16 베이스 가중치 샤딩)
    - Full  : 단일 GPU / ZeRO-Offload(1GPU) / DDP / ZeRO-1,2,3 / ZeRO-2+offload / ZeRO-3+offload
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field, replace
from itertools import product

from .hardware import GPUSpec, custom_gpu
from .memory import (TrainConfig, MemoryBreakdown, Method, estimate, get_default_safety, GB,
                     cpu_ram_needed_gb)
from .models import ModelSpec, get_model
from .throughput import ThroughputEstimate, estimate_throughput


# ---------------------------------------------------------------------------
# 입력 / 출력
# ---------------------------------------------------------------------------
@dataclass
class PlanRequest:
    model: str
    gpu: GPUSpec
    gpus: int = 1
    method: Method = "qlora"
    seq_len: int = 2048
    global_batch: int = 16            # optimizer step 1회당 시퀀스 수 (모든 GPU·누적 합)
    lora_rank: int = 16
    lora_targets: str = "attn"
    allow_offload: bool = True
    allow_8bit_optim: bool = True
    safety: float | None = None       # None → 보정 권장값 또는 0.90
    cpu_ram_gb: float | None = None   # 노드 CPU RAM. 주어지면 offload 설정의 RAM 요구량으로 거름
    dataset_tokens: float | None = None   # 학습 데이터 총 토큰 (시간 예측용)
    epochs: float = 1.0
    # GPU 종류가 섞인 경우 전체 목록 (예: [3090, 3090, 4090, 4090]). 주면 gpu·gpus는 이 목록으로 정해진다
    # (데이터 병렬 계열 추천은 가장 작은 GPU 기준 — 모든 GPU가 같은 몫을 들기 때문)
    gpu_list: list[GPUSpec] | None = None
    link: str | None = None           # GPU 간 연결: nvlink / pcie / pcie-nop2p / network (None=GPU 종류로 추정)
    check_3d: bool = True             # GPU 2장 이상이면 3D 병렬화(TP·PP) 필요 여부 판단

    def __post_init__(self):
        if self.gpu_list:
            self.gpus = len(self.gpu_list)
            kinds = {g.key: g for g in self.gpu_list}
            if len(kinds) == 1:
                self.gpu = self.gpu_list[0]
            else:   # 데이터 병렬은 가장 느린 GPU 속도·가장 작은 메모리에 맞춰짐
                slow = min(kinds.values(), key=lambda g: g.tflops_16bit)
                counts = {k: sum(g.key == k for g in self.gpu_list) for k in kinds}
                self.gpu = replace(slow, mem_gb=min(g.mem_gb for g in kinds.values()),
                                   bf16=all(g.bf16 for g in kinds.values()),
                                   flash=all(g.flash for g in kinds.values()),
                                   label=" + ".join(f"{kinds[k].label} ×{n}" for k, n in counts.items()))


@dataclass
class Recommendation:
    config: TrainConfig
    grad_accum: int
    strategy: str                     # single / ddp / zero1 / zero2 / zero3 / zero-offload / zero3-offload
    dtype: str                        # bf16 / fp16
    memory: MemoryBreakdown
    pred_mem_gb: float
    cap_gb: float                     # GPU 메모리 × safety
    throughput: ThroughputEstimate
    hours: tuple[float, float, float] | None
    reason: str
    tags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return self.pred_mem_gb <= self.cap_gb

    @property
    def headroom_gb(self) -> float:
        return self.cap_gb - self.pred_mem_gb

    @property
    def effective_batch(self) -> int:
        return self.config.batch_size * max(self.config.gpus, 1) * self.grad_accum

    # 하위호환(v0 API)
    @property
    def score(self) -> float:
        return self.throughput.tokens_per_s


@dataclass
class PlanResult:
    request: PlanRequest
    recommendations: list[Recommendation]
    naive: Recommendation
    n_candidates: int
    n_feasible: int
    safety: float
    diagnosis: list[str] = field(default_factory=list)
    candidates: list[Recommendation] = field(default_factory=list)   # 평가한 전체 후보 (시각화용)
    notes: list[str] = field(default_factory=list)                   # 모든 추천에 공통인 안내
    parallel: "ParallelDecision | None" = None                       # GPU 2장 이상: 3D 병렬화 판단

    @property
    def best(self) -> Recommendation | None:
        return self.recommendations[0] if self.recommendations else None


# ---------------------------------------------------------------------------
# 후보 생성
# ---------------------------------------------------------------------------
STRATEGIES = {
    # 이름: (zero_stage, offload_optim, offload_params, 멀티GPU 전용?, 단일GPU 허용?)
    "single":        (0, False, False, False, True),
    "zero-offload":  (2, True, False, False, True),    # ZeRO-2 + optimizer CPU (1GPU에서도 유효)
    "zero3-offload": (3, True, True, False, True),     # ZeRO-Infinity: 파라미터까지 CPU
    "ddp":           (0, False, False, True, False),
    "zero1":         (1, False, False, True, False),
    "zero2":         (2, False, False, True, False),
    "zero3":         (3, False, False, True, False),
}
ALLOWED = {
    "qlora": {"single", "ddp"},
    "lora": {"single", "ddp", "zero3"},
    "full": set(STRATEGIES),
}
MICRO_BATCHES = [1, 2, 4, 8, 16, 32]


def _strategies(req: PlanRequest) -> list[str]:
    out = []
    for name in ALLOWED[req.method]:
        zs, off_o, off_p, multi_only, single_ok = STRATEGIES[name]
        if req.gpus == 1 and not single_ok:
            continue
        if req.gpus > 1 and name == "single":
            continue
        if (off_o or off_p) and not req.allow_offload:
            continue
        out.append(name)
    return sorted(out)


def _make_config(req: PlanRequest, strategy: str, micro: int, ckpt: bool, optim: str) -> TrainConfig:
    zs, off_o, off_p = STRATEGIES[strategy][:3]
    return TrainConfig(model=req.model, method=req.method, gpus=req.gpus, batch_size=micro,
                       seq_len=req.seq_len, grad_checkpointing=ckpt, optim=optim,
                       zero_stage=zs, cpu_offload_optim=off_o, cpu_offload_params=off_p,
                       lora_rank=req.lora_rank, lora_targets=req.lora_targets,
                       dtype="bf16" if req.gpu.bf16 else "fp16", flash_attn=req.gpu.flash)


def _grad_accum(req: PlanRequest, micro: int) -> int:
    return max(1, math.ceil(req.global_batch / (micro * req.gpus)))


def _margin(req: PlanRequest) -> float:
    from .autospec import structure_of
    try:
        return structure_of(req.model).margin(req.method)
    except Exception:
        return 1.0


def _evaluate(req: PlanRequest, spec: ModelSpec, cfg: TrainConfig, strategy: str,
              safety: float, margin: float = 1.0) -> Recommendation:
    ga = _grad_accum(req, cfg.batch_size)
    mem = estimate(cfg, spec)
    tp = estimate_throughput(cfg, spec, req.gpu, ga)
    total_tokens = req.dataset_tokens * req.epochs if req.dataset_tokens else None
    rec = Recommendation(
        config=cfg, grad_accum=ga, strategy=strategy, dtype="bf16" if req.gpu.bf16 else "fp16",
        memory=mem, pred_mem_gb=round(mem.total / GB * margin, 2), cap_gb=round(req.gpu.mem_gb * safety, 2),
        throughput=tp, hours=tp.hours_for(total_tokens) if total_tokens else None,
        reason=explain(cfg, ga, strategy),
    )
    if rec.effective_batch != req.global_batch:
        rec.warnings.append(f"실제 global batch = {rec.effective_batch} (목표 {req.global_batch}; "
                            f"micro×GPU×누적으로 정확히 나눠지지 않음)")
    ram = cpu_ram_needed_gb(cfg, spec)
    if ram:
        rec.warnings.append(f"CPU offload: 노드에 CPU RAM 약 {ram:.0f}GB 이상 필요 (부족하면 프로세스가 죽음)")
    if cfg.method == "full" and strategy in ("single", "ddp") and req.gpu.bf16:
        rec.warnings.append("순수 bf16 학습(fp32 master 없음): 메모리는 적지만 아주 작은 lr에선 "
                            "업데이트가 묻힐 수 있음. 정밀도가 중요하면 ZeRO 전략(fp32 master) 선택")
    if rec.fits and rec.headroom_gb < 0.05 * req.gpu.mem_gb:
        rec.warnings.append("메모리 여유가 5% 미만: 데이터에 긴 샘플이 섞이면 OOM 가능")
    return rec


def explain(cfg: TrainConfig, ga: int, strategy: str) -> str:
    parts = [f"micro-batch {cfg.batch_size}", f"누적 {ga}",
             "checkpointing " + ("on" if cfg.grad_checkpointing else "off")]
    if cfg.method == "full" and cfg.optim == "adamw8bit":
        parts.append("8bit AdamW")
    label = {"single": "단일 GPU", "ddp": "DDP", "zero1": "ZeRO-1", "zero2": "ZeRO-2",
             "zero3": "ZeRO-3", "zero-offload": "ZeRO-2 + optimizer CPU offload",
             "zero3-offload": "ZeRO-3 + 파라미터·optimizer CPU offload"}[strategy]
    parts.append(label)
    return ", ".join(parts)


def naive_config(req: PlanRequest) -> Recommendation:
    """도구 없이 흔히 쓰는 설정: HF Trainer 기본값(per_device_train_batch_size=8),
    gradient checkpointing 끔, 일반 AdamW, 멀티 GPU면 DDP. 평가(OOM 회피율)의 기준선."""
    spec = get_model(req.model)
    micro = min(8, max(1, req.global_batch // req.gpus))
    strategy = "single" if req.gpus == 1 else "ddp"
    cfg = _make_config(req, strategy, micro, ckpt=False, optim="adamw")
    safety = req.safety if req.safety is not None else get_default_safety()
    rec = _evaluate(req, spec, cfg, strategy, safety, _margin(req))
    rec.tags.append("기본값(naive)")
    return rec


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def plan(req: PlanRequest, top_k: int = 5) -> PlanResult:
    spec = get_model(req.model)
    safety = req.safety if req.safety is not None else get_default_safety()
    optims = ["adamw", "adamw8bit"] if (req.method == "full" and req.allow_8bit_optim) else ["adamw"]
    micros = [b for b in MICRO_BATCHES if b * req.gpus <= max(req.global_batch, req.gpus)] or [1]

    from .autospec import structure_of
    info = structure_of(req.model)
    margin = info.margin(req.method)
    cands = [_evaluate(req, spec, _make_config(req, st, mb, ck, op), st, safety, margin)
             for st, mb, ck, op in product(_strategies(req), micros, [True, False], optims)
             # 8bit AdamW(bitsandbytes)는 DeepSpeed ZeRO와 함께 쓰지 않는다 (ZeRO는 자체 optimizer)
             if not (op == "adamw8bit" and st not in ("single", "ddp"))]
    feasible = [r for r in cands if r.fits and _ram_ok(req, spec, r)]
    # 예측 처리량 높은 순. 비슷하면(2% 이내) 더 단순한 전략·더 큰 메모리 여유를 선호
    simplicity = {"single": 0, "ddp": 0, "zero1": 1, "zero2": 1, "zero3": 2,
                  "zero-offload": 3, "zero3-offload": 4}
    feasible.sort(key=lambda r: (-round(math.log(r.throughput.tokens_per_s) / math.log(1.02)),
                                 simplicity[r.strategy], r.config.optim != "adamw",
                                 r.pred_mem_gb))
    recs = _dedupe(feasible)[:top_k]
    if recs:
        recs[0].tags.append("최고 처리량")
        safest = max(recs, key=lambda r: r.headroom_gb)
        if safest is not recs[0]:
            safest.tags.append("메모리 여유 최대")

    result = PlanResult(request=req, recommendations=recs, naive=naive_config(req),
                        n_candidates=len(cands), n_feasible=len(feasible), safety=safety,
                        candidates=cands)
    if not req.gpu.bf16:
        result.notes.append(f"{req.gpu.label}은 bf16 미지원 → fp16 혼합정밀(loss scaling)로 학습합니다.")
    if not req.gpu.flash:
        result.notes.append("FlashAttention 미지원 GPU → attention 행렬(seq²)이 메모리에 남아 긴 시퀀스·checkpointing off에서 "
                            "메모리가 크게 늘어납니다 (Colab T4 실측으로 확인).")
    if req.gpu.key == "custom":
        result.notes.append("사용자 지정 GPU: 처리량은 100 TFLOPS 가정의 참고값입니다.")
    if info.kind != "verified":
        extra = f" 예측 메모리에 ×{margin:.2f} 여유 포함." if margin > 1 else ""
        result.notes.append(info.note + extra)
    if not recs:
        result.diagnosis = diagnose(req, cands, safety, spec)
    if req.gpus > 1 and req.check_3d:
        result.parallel = _decide_3d(req, spec, safety)
        if result.parallel.verdict == "none" and recs:
            result.notes.append("GPU 메모리만으로 나누는 방법(DP·FSDP·TP·PP)은 모두 부족 → 아래 추천처럼 CPU offload가 필요합니다.")
        elif result.parallel.verdict == "3d" and not recs:
            result.diagnosis.insert(0, "데이터 병렬(DDP·ZeRO)로는 안 되지만 모델을 나누면 됩니다 → " + result.parallel.reason)
    return result


def _decide_3d(req: PlanRequest, spec: ModelSpec, safety: float):
    from .parallel import decide, gpu_from_spec, default_link
    specs = req.gpu_list or [req.gpu] * req.gpus
    link = req.link or default_link(specs)
    gb = max(req.global_batch, len(specs))
    return decide(spec, req.method, [gpu_from_spec(g) for g in specs], link=link, seq_len=req.seq_len,
                  global_batch=gb, safety=safety)


def _ram_ok(req: PlanRequest, spec: ModelSpec, r: Recommendation) -> bool:
    return req.cpu_ram_gb is None or cpu_ram_needed_gb(r.config, spec) <= req.cpu_ram_gb


def _dedupe(recs: list[Recommendation]) -> list[Recommendation]:
    """예측이 사실상 같은 설정(예: 1GPU에서 ZeRO-1≈ZeRO-2)은 하나만 남긴다."""
    out, seen = [], set()
    for r in recs:
        key = (round(r.pred_mem_gb, 1), round(r.throughput.tokens_per_s, -1), r.config.batch_size,
               r.config.grad_checkpointing)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def diagnose(req: PlanRequest, cands: list[Recommendation], safety: float,
             spec: ModelSpec | None = None) -> list[str]:
    """들어가는 설정이 하나도 없을 때: 가장 가까운 설정과 해결책을 제시.
    변경이 작은 것부터 시도: (1) seq_len만 줄이기 (2) method만 가볍게 (3) 둘 다 (4) GPU 수 늘리기."""
    tips = []
    spec = spec or get_model(req.model)
    ram_blocked = [r for r in cands if r.fits]          # GPU엔 들어가지만 CPU RAM 부족으로 제외됨
    if ram_blocked:
        need = min(cpu_ram_needed_gb(r.config, spec) for r in ram_blocked)
        tips.append(f"GPU 메모리에 들어가는 설정은 CPU offload뿐인데, CPU RAM이 약 {need:.0f}GB 필요합니다 "
                    f"(입력한 RAM {req.cpu_ram_gb:g}GB).")
    else:
        closest = min(cands, key=lambda r: r.pred_mem_gb)
        tips.append(f"가장 메모리를 적게 쓰는 설정({closest.reason})도 예측 {closest.pred_mem_gb:.1f}GB로 "
                    f"허용치 {closest.cap_gb:.1f}GB(GPU {req.gpu.mem_gb:.1f}GB × {safety})를 넘습니다.")
    order = ["full", "lora", "qlora"]
    lighter = order[order.index(req.method) + 1:]
    seqs = [s for s in (4096, 2048, 1024, 512, 256) if s < req.seq_len]

    def first_fit(variants):
        for label, r in variants:
            alt = plan(r, top_k=1)
            if alt.best:
                return label, alt.best
        return None

    single_hit = False
    for label_fmt, variants in [
        ("seq_len을 **{s}** 로 줄이기", [({"s": s}, replace(req, seq_len=s)) for s in seqs]),
        ("method를 **{m}** 로 바꾸기", [({"m": m}, replace(req, method=m)) for m in lighter]),
        ("method **{m}** + seq_len **{s}**",
         [({"m": m, "s": s}, replace(req, method=m, seq_len=s)) for m in lighter for s in seqs]),
    ]:
        if "+" in label_fmt and single_hit:
            continue                      # 하나만 바꿔도 되면 조합은 제안하지 않음
        hit = first_fit(variants)
        if hit:
            single_hit = True
            kw, best = hit
            tips.append(f"→ {label_fmt.format(**kw)} → 예측 {best.pred_mem_gb:.1f}GB ({best.reason})")
    if req.gpus < 8 and req.method != "qlora":
        hit = first_fit([({"n": n}, replace(req, gpus=n, global_batch=max(req.global_batch, n)))
                         for n in (2, 4, 8) if n > req.gpus])
        if hit:
            kw, best = hit
            tips.append(f"→ 같은 GPU **{kw['n']}장** → 예측 GPU당 {best.pred_mem_gb:.1f}GB ({best.reason})")
    if len(tips) == 1:
        tips.append("→ 이 GPU로는 스코프 내 어떤 조합도 들어가지 않습니다. 더 큰 GPU가 필요합니다.")
    return tips


# ---------------------------------------------------------------------------
# v0 호환 API
# ---------------------------------------------------------------------------
def recommend(model_key: str, gpu_mem_gb: float, method: Method = "qlora",
              gpus: int = 1, seq_len: int = 2048, optim: str = "adamw",
              top_k: int = 3, safety: float = 0.90, global_batch: int = 16) -> list[Recommendation]:
    """간단 API: GPU 메모리(GB)만으로 추천 top_k."""
    req = PlanRequest(model=model_key, gpu=custom_gpu(gpu_mem_gb), gpus=gpus, method=method,
                      seq_len=seq_len, global_batch=global_batch, safety=safety,
                      allow_8bit_optim=(optim == "adamw8bit") or method == "full")
    return plan(req, top_k=top_k).recommendations
