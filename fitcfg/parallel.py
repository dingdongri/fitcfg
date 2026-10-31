"""3D 병렬화(TP × PP × DP/FSDP) 판단 — GPU 여러 장일 때 '어떻게 나눌지'와 '3D가 필요한지'.

    from fitcfg.parallel import decide, gpu_from_spec
    d = decide(get_model("llama-3.1-8b"), "lora", [gpu_from_spec(get_gpu("rtx3090"))]*2 + [gpu_from_spec(get_gpu("rtx4090"))]*2,
               link="pcie-nop2p", seq_len=2048, global_batch=32)
    print(d.verdict, d.reason, d.best.label)

방법 (scripts/sim_3d.py 연구 결과를 패키지로 옮긴 것):
  축 열거(tp, pp, dp, mbs, ckpt, FSDP 여부) → GPU 배치(TP 묶음은 같은 종류, 느린/빠른 순 두 가지)
  → 동적 계획법으로 층 나누기(가장 느린 stage 최소, 메모리 제약) → 1F1B 시간 + 기울기 동기화 + optimizer update
검증 (gfarm A100, 실제 GPU rank 0 측정 72개):
  - GPU 1장 몫 메모리 식: Full 3% 안쪽, LoRA는 과대(안전한 쪽, 최대 +25%) · 메모리 부족 판정 72/72
  - 계산 시간: 실측 64개로 효율을 다시 맞춤 (MEASURED, 오차 11.5%)
  - 통신 시간은 실측 못 함(MIG) → 겹침 낙관/비관 두 가정을 모두 계산해 결론이 갈리면 그렇다고 알림
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field

from .models import ModelSpec
from .memory import DEFAULT_BY_METHOD, adapter_params, TrainConfig

GiB = 1024 ** 3
SEQ = 2048
GLOBAL_BATCH = 32            # 한 optimizer step의 시퀀스 수
LAT = 10e-6                  # 통신 1회 지연(초)
OVERHEAD_GB = 1.0            # CUDA 컨텍스트·NCCL 버퍼
SAFETY = 0.90
EFF = {"full": 0.38, "lora": 0.33, "qlora": 0.20}   # 사전값 (연구 스크립트 기본). decide()는 MEASURED를 씀
PRIOR = {"EFF": dict(EFF), "T_HALF_BASE": 1024}

@dataclass(frozen=True)
class GPU:
    name: str
    mem_gb: float      # GiB
    tflops: float      # 16bit dense
    hbm: float         # GB/s (optimizer update용)


RTX3090 = GPU("3090", 23.69, 71.0, 936)
RTX4090 = GPU("4090", 23.64, 165.0, 1008)
A100_40 = GPU("A100-40", 39.39, 312.0, 1555)
A100_80 = GPU("A100-80", 79.15, 312.0, 2039)




# ---------------------------------------------------------------------------
# 모델 한 층·임베딩의 크기/연산량
# ---------------------------------------------------------------------------
def layer_params(m: ModelSpec) -> float:
    return m.linear_params / m.n_layers


def weight_bytes(method: str) -> tuple[float, float]:
    """(층 가중치 B/param, 임베딩·lm_head B/param) — 가중치만."""
    return {"full": (2.0, 2.0), "lora": (2.0, 2.0), "qlora": (0.5 + 0.016, 4.0)}[method]


def static_bytes(method: str) -> tuple[float, float]:
    """학습 중 상주하는 바이트/param (가중치+기울기+옵티마이저). Full = bf16 w·g + fp32 master·m·v."""
    return {"full": (16.0, 16.0), "lora": (2.0, 2.0), "qlora": (0.516, 4.0)}[method]


def layer_flops_per_token(m: ModelSpec, method: str, ckpt: bool) -> float:
    H, KV, I = m.hidden, m.kv_dim, m.intermediate
    lin = 2 * (2 * H * H + 2 * H * KV + 3 * H * I)
    attn = 2 * SEQ * H
    fwd = lin + attn
    tot = 3 * fwd if method == "full" else fwd + lin + 2 * attn
    return tot + (fwd if ckpt else 0)


def head_flops_per_token(m: ModelSpec, method: str) -> float:
    f = 2 * m.vocab * m.hidden
    return 3 * f if method == "full" else 2 * f


# ---------------------------------------------------------------------------
# 한 stage(=연속 층 묶음)의 시간과 메모리
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    m: ModelSpec
    method: str
    tp: int
    pp: int
    dp: int
    mbs: int
    ckpt: bool
    fsdp: bool
    bw: float
    M: int            # replica당 micro-batch 수


T_HALF_BASE = 1024          # util의 절반 지점 토큰 수 (hidden 4096 기준)


def util(tokens: float, hidden_eff: float) -> float:
    t_half = T_HALF_BASE * (4096 / hidden_eff)
    return tokens / (tokens + t_half)


# 실제 GPU(gfarm A100 MIG) rank 0 측정 64개로 맞춘 계산 시간 상수 (scripts/fit_sim_compute.py, 2026-10-21)
#   통신 없는 forward+backward 기준 64개 MAPE 11.5%. 기존 사전값은 계산을 1.8~2.5배 느리게 잡았음(통신 비중 과소평가).
#   A100에서 잰 효율을 다른 GPU(4090·3090)에도 쓰는 것은 가정.
MEASURED = {"EFF": {"full": 0.74, "lora": 0.68, "qlora": 0.55}, "T_HALF_BASE": 512}   # qlora = lora × 0.80 (단일 GPU 실측 비율)


def stage_time(c: Ctx, n_layers: int, gpu: GPU, first: bool, last: bool, overlap: bool) -> tuple[float, float, float]:
    """micro-batch 1개가 이 stage를 지나는 시간 (fwd+bwd 합), 그중 bwd 계산, FSDP 노출 통신."""
    tok = c.mbs * SEQ
    speed = gpu.tflops * 1e12 * EFF[c.method] * util(tok, c.m.hidden / c.tp)
    comp = n_layers * layer_flops_per_token(c.m, c.method, c.ckpt) * tok / c.tp / speed
    if last:
        comp += head_flops_per_token(c.m, c.method) * tok / c.tp / speed
    bwd = comp * (2 / 3 if c.method == "full" else 0.5)
    # TP: 층마다 fwd 2번 + bwd 2번 (+checkpointing 재계산 2번) all-reduce — 노출
    t_tp = 0.0
    if c.tp > 1:
        n_ar = 4 + (2 if c.ckpt else 0)
        nbytes = tok * c.m.hidden * 2
        t_tp = n_layers * n_ar * (2 * (c.tp - 1) / c.tp * nbytes / (c.bw * 1e9) + 2 * (c.tp - 1) * LAT)
    # PP: 다음 stage로 활성값 넘기기 (forward, 노출)
    t_p2p = 0.0 if last or c.pp == 1 else tok * c.m.hidden * 2 / (c.bw * 1e9) + LAT
    # FSDP: micro-batch마다 이 stage 가중치를 fwd·bwd에 한 번씩 all-gather
    t_fsdp = 0.0
    if c.fsdp and c.dp > 1:
        wb_layer, wb_emb = weight_bytes(c.method)
        p = n_layers * layer_params(c.m) * wb_layer / c.tp
        if first or last:
            p += c.m.vocab * c.m.hidden * wb_emb / c.tp
        ag = (c.dp - 1) / c.dp * p / (c.bw * 1e9)
        fwd_c = comp - bwd
        t_fsdp = (max(0.0, ag - fwd_c) + max(0.0, ag - bwd)) if overlap else 2 * ag
    return comp + t_tp + t_p2p + t_fsdp, bwd, t_fsdp


# FSDP에서 임베딩·lm_head를 어떻게 감싸는가 (scripts/trace_parallel.py 검증으로 추가)
#   "root"  : 층만 따로 감싸고 임베딩·lm_head는 루트 단위 → 루트는 forward~backward 내내 '나눠지지 않은 채' 남음
#   "split" : 임베딩, (norm+lm_head)를 각각 따로 감쌈 → 필요한 순간에만 하나씩 모음  ← fitcfg가 생성할 방식
#   "ideal" : 반영 안 함 (검증 전 식)
FSDP_WRAP = "split"
TP_UNSPLIT = 0.5      # TP로 안 나뉘는 활성값 비율 (층 입력·norm 등) — trace_parallel로 확인
CKPT_MIN = 2.0        # checkpointing 시 층 입력 저장 최소 바이트/(토큰·hidden) = bf16 hidden 1벌


def stage_mem_parts(c: Ctx, n_layers: int, stage_id: int, first: bool, last: bool,
                    shard_vocab: bool = True, wrap: str | None = None) -> dict:
    """GPU 1장의 최대 메모리 구성 (GiB): static(가중치·기울기·옵티마이저) / gather(FSDP가 모은 가중치) / acts / logits.
    shard_vocab=False: TP에서 임베딩·lm_head를 쪼개지 않는 경우."""
    wrap = wrap or FSDP_WRAP
    k = DEFAULT_BY_METHOD[c.method]
    m, tok = c.m, c.mbs * SEQ
    tpv = c.tp if shard_vocab else 1
    sb_layer, sb_emb = static_bytes(c.method)
    wb_layer, wb_emb = weight_bytes(c.method)
    params_layer = n_layers * layer_params(m) / c.tp
    n_emb = (1 if first else 0) + (1 if last else 0)
    if m.tied_embeddings and first and last:
        n_emb = 1                                                           # 임베딩 = lm_head (한 번만)
    params_emb = (m.vocab * m.hidden / tpv) * n_emb
    static = params_layer * sb_layer + params_emb * sb_emb
    if c.method != "full":           # LoRA 어댑터: fp32 w·g + Adam = 16 B
        static += adapter_params(m, TrainConfig(model="x")) * n_layers / m.n_layers / c.tp * 16
    gather = 0.0
    if c.fsdp and c.dp > 1:
        static = static / c.dp
        gather = 2 * layer_params(m) / c.tp * wb_layer                      # 모은 층 2개(현재+미리 받기)
        gb_emb = min(wb_emb, 2.0)                                            # 모을 때는 계산 정밀도(bf16)로 모음
        if wrap == "root":
            gather += params_emb * gb_emb                                    # 루트(임베딩·lm_head)는 계속 모여 있음
        elif wrap == "split" and n_emb:                                      # 층 2개와 임베딩/lm_head는 동시에 모이지 않음
            gather = max(gather, (m.vocab * m.hidden / tpv) * gb_emb)
    elif c.dp > 1 and c.method == "full":
        static += params_layer * 2                                          # DDP 기울기 버킷
    # 활성값: TP로 나뉘는 부분(attention·MLP 내부)과 안 나뉘는 부분(층 입력·norm, 약 30%)
    full_act = tok * (k.act_h * m.hidden * (TP_UNSPLIT + (1 - TP_UNSPLIT) / c.tp) + k.act_i * m.intermediate / c.tp)
    in_flight = min(c.pp - stage_id, c.M)                                   # 1F1B: 앞 stage일수록 많이 보관
    if c.ckpt:
        acts = in_flight * n_layers * tok * max(k.ckpt_h, CKPT_MIN) * m.hidden + full_act
    else:
        acts = in_flight * n_layers * full_act
    logits = tok * m.vocab * k.logit_bytes / tpv if last else 0.0
    if c.fsdp and c.dp > 1 and c.method == "full" and n_emb and wrap != "ideal":
        # 역전파 끝: 마지막으로 줄어드는(reduce-scatter) 단위의 기울기가 bf16 + fp32(축소용) 사본으로 잠시 통째로 존재
        unit = params_emb if wrap == "root" else max(layer_params(m) / c.tp, m.vocab * m.hidden / tpv)
        spike = unit * 6
        if spike > gather + acts + logits:
            return dict(static=static / GiB, gather=spike / GiB, acts=0.0, logits=0.0)
    return dict(static=static / GiB, gather=gather / GiB, acts=acts / GiB, logits=logits / GiB)


def stage_mem(c: Ctx, n_layers: int, stage_id: int, first: bool, last: bool) -> float:
    """이 stage를 맡은 GPU 1장의 최대 메모리 (GiB)."""
    return sum(stage_mem_parts(c, n_layers, stage_id, first, last).values()) + OVERHEAD_GB


# ---------------------------------------------------------------------------
# 층 나누기 (병목 stage 시간 최소, 메모리 제약) — 인턴 과제의 linear-partition DP와 같은 아이디어
# ---------------------------------------------------------------------------
def partition(c: Ctx, stage_gpus: list[list[GPU]], overlap: bool):
    L, pp = c.m.n_layers, c.pp
    INF = float("inf")
    best = [[INF] * (L + 1) for _ in range(pp + 1)]
    cut = [[-1] * (L + 1) for _ in range(pp + 1)]
    best[0][0] = 0.0

    memo = {}

    def cost(n, s):
        if (n, s) in memo:
            return memo[(n, s)]
        memo[(n, s)] = v = _cost(n, s)
        return v

    def _cost(n, s):
        first, last = s == 0, s == pp - 1
        cap = min(g.mem_gb for g in stage_gpus[s]) * SAFETY
        if stage_mem(c, n, s, first, last) > cap:
            return INF
        return max(stage_time(c, n, g, first, last, overlap)[0] for g in stage_gpus[s])

    for p in range(1, pp + 1):
        for i in range(p, L + 1):
            if p == pp and i != L:
                continue
            for j in range(p - 1, i):
                if best[p - 1][j] == INF:
                    continue
                t = cost(i - j, p - 1)
                v = max(best[p - 1][j], t)
                if v < best[p][i]:
                    best[p][i], cut[p][i] = v, j
    if best[pp][L] == INF:
        return None
    sizes, i = [], L
    for p in range(pp, 0, -1):
        j = cut[p][i]
        sizes.append(i - j)
        i = j
    return sizes[::-1]


def step_time(c: Ctx, sizes, stage_gpus, overlap: bool) -> float:
    times, bwds, peaks = [], [], []
    for s, n in enumerate(sizes):
        first, last = s == 0, s == c.pp - 1
        r = [stage_time(c, n, g, first, last, overlap) for g in stage_gpus[s]]
        t, b, _ = max(r, key=lambda x: x[0])
        times.append(t)
        bwds.append(b)
    pipe = sum(times) + (c.M - 1) * max(times)
    # 기울기 동기화: DDP all-reduce 또는 FSDP reduce-scatter (step당 1번)
    comm = 0.0
    if c.dp > 1:
        for s, n in enumerate(sizes):
            if c.method == "full":
                gbytes = n * layer_params(c.m) / c.tp * 2
                if s in (0, c.pp - 1):
                    gbytes += c.m.vocab * c.m.hidden / c.tp * 2
            else:
                gbytes = adapter_params(c.m, TrainConfig(model="x")) * n / c.m.n_layers / c.tp * 4
            factor = (c.dp - 1) / c.dp * (1 if c.fsdp else 2)
            t = factor * gbytes / (c.bw * 1e9) + 2 * (c.dp - 1) * LAT
            comm = max(comm, max(0.0, t - bwds[s]) if overlap else t)
    # optimizer update (메모리 대역폭 한계)
    upd = 0.0
    for s, n in enumerate(sizes):
        p = (n * layer_params(c.m) if c.method == "full" else adapter_params(c.m, TrainConfig(model="x")) * n / c.m.n_layers) / c.tp
        if c.fsdp and c.dp > 1:
            p /= c.dp
        upd = max(upd, p * 16 * 3 / (min(g.hbm for g in stage_gpus[s]) * 1e9))
    return pipe + comm + upd


# ---------------------------------------------------------------------------
# GPU 배치: TP 묶음은 같은 종류, stage마다 종류를 모으는 두 순서(느린 것 먼저/빠른 것 먼저) 시도
# ---------------------------------------------------------------------------
def mappings(gpus: list[GPU], tp: int, pp: int, dp: int):
    by_type: dict[str, list[GPU]] = {}
    for g in gpus:
        by_type.setdefault(g.name, []).append(g)
    groups = []
    for name, gs in by_type.items():
        if len(gs) % tp:
            return []
        groups += [gs[0]] * (len(gs) // tp)          # TP 묶음 하나를 대표 GPU로
    outs = []
    for order in (sorted(groups, key=lambda g: g.tflops), sorted(groups, key=lambda g: -g.tflops)):
        stage_gpus = [order[s * dp:(s + 1) * dp] for s in range(pp)]
        if stage_gpus not in outs:
            outs.append(stage_gpus)
    return outs


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def search(m: ModelSpec, method: str, gpus: list[GPU], bw: float, overlap: bool):
    """모든 유효 조합 → (시간, 설명) 목록."""
    N = len(gpus)
    res = []
    for tp in [t for t in (1, 2, 4, 8) if N % t == 0]:
        for pp in divisors(N // tp):
            if pp > m.n_layers:
                continue
            dp = N // (tp * pp)
            for fsdp in ([False, True] if dp > 1 else [False]):
                for mbs in (1, 2, 4, 8):
                    if GLOBAL_BATCH % (dp * mbs):
                        continue
                    M = GLOBAL_BATCH // (dp * mbs)
                    for ckpt in (True, False):
                        c = Ctx(m, method, tp, pp, dp, mbs, ckpt, fsdp, bw, M)
                        for sg in mappings(gpus, tp, pp, dp):
                            sizes = partition(c, sg, overlap)
                            if sizes is None:
                                continue
                            t = step_time(c, sizes, sg, overlap)
                            res.append((t, dict(tp=tp, pp=pp, dp=dp, fsdp=fsdp, mbs=mbs, ckpt=ckpt,
                                                layers=sizes, stages=[s[0].name for s in sg])))
    return sorted(res, key=lambda x: x[0])


def label(d) -> str:
    if d is None:
        return "불가"
    kind = []
    if d["dp"] > 1:
        kind.append(f"{'FSDP' if d['fsdp'] else 'DP'}{d['dp']}")
    if d["tp"] > 1:
        kind.append(f"TP{d['tp']}")
    if d["pp"] > 1:
        lay = "/".join(map(str, d["layers"]))
        kind.append(f"PP{d['pp']}[{lay}]")
    s = "×".join(kind) or "단일"
    return f"{s} · mbs{d['mbs']} · ckpt {'on' if d['ckpt'] else 'off'}"


# ===========================================================================
# 공개 API: 3D 병렬화 판단
# ===========================================================================
# GPU 사이 통신 실효 대역폭 (GB/s, all-reduce 기준) — 공개 측정값 기반 가정 (실측 전)
LINKS = {
    "nvlink": 230.0,       # A100·H100 SXM (NVLink)
    "pcie": 25.0,          # 서버용 GPU, PCIe Gen4 GPU 간 직접 통신(P2P) 가능
    "pcie-nop2p": 15.0,    # 게임용 GPU(RTX 3090·4090): GPU 간 직접 통신 막힘 → CPU 메모리 경유
    "network": 5.0,        # GPU가 다른 컴퓨터에 있음 (일반 이더넷/저속 네트워크)
}
LINK_LABEL = {"nvlink": "NVLink", "pcie": "PCIe (GPU 간 직접 통신)", "pcie-nop2p": "PCIe (CPU 경유)",
              "network": "서버 간 네트워크"}
# GPU 메모리 대역폭 (GB/s) — optimizer update 시간용
_HBM = {"t4": 300, "p100": 732, "l4": 300, "a10g": 600, "rtx3090": 936, "rtx4090": 1008,
        "a100-40": 1555, "a100-80": 2039, "h100": 3350}

GAIN_RECOMMEND = 1.10     # 3D가 DP·FSDP보다 이만큼 빨라야 추천 (복잡도·구현 위험 대비)
GAIN_NEGLIGIBLE = 1.05    # 이보다 작으면 '차이 없음'


def gpu_from_spec(spec) -> GPU:
    """fitcfg.hardware.GPUSpec → 3D 계산용 GPU."""
    hbm = _HBM.get(spec.key)
    if hbm is None:
        parent = spec.key.split("-mig")[0]
        hbm = _HBM.get(parent, 1000) * (3 / 7 if "-mig" in spec.key else 1)
    return GPU(spec.key, spec.mem_gb, spec.tflops_16bit, hbm)


def default_link(specs) -> str:
    """GPU 종류로 연결 방식 추정 (사용자가 모르면). 게임용 RTX → CPU 경유 PCIe, A100·H100 → NVLink."""
    keys = {s.key for s in specs}
    if any(k.startswith("rtx") for k in keys):
        return "pcie-nop2p"
    if keys and all(k.split("-mig")[0] in ("a100-40", "a100-80", "h100") for k in keys) and len(keys) == 1:
        return "nvlink"
    return "pcie"


@dataclass
class ParallelPlan:
    time_s: float                 # optimizer step 1회 시간 (예측)
    tokens_per_s: float
    tp: int
    pp: int
    dp: int
    fsdp: bool
    mbs: int
    ckpt: bool
    layers: list[int]             # stage별 층 수
    stages: list[str]             # stage별 GPU 종류
    mem_gb: list[float] = field(default_factory=list)   # stage별 GPU 1장 예측 메모리

    @property
    def is_3d(self) -> bool:
        return self.tp > 1 or self.pp > 1

    @property
    def label(self) -> str:
        return label(dict(tp=self.tp, pp=self.pp, dp=self.dp, fsdp=self.fsdp, mbs=self.mbs,
                          ckpt=self.ckpt, layers=self.layers))


@dataclass
class ParallelDecision:
    verdict: str                  # "3d" (3D 추천) | "dp" (DP·FSDP로 충분) | "none" (어떤 조합도 안 들어감)
    reason: str
    best: ParallelPlan | None             # 전체 최선 (통신 겹침 낙관 가정)
    best_dp: ParallelPlan | None          # DP·FSDP만 쓴 최선 (낙관)
    gain_opt: float | None                # best_dp 시간 ÷ best 시간 (낙관)
    gain_pess: float | None               # 같은 비교, 통신이 안 겹친다는 가정
    link: str
    n_gpus: int
    mixed: bool
    notes: list[str] = field(default_factory=list)
    top: list[ParallelPlan] = field(default_factory=list)


@contextmanager
def _settings(seq: int, global_batch: int, safety: float, measured: bool = True):
    g = globals()
    saved = {k: g[k] for k in ("SEQ", "GLOBAL_BATCH", "SAFETY", "T_HALF_BASE")}
    saved_eff = dict(EFF)
    try:
        g.update(SEQ=seq, GLOBAL_BATCH=global_batch, SAFETY=safety)
        if measured:
            EFF.update(MEASURED["EFF"])
            g["T_HALF_BASE"] = MEASURED["T_HALF_BASE"]
        yield
    finally:
        g.update(saved)
        EFF.clear()
        EFF.update(saved_eff)


def _to_plan(m: ModelSpec, method: str, r, bw: float) -> ParallelPlan:
    t, d = r
    c = Ctx(m, method, d["tp"], d["pp"], d["dp"], d["mbs"], d["ckpt"], d["fsdp"], bw, GLOBAL_BATCH // (d["dp"] * d["mbs"]))
    mem = [round(stage_mem(c, n, s, s == 0, s == d["pp"] - 1), 2) for s, n in enumerate(d["layers"])]
    return ParallelPlan(time_s=t, tokens_per_s=GLOBAL_BATCH * SEQ / t, tp=d["tp"], pp=d["pp"], dp=d["dp"],
                        fsdp=d["fsdp"], mbs=d["mbs"], ckpt=d["ckpt"], layers=list(d["layers"]),
                        stages=list(d["stages"]), mem_gb=mem)


def decide(m: ModelSpec, method: str, gpus: list[GPU], link: str | float = "pcie", seq_len: int = 2048,
           global_batch: int = 32, safety: float = 0.90, allow_tp: bool | None = None,
           top_k: int = 5) -> ParallelDecision:
    """GPU 여러 장에서 3D 병렬화가 필요한가. 필요한 두 경우만 3D를 추천한다:
    (1) DP·FSDP로는 메모리가 모자랄 때  (2) GPU 종류가 섞여 층을 나눠 주는 게 확실히 빠를 때(+통신이 느린 경우)."""
    bw = LINKS[link] if isinstance(link, str) else float(link)
    link_name = link if isinstance(link, str) else f"{bw:g}GB/s"
    if allow_tp is None:
        allow_tp = method != "qlora"            # 4bit(bitsandbytes) 가중치는 TP 분할을 지원하지 않음
    mixed = len({g.name for g in gpus}) > 1
    n = len(gpus)
    with _settings(seq_len, global_batch, safety):
        res = {}
        for ov in (True, False):
            r = [x for x in search(m, method, gpus, bw, ov) if allow_tp or x[1]["tp"] == 1]
            res[ov] = r
        best = _to_plan(m, method, res[True][0], bw) if res[True] else None
        dp_opt = [x for x in res[True] if x[1]["tp"] == 1 and x[1]["pp"] == 1]
        dp_pes = [x for x in res[False] if x[1]["tp"] == 1 and x[1]["pp"] == 1]
        best_dp = _to_plan(m, method, dp_opt[0], bw) if dp_opt else None
        top = [_to_plan(m, method, x, bw) for x in res[True][:top_k]]
        g_opt = dp_opt[0][0] / res[True][0][0] if (dp_opt and res[True]) else None
        g_pes = dp_pes[0][0] / res[False][0][0] if (dp_pes and res[False]) else None

    notes = [f"GPU 간 연결: {LINK_LABEL.get(link_name, link_name)} (실효 {bw:g}GB/s 가정 — 통신 시간은 실측 전)"]
    if not allow_tp:
        notes.append("QLoRA(4bit)는 TP 분할을 지원하지 않아 TP 후보는 제외")
    if best is None:
        verdict, reason = "none", (f"GPU {n}장으로도 어떤 나누기 조합도 메모리에 들어가지 않습니다 "
                                   f"(GPU당 허용 {min(g.mem_gb for g in gpus) * safety:.1f}GB).")
    elif best_dp is None:
        verdict = "3d"
        reason = (f"DP·FSDP로는 GPU 1장에 메모리가 모자랍니다 → 모델을 나눠야 합니다: {best.label}. "
                  f"(3D가 필요한 경우 ①: 메모리 부족)")
    else:
        go, gp = g_opt or 1.0, g_pes or 1.0
        alt3d = next((p for p in top if p.is_3d), None) if not best.is_3d else best
        if go >= GAIN_RECOMMEND and gp >= GAIN_RECOMMEND:
            verdict = "3d"
            why = ("GPU 종류가 섞여 있어 빠른 GPU에 층을 더 주면 느린 GPU를 기다리는 시간이 줄어듭니다"
                   if mixed and best.pp > 1 else
                   "GPU 간 연결이 느려 DP·FSDP의 가중치·기울기 통신이 병목입니다")
            reason = (f"3D 추천: {best.label} — DP·FSDP 최선보다 {go:.2f}~{gp:.2f}배 빠름. {why}. "
                      f"(3D가 필요한 경우 ②)")
        elif gp >= GAIN_RECOMMEND:
            verdict = "dp"
            alt = f"3D({alt3d.label})" if alt3d else "3D"
            reason = (f"기본은 DP·FSDP({best_dp.label}). 단, 통신과 계산이 겹치지 않는 환경이면 {alt}가 "
                      f"{gp:.2f}배 빠를 수 있습니다 — 통신 가정에 따라 결론이 갈리는 경우입니다.")
            notes.append("결론이 통신 가정에 따라 갈림 → 실제 환경에서 두 설정을 짧게 돌려 비교 권장")
        elif max(go, gp) < GAIN_NEGLIGIBLE:
            verdict = "dp"
            reason = (f"DP·FSDP로 충분합니다 ({best_dp.label}). 3D로 나눠도 이득이 "
                      f"{max(go, gp):.2f}배 이하라 통신·구현 부담만 늘어납니다.")
        else:
            verdict = "dp"
            reason = f"DP·FSDP로 충분합니다 ({best_dp.label}). 3D는 {go:.2f}~{gp:.2f}배로 이득이 작습니다."
    if mixed and best_dp is not None:
        notes.append("GPU 종류가 섞인 경우 DP·FSDP는 가장 느린 GPU 속도에 맞춰집니다")
    return ParallelDecision(verdict=verdict, reason=reason, best=best, best_dp=best_dp, gain_opt=g_opt,
                            gain_pess=g_pes, link=link_name, n_gpus=n, mixed=mixed, notes=notes, top=top)
