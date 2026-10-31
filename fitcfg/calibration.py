"""cost model 보정(calibration) 로직 — Phase 3의 심장.

실측 CSV(scripts/measure.py 출력)로 memory.py 의 CostConstants 를 fit 하고,
예측 vs 실측 정확도(MAPE 등)를 계산한다. 표준 라이브러리만 사용 (GPU·numpy 불필요).

보정 방식: "사전값 쪽으로 당기는 가중 ridge 선형회귀" + peak 시점 교대 추정
    memory.py 의 peak 식은 두 시점의 max 이고, 각 시점의 식은 보정 상수에 대해 **선형**이다.
        역전파 시점 : known + act_h·[BSH] + act_i·[BSI] + ckpt_h·[BSLH] + logit_bytes·[BSV] + overhead
        step 시점   : known + grads + opt_temp·[학습 파라미터 수] + overhead
    → (1) 현재 상수로 행마다 어느 시점이 peak인지 정하고 (2) 그 시점의 선형식으로 ridge를 풀고
      (3) 시점 배정이 바뀌지 않을 때까지 반복한다. 여기에 두 가지를 더한다.
      (1) 상대오차 가중(1/y): 목표 지표가 MAPE 이므로 작은 설정의 오차도 똑같이 중요.
      (2) 사전값 방향 ridge (λ·Σ(θ/θ₀−1)²): 데이터가 적거나 두 항이 겹쳐 구분이
          안 될 때(예: 모델 1개만 측정하면 activation과 logits가 둘 다 B·S에 비례)
          값이 발산하지 않고 문헌값 근처에 머문다. 데이터가 충분하면 데이터가 이긴다.

2단계로 푼다.
    Stage A  QLoRA weights (mem_model_gb 열):  nf4_extra, qlora_nonquant_bytes
    Stage B  peak (device_peak_gb 또는 peak_alloc_gb 열), method별:
             act_h, act_i, ckpt_h, logit_bytes, opt_temp, overhead_gb

데이터 출처 두 가지를 같은 코드로 다룬다.
    - GPU 실측 (measure.py)        : 모든 상수 + overhead
    - FakeTensor 추적 (trace_sweep): 텐서 항만 (overhead 없음 → --target alloc).
      추적 행은 가중치를 추적값(mem_model_gb)으로 대체한다 (QLoRA 에뮬레이션은 가중치가 16bit라서).
"""
from __future__ import annotations
import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

from .memory import CostConstants, DEFAULT_CONSTANTS, TrainConfig, estimate, features, GB
from .models import ModelSpec, MODELS

Constants = dict[str, CostConstants]    # {"default": ..., "qlora": ..., ...}


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
def _f(x: str) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


@dataclass
class Measurement:
    cfg: TrainConfig
    spec: ModelSpec
    ok: bool
    oom: bool
    device_peak_gb: float | None
    peak_alloc_gb: float | None
    mem_model_gb: float | None
    mem_steady_gb: float | None
    gpu_total_gb: float | None
    gpu_name: str
    run_id: str
    tokens_per_s: float | None = None
    grad_accum: int = 1
    source: str = "measure"          # measure | trace | bundle

    @property
    def is_trace(self) -> bool:
        return self.gpu_name.startswith("trace")

    def target(self, kind: str) -> float | None:
        return self.device_peak_gb if kind == "device" else self.peak_alloc_gb

    @property
    def label(self) -> str:
        c = self.cfg
        return (f"{c.model}/{c.method} b{c.batch_size} s{c.seq_len} "
                f"ckpt{int(c.grad_checkpointing)} {c.optim} r{c.lora_rank}{c.lora_targets[0]}")


def _flash(gpu_name: str) -> bool:
    """측정 행의 GPU가 FlashAttention을 쓸 수 있었는지. 추적(trace)·모르는 GPU는 True."""
    from .hardware import match_gpu
    g = match_gpu(gpu_name)
    return True if g is None else g.flash


def load_runs(path: str | Path) -> tuple[list[Measurement], list[str]]:
    """scripts/run_bundle.py 가 남긴 bundle_runs.jsonl → Measurement 목록 (실제 학습 번들 실행 결과)."""
    import json
    out, warns = [], []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r.get("model_key") or "").lower()
        if key not in MODELS:
            warns.append(f"{path}:{i}: 알 수 없는 모델 '{key}' → 건너뜀")
            continue
        cfg = TrainConfig(
            model=key, method=r["method"], gpus=int(r.get("gpus", 1)), batch_size=int(r["micro_batch"]),
            seq_len=int(r["seq_len"]), grad_checkpointing=bool(r["gradient_checkpointing"]),
            optim=r.get("optim_label", "adamw"), zero_stage=int(r.get("zero_stage", 0)),
            cpu_offload_optim=bool(r.get("cpu_offload_optim")), cpu_offload_params=bool(r.get("cpu_offload_params")),
            lora_rank=int(r.get("lora_rank", 16)), lora_targets=r.get("lora_targets", "attn"),
            dtype=r.get("dtype", "bf16"), flash_attn=_flash(r.get("gpu_name") or ""),
        )
        out.append(Measurement(
            cfg=cfg, spec=MODELS[key], ok=bool(r.get("ok")), oom=(r.get("note") or "").upper().startswith("OOM"),
            device_peak_gb=r.get("device_peak_gb"), peak_alloc_gb=r.get("peak_alloc_gb"),
            mem_model_gb=None, mem_steady_gb=None, gpu_total_gb=r.get("gpu_total_gb"),
            gpu_name=r.get("gpu_name") or "", run_id=f"{r.get('bundle', 'bundle')}#{i}",
            tokens_per_s=r.get("tokens_per_s"), grad_accum=int(r.get("grad_accum", 1)), source="bundle",
        ))
    return out, warns


def load_measurements(path: str | Path) -> tuple[list[Measurement], list[str]]:
    """CSV → Measurement 목록. 두 번째 반환값은 건너뛴 행에 대한 경고."""
    out, warns = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            key = (r.get("model") or "").lower()
            if key not in MODELS:
                warns.append(f"line {i}: 알 수 없는 모델 '{key}' → 건너뜀")
                continue
            cfg = TrainConfig(
                model=key, method=r["method"], gpus=int(r.get("gpus") or 1),
                batch_size=int(r["batch"]), seq_len=int(r["seq"]),
                grad_checkpointing=bool(int(r["ckpt"])), optim=r.get("optim") or "adamw",
                zero_stage=int(r.get("zero") or 0), lora_rank=int(r.get("lora_rank") or 16),
                lora_targets=r.get("lora_targets") or "attn",
                dtype="fp16" if (r.get("dtype") or "").startswith("fp16") else "bf16",
                flash_attn=_flash(r.get("gpu_name") or ""),
            )
            note = (r.get("note") or "").upper()
            out.append(Measurement(
                cfg=cfg, spec=MODELS[key], ok=r.get("ok") == "1", oom=note.startswith("OOM"),
                device_peak_gb=_f(r.get("device_peak_gb")), peak_alloc_gb=_f(r.get("peak_alloc_gb")),
                mem_model_gb=_f(r.get("mem_model_gb")), mem_steady_gb=_f(r.get("mem_steady_gb")),
                gpu_total_gb=_f(r.get("gpu_total_gb")), gpu_name=r.get("gpu_name") or "",
                run_id=r.get("run_id") or f"line{i}",
                tokens_per_s=_f(r.get("tokens_per_s")),
                source="trace" if (r.get("gpu_name") or "").startswith("trace") else "measure",
            ))
    return out, warns


# ---------------------------------------------------------------------------
# 작은 선형대수 (numpy 없이)
# ---------------------------------------------------------------------------
def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    """가우스 소거 (부분 피벗). A는 작고(≤4×4) 양의 정부호(ridge 덕분)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        if abs(p) < 1e-18:
            raise ValueError("singular system")
        for r in range(col + 1, n):
            k = M[r][col] / p
            for c in range(col, n + 1):
                M[r][c] -= k * M[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][c] * x[c] for c in range(r + 1, n))) / M[r][r]
    return x


def ridge_to_prior(X: list[list[float]], y: list[float], w: list[float],
                   prior: list[float], lam: float) -> list[float]:
    """min Σ wᵢ²(yᵢ − Xᵢθ)² + λ Σ (θⱼ/sⱼ − θ₀ⱼ/sⱼ)²,  sⱼ = |θ₀ⱼ| (0이면 1).

    열을 사전값으로 스케일해서 푼다 → λ가 '사전값 대비 몇 %' 단위로 해석된다.
    데이터가 전혀 없는 열(전부 0)은 사전값이 그대로 나온다."""
    k = len(prior)
    s = [abs(p) if p != 0 else 1.0 for p in prior]
    z0 = [p / sj for p, sj in zip(prior, s)]
    A = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for xi, yi, wi in zip(X, y, w):
        xs = [xi[j] * s[j] for j in range(k)]
        w2 = wi * wi
        for a in range(k):
            b[a] += w2 * xs[a] * yi
            for c in range(k):
                A[a][c] += w2 * xs[a] * xs[c]
    for j in range(k):
        A[j][j] += lam
        b[j] += lam * z0[j]
    z = _solve(A, b)
    return [zj * sj for zj, sj in zip(z, s)]


# ---------------------------------------------------------------------------
# 보정
# ---------------------------------------------------------------------------
def prior_constants(target: str) -> CostConstants:
    """보정의 출발점. alloc 타깃이면 CUDA 컨텍스트가 빠지므로 overhead 사전값을 0으로."""
    return replace(DEFAULT_CONSTANTS, overhead_gb=0.0) if target == "alloc" else DEFAULT_CONSTANTS


def fit_weights(meas: list[Measurement], prior: CostConstants, lam: float) -> CostConstants:
    """Stage A: QLoRA 로드 직후 메모리로 nf4_extra, qlora_nonquant_bytes 보정."""
    from .memory import adapter_params, BYTES
    # 추적(trace) 행은 QLoRA 가중치를 16bit로 에뮬레이션하므로 가중치 보정에 쓰지 않는다
    rows = [m for m in meas if m.ok and m.cfg.method == "qlora" and m.mem_model_gb and not m.is_trace]
    if not rows:
        return prior
    X, y, w = [], [], []
    for m in rows:
        lin, emb = m.spec.linear_params, m.spec.embed_params
        fixed = lin * BYTES["nf4"] + adapter_params(m.spec, m.cfg) * 4.0
        X.append([lin / GB, emb / GB])
        y.append(m.mem_model_gb - fixed / GB)
        w.append(1.0 / m.mem_model_gb)
    nf4_extra, nonq = ridge_to_prior(X, y, w, [prior.nf4_extra, prior.qlora_nonquant_bytes], lam)
    return replace(prior, nf4_extra=max(nf4_extra, 0.0), qlora_nonquant_bytes=max(nonq, 0.5))


PEAK_FIELDS = ["act_h", "act_i", "ckpt_h", "logit_bytes", "opt_temp", "attn_sq_bytes", "overhead_gb"]


def _weights_fix(m: Measurement, f) -> float:
    """추적 행은 가중치를 추적값으로 대체 (bytes 차이). 실측 행은 0."""
    if m.is_trace and m.mem_model_gb is not None:
        return m.mem_model_gb * GB - f.known_weights
    return 0.0


def _row_design(m: Measurement, f, phase: str) -> tuple[list[float], float]:
    """(기저 벡터[GB], 상수와 무관한 부분[GB]) — PEAK_FIELDS 순서."""
    fix = _weights_fix(m, f)
    base = (f.known_weights + f.known_optimizer + f.known_persistent + fix) / GB
    if phase == "fwd_bwd":
        x = [f.act_h_basis / GB, f.act_i_basis / GB, f.ckpt_basis / GB, f.logit_basis / GB, 0.0,
             f.attn_sq_basis / GB, 1.0]
    else:
        x = [0.0, 0.0, 0.0, 0.0, f.temp_basis / GB, 0.0, 1.0]
        base += f.known_gradients / GB
    return x, base


def fit_peak(meas: list[Measurement], base: CostConstants, target: str,
             lam: float, iters: int = 10, fixed: dict | None = None) -> CostConstants:
    """Stage B: peak 메모리로 활성값 / logits / optimizer 임시 / overhead 상수 보정.
    fixed: {상수명: 값} — 보정하지 않고 고정할 상수 (절제 실험용)."""
    fixed = fixed or {}
    base = replace(base, **fixed)
    free = [i for i, k in enumerate(PEAK_FIELDS) if k not in fixed]
    rows = [m for m in meas if m.ok and m.target(target)]
    if not rows:
        return base
    c = base
    prior = [getattr(base, PEAK_FIELDS[i]) for i in free]
    feats = [features(m.cfg, m.spec, base) for m in rows]
    phases = None
    for _ in range(iters):
        new_phases = ["fwd_bwd" if f.fwd_bwd(c) >= f.optim_step(c) else "optim_step" for f in feats]
        if new_phases == phases:
            break
        phases = new_phases
        X, y, w = [], [], []
        for m, f, ph in zip(rows, feats, phases):
            x, known = _row_design(m, f, ph)
            known += sum(x[i] * fixed[k] for i, k in enumerate(PEAK_FIELDS) if k in fixed)
            yt = m.target(target)
            X.append([x[i] for i in free])
            y.append(yt - known)
            w.append(1.0 / yt)
        theta = ridge_to_prior(X, y, w, prior, lam)
        c = replace(base, **{PEAK_FIELDS[i]: max(v, 0.0) for i, v in zip(free, theta)})
    return c


def fit(meas: list[Measurement], target: str = "device", lam: float = 0.01,
        group_by_method: bool = True, min_group: int = 6, fixed: dict | None = None,
        use_method_priors: bool = True) -> Constants:
    """전체 보정. 반환: {"default": 공통, "<method>": method별(데이터 충분할 때만)}.

    사전값: 공통은 DEFAULT_CONSTANTS, method별은 DEFAULT_BY_METHOD(FakeTensor 추적 기반, 있으면).
    overhead는 장치 수준 값이라 공통 보정값을 method별 보정의 출발점으로 쓴다."""
    from .memory import DEFAULT_BY_METHOD
    prior = prior_constants(target)
    wconst = fit_weights(meas, prior, lam)
    out: Constants = {"default": fit_peak(meas, wconst, target, lam, fixed=fixed)}
    if group_by_method:
        for method in sorted({m.cfg.method for m in meas}):
            grp = [m for m in meas if m.cfg.method == method and m.ok and m.target(target)]
            if len(grp) < min_group:
                continue
            start = out["default"]
            mp = DEFAULT_BY_METHOD.get(method) if use_method_priors else None
            if mp is not None:
                keep = {k: getattr(mp, k) for k in PEAK_FIELDS if k != "overhead_gb"}
                start = replace(start, **keep)
            out[method] = fit_peak(grp, start, target, lam, fixed=fixed)
    return out


def predict_gb(m: Measurement, consts: Constants) -> float:
    c = consts.get(m.cfg.method) or consts["default"]
    if m.is_trace:
        f = features(m.cfg, m.spec, c)
        fix = _weights_fix(m, f)
        return (max(f.fwd_bwd(c), f.optim_step(c)) + fix) / GB + c.overhead_gb
    return estimate(m.cfg, m.spec, c).total / GB


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
@dataclass
class Metrics:
    n: int
    mape: float            # 평균 절대 % 오차
    median_ape: float
    max_ape: float
    under_rate: float      # 과소예측(=OOM 위험) 비율
    p95_under: float       # 과소예측 크기의 95퍼센타일 (actual/pred − 1)

    def row(self) -> str:
        return (f"{self.n} | {self.mape:.1f}% | {self.median_ape:.1f}% | {self.max_ape:.1f}% | "
                f"{self.under_rate*100:.0f}% | {self.p95_under*100:.1f}%")


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def metrics(pairs: Iterable[tuple[float, float]]) -> Metrics:
    """pairs: (pred, actual)."""
    pairs = [(p, a) for p, a in pairs if a]
    if not pairs:
        return Metrics(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    ape = [abs(p - a) / a * 100 for p, a in pairs]
    under = [a / p - 1 for p, a in pairs]
    return Metrics(
        n=len(pairs), mape=sum(ape) / len(ape), median_ape=_pct(ape, 0.5), max_ape=max(ape),
        under_rate=sum(u > 0 for u in under) / len(under), p95_under=max(_pct(under, 0.95), 0.0),
    )


FitFn = Callable[[list[Measurement]], Constants]


def cross_validate(meas: list[Measurement], split: str, target: str,
                   fit_fn: FitFn) -> list[tuple[Measurement, float]] | None:
    """held-out 예측. split='loo'(하나씩 빼기) | 'model'(모델 하나를 통째로 빼기).
    반환: [(측정, held-out 예측 GB)] — 'model' 분할이 불가능하면(모델 1개) None."""
    rows = [m for m in meas if m.ok and m.target(target)]
    out = []
    if split == "loo":
        for i, m in enumerate(rows):
            train = [r for j, r in enumerate(rows) if j != i]
            out.append((m, predict_gb(m, fit_fn(train))))
    elif split == "model":
        models = sorted({m.cfg.model for m in rows})
        if len(models) < 2:
            return None
        for key in models:
            consts = fit_fn([r for r in rows if r.cfg.model != key])
            out += [(m, predict_gb(m, consts)) for m in rows if m.cfg.model == key]
    else:
        raise ValueError(split)
    return out


def suggest_safety(cv_pairs: list[tuple[float, float]]) -> float:
    """held-out 과소예측 p95를 덮는 안전계수. fits(..., safety=이값) 에 사용."""
    u = metrics(cv_pairs).p95_under
    return round(min(0.98, max(0.70, 1.0 / (1.0 + u))), 3)


@dataclass
class OOMCheck:
    n: int
    correct: int
    missed_oom: int        # 실제 OOM인데 '들어간다'고 예측 (가장 위험)
    false_alarm: int       # 실제 성공인데 'OOM'이라고 예측 (보수적)


def oom_check(meas: list[Measurement], consts: Constants, safety: float = 1.0) -> OOMCheck:
    """'예측 > GPU 용량×safety 이면 OOM' 판정이 실제 결과와 맞는지."""
    rows = [m for m in meas if m.gpu_total_gb and (m.ok or m.oom)]
    c = OOMCheck(len(rows), 0, 0, 0)
    for m in rows:
        pred_oom = predict_gb(m, consts) > m.gpu_total_gb * safety
        if pred_oom == m.oom:
            c.correct += 1
        elif m.oom:
            c.missed_oom += 1
        else:
            c.false_alarm += 1
    return c


# ---------------------------------------------------------------------------
# 처리량 보정
# ---------------------------------------------------------------------------
def _tp_pairs(rows, gpu, c):
    from .throughput import estimate_throughput
    return [(estimate_throughput(m.cfg, m.spec, gpu, m.grad_accum, c).tokens_per_s, m.tokens_per_s)
            for m in rows]


TP_GRID = dict(
    t_half=[8 * 2 ** (k / 2) for k in range(0, 17)],           # 8 ~ 2048 토큰
    alpha=[0.0, 0.1, 0.2, 0.3, 0.4, 0.6],
    recompute=[0.6, 0.7, 0.8, 0.9, 1.0],
    attn_nonflash=[1.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0],
    opt_bpp=[0.0, 16.0, 32.0, 48.0, 64.0, 96.0, 128.0],
    opt8_bpp=[0.0, 16.0, 32.0, 64.0, 96.0, 128.0, 192.0],
)


def _tp_row_parts(m: Measurement, gpu) -> dict:
    """한 측정의 시간 식을 상수별로 분해 (격자탐색을 빠르게): FLOPs(κ,γ) = F0 + (κ−1)·Fa + γ·(Rl + κ·Ra)."""
    from .throughput import DEFAULT_TP, flops_per_token
    from .memory import adapter_params
    def F(k, g):
        return flops_per_token(m.cfg, m.spec, replace(DEFAULT_TP, attn_nonflash=k, recompute=g)).total
    f00, f20, f01, f21 = F(1, 0), F(2, 0), F(1, 1), F(2, 1)
    ra = (f21 - f20) - (f01 - f00)
    trainable = m.spec.n_params if m.cfg.method == "full" else adapter_params(m.spec, m.cfg)
    return dict(F0=f00, Fa=f20 - f00, Rl=(f01 - f00) - ra, Ra=ra, T=m.cfg.batch_size * m.cfg.seq_len,
                H=m.spec.hidden, size=(m.spec.hidden * m.spec.intermediate) / (4096 * 14336),
                opt=trainable / (gpu.mem_bw_gbs * 1e9), eight=m.cfg.optim == "adamw8bit",
                ga=max(m.grad_accum or 1, 1), peak=gpu.tflops_16bit * 1e12, meth=m.cfg.method,
                y=m.tokens_per_s)


def fit_throughput(meas: list[Measurement]) -> dict:
    """GPU별로 처리량 상수를 격자탐색으로 fit (v2).

    시간 = grad_accum × 계산시간 + optimizer 시간
      계산시간 = FLOPs(κ,γ) × 토큰 ÷ (peak × eff_method × 활용도(t_half) × 크기효율(α))
      optimizer 시간 = 학습 파라미터 × bpp ÷ 메모리 대역폭
    (t_half, α, γ, κ, bpp)는 격자, eff_method는 '실측 시간 − optimizer 시간'에 대한 닫힌 해(로그 평균).
    데이터로 정할 수 없는 항(예: checkpointing 측정이 없으면 γ, Full 측정이 없으면 bpp)은 기본값으로 둔다.
    flash 지원 GPU 중 측정이 가장 많은 것의 상수는 'default'로도 저장 → 측정 안 한 GPU(4090 등)에 사용.
    반환: {gpu_key: ThroughputConstants}
    """
    import itertools
    from .hardware import match_gpu
    from .throughput import DEFAULT_TP
    out, by_gpu, gpus = {}, {}, {}
    for m in meas:
        g = match_gpu(m.gpu_name)
        if m.ok and m.tokens_per_s and g:
            by_gpu.setdefault(g.key, []).append(m)
            gpus[g.key] = g
    for key, rows in by_gpu.items():
        if len(rows) < 3:
            continue
        gpu = gpus[key]
        parts = [_tp_row_parts(m, gpu) for m in rows]
        has = dict(recompute=any(m.cfg.grad_checkpointing for m in rows),
                   attn_nonflash=any(not m.cfg.flash_attn for m in rows),
                   opt_bpp=any(p["opt"] > 1e-3 and not p["eight"] and p["meth"] == "full" for p in parts),
                   opt8_bpp=any(p["eight"] and p["meth"] == "full" for p in parts))
        grid = {k: (v if has.get(k, True) else [getattr(DEFAULT_TP, k)]) for k, v in TP_GRID.items()}
        if len({p["size"] for p in parts}) < 2:                  # 모델 크기가 하나뿐이면 α를 정할 수 없음
            grid["alpha"] = [DEFAULT_TP.alpha]
        util = {th: [p["T"] / (p["T"] + th * 4096 / p["H"]) for p in parts] for th in grid["t_half"]}
        sizef = {a: [p["size"] ** a for p in parts] for a in grid["alpha"]}
        best = None
        for g_, k_ in itertools.product(grid["recompute"], grid["attn_nonflash"]):
            work = [(p["F0"] + (k_ - 1) * p["Fa"] + g_ * (p["Rl"] + k_ * p["Ra"])) * p["T"] / p["peak"] for p in parts]
            for b_, b8_ in itertools.product(grid["opt_bpp"], grid["opt8_bpp"]):
                opt = [p["opt"] * (b8_ if p["eight"] else b_) for p in parts]
                comp_act = [p["T"] * p["ga"] / p["y"] - o for p, o in zip(parts, opt)]   # 실측 계산 시간 × ga
                if min(comp_act) <= 0:
                    continue
                for th, a in itertools.product(grid["t_half"], grid["alpha"]):
                    U, Z = util[th], sizef[a]
                    base = [w / (u * z) for w, u, z in zip(work, U, Z)]           # eff=1 계산 시간
                    logs: dict[str, list[float]] = {}
                    for p, b, ca in zip(parts, base, comp_act):
                        logs.setdefault(p["meth"], []).append(math.log(b * p["ga"] / ca))
                    effs = {mm: math.exp(sum(v) / len(v)) for mm, v in logs.items()}
                    sse = 0.0
                    for p, b, o in zip(parts, base, opt):
                        t = p["ga"] * b / effs[p["meth"]] + o
                        sse += math.log(p["T"] * p["ga"] / t / p["y"]) ** 2
                    if best is None or sse < best[0]:
                        best = (sse, dict(t_half=th, alpha=a, recompute=g_, attn_nonflash=k_,
                                          opt_bpp=b_, opt8_bpp=b8_), effs)
        _, prm, effs = best
        c = replace(DEFAULT_TP, **prm)
        ratios = [effs[k] / DEFAULT_TP.eff(k) for k in effs]
        gscale = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        for meth in ("full", "lora", "qlora"):
            c = c.with_eff(meth, effs.get(meth, DEFAULT_TP.eff(meth) * gscale))
        res = sorted(abs(math.log(p / a)) for p, a in _tp_pairs(rows, gpu, c))
        p90 = res[min(len(res) - 1, int(0.9 * len(res)))]
        out[key] = replace(c, band=round(min(1.6, max(1.1, math.exp(p90) * 1.15)), 3))
    flash_keys = [k for k in out if gpus[k].flash]
    if flash_keys:
        ref = max(flash_keys, key=lambda k: len(by_gpu[k]))
        # 측정 안 한 GPU: 같은 세대(FlashAttention 지원) 실측 상수 + 넓은 범위 (GPU마다 효율이 다를 수 있음)
        out["default"] = replace(out[ref], band=max(out[ref].band, 1.5))
    return out


def throughput_metrics(meas: list[Measurement], tp: dict) -> dict:
    """GPU별 처리량 MAPE: 사전값 vs 보정값 (in-sample)."""
    from .hardware import match_gpu
    from .throughput import DEFAULT_TP
    res = {}
    for key, c in tp.items():
        rows = [m for m in meas if m.ok and m.tokens_per_s and (g := match_gpu(m.gpu_name)) and g.key == key]
        if not rows:
            continue
        gpu = match_gpu(rows[0].gpu_name)
        res[key] = {"before": metrics(_tp_pairs(rows, gpu, DEFAULT_TP)),
                    "after": metrics(_tp_pairs(rows, gpu, c))}
    return res
