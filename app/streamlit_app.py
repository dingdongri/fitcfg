"""fitcfg 웹 UI (Phase 6).

    pip install streamlit
    streamlit run app/streamlit_app.py

입력(모델·GPU·목표) → 추천 설정 + 예측(메모리/처리량/시간) → 학습 번들(zip) 다운로드.
처음 보는 모델은 HF 이름이나 config.json 업로드로 넣는다. HF Trainer에 붙이는 플러그인 코드도 보여 준다.
"""
from __future__ import annotations
import json
import sys
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fitcfg import (GPUS, MODELS, PlanRequest, calibration_info, custom_gpu,  # noqa: E402
                    get_default_safety, load_calibration, plan, reset_calibration, structure_of)
from fitcfg.autospec import register, resolve  # noqa: E402
from fitcfg.generate import build_bundle, zip_bundle  # noqa: E402

# ---------------------------------------------------------------- 상수
MODEL_LABELS = {
    "llama-3.2-1b": "Llama-3.2-1B",
    "qwen2.5-3b": "Qwen2.5-3B",
    "qwen2.5-7b": "Qwen2.5-7B",
    "mistral-7b-v0.3": "Mistral-7B-v0.3",
    "llama-3.1-8b": "Llama-3.1-8B",
}
MODEL_LABELS = {k: v for k, v in MODEL_LABELS.items() if k in MODELS}
UNSEEN = "__unseen__"
KIND_LABELS = {"verified": "실측 보정한 모델", "llama-like": "Llama형 구조 (그대로 예측)",
               "gemma-like": "Gemma형 구조 (여유를 더 둠)", "unknown": "검증 안 된 구조 (여유를 더 둠)"}
METHOD_LABELS = {"qlora": "QLoRA (4bit + 어댑터)", "lora": "LoRA (bf16 + 어댑터)", "full": "Full fine-tuning"}
STRATEGY_LABELS = {"single": "단일 GPU", "ddp": "DDP", "zero1": "ZeRO-1", "zero2": "ZeRO-2", "zero3": "ZeRO-3",
                   "zero-offload": "ZeRO-2 + CPU offload", "zero3-offload": "ZeRO-3 + 파라미터 offload"}
# 메모리 항목 → 색 (검증된 categorical 팔레트, 순서 고정: 항목이 빠져도 색이 바뀌지 않음)
COMPONENTS = [("weights", "가중치"), ("gradients", "그래디언트"), ("optimizer", "옵티마이저"),
              ("activations", "활성값"), ("logits", "logits"), ("overhead", "오버헤드")]
COMP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MUTED = "#8a8984"
CALIB_PATH = REPO / "data" / "calibration.json"

st.set_page_config(page_title="fitcfg — 파인튜닝 설정 추천기", page_icon="🧮", layout="wide")


def metric(col, label, value, delta=None, delta_color="off", help=None):
    """delta를 '보조 설명'으로 쓰므로 화살표를 끈다 (구버전 streamlit은 delta_arrow 인자가 없음)."""
    try:
        col.metric(label, value, delta, delta_color=delta_color, help=help, delta_arrow="off")
    except TypeError:
        col.metric(label, value, delta, delta_color=delta_color, help=help)


# ---------------------------------------------------------------- 보정값
def setup_calibration(use: bool) -> str:
    reset_calibration()
    if use and CALIB_PATH.exists():
        load_calibration(CALIB_PATH)
        info = calibration_info()
        mape = info.get("heldout_mape")
        msg = f"실측 보정 적용 · 측정 {info.get('n_rows')}행"
        if mape is not None:
            msg += f" · held-out 메모리 오차(MAPE) {mape:.1f}%"
        if info.get("throughput_gpus"):
            msg += f" · 처리량 보정 GPU: {', '.join(info['throughput_gpus'])}"
        if info.get("synthetic"):
            msg += " · ⚠️ 합성 데이터"
        return msg
    return ("GPU 실측 보정 전: FakeTensor 추적(1B~8B, 288개 설정)으로 맞춘 상수 사용 · "
            "overhead(CUDA 컨텍스트)는 추정값 — Colab 측정 → calibrate.py 후 자동 적용")


# ---------------------------------------------------------------- 사이드바 입력
with st.sidebar:
    st.header("학습 조건")
    model_choice = st.selectbox("모델", list(MODEL_LABELS) + [UNSEEN], index=list(MODEL_LABELS).index("llama-3.1-8b"),
                                format_func=lambda k: MODEL_LABELS.get(k, "처음 보는 모델 (HF 이름 · config.json)"))
    model, model_error = model_choice, None
    if model_choice == UNSEEN:
        src = st.radio("모델 설계도", ["hf", "file"], horizontal=True,
                       format_func={"hf": "HF 이름", "file": "config.json 업로드"}.get)
        try:
            if src == "hf":
                name = st.text_input("Hugging Face 모델 이름", "Qwen/Qwen3-4B-Base",
                                     help="설계도(config.json)만 읽고 가중치는 받지 않습니다. 승인이 필요한 모델은 "
                                          "HF_TOKEN 환경변수를 설정하거나 config.json을 업로드하세요.").strip()
                if name:
                    with st.spinner("설계도 읽는 중…"):
                        resolve(name)
                    model = name
                else:
                    model_error = "모델 이름을 입력하세요."
            else:
                up = st.file_uploader("config.json", type=["json"])
                if up is None:
                    model_error = "config.json을 올려 주세요 (모델 저장소의 config.json 파일)."
                else:
                    cfg = json.loads(up.getvalue().decode("utf-8"))
                    model = f"upload/{Path(up.name).stem}"
                    register(model, cfg)
        except (KeyError, ValueError, OSError) as e:
            model_error = str(e)
    method = st.radio("학습 방식", list(METHOD_LABELS), format_func=METHOD_LABELS.get)
    gpu_keys = list(GPUS) + ["custom"]
    gpu_key = st.selectbox("GPU", gpu_keys, index=gpu_keys.index("rtx4090"),
                           format_func=lambda k: GPUS[k].label if k in GPUS else "직접 입력")
    if gpu_key == "custom":
        mem = st.number_input("GPU 메모리 (GB)", 4.0, 192.0, 24.0, 1.0)
        tflops = st.number_input("16bit TFLOPS (모르면 그대로)", 5.0, 2000.0, 100.0, 5.0)
        bf16 = st.checkbox("bf16 지원 (Ampere 이상)", True)
        gpu = custom_gpu(mem, tflops, bf16)
    else:
        gpu = GPUS[gpu_key]
    n_gpus = st.select_slider("GPU 수 (단일 노드)", [1, 2, 4, 8], value=1)
    seq_len = st.select_slider("시퀀스 길이 (토큰)", [512, 1024, 2048, 4096, 8192], value=2048)
    global_batch = st.number_input("목표 global batch (시퀀스/스텝)", 1, 1024, 16,
                                   help="optimizer 한 스텝에 들어가는 전체 시퀀스 수. 이 값은 고정하고 "
                                        "micro-batch × GPU × 누적 조합을 바꿔가며 메모리·속도를 최적화합니다.")
    with st.expander("데이터 · 학습 시간"):
        data_mtok = st.number_input("학습 데이터 토큰 수 (백만)", 0.0, 100000.0, 50.0, 10.0,
                                    help="0이면 학습 시간은 계산하지 않습니다.")
        epochs = st.number_input("epoch", 0.1, 20.0, 1.0, 0.5)
    with st.expander("고급 설정"):
        lora_rank = st.select_slider("LoRA rank", [4, 8, 16, 32, 64, 128], value=16,
                                     disabled=method == "full")
        lora_targets = st.radio("LoRA 대상", ["attn", "all"], horizontal=True, disabled=method == "full",
                                format_func={"attn": "attention (q,k,v,o)", "all": "전체 Linear"}.get)
        allow_offload = st.checkbox("CPU offload 허용", True)
        cpu_ram = st.number_input("CPU RAM (GB, 0=모름)", 0.0, 4096.0, 0.0, 8.0,
                                  help="입력하면 RAM이 부족한 offload 설정은 후보에서 뺍니다. Colab 무료는 약 12GB.")
        use_calib = st.checkbox("실측 보정값 사용", True, disabled=not CALIB_PATH.exists())
        calib_msg = setup_calibration(use_calib)
        safety = st.slider("안전계수 (GPU 메모리 중 사용 비율)", 0.70, 1.00, float(get_default_safety()), 0.01,
                           help="예측 오차·데이터 편차를 흡수하는 여유. 보정 리포트의 권장값이 기본입니다.")

if model_error:
    st.title("fitcfg")
    st.info(model_error)
    st.stop()
info = structure_of(model)
model_label = MODEL_LABELS.get(model, model)
req = PlanRequest(model=model, gpu=gpu, gpus=n_gpus, method=method, seq_len=seq_len,
                  global_batch=int(global_batch), lora_rank=lora_rank, lora_targets=lora_targets,
                  allow_offload=allow_offload, cpu_ram_gb=cpu_ram or None, safety=safety,
                  dataset_tokens=data_mtok * 1e6 if data_mtok else None, epochs=epochs)
res = plan(req, top_k=5)

# ---------------------------------------------------------------- 헤더
st.title("fitcfg")
st.markdown("**실행 전에** GPU 메모리·처리량·학습 시간을 예측하고, OOM 없이 가장 빠른 파인튜닝 설정과 "
            "바로 돌릴 수 있는 학습 스크립트를 만들어 줍니다.")
st.caption(f"📐 {calib_msg}")
if info.kind != "verified":
    from fitcfg import get_model
    spec = get_model(model)
    st.caption(f"🧩 {model_label} · {spec.n_params / 1e9:.2f}B · {spec.n_layers}층 · hidden {spec.hidden} · "
               f"어휘 {spec.vocab:,} · {KIND_LABELS.get(info.kind, info.kind)}")
for n in res.notes:
    st.caption(f"ℹ️ {n}")

if not res.best:
    st.error(f"**{model_label} · {METHOD_LABELS[method]}** 은(는) {gpu.label} × {n_gpus}에 "
             f"들어가는 설정이 없습니다 (후보 {res.n_candidates}개 전부 초과).")
    for d in res.diagnosis:
        st.markdown(d)
    st.stop()

# 번들로 만들 추천 선택
recs = res.recommendations
pick = st.radio("추천 설정", range(len(recs)), horizontal=True,
                format_func=lambda i: f"#{i + 1}" + (f" · {recs[i].tags[0]}" if recs[i].tags else ""))
rec = recs[pick]
naive = res.naive

# ---------------------------------------------------------------- 핵심 수치
c1, c2, c3, c4 = st.columns(4)
metric(c1, "예측 peak 메모리 (GPU당)", f"{rec.pred_mem_gb:.1f} GB",
       f"허용치까지 여유 {rec.headroom_gb:.1f} GB",
       help=f"허용치 {rec.cap_gb:.1f} GB = GPU {gpu.mem_gb:.1f} GB × 안전계수 {res.safety}")
tp = rec.throughput
metric(c2, "예측 처리량", f"{tp.tokens_per_s:,.0f} tok/s", f"범위 {tp.low:,.0f}~{tp.high:,.0f}",
       help="처리량은 메모리보다 예측이 어려워 범위로 제시합니다."
                                   + ("" if tp.calibrated else " 이 GPU는 아직 실측 보정 전입니다."))
if rec.hours:
    h, fast, slow = rec.hours
    metric(c3, "예상 학습 시간", f"{h:.1f} 시간" if h >= 1 else f"{h * 60:.0f} 분",
           f"범위 {fast:.1f}~{slow:.1f} 시간")
else:
    metric(c3, "예상 학습 시간", "—", "데이터 토큰 수 입력 시 계산")
if naive.fits:
    metric(c4, "기본값(HF Trainer)으로 돌리면", "들어감", f"예측 {naive.pred_mem_gb:.1f} GB",
           help=f"기본값: {naive.reason}")
else:
    metric(c4, "기본값(HF Trainer)으로 돌리면", "OOM 예상", f"예측 {naive.pred_mem_gb:.1f} GB",
           delta_color="inverse", help=f"기본값: {naive.reason}")

st.subheader(f"#{pick + 1} " + rec.reason)
st.markdown(f"micro-batch **{rec.config.batch_size}** × GPU **{rec.config.gpus}** × 누적 **{rec.grad_accum}** "
            f"= global batch **{rec.effective_batch}** · {rec.dtype} · "
            f"후보 {res.n_candidates}개 중 {res.n_feasible}개가 메모리에 들어감")
for w in rec.warnings:
    st.warning(w, icon="⚠️")


# ---------------------------------------------------------------- 차트: 메모리 분해
def breakdown_rows(r, label):
    gb = r.memory.as_gb()
    return [{"설정": label, "항목": name, "GB": gb[k], "순서": i} for i, (k, name) in enumerate(COMPONENTS)]


rows = breakdown_rows(rec, f"추천 #{pick + 1}") + breakdown_rows(naive, "기본값")
df = pd.DataFrame(rows)
cap = rec.cap_gb
xmax = max(gpu.mem_gb, naive.pred_mem_gb, rec.pred_mem_gb) * 1.05
bars = alt.Chart(df).mark_bar(stroke="white", strokeWidth=2).encode(
    x=alt.X("sum(GB):Q", title="GPU당 메모리 (GB)", scale=alt.Scale(domain=[0, xmax])),
    y=alt.Y("설정:N", title=None, sort=[f"추천 #{pick + 1}", "기본값"],
            scale=alt.Scale(paddingInner=0.35, paddingOuter=0.2)),
    color=alt.Color("항목:N", scale=alt.Scale(domain=[n for _, n in COMPONENTS], range=COMP_COLORS),
                    legend=alt.Legend(orient="top", title=None, columns=3)),
    order=alt.Order("순서:Q"),
    tooltip=["설정", "항목", alt.Tooltip("GB:Q", format=".2f")],
)
rules = pd.DataFrame([{"x": gpu.mem_gb, "라벨": f"GPU {gpu.mem_gb:.1f}GB", "side": "left"},
                      {"x": cap, "라벨": f"허용 {cap:.1f}GB", "side": "right"}])
rule = alt.Chart(rules).mark_rule(strokeDash=[4, 3], color="#52514e", strokeWidth=1.5).encode(
    x="x:Q", tooltip=["라벨"])
# 두 점선이 가까우므로 GPU 라벨은 오른쪽, 허용 라벨은 왼쪽으로 붙인다
rule_text = (
    alt.Chart(rules[rules["side"] == "left"]).mark_text(align="left", dx=5, color="#52514e", fontSize=11)
    .encode(x="x:Q", y=alt.value(-7), text="라벨")
    + alt.Chart(rules[rules["side"] == "right"]).mark_text(align="right", dx=-5, color="#52514e", fontSize=11)
    .encode(x="x:Q", y=alt.value(-7), text="라벨")
)
st.altair_chart((bars + rule + rule_text).properties(height=alt.Step(44)), width="stretch")
PHASE = {"fwd_bwd": "역전파 중(활성값·logits 최대)", "optim_step": "optimizer step 중(gradient·임시버퍼 최대)"}
st.caption(f"막대 = peak 시점의 메모리 구성. 추천 #{pick + 1}의 peak 시점: {PHASE[rec.memory.phase]} · "
           f"기본값: {PHASE[naive.memory.phase]}. 점선 = GPU 용량과 허용치(안전계수 {res.safety}).")

# ---------------------------------------------------------------- 탭
tab_cmp, tab_files, tab_plugin, tab_detail = st.tabs(
    ["후보 비교", "학습 번들 (config · 스크립트)", "HF Trainer에 붙이기", "예측 근거"])

with tab_cmp:
    table = pd.DataFrame([{
        "순위": f"#{i + 1}", "태그": ", ".join(r.tags), "전략": STRATEGY_LABELS[r.strategy],
        "micro-batch": r.config.batch_size, "누적": r.grad_accum,
        "checkpointing": "on" if r.config.grad_checkpointing else "off",
        "optimizer": "8bit AdamW" if r.config.optim == "adamw8bit" else "AdamW",
        "메모리 GB": r.pred_mem_gb, "여유 GB": round(r.headroom_gb, 1),
        "tok/s": round(r.throughput.tokens_per_s), "시간(h)": round(r.hours[0], 1) if r.hours else None,
    } for i, r in enumerate(recs)])
    st.dataframe(table, hide_index=True, width="stretch")

    st.markdown("**탐색한 전체 후보** — 허용치(점선)의 왼쪽만 추천 대상입니다.")
    cdf = pd.DataFrame([{
        "메모리 GB": c.pred_mem_gb, "tok/s": round(c.throughput.tokens_per_s),
        "상태": "추천" if any(c is r for r in recs) else ("들어감" if c.fits else "초과"),
        "설정": c.reason,
    } for c in res.candidates])
    status_scale = alt.Scale(domain=["추천", "들어감", "초과"], range=["#2a78d6", "#86b6ef", MUTED])
    pts = alt.Chart(cdf).mark_circle(size=90, stroke="white", strokeWidth=1.5, opacity=0.95).encode(
        x=alt.X("메모리 GB:Q", title="예측 메모리 (GB, GPU당)"),
        y=alt.Y("tok/s:Q", title="예측 처리량 (tok/s)"),
        color=alt.Color("상태:N", scale=status_scale, legend=alt.Legend(orient="top", title=None)),
        tooltip=["설정", "메모리 GB", "tok/s", "상태"],
    )
    cap_rule = alt.Chart(pd.DataFrame([{"x": cap}])).mark_rule(strokeDash=[4, 3], color="#52514e").encode(x="x:Q")
    st.altair_chart((pts + cap_rule).properties(height=320), width="stretch")

with tab_files:
    col_a, col_b, col_c = st.columns([2, 1, 1])
    dataset = col_a.text_input("데이터셋 (jsonl 경로 또는 HF 데이터셋 이름)", "data.jsonl")
    text_field = col_b.text_input("텍스트 필드", "text")
    lr = col_c.number_input("learning rate", 1e-7, 1e-2, 1e-5 if method == "full" else 2e-4, format="%.1e")
    files = build_bundle(rec, req, dataset=dataset, text_field=text_field, learning_rate=lr)
    fname = f"fitcfg_{re.sub(r'[^A-Za-z0-9._-]+', '-', model)}_{method}_{gpu.key}x{n_gpus}.zip"
    st.download_button("⬇️ 학습 번들 다운로드 (.zip)", zip_bundle(files, folder=fname[:-4]), fname,
                       "application/zip", type="primary")
    st.caption("압축을 풀고 `pip install -r requirements.txt` → `bash launch.sh`. "
               "학습이 끝나면 `outputs/fitcfg_run.json`에 실측 메모리·처리량이 남아 예측을 검증할 수 있습니다.")
    order = ["README.md", "launch.sh", "train.py", "ds_config.json", "requirements.txt", "fitcfg_plan.json"]
    names = [n for n in order if n in files]
    for name, t in zip(names, st.tabs(names)):
        lang = {"py": "python", "sh": "bash", "json": "json", "md": "markdown", "txt": "text"}[name.rsplit(".", 1)[1]]
        with t:
            if name == "README.md":
                # 페이지 안에서는 제목 크기를 줄여 보여준다 (파일 내용은 그대로)
                shown = re.sub(r"^(#{1,2}) ", lambda m: "#" * (len(m.group(1)) + 3) + " ", files[name], flags=re.M)
                with st.container(border=True):
                    st.markdown(shown)
            else:
                st.code(files[name], language=lang)

with tab_plugin:
    c = rec.config
    targs = [f"per_device_train_batch_size={c.batch_size}", f"gradient_accumulation_steps={rec.grad_accum}",
             f"gradient_checkpointing={c.grad_checkpointing}"]
    if c.optim == "adamw8bit":
        targs.append('optim="adamw_bnb_8bit"')
    targs.append("bf16=True" if rec.dtype == "bf16" else "fp16=True")
    st.markdown("이미 학습 코드가 있다면 번들 대신 **콜백 한 줄**만 붙이면 됩니다. 학습 시작 직전에 지금 설정의 메모리를 "
                "예측해 OOM이면 미리 알려 주고(들어가는 설정 제안), 끝나면 예측 vs 실측을 `fitcfg_run.json`에 남깁니다.")
    st.code("pip install \"fitcfg[hf] @ git+https://github.com/dingdongri/fitcfg\"", language="bash")
    st.code(f"""from transformers import Trainer, TrainingArguments
from fitcfg.integrations.hf import FitcfgCallback

args = TrainingArguments(
    output_dir="outputs",
    {(","+chr(10)+"    ").join(targs)},
)
trainer = Trainer(model=model, args=args, train_dataset=ds,
                  callbacks=[FitcfgCallback(seq_len={seq_len})])   # strict=True: OOM 예상 시 시작 전에 중단
trainer.train()""", language="python")
    st.caption(f"위 TrainingArguments 값은 추천 #{pick + 1} 설정입니다. 콜백은 모델·TrainingArguments에서 학습 방식"
               "(Full/LoRA/QLoRA)·GPU·FSDP/ZeRO를 자동으로 읽으므로 seq_len만 알려 주면 됩니다.")
    if info.kind != "verified":
        st.markdown("**GPU 없이 직접 측정 (처음 보는 모델)**")
        st.caption("CPU에서 가짜 텐서로 학습 2스텝을 흉내 내 GPU 메모리를 잽니다 (실제 계산·가중치 다운로드 없음). "
                   "실측 8개 계열에서 LoRA·Full 1~2%, QLoRA는 'LoRA × 1.3 + 4bit 가중치' 규칙으로 5% 안팎. "
                   "torch·transformers·peft가 필요하고 수십 초~몇 분 걸립니다.")
        if st.button(f"추천 #{pick + 1} 설정을 CPU로 측정"):
            try:
                from fitcfg.trace import measure_cpu
                with st.spinner("추적 중…"):
                    r = measure_cpu(model, method, c.batch_size, seq_len, c.grad_checkpointing, lora_rank)
                d = (rec.pred_mem_gb - r["device_peak_gb"]) / r["device_peak_gb"] * 100
                metric(st, "CPU 측정 (GPU에서 예상 peak)", f"{r['device_peak_gb']:.2f} GB",
                       f"예측 {rec.pred_mem_gb:.2f} GB ({d:+.1f}%) · 허용 {rec.cap_gb:.1f} GB")
                if r["device_peak_gb"] > rec.cap_gb:
                    st.warning("CPU 측정값이 허용치를 넘습니다 → 다음 순위 추천이나 checkpointing을 고려하세요.", icon="⚠️")
            except ImportError as e:
                st.error(f"CPU 측정에는 torch·transformers·peft가 필요합니다: {e}")
            except Exception as e:  # noqa: BLE001
                st.error(f"측정 실패: {type(e).__name__}: {e}")

with tab_detail:
    st.markdown("**메모리 분해 (GB, GPU당)**")
    st.dataframe(pd.DataFrame([
        {"항목": name, f"추천 #{pick + 1}": rec.memory.as_gb()[k], "기본값": naive.memory.as_gb()[k]}
        for k, name in COMPONENTS] + [
        {"항목": "합계", f"추천 #{pick + 1}": rec.pred_mem_gb, "기본값": naive.pred_mem_gb}]),
        hide_index=True, width="stretch")
    st.markdown(f"""
**어떻게 예측하나**
- `peak = 가중치 + 옵티마이저 상태 + max(활성값 + logits, 그래디언트 + optimizer 임시버퍼) + 오버헤드`
  — 역전파 중과 optimizer step 중 **더 큰 시점**이 peak입니다.
- 가중치·그래디언트·옵티마이저는 파라미터 수 × dtype 바이트로 계산 (QLoRA는 4bit, 임베딩/lm_head는 fp32로 따로).
- 활성값은 레이어·토큰당 `a·hidden + b·MLP폭`, logits는 `batch × seq × vocab` 비례 (vocab이 큰 Llama-3/Qwen에서 매우 큼).
- 상수는 **GPU 실측 140개**(A100 MIG · Colab T4)로 보정: 보정에 안 쓴 모델 예측 오차 4.9%, 안전계수 적용 시 놓친 OOM 0.
  추천 설정으로 실제 학습해 보면 17/17 성공 (HF 기본값은 6/6 OOM).
- 처음 보는 모델은 설계도(config.json)로 계산하고 구조 유형에 따라 여유를 둡니다 (처음 보는 8개 계열 실측 평균 4.4%).
- 처리량 = GPU peak TFLOPS × 효율 × 활용도 × 크기 효율 ÷ 토큰당 학습 FLOPs + optimizer step 시간(메모리 대역폭).
  모델 하나를 빼고 맞춘 오차 3.9%, 실제 학습 번들 4.0% (A100 MIG). 실측 안 한 GPU는 같은 세대 실측값을 쓰고 범위를 넓게 표시.
- 기본값(HF Trainer) = micro-batch 8, checkpointing off, AdamW — OOM 회피율 평가의 기준선.

**이 GPU:** {gpu.label} · 사용 가능 {gpu.mem_gb:.1f} GB · 16bit {gpu.tflops_16bit:g} TFLOPS · 메모리 대역폭 {gpu.mem_bw_gbs:g} GB/s · bf16 {'지원' if gpu.bf16 else '미지원'}
""")
