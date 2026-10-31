# fitcfg — 소규모 GPU 파인튜닝 설정 추천기

개인·학생·소규모 랩(GPU 1~8장)을 위한 도구. **모델·GPU·학습 방법을 넣으면, 학습을 돌리기 전에
GPU 메모리와 속도를 예측하고, OOM 없이 가장 빠른 설정을 골라 바로 실행할 학습 스크립트까지 만든다.**
GPU가 여러 장이면 모델을 GPU에 나눠 올리는 3D 병렬화(TP·PP)가 필요한지도 판단한다. (캡스톤 디자인 프로젝트)

| 검증 (gfarm A100 MIG 3g.40gb + Colab T4 실측) | 결과 |
|---|---|
| 메모리 예측 오차 (보정에 안 쓴 모델로 확인) | 평균 4.9% |
| 추천 설정으로 실제 학습 (6개 시나리오) | 17/17 성공 · HF 기본값은 6/6 OOM |
| 추천 1위가 실제로 가장 빠름 | 6/6 |
| 처음 보는 모델 7종 (설계도만으로 예측) | 평균 4% · 안전계수 적용 시 놓친 OOM 0 |
| HF Trainer 플러그인을 실제 8B·7B·1B에 붙여서 | 학습 방식 자동 감지 7/7 · 메모리 1.6~2.9% · OOM 설정은 학습 전 7초 만에 중단 |
| 긴 시퀀스 4K~16K (보정에 안 씀) | 메모리 2.1% · OOM 판정 18/18 · 처리량 3.3% |
| GPU 2장 (Kaggle T4 ×2: DDP·ZeRO-2·ZeRO-3) | 추천 7/7 성공 · 기본값 3/3 OOM · LoRA·QLoRA 2장 = 1장의 2.0배 · 메모리 DDP 3.6~4.6% |
| GPU 없이 CPU로 직접 측정 (처음 보는 구조 8개 계열) | LoRA·Full 1~2% |
| 처리량 예측 (모델 빼고 맞추기 / 실제 학습 번들) | 3.9% / 4.0% (최대 7%) |

## 설치

```bash
pip install "git+https://github.com/dingdongri/fitcfg"          # 코어: 표준 라이브러리만 (GPU·torch 불필요)
pip install "fitcfg[hf] @ git+https://github.com/dingdongri/fitcfg"   # + HF Trainer 플러그인 · CPU 측정
pip install "fitcfg[ui] @ git+https://github.com/dingdongri/fitcfg"   # + 웹 UI
```
저장소를 받아서 쓸 때는 `python -m fitcfg ...` 그대로 동작한다.

## 쓰는 법 1 — HF Trainer 플러그인 (학습 코드에 한 줄)

```python
from fitcfg.integrations.hf import FitcfgCallback

trainer = Trainer(model=model, args=args, train_dataset=ds,
                  callbacks=[FitcfgCallback(seq_len=2048)])   # strict=True면 OOM 예상 시 시작 전에 멈춤
trainer.train()
```
```
[fitcfg] meta-llama/Llama-3.1-8B · qlora · micro-batch 8 × 누적 2 × GPU 1 · seq 2048 · checkpointing off
[fitcfg] 예측 GPU당 peak 167.2GB / 안전 한도 21.2GB (RTX 4090 24GB)
[fitcfg] ⚠️ OOM 예상 … 들어가는 추천 설정: micro-batch 2, 누적 8, checkpointing on, 단일 GPU (예측 18.6GB)
         → TrainingArguments(per_device_train_batch_size=2, gradient_accumulation_steps=8, gradient_checkpointing=True)
```
- 학습 직전에 모델·`TrainingArguments`에서 방법(Full/LoRA/QLoRA), micro-batch, checkpointing, GPU 수·FSDP/ZeRO,
  optimizer를 자동으로 읽어 예측한다. 몇 분 돌다 OOM 나는 대신 몇 초 만에 알려 준다.
- 학습이 끝나면 예측 vs 실측(peak 메모리·처리량)을 `output_dir/fitcfg_run.json`에 남긴다.
- GPU가 없거나 모르는 GPU면 조용히 건너뛰고 학습은 그대로 진행한다.
- GPU 없이 흐름만 보려면: `python examples/hf_plugin_example.py` (아주 작은 모델, CPU)

## 쓰는 법 2 — CLI

```bash
fitcfg plan --model llama-3.1-8b --gpu rtx4090 --method qlora --data-tokens 5e7       # 추천 + 학습 시간
fitcfg plan --model llama-3.2-1b --gpu t4 --method lora --out runs/lora_t4            # 학습 번들 생성
fitcfg plan --model Qwen/Qwen3-4B --gpu rtx4090 --method lora                         # 처음 보는 모델 (HF 이름)
fitcfg plan --model llama-3.1-8b --gpu-mix rtx3090:2,rtx4090:2 --method lora --global-batch 32   # GPU 여러 종류
fitcfg measure-cpu --model Qwen/Qwen3-4B --method qlora --ckpt --gpu-mem 24          # GPU 없이 직접 측정
fitcfg gpus | fitcfg models
```
```python
from fitcfg import plan, PlanRequest, get_gpu
res = plan(PlanRequest(model="llama-3.1-8b", gpu=get_gpu("rtx4090"), method="qlora"))
print(res.best.reason, res.best.pred_mem_gb, res.best.throughput.tokens_per_s)
```
웹 UI(데모용): `streamlit run app/streamlit_app.py` — 모델·GPU 선택(처음 보는 모델은 HF 이름이나 config.json 업로드)
→ 추천·메모리 분해 차트 → 학습 번들 zip 다운로드 · HF Trainer에 붙일 코드 · 처음 보는 모델 CPU 측정 버튼

## 무엇을 하나

1. **예측** (GPU 불필요) — GPU당 peak 메모리(가중치·그래디언트·옵티마이저·활성값·logits·오버헤드로 분해),
   처리량(tok/s, 범위), 학습 시간. 식의 상수는 GPU 실측으로 보정 (`data/calibration.json`, 140개 측정).
   - 처리량 = peak × 효율 × 활용도 × 크기 효율 ÷ 토큰당 FLOPs + optimizer step 시간(메모리 대역폭) — `reports/throughput_report.md`
2. **추천** — global batch는 그대로 두고 micro-batch·누적·checkpointing·8bit optimizer·DDP/ZeRO/offload를
   모두 따져 보고, 메모리에 들어가는 설정 중 가장 빠른 것. 안 들어가면 "QLoRA로 / seq 1024로 / GPU 2장이면 들어감" 진단.
3. **3D 병렬화 판단** (GPU 2장 이상) — TP·PP·DP 조합과 층 배분을 탐색해 데이터 병렬(DDP·FSDP) 최선과 비교.
   필요한 경우는 두 가지뿐이라는 결론을 그대로 규칙으로 씀:
   ① 메모리 — DP·FSDP로는 GPU 한 장에 안 들어갈 때 (예: A100-40 ×4로 8B Full)
   ② 속도 — GPU 종류가 섞였거나 연결이 느릴 때 (예: 3090 ×2 + 4090 ×2: 빠른 GPU에 층을 더 줘서 1.5~2.2배)
   통신을 겹치는 정도 가정(낙관·비관) 두 가지 모두에서 1.1배 이상 빠를 때만 추천한다.
4. **처음 보는 모델** — HF 이름이나 `config.json`만 있으면 설계도를 읽어 파라미터 수(실제와 0.01% 이내)와 구조를 계산.
   구조 유형에 따라 처리: Llama형 그대로 · Gemma형/모르는 구조는 여유를 더 두고 `measure-cpu` 권장 · MoE는 지원 안 함.
5. **생성** — `train.py`(HF Trainer) + `ds_config.json` + `launch.sh`. 학습 후 실측을 `fitcfg_run.json`에 저장.

## 구조
```
fitcfg/
  fitcfg/                 # 코어 (표준 라이브러리만)
    memory.py             # 메모리 예측 (보정 가능한 상수)
    throughput.py         # 처리량·시간 예측
    recommend.py          # 설정 탐색·추천·진단 (plan)
    parallel.py           # 3D 병렬화 판단 (TP·PP·DP 탐색, 층 배분, 1F1B 시간 모델)
    autospec.py           # 처음 보는 모델: config.json → 사양 + 구조 유형·여유
    trace.py              # GPU 없이 CPU 가짜 텐서로 학습 메모리 추적 (torch 필요)
    generate.py           # 학습 번들 생성 (templates/train.py.tmpl)
    integrations/hf.py    # HF Trainer 플러그인 FitcfgCallback
    calibration.py        # 실측 CSV → 보정 상수
    cli.py                # fitcfg / python -m fitcfg
  app/streamlit_app.py    # 웹 UI
  scripts/                # 측정 스윕·보정·시뮬레이션·리포트 (GPU 서버용 포함)
  examples/               # demo.py (예측·추천), hf_plugin_example.py (플러그인)
  tests/                  # 53개 (GPU 불필요): python -m pytest -q   · .github/workflows/tests.yml (torch 없이 / CPU torch로)
  data/  reports/         # 실측 데이터·보정 결과·리포트
```

## 실측으로 확인한 것 (`reports/`)
- `mig_report.md` — A100 MIG 실측 140개로 보정. held-out 4.9%, 안전계수 0.895로 놓친 OOM 0
- `round4_report.md` — 추천대로 실제 학습(17/17 성공), 처음 보는 모델 4종, QLoRA 규칙 "LoRA × 1.3"
- `round5_report.md` — 플러그인 실제 GPU · CPU offload · 긴 시퀀스 · 처음 보는 구조 식 보완
- `multi_gpu_report.md` — GPU 2장 실측(Kaggle T4 ×2): 통신 시간 식, GPU 1장 설정의 DataParallel 버그 수정
- `throughput_report.md` — 처리량 식 v2 (크기 효율·재계산 비용·optimizer 시간): A100 MIG 오차 12.6% → 3.9%
- `e2e_report.md` — 시나리오별 기본값 vs 추천 1~3위 실측
- `sim_3d_report_recal.md` — 하드웨어 5종 × 모델 2종 × 방법 3종의 3D 병렬화 판단 표

## 알려진 한계
- **GPU 여러 장의 통신 시간은 실측 전** — 3D 판단은 단일 GPU 실측으로 보정한 계산 시간 + 연결 대역폭 가정.
  그래서 결과를 낙관·비관 두 값으로 보여 주고, 둘 다 1.1배 이상일 때만 추천한다.
- 3D 추천은 계획(`parallel_plan.json`)까지만 생성. TP·PP 실행 스크립트 자동 생성은 아직 없다 (번들은 DP 최선 설정).
- Gemma 계열은 구조 항(추가 norm·softcap) 반영 후에도 QLoRA에서 최대 18% 적게 나올 수 있어 여유 ×1.15~1.25 유지
  (실측상 놓친 OOM 0, 대신 다소 보수적).
- CPU offload·ZeRO-3는 실측 1~2건 — ZeRO-3 할당기 단편화(예약 = 할당 × 1.76)와 CPU RAM 식은 그 결과로 보수적으로 잡음.
- GPU 여러 장은 Kaggle T4 ×2(PCIe) 1종만 실측 — 통신 대역폭 4.9 GB/s는 이 결과로 정함. NVLink·3D 병렬화(TP·PP)·혼합 GPU는 계산 기반.
- QLoRA는 CPU로 직접 흉내 낼 수 없어 `measure-cpu`는 "LoRA 측정 × 1.3 + 4bit 가중치" 규칙 (8계열 실측 5% 안팎).
- 처리량은 A100 MIG·T4만 실측 → 측정 안 한 GPU(RTX 3090·4090 등)는 같은 세대(A100) 실측 상수를 쓰고 범위를 ±50%로 넓게 표시.
- MoE 모델 미지원.

## 라이선스
MIT — 누구나 자유롭게 쓰고 고치고 배포할 수 있음 (저작권 표시만 유지). [LICENSE](LICENSE)
