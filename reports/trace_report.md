# FakeTensor 추적 기반 cost model 검증 (GPU 없이)

- 생성: 2026-09-25 15:52 · 데이터: trace_measurements.csv · 288행 = 모델 ['llama-3.1-8b', 'llama-3.2-1b', 'qwen2.5-3b', 'qwen2.5-7b'] × LoRA/QLoRA(에뮬)/Full × batch × seq × ckpt
- 타깃: 학습 구간 텐서 peak (`max_memory_allocated`와 같은 의미). CUDA 컨텍스트·할당자 캐시는 제외 → GPU 실측으로 보정
- 추적기 자체 검증: 알려진 할당 패턴(할당·view·in-place·해제)에서 cur/peak 정확히 일치 (tests/test_trace.py)

## 1. 정확도 비교 (MAPE, 텐서 peak)

| 모델 식 / 상수 | n | MAPE | median | max | 과소예측 비율 | 과소예측 p95 |
|---|---|---|---|---|---|---|
| v0.1 식 + 문헌 상수 | 288 | 22.5% | 20.0% | 51.6% | 99% | 79.1% |
| v0.2 식 + 사전 추정값(fit 전) | 288 | 4.7% | 3.6% | 37.8% | 43% | 9.0% |
| v0.2 식 fit (in-sample) | 288 | 0.8% | 0.6% | 2.8% | 42% | 2.1% |
| v0.2 식 fit · LOO | 288 | 0.8% | 0.6% | 3.1% | 42% | 2.3% |
| v0.2 식 fit · 모델 제외 ★ | 288 | 1.1% | 0.8% | 4.5% | 37% | 2.4% |
| v0.2에서 I항 제거 · 모델 제외 | 288 | 4.4% | 3.9% | 17.0% | 41% | 7.5% |

**읽는 법**: '모델 제외'는 그 모델의 데이터를 전혀 보지 않고 다른 모델로만 fit 한 뒤 예측한 값 — 새 모델로 일반화되는지를 본다. I항 제거 비교는 활성값을 MLP 폭(I)으로 분리한 설계의 근거.

## 2. 모델 제외 검증 — 모델별

| 모델 | n | MAPE | max | 과소예측 |
|---|---|---|---|---|
| llama-3.1-8b | 72 | 1.6% | 4.5% | 22% |
| llama-3.2-1b | 72 | 1.2% | 3.9% | 35% |
| qwen2.5-3b | 72 | 0.8% | 2.6% | 62% |
| qwen2.5-7b | 72 | 0.9% | 2.7% | 28% |

## 3. 모델 제외 검증 — method별

| method | n | MAPE | max | 과소예측 |
|---|---|---|---|---|
| full | 96 | 1.1% | 4.2% | 35% |
| lora | 96 | 1.0% | 4.3% | 29% |
| qlora | 96 | 1.3% | 4.5% | 46% |

## 4. fit 된 상수 (→ fitcfg/priors.json)

| 상수 | default | full | lora | qlora |
|---|---|---|---|---|
| act_h | 25.54 | 26.33 | 25.04 | 25.89 |
| act_i | 6.682 | 8.25 | 6.307 | 6.395 |
| ckpt_h | 1.966 | 1.425 | 0.8215 | 3.713 |
| logit_bytes | 13.66 | 13.31 | 13.74 | 13.5 |
| opt_temp | 2.082 | 2.093 | 2.082 | 2.082 |
| overhead_gb | 0.1725 | 0.1382 | 0 | 0.4883 |

## 5. peak 시점

- full: optimizer step 시점이 peak인 설정 55/96
- lora: optimizer step 시점이 peak인 설정 0/96
- qlora: optimizer step 시점이 peak인 설정 0/96

## 6. 한계

- QLoRA는 bitsandbytes 없이 에뮬레이션(16bit Linear + fp32 norm/임베딩/lm_head) → 4bit 역양자화 임시버퍼는 미포함
- CPU의 SDPA(flash-for-CPU) 경로 기준. CUDA flash/efficient 커널의 임시 버퍼와 약간 다를 수 있음
- overhead(CUDA 컨텍스트, 캐시 할당자 단편화)는 이 방법으로 알 수 없음 → Colab 실측 필요
