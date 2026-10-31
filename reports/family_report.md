# 처음 보는 모델 계열 일반화 검증 (FakeTensor, GPU 없이)

- 상수: 기존 4개 모델(Llama-3.2-1B, Qwen2.5-3B/7B, Llama-3.1-8B) 추적 288행으로만 보정
- 대상: 보정에 쓰지 않은 Mistral-7B-v0.3, Phi-3-mini, Gemma-2-2B, Gemma-3-1B
- 비교 기준: 기존 4개 모델끼리 '모델 하나 빼고' 검증한 오차 = 1.1%

### 모델별

| 구분 | n | MAPE | 최대 | 과소예측 비율 |
|---|---|---|---|---|
| gemma-2-2b (eager) | 48 | 15.3% | 41.2% | 96% |
| gemma-2-2b (sdpa) | 48 | 9.5% | 19.0% | 96% |
| gemma-3-1b (sdpa) | 48 | 4.2% | 14.5% | 100% |
| mistral-7b-v0.3 (sdpa) | 48 | 1.6% | 7.1% | 6% |
| phi-3-mini (sdpa) | 48 | 2.4% | 8.4% | 25% |

### 모델 × 학습 방식

| 구분 | n | MAPE | 최대 | 과소예측 비율 |
|---|---|---|---|---|
| gemma-2-2b (eager) · full | 16 | 12.5% | 41.2% | 88% |
| gemma-2-2b (eager) · lora | 16 | 15.8% | 40.7% | 100% |
| gemma-2-2b (eager) · qlora | 16 | 17.5% | 40.5% | 100% |
| gemma-2-2b (sdpa) · full | 16 | 7.9% | 19.0% | 88% |
| gemma-2-2b (sdpa) · lora | 16 | 9.2% | 13.5% | 100% |
| gemma-2-2b (sdpa) · qlora | 16 | 11.3% | 13.7% | 100% |
| gemma-3-1b (sdpa) · full | 16 | 5.4% | 14.5% | 100% |
| gemma-3-1b (sdpa) · lora | 16 | 3.0% | 7.6% | 100% |
| gemma-3-1b (sdpa) · qlora | 16 | 4.1% | 8.5% | 100% |
| mistral-7b-v0.3 (sdpa) · full | 16 | 1.0% | 1.9% | 6% |
| mistral-7b-v0.3 (sdpa) · lora | 16 | 0.6% | 1.9% | 12% |
| mistral-7b-v0.3 (sdpa) · qlora | 16 | 3.3% | 7.1% | 0% |
| phi-3-mini (sdpa) · full | 16 | 2.1% | 8.4% | 38% |
| phi-3-mini (sdpa) · lora | 16 | 0.7% | 1.8% | 38% |
| phi-3-mini (sdpa) · qlora | 16 | 4.3% | 6.8% | 0% |

### 모델 × checkpointing

| 구분 | n | MAPE | 최대 | 과소예측 비율 |
|---|---|---|---|---|
| gemma-2-2b (eager) · ckpt0 | 24 | 22.7% | 41.2% | 96% |
| gemma-2-2b (eager) · ckpt1 | 24 | 7.9% | 11.5% | 96% |
| gemma-2-2b (sdpa) · ckpt0 | 24 | 10.7% | 19.0% | 96% |
| gemma-2-2b (sdpa) · ckpt1 | 24 | 8.3% | 12.4% | 96% |
| gemma-3-1b (sdpa) · ckpt0 | 24 | 6.8% | 14.5% | 100% |
| gemma-3-1b (sdpa) · ckpt1 | 24 | 1.5% | 2.9% | 100% |
| mistral-7b-v0.3 (sdpa) · ckpt0 | 24 | 1.6% | 3.9% | 8% |
| mistral-7b-v0.3 (sdpa) · ckpt1 | 24 | 1.6% | 7.1% | 4% |
| phi-3-mini (sdpa) · ckpt0 | 24 | 2.6% | 8.4% | 25% |
| phi-3-mini (sdpa) · ckpt1 | 24 | 2.2% | 6.8% | 25% |

### 모델 × 시퀀스 길이

| 구분 | n | MAPE | 최대 | 과소예측 비율 |
|---|---|---|---|---|
| gemma-2-2b (eager) · seq 512 | 12 | 7.9% | 14.4% | 83% |
| gemma-2-2b (eager) · seq1024 | 12 | 11.3% | 20.5% | 100% |
| gemma-2-2b (eager) · seq2048 | 12 | 17.6% | 30.1% | 100% |
| gemma-2-2b (eager) · seq4096 | 12 | 24.3% | 41.2% | 100% |
| gemma-2-2b (sdpa) · seq 512 | 12 | 6.6% | 10.8% | 83% |
| gemma-2-2b (sdpa) · seq1024 | 12 | 8.2% | 13.4% | 100% |
| gemma-2-2b (sdpa) · seq2048 | 12 | 10.5% | 16.4% | 100% |
| gemma-2-2b (sdpa) · seq4096 | 12 | 12.5% | 19.0% | 100% |
| gemma-3-1b (sdpa) · seq 512 | 12 | 2.6% | 8.3% | 100% |
| gemma-3-1b (sdpa) · seq1024 | 12 | 3.8% | 10.5% | 100% |
| gemma-3-1b (sdpa) · seq2048 | 12 | 4.6% | 12.4% | 100% |
| gemma-3-1b (sdpa) · seq4096 | 12 | 5.8% | 14.5% | 100% |
| mistral-7b-v0.3 (sdpa) · seq 512 | 12 | 1.4% | 3.5% | 0% |
| mistral-7b-v0.3 (sdpa) · seq1024 | 12 | 1.7% | 4.0% | 0% |
| mistral-7b-v0.3 (sdpa) · seq2048 | 12 | 1.8% | 5.4% | 0% |
| mistral-7b-v0.3 (sdpa) · seq4096 | 12 | 1.6% | 7.1% | 25% |
| phi-3-mini (sdpa) · seq 512 | 12 | 2.1% | 5.0% | 0% |
| phi-3-mini (sdpa) · seq1024 | 12 | 2.3% | 5.5% | 17% |
| phi-3-mini (sdpa) · seq2048 | 12 | 2.2% | 6.2% | 25% |
| phi-3-mini (sdpa) · seq4096 | 12 | 2.8% | 8.4% | 58% |
