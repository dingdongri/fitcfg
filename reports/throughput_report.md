# 처리량 예측 v2 (2026-10-30) — 단일 GPU 실측으로 식 개선

v1 식: `처리량 = peak × 효율(방법) × 활용도(토큰 수) ÷ 토큰당 FLOPs`. v2에서 실측 잔차가 보인 세 가지를 추가:

| 추가 항 | 근거 (v1 잔차) | 값 (A100 MIG) |
|---|---|---|
| 크기 효율 α: 효율 × (H·I ÷ Llama-8B)^α | 1B·3B는 과대, 7B·8B는 과소 예측 | α = 0.2 |
| 재계산 비용 γ (forward 대비) | checkpointing 켬 과소 / 끔 과대 | γ = 0.8 |
| optimizer step = 학습 파라미터 × 바이트 ÷ 대역폭 | 1B Full: 측정(누적 1) 과대·번들(누적 8~16) 과소 | AdamW 48 B/param, 8bit 96 B/param |
| FlashAttention 없는 GPU의 attention 비용 κ | T4에서 seq가 길수록 과대 예측 | κ = 16 (T4) |

## 오차 (평균 절대 오차 %, 괄호는 최대)

| GPU | 모델 수 | in-sample v1 → v2 | 모델 빼고 맞추기 v1 → v2 | 실제 학습 번들 v1 → v2 |
|---|---|---|---|---|
| t4 | 2 | 19.4 (64) → **4.3** (14) | 29.1 (103) → **23.9** (105) | — |
| a100-mig3g40 | 5 | 10.4 (34) → **3.2** (24) | 12.6 (40) → **3.9** (27) | 6.9 (22) → **4.0** (7) |

- 실제 학습 번들(HF Trainer + 실제 데이터, 누적 4~16)은 fit에 쓰지 않은 검증 데이터
- T4는 모델이 2개뿐이라 '모델 빼고 맞추기'가 곧 '한 모델로 다른 모델 맞추기' → 참고용
- 측정 안 한 GPU(RTX 3090·4090 등)는 A100 MIG 상수를 쓰고 범위를 ±50%로 넓게 표시 (`default`)

## 상수

```json
{
 "t4": {
  "eff_full": 0.4618,
  "eff_lora": 0.4623,
  "eff_qlora": 0.4061,
  "t_half": 22.6274,
  "ddp_eff": 0.9,
  "zero3_eff": 0.8,
  "cpu_adam_params_per_s": 500000000.0,
  "band": 1.283,
  "alpha": 0.6,
  "recompute": 0.9,
  "attn_nonflash": 16.0,
  "opt_bpp": 0.0,
  "opt8_bpp": 64.0
 },
 "a100-mig3g40": {
  "eff_full": 0.7391,
  "eff_lora": 0.5717,
  "eff_qlora": 0.4704,
  "t_half": 128.0,
  "ddp_eff": 0.9,
  "zero3_eff": 0.8,
  "cpu_adam_params_per_s": 500000000.0,
  "band": 1.225,
  "alpha": 0.2,
  "recompute": 0.8,
  "attn_nonflash": 1.0,
  "opt_bpp": 48.0,
  "opt8_bpp": 96.0
 },
 "default": {
  "eff_full": 0.7391,
  "eff_lora": 0.5717,
  "eff_qlora": 0.4704,
  "t_half": 128.0,
  "ddp_eff": 0.9,
  "zero3_eff": 0.8,
  "cpu_adam_params_per_s": 500000000.0,
  "band": 1.5,
  "alpha": 0.2,
  "recompute": 0.8,
  "attn_nonflash": 1.0,
  "opt_bpp": 48.0,
  "opt8_bpp": 96.0
 }
}
```
