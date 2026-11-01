# GPU 2장 실측 (2026-10-31, Kaggle 무료 Tesla T4 ×2, PCIe) — 처음으로 진짜 GPU 여러 장

도구가 만든 학습 번들 그대로(HF Trainer · alpaca · 12스텝), 5개 시나리오. 원자료 `data/bundle_runs_kaggle.jsonl`,
시나리오별 표 `reports/e2e_kaggle_report.md`, 노트북 `notebooks/kaggle_t4x2.ipynb`.

## 요약

| 항목 | 결과 |
|---|---|
| 추천 설정 학습 성공 | **7/7** (DDP 3 · ZeRO-2 1 · ZeRO-3 1 · GPU 1장 2) |
| HF 기본값 | 3/3 OOM (예측대로) |
| 메모리 예측 (GPU당) | DDP·GPU 1장 +3.6~+5.9% (모두 약간 크게 = 안전) · ZeRO-3 +16% · ZeRO-2 −9% → 수정 후 −2.5% |
| GPU 2장 확장 효율 (LoRA·QLoRA DDP) | GPU 1장 대비 **정확히 2.0배** (1B LoRA 1,615 vs 808 · 3B QLoRA 800 vs 400 tok/s) |
| 속도 예측 | QLoRA ±0.4% · ZeRO-3 +2% · Full ZeRO-2 +35% → 수정 후 +0.5% · 1B LoRA는 1장·2장 모두 +14% (T4 기본 보정 편향, 확장 문제 아님) |
| 추천 순위 | 예측 차이 3% 이상인 쌍 2/2 정답 · 1·2위는 실측 1.5% 차이(사실상 동률) |

## 발견과 수정

1. **GPU 1장 설정을 GPU 여러 장 머신에서 실행하면** HF Trainer가 자동으로 `nn.DataParallel`로 모든 GPU를 씀 →
   메모리 +50%, 속도 1/4 (1차 실행: Qwen-3B QLoRA 13.0GB·190 tok/s vs 예측 8.7GB·401). 생성 `launch.sh`가 GPU 1장으로 고정
   (`CUDA_VISIBLE_DEVICES`) → 2차 실행 8.2GB·400 tok/s로 예측과 일치. **실제 사용자도 겪을 버그를 실측으로 찾음.**
2. **통신 시간을 식에 넣음** (`throughput.py`): 예전엔 "장수가 2배면 효율 0.9배" 문헌 배수.
   이제 '보내는 바이트 ÷ 실효 대역폭' — DDP는 학습 파라미터 기울기를 optimizer step마다, ZeRO-2는 마이크로스텝마다,
   ZeRO-3은 파라미터 all-gather까지. LoRA는 기울기가 작아 통신이 거의 0 → 실측 2.0배와 일치,
   Full ZeRO-2는 통신이 시간의 1/3 → +35% → +0.5%. 실효 대역폭 4.9 GB/s(PCIe, P2P 없음)는 이 1건으로 정함 → 다른 머신에서 재확인 필요.
3. **DeepSpeed ZeRO-1/2의 all-gather 버퍼**(기본 5억 원소, 16bit ≈ 0.93GB)를 메모리 식에 추가 → ZeRO-2 −9% → −2.5%.
4. Kaggle 이미지의 torchao 0.10이 최신 transformers·peft LoRA와 충돌 → 노트북에서 제거 (Colab 때와 같은 문제).

## 한계
- T4 ×2(PCIe) 한 종류. NVLink(A100·H100) 대역폭은 문헌값 100 GB/s, 3D 병렬화(TP·PP)·혼합 GPU는 여전히 계산 기반.
- ZeRO-3 메모리는 단편화 여유를 크게 잡아 +16% (안전한 쪽). 실측 2건(MIG offload·T4 LoRA)뿐.

## 추가: HF Trainer 플러그인을 GPU 2장(torchrun DDP)에 붙여서 (`scripts/plugin_ddp_check.py`, `data/plugin_runs_ddp.jsonl`)

| 시나리오 | 감지 | 결과 | 예측 GB | 실측 GB | 오차 |
|---|---|---|---|---|---|
| 1B LoRA mb2 · GPU 2장 | lora · GPU 2장 | 학습 완료 | 11.15 | 10.85 | +2.7% |
| 1B LoRA mb2 · GPU 1장 | lora · GPU 1장 | 학습 완료 | 11.14 | 10.64 | +4.7% |
| 3B QLoRA mb1 · GPU 2장 | qlora · GPU 2장 | 학습 완료 | 8.73 | 8.43 | +3.6% |
| 1B LoRA mb8 ckpt off · GPU 2장 · strict | lora · GPU 2장 | **학습 전 중단 (2.5초)** | 114.5 | — | — |

- GPU 수·학습 방식·checkpointing을 torchrun 환경에서도 정확히 읽음 (4/4), 메모리 예측 +2.7~+4.7% (안전한 쪽)
- strict 중단 시 제안한 설정(micro-batch 1 × 누적 32 × GPU 2)은 원래 global batch 64를 그대로 유지
- **발견 → 플러그인 기능 추가**: 테스트 스크립트가 T4에서 bf16을 켰는데(최신 PyTorch는 구형 GPU에서도 bf16을
  '에뮬레이션'으로 허용) 오류 없이 약 3배 느려짐 (같은 설정 fp16 번들 808 tok/s vs bf16 264 tok/s). 사용자도 쉽게 하는 실수라
  플러그인이 bf16 미지원 GPU에서 bf16을 켜면 "fp16=True 권장" 경고를 내도록 함. (GPU 2장 / 1장 속도비는 1.98배로 확장 결과와 일치)

## 추가: T4 처리량 재보정 검토
- Kaggle 1B LoRA(seq 2048)가 +14% 높게 예측된 건 Colab에서 잰 같은 GPU·같은 설정(865 tok/s)보다 Kaggle(808)이 7% 느린 환경 차이가 절반
- Kaggle 데이터를 넣어 다시 맞춰도 13.8% → 12.3%로 거의 그대로 → 보정값은 유지하고 "같은 GPU라도 환경에 따라 ±7%"를 한계로 기록
