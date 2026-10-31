# 4차 실측 (2026-10-29~30, gfarm A100 MIG 3g.40gb) — 추천대로 학습하면 되는가 + 처음 보는 모델 QLoRA

## 1. 추천 설정으로 실제 학습 (도구가 만든 학습 스크립트 그대로)

- 6개 시나리오: Llama-3.1-8B QLoRA·LoRA (2048), Qwen2.5-7B QLoRA (4096)·LoRA (2048), Mistral-7B LoRA (4096), Llama-3.2-1B Full (2048)
- 각 시나리오: HF Trainer 기본값 + fitcfg 추천 1~3위, 실제 데이터(alpaca) · HF Trainer · 10 스텝 (앞 3스텝은 속도 측정 제외)

| 항목 | 결과 |
|---|---|
| HF 기본값 → 메모리 부족 | **6/6** |
| 추천 설정 학습 성공 | **17/17** |
| 추천 설정 peak 메모리 예측 오차 | 평균 3.1% (−7.8 ~ +7.0%) |
| 추천 1위가 실제로 가장 빠름 | **6/6** |
| 추천끼리 빠르기 순서 (예측 차이 3% 이상인 쌍) | **11/11** |
| 속도 예측 오차 | 평균 6.9% (7B·8B는 대부분 ±6%, 1B Full만 −17~−22%: 실제가 더 빠름) |

- 한도 근처 사례: Qwen-7B LoRA 추천 2위 실측 34.94GB (안전 한도 35.1GB, GPU 39.25GB) — 예측 −4.5%를 안전계수 0.895가 정확히 흡수
- 메모리를 많이 쓰는 설정에서 예측이 조금 작게 나오는 경향(−3~−8%) → 안전계수를 유지해야 하는 근거
- 상세: `reports/e2e_report.md`, 원자료 `data/bundle_runs_mig.jsonl`

## 2. 처음 보는 모델 4종 (보정에 안 씀) — 63개 설정 (성공 59 · OOM 4)

| 모델 | 구조 특징 | 예측식 LoRA | 예측식 QLoRA | CPU 흉내 LoRA |
|---|---|---|---|---|
| SmolLM2-1.7B | Llama형, 어휘 49K | 1.6% | 6.5% (전부 과대) | 0.9% |
| OLMo-2 1B | 정규화 위치 다름, q·k 정규화 | 4.8% | 5.8% | 0.6% |
| Granite-3.3 2B | 배율 상수, 40층 | 2.8% | 4.5% (전부 과대) | 2.2% |
| Qwen3-4B | q·k 정규화, head_dim ≠ hidden/heads | 4.6% | 4.7% (전부 과소) | 1.9% |
| **전체** | | | **4.4%** (59개) | **1.4%** (30개) |

- OOM 판정: 63개 중 62개 정답, 안전계수 0.895 적용 시 **놓친 OOM 0**
- CPU 흉내 실행(GPU 없이)은 처음 보는 구조에서도 LoRA 1~2% → 지난번 Gemma(0.3~0.7%)와 합쳐 **8개 계열 모두 확인**

### QLoRA 규칙: "QLoRA 학습 중 메모리 ≈ LoRA의 1.3배"

- QLoRA는 4bit라 CPU로 흉내 불가 → CPU로 LoRA를 흉내 낸 값에 비율을 곱하고, 가중치(4bit, ±0.3% 정확)를 더함
- 비율 (학습 중 메모리 = peak − 모델 가중치):

| 계열 | checkpointing 끔 | 켬 |
|---|---|---|
| 기존 4계열 (Gemma-2·3, Phi-3, Mistral) 평균 | 1.27 | 1.31 |
| 새 4계열 (SmolLM2, OLMo-2, Granite, Qwen3) | 1.23 ~ 1.35 | 1.17 ~ 1.30 |
| **8계열 전체** | **1.29 (±0.06)** | 1.27 (±0.15) |

- 기존 4계열로 정한 비율로 새 4계열 예측 (진짜 held-out): 평균 5.2% (checkpointing 끔 3.6%, 켬 6.5%), 대부분 과대예측(안전한 쪽)
- 결론: 처음 보는 구조 + QLoRA는 **CPU LoRA 흉내 × 1.3 + 4bit 가중치**로 5% 안팎. checkpointing 켬은 흔들림이 커서 여유를 조금 더 둠

## 3. 서버 환경에서 겪은 문제 (재현용, `scripts/setup_gfarm.sh`에 모두 반영)

| 증상 | 원인 | 조치 |
|---|---|---|
| `Too few arguments for numpy.ndarray` | 이미지의 numpy 1.22 | numpy·pandas·scipy·sklearn 올림 |
| `Could not import module 'Trainer'/'BloomPreTrainedModel'` | librosa가 numba(제거됨)에 의존 | librosa·RAPIDS(cudf·cupy) 제거 |
| `KeyError: 'maxdepth'` | 구버전 fsspec | datasets·huggingface_hub·fsspec·pyarrow 범위 지정 설치 |
| 같은 실험 2개 동시 실행 | 터미널에서 두 번 실행 | `[1]`/`[2]` 작업 번호 확인 후 하나만 |
