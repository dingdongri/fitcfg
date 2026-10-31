"""Model architecture presets.

cost model이 메모리를 계산하려면 모델의 구조 정보(파라미터 수, 레이어 수,
hidden size 등)가 필요하다. 여기에 스코프에 포함한 계열만 등록한다.

새 모델을 추가하려면 아래 dict에 한 줄 추가하면 된다.
(값은 각 모델의 HF config.json 기준)
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str              # HF repo id
    n_params: int          # 전체 파라미터 수 (embedding 포함, tied면 1번만 셈)
    n_layers: int          # transformer 블록 수 (L)
    hidden: int            # hidden size (H)
    n_heads: int           # attention head 수
    n_kv_heads: int        # KV head 수 (GQA면 n_heads보다 작음)
    vocab: int             # vocabulary 크기 (V)
    intermediate: int      # MLP intermediate size (I)
    tied_embeddings: bool = False   # 입력 임베딩과 lm_head가 같은 행렬을 공유하는지
    attn_head_dim: int | None = None  # head 하나의 차원. None이면 hidden/heads (Qwen3·Gemma는 다름)

    @property
    def head_dim(self) -> int:
        return self.attn_head_dim or self.hidden // self.n_heads

    @property
    def q_dim(self) -> int:
        """q_proj 출력 차원 = attention 내부 폭 (보통 hidden과 같지만 Qwen3-4B는 4096 vs 2560)."""
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        """k_proj / v_proj 의 출력 차원 (GQA)."""
        return self.n_kv_heads * self.head_dim

    @property
    def embed_params(self) -> int:
        """임베딩 + lm_head 파라미터 수.
        QLoRA에서 이 부분은 4bit 양자화되지 않는다(bitsandbytes가 lm_head를 건너뛰고,
        임베딩은 Linear가 아님). 그래서 메모리 계산에서 따로 다룬다."""
        return self.vocab * self.hidden * (1 if self.tied_embeddings else 2)

    @property
    def linear_params(self) -> int:
        """임베딩/lm_head를 제외한 나머지(= 대부분 transformer 블록의 Linear)."""
        return self.n_params - self.embed_params


# 스코프: 최신 오픈모델의 표준 구조(디코더 전용 + GQA + SwiGLU)를 대표하는 2계열.
MODELS: dict[str, ModelSpec] = {
    "llama-3.1-8b": ModelSpec(
        name="meta-llama/Llama-3.1-8B",
        n_params=8_030_000_000,
        n_layers=32, hidden=4096, n_heads=32, n_kv_heads=8,
        vocab=128_256, intermediate=14_336, tied_embeddings=False,
    ),
    "qwen2.5-7b": ModelSpec(
        name="Qwen/Qwen2.5-7B",
        n_params=7_620_000_000,
        n_layers=28, hidden=3584, n_heads=28, n_kv_heads=4,
        vocab=152_064, intermediate=18_944, tied_embeddings=False,
    ),
    # --- 초반 검증(무료 GPU)용 소형 모델. 같은 계열이라 방법론 검증에 적합 ---
    # 소형 모델은 입력 임베딩과 lm_head를 공유(tied)한다.
    "llama-3.2-1b": ModelSpec(
        name="meta-llama/Llama-3.2-1B",
        n_params=1_240_000_000,
        n_layers=16, hidden=2048, n_heads=32, n_kv_heads=8,
        vocab=128_256, intermediate=8192, tied_embeddings=True,
    ),
    # --- 어휘(vocab)가 작은 모델 (32K). 128K~152K 모델만으로는 어휘 비례 항과 층 비례 항이 구분되지 않아 추가 (2026-10-21) ---
    "mistral-7b-v0.3": ModelSpec(
        name="mistralai/Mistral-7B-v0.3",   # gated → 측정은 공개 사본 unsloth/mistral-7b-v0.3
        n_params=7_248_023_552,
        n_layers=32, hidden=4096, n_heads=32, n_kv_heads=8,
        vocab=32_768, intermediate=14_336, tied_embeddings=False,
    ),
    "qwen2.5-3b": ModelSpec(
        name="Qwen/Qwen2.5-3B",
        n_params=3_090_000_000,
        n_layers=36, hidden=2048, n_heads=16, n_kv_heads=2,
        vocab=151_936, intermediate=11_008, tied_embeddings=True,
    ),
}


# 일반화 검증 전용(보정에 쓰지 않음 — calibration.py는 MODELS만 읽는다): 처음 보는 계열을 실측해 오차를 보는 용도.
# 구조 값은 config.json, 파라미터 수는 meta 장치에서 센 값. 이름은 승인 없이 받을 수 있는 공개 사본.
EVAL_MODELS: dict[str, ModelSpec] = {
    "phi-3-mini": ModelSpec(       # q·k·v가 한 행렬(qkv_proj), gate·up도 한 행렬(gate_up_proj)
        name="microsoft/Phi-3-mini-4k-instruct", n_params=3_821_079_552,
        n_layers=32, hidden=3072, n_heads=32, n_kv_heads=32, vocab=32_064, intermediate=8192),
    "gemma-2-2b": ModelSpec(       # head_dim 256 ≠ hidden/heads, logit softcapping
        name="unsloth/gemma-2-2b", n_params=2_614_341_888,
        n_layers=26, hidden=2304, n_heads=8, n_kv_heads=4, vocab=256_000, intermediate=9216,
        tied_embeddings=True, attn_head_dim=256),
    # --- 2026-10-29 추가: QLoRA 일반화 규칙 검증용 (구조가 조금씩 다른 최신 계열) ---
    "qwen3-4b": ModelSpec(         # q·k 정규화(qk-norm), head_dim 128 ≠ hidden/heads(80)
        name="Qwen/Qwen3-4B-Base", n_params=4_022_468_096,
        n_layers=36, hidden=2560, n_heads=32, n_kv_heads=8, vocab=151_936, intermediate=9728,
        tied_embeddings=True, attn_head_dim=128),
    "olmo2-1b": ModelSpec(         # 정규화 위치가 다름(층 출력 뒤), q·k 정규화
        name="allenai/OLMo-2-0425-1B", n_params=1_484_916_736,
        n_layers=16, hidden=2048, n_heads=16, n_kv_heads=16, vocab=100_352, intermediate=8192),
    "smollm2-1.7b": ModelSpec(     # Llama형, 작은 어휘(49K)
        name="HuggingFaceTB/SmolLM2-1.7B", n_params=1_711_376_384,
        n_layers=24, hidden=2048, n_heads=32, n_kv_heads=32, vocab=49_152, intermediate=8192,
        tied_embeddings=True),
    "granite-3.3-2b": ModelSpec(   # Llama형 + 배율 상수(임베딩·잔차·logits), 층 40개
        name="ibm-granite/granite-3.3-2b-base", n_params=2_533_525_504,
        n_layers=40, hidden=2048, n_heads=32, n_kv_heads=8, vocab=49_152, intermediate=8192,
        tied_embeddings=True),
    "gemma-3-1b": ModelSpec(
        name="unsloth/gemma-3-1b-pt", n_params=999_885_952,
        n_layers=26, hidden=1152, n_heads=4, n_kv_heads=1, vocab=262_144, intermediate=6912,
        tied_embeddings=True, attn_head_dim=256),
}


def get_model(key: str) -> ModelSpec:
    """등록 모델 키(예: llama-3.1-8b) 또는 처음 보는 모델(HF 이름 org/model · config.json 경로).
    처음 보는 모델은 설계도(config.json)만 읽어 구조를 만든다 (fitcfg.autospec)."""
    k = key.lower()
    if k in EVAL_MODELS:
        return EVAL_MODELS[k]
    if k in MODELS:
        return MODELS[k]
    if "/" in key or key.endswith(".json"):
        from .autospec import resolve
        return resolve(key)[0]
    raise KeyError(f"unknown model '{key}'. available: {list(MODELS)} 또는 HF 이름(org/model)·config.json 경로")
