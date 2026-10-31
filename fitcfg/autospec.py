"""처음 보는 모델 자동 처리 — 모델 설계도(config.json)만으로 예측 + 구조 유형에 따른 여유.

    from fitcfg import get_model, structure_of
    spec = get_model("Qwen/Qwen3-4B-Base")        # HF 이름·로컬 config.json·폴더 모두 가능
    info = structure_of("Qwen/Qwen3-4B-Base")     # 구조 유형·여유 계수·안내 문구

구조 유형 (gfarm A100 실측으로 정한 처리 — reports/round4_report.md, reports/mig_report.md)
    verified   : 직접 측정해 보정한 모델 (Llama·Qwen2.5·Mistral)            → 그대로 (오차 3~5%)
    llama-like : 디코더 + GQA + SwiGLU 계열 (Phi-3·Qwen3·OLMo-2·SmolLM2·Granite 등)
                 처음 보는 8개 계열 실측: 예측식 평균 4.4%, 안전계수로 놓친 OOM 0 → 그대로 + 안내
    gemma-like : Gemma 계열 (head_dim 256, 큰 어휘, softcapping) — 예측식이 9~23% 적게 예측
                 → 예측 메모리에 여유(LoRA·Full ×1.15, QLoRA ×1.25) + CPU 직접 측정 권장
    moe        : 전문가 혼합(Mixtral·Qwen-MoE·DeepSeek-V2 등) → 지원 안 함 (메모리 구조가 다름)
    unknown    : 위에 없는 구조 → 여유(×1.10 / ×1.20) + CPU 직접 측정 권장
"""
from __future__ import annotations
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .models import ModelSpec

LLAMA_LIKE = {"llama", "mistral", "qwen2", "qwen3", "phi3", "olmo2", "olmo", "granite", "smollm3", "yi",
              "deepseek", "internlm2", "internlm3", "exaone", "cohere", "starcoder2", "stablelm", "glm", "glm4",
              "ministral", "seed_oss", "hunyuan_v1_dense", "apertus"}
GEMMA_LIKE = {"gemma", "gemma2", "gemma3", "gemma3_text"}
MOE_KEYS = ("num_local_experts", "num_experts", "n_routed_experts", "moe_intermediate_size", "num_experts_per_tok")

MARGINS = {   # (LoRA·Full, QLoRA) — 예측 메모리에 곱함
    "verified": (1.0, 1.0), "llama-like": (1.0, 1.0), "gemma-like": (1.15, 1.25), "unknown": (1.10, 1.20),
}


@dataclass(frozen=True)
class StructureInfo:
    kind: str                 # verified / llama-like / gemma-like / moe / unknown
    model_type: str
    note: str
    margin_lora: float = 1.0
    margin_qlora: float = 1.0

    def margin(self, method: str) -> float:
        return self.margin_qlora if method == "qlora" else self.margin_lora

    @property
    def supported(self) -> bool:
        return self.kind != "moe"


_CACHE: dict[str, tuple[ModelSpec, StructureInfo]] = {}
_CFG: dict[str, dict] = {}          # 등록·조회한 설계도 원본 (CPU 측정이 다시 쓰도록)


# ---------------------------------------------------------------------------
# 설계도 읽기
# ---------------------------------------------------------------------------
def fetch_config(name: str, token: str | None = None, timeout: float = 20.0) -> dict:
    """로컬 config.json / 모델 폴더 / Hugging Face 이름(org/model) → config dict.
    승인(gated) 모델은 HF_TOKEN 환경변수(또는 token 인자)가 필요하다. 가중치는 받지 않는다."""
    if name in _CFG:
        return _CFG[name]
    p = Path(name).expanduser()
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    if p.is_dir() and (p / "config.json").exists():
        return json.loads((p / "config.json").read_text(encoding="utf-8"))
    url = f"https://huggingface.co/{name}/resolve/main/config.json"
    req = urllib.request.Request(url)
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        hint = " (승인이 필요한 모델이면 HF_TOKEN 환경변수를 설정하세요)" if e.code in (401, 403) else ""
        raise KeyError(f"'{name}'의 config.json을 가져오지 못했습니다: HTTP {e.code}{hint}") from e


def _text_cfg(cfg: dict) -> dict:
    """멀티모달 모델은 text_config 안에 언어 모델 구조가 있다."""
    if "text_config" in cfg and isinstance(cfg["text_config"], dict) and "hidden_size" not in cfg:
        out = dict(cfg["text_config"])
        out.setdefault("model_type", cfg.get("model_type", ""))
        return out
    return cfg


def classify(cfg: dict, verified: bool = False) -> StructureInfo:
    c = _text_cfg(cfg)
    mt = str(c.get("model_type", "")).lower()
    if verified:
        return StructureInfo("verified", mt, "직접 측정해 보정한 모델 (gfarm A100 · Colab T4 실측)")
    if any(k in c and c[k] for k in MOE_KEYS) or "moe" in mt:
        return StructureInfo("moe", mt, "전문가 혼합(MoE) 모델은 메모리 구조가 달라 지원하지 않습니다.")
    if mt in GEMMA_LIKE:
        lo, q = MARGINS["gemma-like"]
        return StructureInfo("gemma-like", mt,
                             "Gemma 계열은 예측식이 9~23% 적게 예측(실측) → 여유를 더 두었습니다. "
                             "정확한 값은 `fitcfg measure-cpu`로 확인하세요.", lo, q)
    if mt in LLAMA_LIKE:
        return StructureInfo("llama-like", mt,
                             "처음 보는 모델(Llama형 구조): 처음 보는 8개 계열 실측 기준 오차 2~6%, 안전 여유로 OOM 0건.")
    lo, q = MARGINS["unknown"]
    return StructureInfo("unknown", mt or "?",
                         f"검증하지 않은 구조({mt or '알 수 없음'}) → 여유를 더 두었습니다. "
                         f"`fitcfg measure-cpu`로 직접 확인을 권장합니다.", lo, q)


def spec_from_config(cfg: dict, name: str) -> ModelSpec:
    """config.json → ModelSpec. 파라미터 수는 구조에서 계산 (가중치 다운로드 없음, 실제 값과 ±1%)."""
    c = _text_cfg(cfg)
    H = int(c["hidden_size"])
    L = int(c.get("num_hidden_layers") or c["n_layer"])
    nh = int(c.get("num_attention_heads") or c["n_head"])
    nkv = int(c.get("num_key_value_heads") or nh)
    hd = int(c.get("head_dim") or H // nh)
    I = int(c.get("intermediate_size") or 4 * H)
    V = int(c["vocab_size"])
    tied = bool(c.get("tie_word_embeddings", True if str(c.get("model_type", "")).startswith("gemma") else False))
    attn = H * nh * hd + 2 * H * nkv * hd + nh * hd * H
    if c.get("attention_bias") or str(c.get("model_type", "")) == "qwen2":
        attn += nh * hd + 2 * nkv * hd
    mlp = 3 * H * I                                   # gate·up·down (SwiGLU / GeGLU)
    per_layer = attn + mlp + 4 * H
    n = L * per_layer + V * H * (1 if tied else 2) + H
    return ModelSpec(name=name, n_params=int(n), n_layers=L, hidden=H, n_heads=nh, n_kv_heads=nkv,
                     vocab=V, intermediate=I, tied_embeddings=tied,
                     attn_head_dim=hd if hd != H // nh else None)


def resolve(name: str, token: str | None = None) -> tuple[ModelSpec, StructureInfo]:
    """등록 안 된 모델 이름 → (ModelSpec, StructureInfo). 한 번 읽으면 캐시."""
    key = name.strip()
    if key in _CACHE:
        return _CACHE[key]
    cfg = fetch_config(key, token)
    info = classify(cfg)
    if not info.supported:
        raise ValueError(f"{key}: {info.note}")
    spec = spec_from_config(cfg, key)
    _CACHE[key] = (spec, info)
    _CFG[key] = cfg
    return spec, info


def register(name: str, cfg: dict) -> tuple[ModelSpec, StructureInfo]:
    """설계도 dict를 직접 등록 (오프라인·테스트용)."""
    info = classify(cfg)
    if not info.supported:
        raise ValueError(f"{name}: {info.note}")
    _CACHE[name] = (spec_from_config(cfg, name), info)
    _CFG[name] = cfg
    return _CACHE[name]


# 이미 아는 모델의 구조 유형
_KNOWN = {
    "phi-3-mini": "phi3", "qwen3-4b": "qwen3", "olmo2-1b": "olmo2", "smollm2-1.7b": "llama",
    "granite-3.3-2b": "granite", "gemma-2-2b": "gemma2", "gemma-3-1b": "gemma3_text",
}


def structure_of(key: str) -> StructureInfo:
    from .models import MODELS
    k = key.lower()
    if k in MODELS:
        return classify({"model_type": "llama"}, verified=True)
    if k in _KNOWN:
        return classify({"model_type": _KNOWN[k]})
    if key in _CACHE:
        return _CACHE[key][1]
    return resolve(key)[1]
