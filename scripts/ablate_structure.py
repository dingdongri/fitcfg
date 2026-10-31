"""처음 보는 구조의 메모리 항을 GPU 없이 재는 절제 실험 (FakeTensor 추적, 2026-10-31).
결과 → fitcfg/memory.py 의 act_q(attention 폭) · norm_bytes(추가 norm) · softcap_bytes(logits softcapping)

    python scripts/ablate_structure.py          # CPU 10~20분

방법: 한 가지만 바꾼 설정을 층 2개·4개로 추적해 차이를 내면 '층 1개·토큰 1개당 저장 바이트'가 나온다.
  ① hidden 고정, head 수만 바꿈 → q_dim 1024/2048/4096 → 기울기 = act_q (측정 10.1 B)
  ② Gemma-2 층 vs 같은 크기 Llama 층 → 차이 / hidden = 추가 norm 2개분 (측정 8.0 B/H → norm 1개 4 B)
  ③ Gemma-2 logits softcapping 켬/끔 → 차이 / (B·S·V) (측정 2.0 B)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fitcfg.trace import trace_hf  # noqa: E402

B, S = 1, 1024


def act(cfg, method="lora", ckpt=False):
    r = trace_hf(cfg, method, B, S, ckpt, steps=1)
    return (r.peak_alloc_gb - r.mem_model_gb) * 2 ** 30


def per_layer(cfg, method="lora"):
    return (act(dict(cfg, num_hidden_layers=4), method) - act(dict(cfg, num_hidden_layers=2), method)) / 2 / (B * S)


# ① attention 폭
base = dict(model_type="llama", hidden_size=2048, intermediate_size=5632, num_key_value_heads=4, head_dim=128,
            vocab_size=32000, tie_word_embeddings=False, max_position_embeddings=4096)
pts = []
for nh in (8, 16, 32):
    v = per_layer(dict(base, num_attention_heads=nh))
    pts.append((nh * 128, v))
    print(f"① q_dim {nh * 128:5d}: {v:9.1f} B/토큰·층", flush=True)
slope = (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])
print(f"   → act_q ≈ {slope:.1f} B / (토큰·층·q_dim 1)")

# ② Gemma-2 추가 norm, ③ softcapping
g2 = dict(model_type="gemma2", vocab_size=256000, hidden_size=2304, intermediate_size=9216, num_attention_heads=8,
          num_key_value_heads=4, head_dim=256, sliding_window=4096, final_logit_softcapping=30.0,
          attn_logit_softcapping=50.0, query_pre_attn_scalar=256, hidden_activation="gelu_pytorch_tanh",
          tie_word_embeddings=True, max_position_embeddings=8192)
ll = dict(model_type="llama", vocab_size=256000, hidden_size=2304, intermediate_size=9216, num_attention_heads=8,
          num_key_value_heads=4, head_dim=256, tie_word_embeddings=True, max_position_embeddings=8192)
g_layer = per_layer(dict(g2, final_logit_softcapping=None))
l_layer = per_layer(ll)
print(f"② Gemma-2 층 {g_layer:.1f} vs 같은 크기 Llama 층 {l_layer:.1f} B/토큰 → 추가 {(g_layer - l_layer) / 2304:.1f} B/H")
on = act(dict(g2, num_hidden_layers=2))
off = act(dict(g2, num_hidden_layers=2, final_logit_softcapping=None))
print(f"③ softcapping 추가 {(on - off) / (B * S * g2['vocab_size']):.2f} B / logits 원소")
