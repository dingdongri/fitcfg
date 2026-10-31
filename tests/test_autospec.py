"""처음 보는 모델 자동 처리 (fitcfg.autospec) — 네트워크·GPU 불필요.  실행: python -m pytest -q tests/test_autospec.py"""
import json, os, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fitcfg import PlanRequest, get_gpu, get_model, plan, reset_calibration, register, structure_of
from fitcfg.autospec import classify, spec_from_config, fetch_config, resolve

QWEN3_4B = dict(model_type="qwen3", vocab_size=151936, hidden_size=2560, intermediate_size=9728,
                num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                tie_word_embeddings=True)
GEMMA2_2B = dict(model_type="gemma2", vocab_size=256000, hidden_size=2304, intermediate_size=9216,
                 num_hidden_layers=26, num_attention_heads=8, num_key_value_heads=4, head_dim=256,
                 tie_word_embeddings=True)
LLAMA_8B = dict(model_type="llama", vocab_size=128256, hidden_size=4096, intermediate_size=14336,
                num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8, tie_word_embeddings=False)
QWEN25_7B = dict(model_type="qwen2", vocab_size=152064, hidden_size=3584, intermediate_size=18944,
                 num_hidden_layers=28, num_attention_heads=28, num_key_value_heads=4, tie_word_embeddings=False)
MIXTRAL = dict(model_type="mixtral", vocab_size=32000, hidden_size=4096, intermediate_size=14336,
               num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8, num_local_experts=8)


def setup_function(_):
    reset_calibration()


@pytest.mark.parametrize("cfg,n_true", [
    (QWEN3_4B, 4_022_468_096),     # HF 실제 파라미터 수 (meta 장치에서 센 값)
    (GEMMA2_2B, 2_614_341_888),
    (LLAMA_8B, 8_030_261_248),
    (QWEN25_7B, 7_615_616_512),
])
def test_param_count_from_config_only(cfg, n_true):
    spec = spec_from_config(cfg, "x")
    assert abs(spec.n_params - n_true) / n_true < 1e-3


def test_classify_kinds():
    assert classify(QWEN3_4B).kind == "llama-like"
    g = classify(GEMMA2_2B)
    assert g.kind == "gemma-like" and g.margin("lora") == 1.15 and g.margin("qlora") == 1.25
    assert classify(MIXTRAL).kind == "moe" and not classify(MIXTRAL).supported
    u = classify(dict(QWEN3_4B, model_type="brand-new-arch"))
    assert u.kind == "unknown" and u.margin("lora") > 1.0
    assert structure_of("llama-3.1-8b").kind == "verified"
    assert structure_of("gemma-2-2b").kind == "gemma-like"


def test_multimodal_text_config_is_used():
    cfg = {"model_type": "gemma3", "text_config": dict(GEMMA2_2B, model_type="gemma3_text")}
    assert spec_from_config(cfg, "mm").hidden == 2304
    assert classify(cfg).kind == "gemma-like"


def test_moe_is_rejected():
    with pytest.raises(ValueError):
        register("test/mixtral", MIXTRAL)


def test_local_config_file_and_get_model(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(QWEN3_4B))
    assert fetch_config(str(tmp_path))["hidden_size"] == 2560        # 폴더
    spec = get_model(str(p))                                         # config.json 경로
    assert spec.n_layers == 36 and resolve(str(p))[1].kind == "llama-like"


def test_registered_unseen_model_plans_and_gemma_gets_margin():
    register("test/qwen3-4b", QWEN3_4B)
    register("test/gemma2-2b", GEMMA2_2B)
    assert get_model("test/qwen3-4b").n_layers == 36
    q = plan(PlanRequest(model="test/qwen3-4b", gpu=get_gpu("rtx4090"), method="lora"))
    assert q.best is not None and any("처음 보는 모델" in n for n in q.notes)
    # 같은 설정을 등록 모델 이름으로 예측한 값과 비교: Gemma형은 ×1.15 여유가 붙어야 함
    g_unseen = plan(PlanRequest(model="test/gemma2-2b", gpu=get_gpu("rtx4090"), method="lora"))
    g_known = plan(PlanRequest(model="gemma-2-2b", gpu=get_gpu("rtx4090"), method="lora"))
    assert any("×1.15" in n for n in g_unseen.notes)
    a = {r.reason: r.pred_mem_gb for r in g_unseen.recommendations}
    b = {r.reason: r.pred_mem_gb for r in g_known.recommendations}
    common = set(a) & set(b)
    assert common and all(a[k] == pytest.approx(b[k], rel=0.01) for k in common)   # 둘 다 gemma-like 여유


def test_unknown_name_without_network_raises_keyerror(monkeypatch):
    import urllib.request
    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 404, "nf", None, None)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(KeyError):
        get_model("nobody/does-not-exist")


def test_registered_config_is_reused_for_cpu_measurement():
    register("upload/my-model", QWEN3_4B)
    assert fetch_config("upload/my-model")["hidden_size"] == 2560     # 네트워크 없이 등록한 설계도 재사용
