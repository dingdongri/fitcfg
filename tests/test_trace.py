"""FakeTensor 메모리 추적기 검증. torch/transformers/peft가 없으면 건너뜀.  실행: python -m pytest -q"""
import os, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402
from fitcfg.trace import _make_tracker, trace_step  # noqa: E402
from fitcfg import get_model  # noqa: E402

MB = 2 ** 20


def test_tracker_counts_alloc_view_inplace_free():
    T = _make_tracker()()
    with FakeTensorMode(), T:
        x = torch.empty(1024, 1024)          # 4MB
        y = x * 2                            # +4MB
        v = y.view(-1)                       # view → 0
        y.add_(1)                            # in-place → 0
        del x                                # -4MB
        z = torch.empty(512, 1024)           # +2MB
        assert T.cur == 6 * MB and T.peak == 8 * MB
        del y, v, z
        assert T.cur == 0


def test_trace_small_model_weights_exact():
    # 1개 레이어, vocab 1000인 축소 Llama-3.2-1B → 가중치 바이트가 구조 계산과 정확히 일치해야 함
    spec = get_model("llama-3.2-1b")
    r = trace_step("llama-3.2-1b", "full", batch=1, seq=64, ckpt=False, n_layers=1, vocab=1000)
    H, I, KV = spec.hidden, spec.intermediate, spec.kv_dim
    params = 1000 * H + (2 * H * H + 2 * H * KV + 3 * H * I) + 3 * H     # tied embed + 1 layer + norms
    assert abs(r.mem_model_gb * 2 ** 30 - params * 2) < 1e5
    assert r.peak_alloc_gb > r.mem_steady_gb > r.mem_model_gb    # 학습 중 peak > (가중치+optimizer) > 가중치


def test_trace_checkpointing_reduces_peak():
    kw = dict(batch=2, seq=256, n_layers=4, vocab=1000)
    on = trace_step("llama-3.2-1b", "lora", ckpt=True, **kw).peak_alloc_gb
    off = trace_step("llama-3.2-1b", "lora", ckpt=False, **kw).peak_alloc_gb
    assert on < off
