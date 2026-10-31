"""cost model 기본 검증 (GPU 불필요).  실행: python -m pytest -q  또는  python tests/test_memory.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fitcfg import TrainConfig, estimate, get_model, recommend


def test_qlora_less_than_full():
    m = get_model("llama-3.1-8b")
    q = estimate(TrainConfig("llama-3.1-8b", method="qlora"), m).total
    f = estimate(TrainConfig("llama-3.1-8b", method="full"), m).total
    assert q < f, "QLoRA should use far less memory than full FT"


def test_batch_increases_memory():
    m = get_model("llama-3.1-8b")
    small = estimate(TrainConfig("llama-3.1-8b", batch_size=1), m).total
    big = estimate(TrainConfig("llama-3.1-8b", batch_size=8), m).total
    assert big > small


def test_checkpointing_reduces_memory():
    m = get_model("llama-3.1-8b")
    on = estimate(TrainConfig("llama-3.1-8b", batch_size=8, grad_checkpointing=True), m).total
    off = estimate(TrainConfig("llama-3.1-8b", batch_size=8, grad_checkpointing=False), m).total
    assert on < off


def test_zero3_shards_across_gpus():
    m = get_model("llama-3.1-8b")
    one = estimate(TrainConfig("llama-3.1-8b", method="full", gpus=1, zero_stage=3), m).total
    two = estimate(TrainConfig("llama-3.1-8b", method="full", gpus=2, zero_stage=3), m).total
    assert two < one


def test_recommend_fits_budget():
    recs = recommend("llama-3.1-8b", gpu_mem_gb=24, method="qlora")
    assert len(recs) > 0
    assert all(r.pred_mem_gb <= 24 * 0.90 for r in recs)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"PASS {name}")
    print("all passed")
