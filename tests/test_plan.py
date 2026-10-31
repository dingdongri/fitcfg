"""추천(plan)·처리량·번들 생성 검증 (GPU 불필요).  실행: python -m pytest -q"""
import json, os, sys, tempfile, zipfile, io, py_compile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fitcfg import (PlanRequest, TrainConfig, get_gpu, get_model, plan, estimate_throughput,
                    reset_calibration, estimate)
from fitcfg.generate import build_bundle, write_bundle, zip_bundle
from fitcfg.memory import cpu_ram_needed_gb


def setup_function(_):
    reset_calibration()


def test_best_fits_and_is_fastest_feasible():
    res = plan(PlanRequest(model="llama-3.1-8b", gpu=get_gpu("rtx4090"), method="qlora"))
    assert res.best and res.best.fits
    feas = [c for c in res.candidates if c.fits]
    assert res.best.throughput.tokens_per_s >= max(c.throughput.tokens_per_s for c in feas) / 1.03


def test_global_batch_is_preserved():
    res = plan(PlanRequest(model="llama-3.2-1b", gpu=get_gpu("t4"), method="qlora", global_batch=32))
    for r in res.recommendations:
        assert r.effective_batch == 32


def test_naive_ooms_where_tool_finds_config():
    # 1B QLoRA, T4, seq 1024: HF Trainer 기본값(batch 8, ckpt off)은 OOM 예측, fitcfg는 해결
    res = plan(PlanRequest(model="llama-3.2-1b", gpu=get_gpu("t4"), method="qlora", seq_len=1024))
    assert not res.naive.fits and res.best is not None


def test_diagnosis_when_nothing_fits():
    res = plan(PlanRequest(model="qwen2.5-7b", gpu=get_gpu("t4"), method="lora", seq_len=4096))
    assert res.best is None and len(res.diagnosis) >= 2
    assert any("qlora" in d for d in res.diagnosis)


def test_cpu_ram_filters_offload():
    req = PlanRequest(model="llama-3.1-8b", gpu=get_gpu("rtx4090"), method="full")
    assert plan(req).best is not None                        # RAM 모르면 offload 허용
    res = plan(PlanRequest(**{**req.__dict__, "cpu_ram_gb": 32}))
    assert res.best is None and "CPU RAM" in res.diagnosis[0]


def test_8bit_only_without_zero():
    res = plan(PlanRequest(model="llama-3.1-8b", gpu=get_gpu("a100-40"), gpus=4, method="full"))
    for c in res.candidates:
        if c.config.optim == "adamw8bit":
            assert c.strategy in ("single", "ddp")


def test_fp16_full_uses_fp32_weights():
    m = get_model("llama-3.2-1b")
    bf = estimate(TrainConfig("llama-3.2-1b", method="full", dtype="bf16"), m)
    fp = estimate(TrainConfig("llama-3.2-1b", method="full", dtype="fp16"), m)
    assert fp.weights == 2 * bf.weights


def test_throughput_monotonic():
    m, g = get_model("llama-3.1-8b"), get_gpu("a100-80")
    base = TrainConfig("llama-3.1-8b", method="lora", batch_size=4)
    t = lambda **kw: estimate_throughput(TrainConfig(**{**base.__dict__, **kw}), m, g).tokens_per_s
    assert t(grad_checkpointing=False) > t(grad_checkpointing=True)      # 재계산 비용
    assert t(gpus=2) > t(gpus=1)                                          # 데이터 병렬
    assert t(cpu_offload_optim=True, method="full") < t(method="full")   # offload는 느림
    e = estimate_throughput(base, m, g)
    assert e.low < e.tokens_per_s < e.high


def test_offload_ram_estimate():
    cfg = TrainConfig("llama-3.1-8b", method="full", zero_stage=2, cpu_offload_optim=True)
    assert 120 < cpu_ram_needed_gb(cfg, get_model("llama-3.1-8b")) < 200


def test_bundle_single_gpu():
    req = PlanRequest(model="llama-3.2-1b", gpu=get_gpu("t4"), method="qlora")
    files = build_bundle(plan(req).best, req)
    assert "ds_config.json" not in files
    assert "python train.py" in files["launch.sh"]
    assert "bitsandbytes" in files["requirements.txt"]
    d = write_bundle(files, tempfile.mkdtemp())
    py_compile.compile(str(d / "train.py"), doraise=True)
    assert "'dtype': 'fp16'" in files["train.py"]              # T4 → fp16


def test_bundle_deepspeed():
    req = PlanRequest(model="llama-3.1-8b", gpu=get_gpu("rtx4090"), gpus=4, method="full")
    rec = plan(req).best
    files = build_bundle(rec, req)
    ds = json.loads(files["ds_config.json"])
    assert ds["zero_optimization"]["stage"] == rec.config.zero_stage
    assert "torchrun" in files["launch.sh"] and "--nproc_per_node 4" in files["launch.sh"]
    z = zipfile.ZipFile(io.BytesIO(zip_bundle(files, "b")))
    assert "b/train.py" in z.namelist()


def test_cli_runs():
    from fitcfg.cli import main
    assert main(["plan", "--model", "llama-3.2-1b", "--gpu", "t4", "--no-calib", "--top", "2"]) == 0
    assert main(["plan", "--model", "qwen2.5-7b", "--gpu", "t4", "--method", "lora", "--seq", "4096",
                 "--no-calib"]) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            setup_function(None); fn(); print(f"PASS {name}")
    print("all passed")
