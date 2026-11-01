"""HF Trainer 플러그인 (fitcfg.integrations.hf) — 작은 모델·CPU로 확인.  실행: python -m pytest -q tests/test_hf_plugin.py"""
import json, os, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments, Trainer
from fitcfg import custom_gpu, get_gpu, reset_calibration
from fitcfg.integrations.hf import FitcfgCallback, detect, predict

TINY = dict(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)


def setup_function(_):
    reset_calibration()


def _model():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(**TINY))


def _args(tmp_path, **kw):
    base = dict(output_dir=str(tmp_path), per_device_train_batch_size=2, gradient_accumulation_steps=2,
                max_steps=6, use_cpu=True, report_to=[], logging_strategy="no", save_strategy="no",
                disable_tqdm=True)
    base.update(kw)
    return TrainingArguments(**base)


class _DS(torch.utils.data.Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, i):
        ids = torch.randint(0, TINY["vocab_size"], (32,))
        return {"input_ids": ids, "labels": ids.clone()}


def test_detect_full_and_lora(tmp_path):
    m = _model()
    d = detect(m, _args(tmp_path, gradient_checkpointing=True, optim="adamw_bnb_8bit"), seq_len=1024)
    assert (d.method, d.batch_size, d.grad_accum, d.seq_len, d.grad_checkpointing, d.optim) == \
        ("full", 2, 2, 1024, True, "adamw8bit")
    peft = pytest.importorskip("peft")
    lm = peft.get_peft_model(m, peft.LoraConfig(r=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    d = detect(lm, _args(tmp_path))
    assert d.method == "lora" and d.lora_rank == 8 and d.lora_targets == "attn" and d.seq_len_guessed
    lm.base_model.model.is_loaded_in_4bit = True            # 4bit 로딩 흉내
    assert detect(lm, _args(tmp_path)).method == "qlora"


def test_predict_matches_formula_scale(tmp_path):
    _, p = predict(_model(), _args(tmp_path), gpu=get_gpu("rtx4090"), seq_len=32)
    assert p.fits and 0 < p.pred_mem_gb < 2 and p.structure == "llama-like"


def test_strict_mode_stops_before_training(tmp_path):
    cb = FitcfgCallback(seq_len=32, gpu=custom_gpu(0.01), strict=True, verbose=False, calibration=None)
    tr = Trainer(model=_model(), args=_args(tmp_path), train_dataset=_DS(), callbacks=[cb])
    with pytest.raises(RuntimeError, match="OOM 예상"):
        tr.train()
    rec = json.loads((tmp_path / "fitcfg_run.json").read_text())
    assert rec["status"] == "stopped_predicted_oom" and rec["prediction"]["fits"] is False


def test_warning_mode_trains_and_logs(tmp_path, capsys):
    cb = FitcfgCallback(seq_len=32, gpu="rtx4090", warmup_steps=2, calibration=None)
    Trainer(model=_model(), args=_args(tmp_path), train_dataset=_DS(), callbacks=[cb]).train()
    out = capsys.readouterr().out
    assert "[fitcfg]" in out and "예측 GPU당 peak" in out
    rec = json.loads((tmp_path / "fitcfg_run.json").read_text())
    assert rec["status"] == "finished" and rec["detected"]["method"] == "full"
    assert rec["measured"]["steps_timed"] == 4 and rec["measured"]["tokens_per_s"] > 0


def test_no_gpu_skips_quietly(tmp_path, monkeypatch):
    import fitcfg.integrations.hf as H
    monkeypatch.setattr(H, "_current_gpu", lambda: None)
    cb = FitcfgCallback(seq_len=32, verbose=False, calibration=None)
    Trainer(model=_model(), args=_args(tmp_path, max_steps=2), train_dataset=_DS(), callbacks=[cb]).train()
    assert cb.prediction is None and cb.detected is not None


def test_bf16_on_gpu_without_bf16_warns(tmp_path):
    args = TrainingArguments(output_dir=str(tmp_path), per_device_train_batch_size=1, use_cpu=True, report_to=[])
    args.bf16 = True                                         # 사용자가 T4에서 bf16을 켠 상황
    _, p = predict(_model(), args, gpu=get_gpu("t4"), seq_len=32)
    assert any("bf16" in w and "fp16=True" in w for w in p.warnings)
    _, p2 = predict(_model(), args, gpu=get_gpu("rtx4090"), seq_len=32)
    assert not p2.warnings
