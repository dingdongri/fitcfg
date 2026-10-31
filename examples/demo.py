"""GPU 없이 돌아가는 데모.
    python examples/demo.py
cost model이 합리적인 숫자를 내는지, 추천이 동작하는지 눈으로 확인하는 용도."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pip 설치 없이 저장소에서 바로 실행
from fitcfg import TrainConfig, estimate, get_model, recommend, load_calibration

# 보정 결과가 있으면 사용 (scripts/calibrate.py 가 생성)
_calib = Path(__file__).resolve().parents[1] / "data" / "calibration.json"
if _calib.exists():
    load_calibration(_calib)
    print(f"[보정값 사용] {_calib}")
else:
    print("[사전추정 상수 사용] data/calibration.json 없음")


def show_breakdown(title, cfg):
    m = get_model(cfg.model)
    bd = estimate(cfg, m).as_gb()
    print(f"\n[{title}]  {cfg.method}, batch={cfg.batch_size}, seq={cfg.seq_len}, "
          f"ckpt={cfg.grad_checkpointing}, gpus={cfg.gpus}, zero={cfg.zero_stage}")
    print(f"  weights={bd['weights']}  grad={bd['gradients']}  "
          f"optim={bd['optimizer']}  act={bd['activations']}  logits={bd['logits']}  "
          f"overhead={bd['overhead']}  =>  TOTAL {bd['total']} GB")


print("=" * 68)
print("메모리 예측 분해 (Llama-3.1-8B)")
print("=" * 68)
show_breakdown("QLoRA / 1x GPU", TrainConfig("llama-3.1-8b", method="qlora", batch_size=4))
show_breakdown("LoRA (bf16 base) / 1x GPU", TrainConfig("llama-3.1-8b", method="lora", batch_size=4))
show_breakdown("Full-FT / 1x GPU", TrainConfig("llama-3.1-8b", method="full", batch_size=4))
show_breakdown("Full-FT / 2x GPU ZeRO-3", TrainConfig("llama-3.1-8b", method="full", batch_size=4, gpus=2, zero_stage=3))

print("\n" + "=" * 68)
print("추천 데모: Llama-3.1-8B, QLoRA, 24GB GPU 1장 (예: RTX 4090)")
print("=" * 68)
for i, r in enumerate(recommend("llama-3.1-8b", gpu_mem_gb=24, method="qlora"), 1):
    print(f"  #{i}  {r.reason}  -> 예측 {r.pred_mem_gb} GB (score {r.score:.2f})")

print("\n추천 데모: Llama-3.1-8B, Full-FT, 80GB GPU 1장 (A100)")
for i, r in enumerate(recommend("llama-3.1-8b", gpu_mem_gb=80, method="full"), 1):
    print(f"  #{i}  {r.reason}  -> 예측 {r.pred_mem_gb} GB (score {r.score:.2f})")
