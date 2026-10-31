"""보정 파이프라인 검증 (GPU 불필요). 합성 데이터로 '숨겨진 상수를 되찾는지' 확인한다.
실행: python -m pytest -q  또는  python tests/test_calibrate.py"""
import os, sys, tempfile, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from functools import partial
from fitcfg import TrainConfig, estimate, get_model, load_calibration, reset_calibration
from fitcfg.calibration import (load_measurements, fit, predict_gb, metrics, cross_validate,
                                ridge_to_prior, prior_constants)
from synthetic import write_synthetic_csv, TRUTH


def _synthetic():
    path = os.path.join(tempfile.mkdtemp(), "syn.csv")
    write_synthetic_csv(path)
    meas, warns = load_measurements(path)
    assert not warns
    return meas


def test_ridge_recovers_exact_linear_system():
    # y = 2·x0 + 3·x1 (노이즈 없음) → λ가 작으면 사전값과 무관하게 정답을 찾아야 함
    X = [[1, 0], [0, 1], [1, 1], [2, 1]]
    y = [2 * a + 3 * b for a, b in X]
    th = ridge_to_prior(X, y, [1.0] * 4, prior=[10.0, 10.0], lam=1e-9)
    assert abs(th[0] - 2) < 1e-6 and abs(th[1] - 3) < 1e-6


def test_ridge_keeps_prior_for_unobserved_column():
    X = [[1, 0], [2, 0], [3, 0]]      # 두 번째 열은 데이터 없음
    th = ridge_to_prior(X, [2, 4, 6], [1.0] * 3, prior=[5.0, 7.0], lam=1e-3)
    assert abs(th[1] - 7.0) < 1e-9


def test_calibration_improves_and_generalizes():
    meas = _synthetic()
    ok = [m for m in meas if m.ok]
    consts = fit(meas)
    before = metrics((predict_gb(m, {"default": prior_constants("device")}), m.device_peak_gb) for m in ok)
    after = metrics((predict_gb(m, consts), m.device_peak_gb) for m in ok)
    assert after.mape < before.mape
    assert after.mape < 3.0
    # 모델 하나를 통째로 빼도 잘 맞아야 함 (새 모델 일반화)
    ho = cross_validate(meas, "model", "device", partial(fit))
    assert metrics((p, m.device_peak_gb) for m, p in ho).mape < 4.0


def test_recovers_hidden_constants_roughly():
    consts = fit(_synthetic())
    q, t = consts["qlora"], TRUTH["qlora"]
    # act_h 와 act_i 는 모델 2개(1B·3B)만으로는 개별 식별이 약하다 → 1B 기준 '레이어·토큰당 활성값'으로 비교
    from fitcfg import get_model
    m = get_model("llama-3.2-1b")
    per = lambda c: c.act_h * m.hidden + c.act_i * m.intermediate  # noqa: E731
    assert abs(per(q) - per(t)) / per(t) < 0.15
    assert abs(q.logit_bytes - t.logit_bytes) / t.logit_bytes < 0.2


def test_throughput_calibration_recovers_truth():
    from fitcfg.calibration import fit_throughput
    from synthetic import TRUTH_TP
    tp = fit_throughput(_synthetic())
    assert "t4" in tp
    for m in ("full", "lora", "qlora"):
        assert abs(tp["t4"].eff(m) - TRUTH_TP.eff(m)) / TRUTH_TP.eff(m) < 0.15


def test_load_calibration_changes_estimate():
    cfg, m = TrainConfig("llama-3.2-1b", batch_size=4, seq_len=1024), get_model("llama-3.2-1b")
    reset_calibration()
    base = estimate(cfg, m).total
    path = os.path.join(tempfile.mkdtemp(), "calib.json")
    with open(path, "w") as f:
        json.dump({"constants": {"default": {"logit_bytes": 16.0}}}, f)
    load_calibration(path)
    try:
        assert estimate(cfg, m).total > base
    finally:
        reset_calibration()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"PASS {name}")
    print("all passed")
