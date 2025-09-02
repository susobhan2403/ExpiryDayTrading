import math
import datetime as dt
from pathlib import Path

import pandas as pd

from src.metrics.core import (
    infer_strike_step,
    pick_atm_strike,
    compute_forward,
    compute_atm_iv,
    compute_pcr,
    compute_iv_stats,
    choose_expiry,
    apply_gates,
)
from src.strategy.decision_table import detect_regime, decide_trade

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def test_chain_metrics_and_percentile():
    df = pd.read_csv(Path(__file__).parent / "fixtures/atm_iv_pcr.csv")
    strikes = df["strike"].tolist()
    step = infer_strike_step(strikes)
    ce_mid = dict(zip(df["strike"], df["ce_mid"]))
    pe_mid = dict(zip(df["strike"], df["pe_mid"]))
    oi_call = dict(zip(df["strike"], df["oi_call"]))
    oi_put = dict(zip(df["strike"], df["oi_put"]))
    F = compute_forward(150.0, 152.0, 0.05, 0.0, 0.02)
    K, diag_atm = pick_atm_strike(F, strikes, step, ce_mid, pe_mid)
    assert K == 150.0
    iv, diag_iv = compute_atm_iv(ce_mid[K], pe_mid[K], F, K, 0.02, 0.05)
    assert iv is not None and abs(iv - 0.2) < 1e-3
    pcr = compute_pcr(oi_put, oi_call, strikes, K, step, m=1)
    assert math.isclose(pcr["PCR_OI_total"], 2.0)
    hist = pd.read_csv(Path(__file__).parent / "fixtures/iv_history_tenor.csv")
    history = list(zip(hist["tau"], hist["iv"]))
    stats = compute_iv_stats(history, 0.25, current_tau=0.02)
    assert math.isclose(stats["percentile"], 50.0)
    assert math.isclose(stats["iv_rank"], 50.0)


def test_tau_rollover_near_close():
    now = dt.datetime(2024, 1, 1, 15, 29, 0, tzinfo=IST)
    exp1 = dt.datetime(2024, 1, 1, 15, 30, tzinfo=IST)
    exp2 = dt.datetime(2024, 1, 8, 15, 30, tzinfo=IST)
    chosen = choose_expiry(now, [exp1, exp2], min_tau_h=0.25)
    assert chosen == exp2


def test_gate_override_decision():
    gate = apply_gates(["ORB", "VOLUME", "OI_DT", "IV_CRUSH"], "NEWS_SHOCK", 0)
    regime = detect_regime(-1.0, 0.2, 0.005)
    decision = decide_trade("LONG", gate, regime)
    assert decision == "LONG"

