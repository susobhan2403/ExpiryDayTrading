import math
import datetime as dt
from pathlib import Path

import pandas as pd

from src.metrics.core import (
    infer_strike_step,
    choose_expiry,
    compute_forward,
    pick_atm_strike,
    implied_vol,
    compute_atm_iv,
    compute_iv_stats,
    compute_pcr,
    apply_gates,
)

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def test_implied_vol_golden():
    price = 10.4506
    T = 1.0
    r = 0.05
    F = 100 * math.exp(r * T)
    iv, diag = implied_vol(price, F, 100.0, T, r, "C")
    assert iv is not None
    assert abs(iv - 0.2) < 1e-4
    assert diag["converged"]


def test_pick_atm_strike_tie_break():
    strikes = [100.0, 150.0]
    step = infer_strike_step(strikes)
    F = 125.0
    ce_mid = {100.0: 1.0, 150.0: 1.5}
    pe_mid = {100.0: 2.0, 150.0: 1.4}
    K, diag = pick_atm_strike(F, strikes, step, ce_mid, pe_mid)
    assert K == 150.0
    assert diag["tie_break"] == "upper"


def test_compute_pcr():
    df = pd.read_csv(Path(__file__).parent / "fixtures/pcr_chain.csv")
    strikes = df["strike"].tolist()
    oi_call = dict(zip(df["strike"], df["oi_call"]))
    oi_put = dict(zip(df["strike"], df["oi_put"]))
    step = infer_strike_step(strikes)
    res = compute_pcr(oi_put, oi_call, strikes, 150.0, step, m=1)
    assert math.isclose(res["PCR_OI_total"], 2.0)
    assert math.isclose(res["PCR_OI_band"], 1600/600)
    assert res["band_count"] == 3


def test_compute_iv_stats():
    df = pd.read_csv(Path(__file__).parent / "fixtures/iv_history.csv")
    hist = df["iv"].tolist()
    stats = compute_iv_stats(hist, 0.22)
    assert stats["percentile"] == 60.0
    assert math.isclose(stats["iv_rank"], 60.0)
    stats2 = compute_iv_stats([0.2, 0.2, 0.2], 0.2)
    assert stats2["iv_rank"] is None
    assert "zero-range" in stats2["reasons"]


def test_choose_expiry_rollover():
    now = dt.datetime(2024, 1, 1, 15, 29, tzinfo=IST)
    exp1 = dt.datetime(2024, 1, 1, 15, 30, tzinfo=IST)
    exp2 = dt.datetime(2024, 1, 8, 15, 30, tzinfo=IST)
    chosen = choose_expiry(now, [exp1, exp2], min_tau_h=0.25)
    assert chosen == exp2


def test_compute_atm_iv():
    S = 100.0
    K = 100.0
    tau = 0.5
    r = 0.05
    q = 0.0
    sigma = 0.2
    F = compute_forward(S, None, r, q, tau)
    sqrtT = math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * tau) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    call_price = math.exp(-r * tau) * (F * N(d1) - K * N(d2))
    put_price = math.exp(-r * tau) * (K * N(-d2) - F * N(-d1))
    atm_iv, diag = compute_atm_iv(call_price, put_price, F, K, tau, r)
    assert atm_iv is not None and abs(atm_iv - sigma) < 1e-4
    assert diag["atm_iv"] == atm_iv


def test_gate_override():
    decision = apply_gates(["ORB", "OI_SHIFT", "IV_CRUSH"], "NEWS_SHOCK", confirming_bars=0)
    assert not decision.muted
    assert decision.override
    assert decision.size_factor == 0.5
