import math
import datetime as dt
from pathlib import Path

import pandas as pd

from src.features.robust_metrics import (
    detect_strike_step,
    pick_expiry,
    compute_forward,
    pick_atm_strike,
    implied_vol_bs,
    compute_atm_iv,
    compute_iv_percentile,
    compute_pcr,
)

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def test_implied_vol_golden():
    price = 10.4506
    iv, diag = implied_vol_bs(price, 100.0, 100.0, 1.0, 0.05, 0.0, "C")
    assert abs(iv - 0.2) < 1e-4
    assert diag["converged"]


def test_pick_expiry_rollover():
    now = dt.datetime(2024, 1, 1, 15, 29, tzinfo=IST)
    exp1 = dt.datetime(2024, 1, 1, 15, 30, tzinfo=IST)
    exp2 = dt.datetime(2024, 1, 8, 15, 30, tzinfo=IST)
    chosen = pick_expiry(now, [exp1, exp2], min_tau_h=0.25)
    assert chosen == exp2


def test_pick_atm_strike_tie_break():
    strikes = [100.0, 150.0]
    step = detect_strike_step(strikes)
    F = 125.0
    ce_mid = {100.0: 1.0, 150.0: 1.5}
    pe_mid = {100.0: 2.0, 150.0: 1.4}
    K, diag = pick_atm_strike(F, strikes, step, ce_mid, pe_mid)
    assert K == 150.0  # straddle nearer theoretical
    assert diag["tie_break"] == "upper"


def test_compute_atm_iv():
    S = 100.0
    K = 100.0
    tau = 0.5
    r = 0.05
    q = 0.0
    sigma = 0.2
    # Generate mid prices from BS formula
    F = compute_forward(S, None, r, q, tau)
    sqrtT = math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * tau) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    call_price = math.exp(-r * tau) * (F * N(d1) - K * N(d2))
    put_price = math.exp(-r * tau) * (K * N(-d2) - F * N(-d1))
    atm_iv, diag = compute_atm_iv(call_price, put_price, S, K, tau, r, q, F)
    assert abs(atm_iv - sigma) < 1e-4
    assert diag["atm_iv"] == atm_iv


def test_compute_iv_percentile_and_rank(tmp_path):
    df = pd.read_csv(Path(__file__).parent / "fixtures/iv_history.csv")
    pct, rank = compute_iv_percentile(df["iv"].tolist(), 0.22)
    assert pct == 60.0
    assert math.isclose(rank, 60.0)
    pct2, rank2 = compute_iv_percentile([0.2, 0.2, 0.2], 0.2)
    assert math.isnan(rank2)


def test_compute_pcr():
    df = pd.read_csv(Path(__file__).parent / "fixtures/pcr_chain.csv")
    strikes = df["strike"].tolist()
    oi_call = dict(zip(df["strike"], df["oi_call"]))
    oi_put = dict(zip(df["strike"], df["oi_put"]))
    step = detect_strike_step(strikes)
    res = compute_pcr(oi_put, oi_call, strikes, 150.0, step, m=1)
    assert math.isclose(res["PCR_OI_total"], 2.0)
    assert math.isclose(res["PCR_OI_band"], 1600/600)
    assert res["band_count"] == 3
