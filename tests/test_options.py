from pathlib import Path

import math
import pandas as pd

from src.features.options import (
    atm_strike,
    atm_iv_from_chain,
    max_pain,
    pcr_from_chain,
)
from src.config import get_rfr


FIXTURE = Path(__file__).parent / "fixtures" / "chain_sample.csv"


def load_chain():
    df = pd.read_csv(FIXTURE)
    strikes = df["strike"].tolist()
    calls = {row["strike"]: {"oi": row["ce_oi"]} for _, row in df.iterrows()}
    puts = {row["strike"]: {"oi": row["pe_oi"]} for _, row in df.iterrows()}
    return {"strikes": strikes, "calls": calls, "puts": puts}


def test_atm_rounds_nearest():
    strikes = [24500, 24550, 24600]
    assert atm_strike(24515, strikes) == 24500
    # tie → higher
    assert atm_strike(24525, strikes) == 24550


def test_max_pain_tiebreak():
    chain = load_chain()
    # Equal pain at 100 and 110; spot closer to 110 so choose 110
    assert max_pain(chain, spot=108, step=10) == 110


def test_atm_iv_solver_bounds():
    chain = {
        "strikes": [100],
        "calls": {
            100: {"bid": 80, "ask": 82, "bid_qty": 1, "ask_qty": 1, "oi": 0}
        },
        "puts": {
            100: {"bid": 80, "ask": 82, "bid_qty": 1, "ask_qty": 1, "oi": 0}
        },
    }
    iv = atm_iv_from_chain(chain, spot=100, minutes_to_exp=1440, risk_free_rate=get_rfr())
    assert math.isnan(iv)


def _make_chain(strikes, ce_ois, pe_ois):
    calls = {k: {"oi": ce} for k, ce in zip(strikes, ce_ois)}
    puts = {k: {"oi": pe} for k, pe in zip(strikes, pe_ois)}
    return {"strikes": strikes, "calls": calls, "puts": puts}


def test_pcr_band_filtering():
    strikes = [19400, 19450, 19500, 19550, 19600, 19650]
    ce = [5, 10, 10, 15, 5, 20]
    pe = [5, 10, 20, 25, 30, 40]
    chain = _make_chain(strikes, ce, pe)
    res = pcr_from_chain(chain, spot=19550, symbol="NIFTY", band_steps=1)
    # uses strikes 19500,19550,19600
    assert math.isclose(res["PCR_OI_band"], (20 + 25 + 30) / (10 + 15 + 5))


def test_pcr_bad_oi_and_min_count():
    strikes = [19400, 19500, 19550, 19600]
    ce = [10, 10, 0, 5]  # 19550 has bad OI
    pe = [10, 20, 5, 30]
    chain = _make_chain(strikes, ce, pe)
    res = pcr_from_chain(chain, spot=19550, symbol="NIFTY", band_steps=1)
    assert math.isclose(res["PCR_OI_band"], (20 + 30) / (10 + 5))

    # only one valid strike -> nan
    ce = [10, 0, 0]
    pe = [20, 5, 0]
    chain = _make_chain([19500, 19550, 19600], ce, pe)
    assert math.isnan(pcr_from_chain(chain, spot=19550, symbol="NIFTY", band_steps=1)["PCR_OI_band"])


def test_atm_strike_uses_step_map_for_truncated_chain():
    # BankNifty step size is 100; chain is missing higher strikes
    strikes = [44000]
    spot = 44150
    assert atm_strike(spot, strikes, symbol="BANKNIFTY") == 44200

