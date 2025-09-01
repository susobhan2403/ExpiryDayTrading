from pathlib import Path

import math
import pandas as pd

from src.features.options import atm_strike, atm_iv_from_chain, max_pain


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
    iv = atm_iv_from_chain(chain, spot=100, minutes_to_exp=1440, risk_free_rate=0.0)
    assert not math.isnan(iv)
    assert iv > 1.5

