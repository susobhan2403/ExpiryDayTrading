import math
from src.features import options_ext as opt

CHAIN = {
    "strikes": [100, 110, 120],
    "calls": {100: {"oi": 10}, 110: {"oi": 20}, 120: {"oi": 5}},
    "puts": {100: {"oi": 5}, 110: {"oi": 15}, 120: {"oi": 25}},
}


def test_forward_atm():
    atm = opt.forward_atm(spot=111, r=0.05, div=0.0, tau=1 / 12, strikes=CHAIN["strikes"])
    assert atm == 110


def test_pcr_variants():
    res = opt.pcr_variants(CHAIN, atm=110, step=10, k=1)
    assert math.isclose(res["pcr_total"], (5 + 15 + 25) / (10 + 20 + 5))
    assert math.isclose(res["pcr_band"], 15 / 20)
