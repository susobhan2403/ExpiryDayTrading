import pytest
from src.features.options import atm_strike_with_tie_high


def test_atm_rounds_up():
    strikes = [24500, 24550, 24600]
    # Spot below 24550 should round up to 24550
    assert atm_strike_with_tie_high(24515, strikes) == 24550


def test_atm_upper_bound():
    strikes = [24500, 24550, 24600]
    # Midway between 24550 and 24600 -> still round up to 24600
    assert atm_strike_with_tie_high(24575, strikes) == 24600

