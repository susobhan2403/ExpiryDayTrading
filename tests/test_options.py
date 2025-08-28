import pytest
from src.features.options import atm_strike_with_tie_high

def test_atm_tie_high():
    strikes = [24500, 24550, 24600]
    # equidistant between 24550 and 24600 -> should pick higher (24600)
    assert atm_strike_with_tie_high(24575, strikes) == 24600

