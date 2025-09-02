import math
from engine import _round_or_nan


def test_round_or_nan_none():
    assert math.isnan(_round_or_nan(None))


def test_round_or_nan_value():
    assert _round_or_nan(1.2345, 2) == 1.23
