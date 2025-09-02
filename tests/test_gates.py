import engine
from src.strategy.gates import RollingZGate
from src.strategy.trend_consensus import TrendResult


def test_zgate_debounce():
    gate = RollingZGate(window=5, threshold=1.0, confirm=2)
    for v in [1, 2, 1, 2, 1]:
        gate.update(v)
    z, mute = gate.update(5)
    assert abs(z) > 1
    assert mute is False
    z, mute = gate.update(5)
    assert mute is True


def test_evaluate_decision_muted():
    trend = TrendResult(direction="BULL", score=0.8, confidence=0.8, last_change_ts=None)
    dr, act = engine.evaluate_decision(trend, threshold=0.7, muted=True)
    assert act is False
    assert dr.trend_direction == "BULL"
