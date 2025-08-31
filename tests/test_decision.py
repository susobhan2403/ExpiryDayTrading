import engine
from src.strategy.trend_consensus import TrendResult


def test_evaluate_decision_threshold():
    trend = TrendResult(direction="BULL", score=0.8, confidence=0.8, last_change_ts=None)
    decision, act = engine.evaluate_decision(trend, threshold=0.7)
    assert act is True
    assert decision.trend_direction == "BULL"
    trend2 = TrendResult(direction="NEUTRAL", score=0.0, confidence=0.5, last_change_ts=None)
    decision2, act2 = engine.evaluate_decision(trend2, threshold=0.7)
    assert act2 is False
    assert decision2.trend_direction == "NEUTRAL"
