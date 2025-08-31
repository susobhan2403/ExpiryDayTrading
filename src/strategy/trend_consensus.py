"""Multi-timeframe trend consensus engine.

This module aggregates trend signals from 1m/5m/10m/15m bars and outputs
an overall direction score. The direction only changes after a configurable
number of consecutive confirmations which dampens noise from spot updates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class TrendResult:
    direction: str
    score: float
    confidence: float
    last_change_ts: Optional[pd.Timestamp]


class TrendConsensus:
    """Track price trend across multiple timeframes.

    Parameters
    ----------
    weights: mapping of timeframe (minutes) to weight.
    threshold: minimum absolute score required to consider a direction.
    confirm: number of consecutive evaluations required to flip direction.
    """

    def __init__(self,
                 weights: Optional[Dict[int, float]] = None,
                 threshold: float = 0.6,
                 confirm: int = 3) -> None:
        self.weights = weights or {1: 0.2, 5: 0.3, 10: 0.25, 15: 0.25}
        self.threshold = threshold
        self.confirm = confirm
        self.last_decision = "NEUTRAL"
        self.last_score = 0.0
        self.last_change_ts: Optional[pd.Timestamp] = None
        self._history: deque[float] = deque(maxlen=confirm)

    def _classify(self, df: pd.DataFrame) -> int:
        """Return +1 for bullish, -1 for bearish, 0 for neutral for a timeframe."""
        if df.empty:
            return 0
        if len(df) < 21:
            return 0
        ema_fast = df["close"].ewm(span=9, adjust=False).mean().iloc[-1]
        ema_slow = df["close"].ewm(span=21, adjust=False).mean().iloc[-1]
        if ema_fast > ema_slow:
            return 1
        if ema_fast < ema_slow:
            return -1
        return 0

    def evaluate(self, spot_1m: pd.DataFrame) -> TrendResult:
        """Evaluate trend using the provided 1-minute OHLCV dataframe."""
        frames: Dict[int, pd.DataFrame] = {
            1: spot_1m,
            5: spot_1m.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(),
            10: spot_1m.resample("10min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(),
            15: spot_1m.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(),
        }
        agg = 0.0
        for tf, df in frames.items():
            w = self.weights.get(tf, 0.0)
            if w <= 0 or df.empty:
                continue
            agg += w * self._classify(df)
        self._history.append(agg)

        direction = self.last_decision
        if len(self._history) == self.confirm:
            if all(h > self.threshold for h in self._history):
                direction = "BULL"
            elif all(h < -self.threshold for h in self._history):
                direction = "BEAR"
            else:
                direction = "NEUTRAL"
        confidence = abs(agg)
        if direction != self.last_decision:
            self.last_decision = direction
            self.last_change_ts = pd.Timestamp.utcnow()
        self.last_score = agg
        return TrendResult(direction=direction, score=agg,
                           confidence=confidence,
                           last_change_ts=self.last_change_ts)
