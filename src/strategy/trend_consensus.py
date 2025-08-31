"""Multi-timeframe trend consensus engine.

This module aggregates trend signals from 1m/5m/10m/15m bars and outputs
an overall direction score. The direction only changes after a configurable
number of consecutive confirmations which dampens noise from spot updates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd


class BarBuffer:
    """Incrementally maintain OHLCV aggregates for multiple timeframes."""

    def __init__(self, timeframes: Optional[Iterable[int]] = None) -> None:
        tfs = sorted(timeframes) if timeframes else [1]
        self.timeframes = tfs
        cols = ["open", "high", "low", "close", "volume"]
        self.frames: Dict[int, pd.DataFrame] = {
            tf: pd.DataFrame(columns=cols) for tf in tfs
        }
        self.last_ts: Optional[pd.Timestamp] = None

    def update(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Ingest new 1m bars and update aggregated frames."""
        if df.empty:
            return self.frames
        if self.last_ts is not None:
            df = df[df.index > self.last_ts]
        for ts, row in df.iterrows():
            self._add_bar(ts, row)
            self.last_ts = ts
        return self.frames

    def _add_bar(self, ts: pd.Timestamp, row: pd.Series) -> None:
        # 1m frame
        if 1 in self.timeframes:
            self.frames[1].loc[ts] = [row.open, row.high, row.low, row.close, row.volume]
        for tf in self.timeframes:
            if tf == 1:
                continue
            start = ts.floor(f"{tf}min")
            frame = self.frames[tf]
            if start in frame.index:
                prev = frame.loc[start]
                frame.at[start, "high"] = max(prev.high, row.high)
                frame.at[start, "low"] = min(prev.low, row.low)
                frame.at[start, "close"] = row.close
                frame.at[start, "volume"] = prev.volume + row.volume
            else:
                frame.loc[start] = [row.open, row.high, row.low, row.close, row.volume]


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
    alpha: smoothing factor for exponential averaging of aggregate score.
    """

    def __init__(self,
                 weights: Optional[Dict[int, float]] = None,
                 threshold: float = 0.6,
                 confirm: int = 3,
                 alpha: float = 0.3) -> None:
        self.weights = weights or {1: 0.2, 5: 0.3, 10: 0.25, 15: 0.25}
        self.threshold = threshold
        self.confirm = confirm
        self.alpha = alpha
        self.last_decision = "NEUTRAL"
        self.last_score = 0.0
        self.smoothed_score = 0.0
        self.last_change_ts: Optional[pd.Timestamp] = None
        self._history: deque[float] = deque(maxlen=confirm)
        self._buffer = BarBuffer(self.weights.keys())

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
        """Evaluate trend using pre-computed OHLCV frames."""
        frames = self._buffer.update(spot_1m)
        agg = 0.0
        for tf, w in self.weights.items():
            df = frames.get(tf, pd.DataFrame())
            if w <= 0 or df.empty:
                continue
            agg += w * self._classify(df)
        self.smoothed_score = self.alpha * agg + (1 - self.alpha) * self.smoothed_score
        self._history.append(self.smoothed_score)

        direction = self.last_decision
        if len(self._history) == self.confirm:
            if all(h > self.threshold for h in self._history):
                direction = "BULL"
            elif all(h < -self.threshold for h in self._history):
                direction = "BEAR"
            else:
                direction = "NEUTRAL"
        confidence = abs(self.smoothed_score)
        if direction != self.last_decision:
            self.last_decision = direction
            self.last_change_ts = pd.Timestamp.utcnow()
        self.last_score = self.smoothed_score
        return TrendResult(direction=direction, score=self.smoothed_score,
                           confidence=confidence,
                           last_change_ts=self.last_change_ts)
