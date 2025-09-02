from __future__ import annotations
import math
from typing import Dict


def atr_stop(entry: float, atr: float, mul: float = 1.5, direction: str = "long") -> float:
    """Initial ATR-based stop."""
    return entry - mul * atr if direction == "long" else entry + mul * atr


def chandelier_trail(high: float, low: float, atr: float, n: int, direction: str) -> float:
    """Chandelier trailing stop."""
    if direction == "long":
        return max(high - n * atr, low)
    return min(low + n * atr, high)


def size_position(capital: float, price: float, target_vol: float, realized_vol: float) -> int:
    """Volatility-scaled position size."""
    vol_units = capital * target_vol / max(realized_vol, 1e-6)
    return int(vol_units / price)


__all__ = ["atr_stop", "chandelier_trail", "size_position"]
