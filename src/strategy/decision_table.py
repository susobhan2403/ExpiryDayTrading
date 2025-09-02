from __future__ import annotations
"""Trade regime detection and decision logic."""

from dataclasses import dataclass
from typing import Literal

from ..metrics.core import GateDecision


@dataclass(frozen=True)
class Regime:
    """Simple representation of market regime."""

    trend: Literal["UP", "DOWN", "FLAT"]
    vol: Literal["HIGH", "MID", "LOW"]
    liquidity: Literal["GOOD", "BAD"]


def detect_regime(trend_score: float, iv: float, spread: float) -> Regime:
    """Classify market regime based on trend, volatility and liquidity.

    Parameters
    ----------
    trend_score: float
        Positive for bullish, negative for bearish.
    iv: float
        Current ATM implied volatility as a decimal.
    spread: float
        Bid/ask spread expressed as a percentage of price.
    """

    if trend_score > 0.5:
        trend = "UP"
    elif trend_score < -0.5:
        trend = "DOWN"
    else:
        trend = "FLAT"

    if iv > 0.25:
        vol = "HIGH"
    elif iv < 0.15:
        vol = "LOW"
    else:
        vol = "MID"

    liquidity = "GOOD" if spread <= 0.01 else "BAD"
    return Regime(trend, vol, liquidity)


def decide_trade(
    direction: Literal["LONG", "SHORT"],
    gate: GateDecision,
    regime: Regime,
) -> Literal["LONG", "SHORT", "NO_TRADE"]:
    """Return final trade decision.

    ``gate.override`` allows trading even when regime trend disagrees with the
    trade direction.  Bad liquidity always suppresses trading.
    """

    if gate.muted or regime.liquidity == "BAD":
        return "NO_TRADE"
    if direction == "LONG" and regime.trend == "DOWN" and not gate.override:
        return "NO_TRADE"
    if direction == "SHORT" and regime.trend == "UP" and not gate.override:
        return "NO_TRADE"
    return direction


__all__ = ["Regime", "detect_regime", "decide_trade"]
