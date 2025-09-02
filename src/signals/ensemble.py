from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class SignalInputs:
    trend: float
    momentum: float
    mean_reversion: float
    breakout: float
    options_flow: float
    basis_change: float
    breadth: float


def ranked_ensemble(inp: SignalInputs) -> Dict[str, float]:
    """Rank weighted sum of individual factors."""
    weights = {
        "trend": 0.2,
        "momentum": 0.15,
        "mean_reversion": -0.1,
        "breakout": 0.25,
        "options_flow": 0.15,
        "basis_change": 0.1,
        "breadth": 0.05,
    }
    score = sum(getattr(inp, k) * w for k, w in weights.items())
    return {"score": score, "direction": "long" if score > 0 else "short"}


__all__ = ["SignalInputs", "ranked_ensemble"]
