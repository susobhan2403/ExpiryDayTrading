from __future__ import annotations
import math

def vol_target_size(target_vol: float, realized_vol: float, base_risk: float = 1.0) -> float:
    rv = max(1e-6, realized_vol)
    return float(base_risk * target_vol / rv)

def capped_kelly(p: float, b: float = 1.0, cap: float = 0.1) -> float:
    """Kelly fraction with cap. p=win probability (0..1), b=odds (default 1)."""
    p = max(0.0, min(1.0, p))
    q = 1.0 - p
    k = (b*p - q) / b
    return float(max(0.0, min(cap, k)))

