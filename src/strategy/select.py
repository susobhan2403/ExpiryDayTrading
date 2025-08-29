from __future__ import annotations
from typing import Tuple

def select_strategy(scp: float, erp: float, ivp: float, mins_to_close: float, zero_gamma_dist: float) -> str:
    """Return a concise strategy suggestion string.
    scp: short-covering prob [0,1]
    erp: expiry reversion prob [0,1]
    ivp: IV percentile (0-100)
    mins_to_close: minutes to market close
    zero_gamma_dist: distance to zero-gamma in points
    """
    try:
        ivp = float(ivp)
    except Exception:
        ivp = 50.0
    trend_bias = scp >= 0.6 and erp < 0.5
    revert_bias = erp >= 0.6 and scp < 0.5
    tight_gamma = abs(zero_gamma_dist) <= 50.0
    if trend_bias:
        if ivp <= 35:
            return "Trend bias: Debit verticals (delta 0.25–0.35)"
        else:
            return "Trend bias: Calendars (term skew) or light debit"
    if revert_bias:
        if ivp >= 60:
            return "Mean-revert: Credit spreads (iron condor/butterfly), risk-defined"
        else:
            return "Cautious: Wait for clearer IVP/Pin signals"
    if tight_gamma:
        return "Gamma flip risk: scalp biased, tighter targets, avoid overnight"
    if mins_to_close < 120 and ivp >= 60:
        return "Late session & IV high: prefer credit, avoid long premium"
    return "Neutral: Small debit or stay flat pending signals"

