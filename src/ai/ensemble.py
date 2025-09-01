from __future__ import annotations
from typing import Dict

def ai_predict_probs(
    symbol: str,
    above_vwap: bool,
    below_vwap: bool,
    macd_up: bool,
    macd_dn: bool,
    bull_flow: bool,
    bear_flow: bool,
    breakout_up: bool,
    breakdown: bool,
    adx5: float,
    dpcr_z: float,
    iv_z: float,
    mph_norm: float,
    VND: float,
) -> Dict[str, float]:
    def nz(x):
        return 0.0 if x!=x else x
    adx_s = max(0.0, (nz(adx5) - 18.0) / 22.0)
    pcr_s_up = max(0.0, nz(dpcr_z)) / 2.0
    pcr_s_dn = max(0.0, -nz(dpcr_z)) / 2.0
    iv_s = min(1.0, abs(nz(iv_z)) / 2.0)
    drift_up = max(0.0, nz(mph_norm))
    drift_dn = max(0.0, -nz(mph_norm))
    range_s = max(0.0, (0.4 - nz(VND))) / max(0.4, 1e-9)

    bull = (0.25*float(above_vwap) + 0.2*float(macd_up) + 0.25*float(bull_flow)
            + 0.15*pcr_s_up + 0.15*drift_up + 0.2*adx_s)
    bear = (0.25*float(below_vwap) + 0.2*float(macd_dn) + 0.25*float(bear_flow)
            + 0.15*pcr_s_dn + 0.15*drift_dn + 0.2*adx_s)
    squeeze = 0.6*float(breakout_up or breakdown) + 0.4*adx_s
    pin = 0.7*range_s + 0.3*(1.0 - adx_s)
    event = 1.0 if abs(nz(iv_z)) >= 2.0 and abs(nz(dpcr_z)) < 0.5 else 0.0

    raw = {
        "Short-cover reversion up": max(0.0, bull),
        "Bear migration": max(0.0, bear),
        "Bull migration / gamma carry": max(0.0, bull + 0.2*drift_up),
        "Pin & decay day (IV crush)": max(0.0, pin),
        "Squeeze continuation (one-way)": max(0.0, squeeze),
        "Event knee-jerk then revert": max(0.0, event),
    }
    vals = list(raw.values())
    if sum(vals) <= 1e-9:
        out = {k: 1.0/len(raw) for k in raw}
    else:
        s = sum(vals)
        out = {k: v/s for k, v in raw.items()}
    return out

def blend_probs(rule_probs: Dict[str,float], ai_probs: Dict[str,float], adx5: float, dpcr_z: float, iv_z: float, mph_norm: float) -> Dict[str,float]:
    def clamp(x,a,b):
        return max(a, min(b, x))
    strength = (
        max(0.0, (adx5 - 18.0)/22.0) + min(1.0, abs(dpcr_z)/2.0) + min(1.0, abs(iv_z)/2.0) + min(1.0, abs(mph_norm))
    ) / 4.0
    alpha = clamp(0.2 + 0.5*strength, 0.2, 0.6)
    keys = rule_probs.keys()
    out = {k: clamp((1-alpha)*rule_probs.get(k,0.0) + alpha*ai_probs.get(k,0.0), 0.0, 1.0) for k in keys}
    s = sum(out.values()) or 1.0
    return {k: round(v/s,3) for k,v in out.items()}

