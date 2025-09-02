from __future__ import annotations
import math
from typing import Dict, Iterable

from .options import max_pain, atm_strike


def max_pain_smoothed(chain: Dict, spot: float, step: int, smooth: int = 3) -> int:
    """Moving-average of Max Pain over ``smooth`` strikes."""
    raw = []
    strikes = sorted(int(k) for k in chain["strikes"])
    for i in range(len(strikes) - smooth + 1):
        sub = {
            "strikes": strikes[i : i + smooth],
            "calls": {k: chain["calls"][k] for k in strikes[i : i + smooth]},
            "puts": {k: chain["puts"][k] for k in strikes[i : i + smooth]},
        }
        raw.append(max_pain(sub, spot, step))
    return int(sum(raw) / len(raw)) if raw else 0


def forward_atm(spot: float, r: float, div: float, tau: float, strikes: Iterable[int]) -> int:
    """ATM strike picked against forward price F=S*exp((r-div)*tau)."""
    forward = spot * math.exp((r - div) * tau)
    return atm_strike(forward, strikes)


def pcr_variants(chain: Dict, atm: int, step: int, k: int = 2) -> Dict[str, float]:
    """Return total, banded and OI-weighted PCR."""
    strikes = chain["strikes"]
    lo = atm - k * step
    hi = atm + k * step
    ce_tot = pe_tot = ce_band = pe_band = ce_w = pe_w = 0
    for s in strikes:
        ce = chain["calls"][s]["oi"]
        pe = chain["puts"][s]["oi"]
        ce_tot += ce
        pe_tot += pe
        ce_w += ce * s
        pe_w += pe * s
        if lo < s < hi:
            ce_band += ce
            pe_band += pe
    res = {
        "pcr_total": pe_tot / ce_tot if ce_tot else math.nan,
        "pcr_band": pe_band / ce_band if ce_band else math.nan,
        "pcr_w": (pe_w / pe_tot) / (ce_w / ce_tot) if ce_tot and pe_tot else math.nan,
    }
    return res


__all__ = ["max_pain_smoothed", "forward_atm", "pcr_variants"]
