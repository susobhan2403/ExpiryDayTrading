from __future__ import annotations
from typing import Dict, Tuple

def _get_oi(node):
    try:
        return int(node.get("oi", 0))
    except Exception:
        return 0

def compute_voi(prev_chain: Dict, curr_chain: Dict, dt_minutes: float, atm: int, step: int) -> Dict[str, float]:
    """
    VOI (velocity of OI) per side near ATM bands and totals.
    Returns per-minute rates to make it interval-agnostic.
    """
    if not prev_chain or not curr_chain or dt_minutes <= 0:
        return {"voi_ce_atm": 0.0, "voi_pe_atm": 0.0, "voi_ce_total": 0.0, "voi_pe_total": 0.0}

    # Collect unions of strikes
    strikes = set(curr_chain.get("calls", {}).keys()) | set(curr_chain.get("puts", {}).keys())
    # Some saved prev chains may have str keys
    prev_calls = prev_chain.get("calls", {})
    prev_puts = prev_chain.get("puts", {})

    # Define ATM bands: +/− one step around ATM
    band_up = atm + step
    band_dn = atm - step

    d_ce_total = 0
    d_pe_total = 0
    d_ce_atm = 0
    d_pe_atm = 0

    for k in strikes:
        try:
            k_int = int(k)
        except Exception:
            continue
        ce_now = _get_oi(curr_chain.get("calls", {}).get(k, {}))
        pe_now = _get_oi(curr_chain.get("puts", {}).get(k, {}))
        ce_prev = _get_oi(prev_calls.get(str(k), prev_calls.get(k, {})))
        pe_prev = _get_oi(prev_puts.get(str(k), prev_puts.get(k, {})))
        d_ce = ce_now - ce_prev
        d_pe = pe_now - pe_prev
        d_ce_total += d_ce
        d_pe_total += d_pe
        if k_int in (band_dn, atm, band_up):
            d_ce_atm += d_ce
            d_pe_atm += d_pe

    rate = 1.0 / dt_minutes
    return {
        "voi_ce_atm": d_ce_atm * rate,
        "voi_pe_atm": d_pe_atm * rate,
        "voi_ce_total": d_ce_total * rate,
        "voi_pe_total": d_pe_total * rate,
    }

def oiwap(chain: Dict) -> float:
    """OI-weighted average strike across calls+puts."""
    if not chain:
        return float("nan")
    total = 0
    num = 0
    for k, v in chain.get("calls", {}).items():
        try:
            ki = int(k); oi = _get_oi(v)
            total += oi
            num += oi * ki
        except Exception:
            pass
    for k, v in chain.get("puts", {}).items():
        try:
            ki = int(k); oi = _get_oi(v)
            total += oi
            num += oi * ki
        except Exception:
            pass
    if total <= 0:
        return float("nan")
    return float(num / total)

def pin_density(chain: Dict, atm: int, step: int, n: int = 3) -> float:
    """
    Pin risk density around ATM: normalized OI around spot over a window of ±n*step.
    """
    if not chain or not chain.get("calls") or not chain.get("puts"):
        return float("nan")
    window = [atm + i*step for i in range(-n, n+1)]
    tot = 0
    focus = 0
    for k in chain.get("calls", {}).keys():
        try:
            ki = int(k)
        except Exception:
            continue
        oi = _get_oi(chain["calls"].get(ki, {})) + _get_oi(chain["puts"].get(ki, {}))
        tot += oi
        if ki in window:
            focus += oi
    if tot <= 0:
        return float("nan")
    return float(focus / tot)

