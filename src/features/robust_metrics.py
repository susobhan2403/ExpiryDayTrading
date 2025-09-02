from __future__ import annotations

"""Robust option metrics for Indian index options.

This module implements forward-based ATM strike selection, implied volatility
solvers and ancillary helpers.  The functions follow the conventions specified
in the project specification and are designed to be deterministic and fail
closed on bad inputs.
"""

import math
import datetime as dt
from collections import Counter
from typing import Dict, Iterable, Mapping, Sequence, Tuple, Literal


def detect_strike_step(strikes: Sequence[float]) -> int:
    """Infer the strike step size from a sequence of strikes.

    The step is computed as the statistical mode of adjacent differences.
    When the mode is ambiguous the smallest difference is returned.  The
    result is always an integer number of points.
    """
    arr = sorted(float(k) for k in strikes)
    if len(arr) < 2:
        return 0
    diffs = [int(round(b - a)) for a, b in zip(arr, arr[1:]) if b > a]
    if not diffs:
        return 0
    step, _ = Counter(diffs).most_common(1)[0]
    return int(step)


def pick_expiry(now_ist: dt.datetime, expiries: Sequence[dt.datetime], min_tau_h: float) -> dt.datetime:
    """Select the nearest non-expired expiry with minimum time-to-expiry guard.

    Parameters
    ----------
    now_ist: timezone-aware ``datetime`` in IST.
    expiries: iterable of expiry datetimes (also in IST).
    min_tau_h: minimum required time to expiry in hours; if the nearest expiry
        falls short the next one is selected.
    """
    future = [e for e in sorted(expiries) if e > now_ist]
    if not future:
        raise ValueError("no valid expiries")
    first = future[0]
    tau_h = (first - now_ist).total_seconds() / 3600.0
    if tau_h < min_tau_h and len(future) > 1:
        return future[1]
    return first


def compute_forward(spot: float, fut_mid: float | None, r: float, q: float, tau_y: float) -> float:
    """Return the forward price using futures when available.

    ``fut_mid`` is preferred when provided; otherwise ``F = S*exp((r-q)*tau)``.
    """
    if fut_mid is not None and fut_mid > 0:
        return float(fut_mid)
    return float(spot * math.exp((r - q) * tau_y))


def _bs_price_forward(F: float, K: float, tau: float, sigma: float, opt_type: str) -> float:
    if tau <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return 0.0
    sqrtT = math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * tau) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    if opt_type == "C":
        return F * N(d1) - K * N(d2)
    else:
        return K * N(-d2) - F * N(-d1)


def pick_atm_strike(F: float, strikes: Sequence[float], step: int,
                     ce_mid: Mapping[float, float], pe_mid: Mapping[float, float]) -> Tuple[float, Dict]:
    """Pick the at-the-money strike against the forward price ``F``.

    The strike nearest to ``F`` is chosen.  When ``F`` lies exactly halfway
    between two strikes the straddle mid (call + put) is compared against a
    theoretical ATM straddle priced with ``σ=20%`` and ``τ=1/52``.  The strike
    whose straddle is closer to the theoretical value wins; any remaining tie
    is resolved in favour of the higher strike.
    """
    ks = sorted(float(k) for k in strikes)
    if not ks:
        return 0.0, {"notes": ["no strikes"]}
    # snap forward to grid for diagnostics
    snapped = round(F / step) * step if step > 0 else F
    # find nearest strikes
    lower = max([k for k in ks if k <= F], default=None)
    upper = min([k for k in ks if k >= F], default=None)
    diag: Dict[str, object] = {"F": F, "snapped": snapped, "candidates": []}
    if lower is None and upper is None:
        return 0.0, diag
    if lower is None or (upper is not None and abs(upper - F) < abs(F - lower)):
        diag["candidates"].append(upper)
        return upper, diag
    if upper is None or abs(F - lower) < abs(upper - F):
        diag["candidates"].append(lower)
        return lower, diag
    # Tie: evaluate straddle mids
    diag["candidates"] = [lower, upper]
    theo = 2.0 * _bs_price_forward(F, F, 1 / 52, 0.2, "C")
    m_lower = ce_mid.get(lower, math.nan) + pe_mid.get(lower, math.nan)
    m_upper = ce_mid.get(upper, math.nan) + pe_mid.get(upper, math.nan)
    diff_lower = abs(m_lower - theo)
    diff_upper = abs(m_upper - theo)
    diag["theo_straddle"] = theo
    diag["straddle_mid"] = {lower: m_lower, upper: m_upper}
    if diff_lower < diff_upper:
        diag["tie_break"] = "lower"
        return lower, diag
    if diff_upper < diff_lower:
        diag["tie_break"] = "upper"
        return upper, diag
    diag["tie_break"] = "higher_preferred"
    return max(lower, upper), diag


def implied_vol_bs(price: float, S_or_F: float, K: float, tau_y: float, r: float, q: float,
                   opt_type: Literal["C", "P"]) -> Tuple[float, Dict]:
    """Black–Scholes IV solver using a bisection/Brent hybrid.

    Parameters
    ----------
    price: option premium (undiscounted).
    S_or_F: spot or forward price.  The forward is derived internally when
        ``q`` is provided.
    tau_y: time to expiry in years.
    opt_type: ``"C"`` for calls or ``"P"`` for puts.
    """
    F = S_or_F * math.exp((r - q) * tau_y)
    df = math.exp(-r * tau_y)
    def f(sig: float) -> float:
        return df * _bs_price_forward(F, K, tau_y, sig, opt_type) - price
    lo, hi = 1e-4, 5.0
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return float("nan"), {"converged": False, "iterations": 0}
    mid = lo
    for i in range(100):
        # Brent step
        if f_hi != f_lo:
            mid = hi - f_hi * (hi - lo) / (f_hi - f_lo)
        else:
            mid = 0.5 * (hi + lo)
        fm = f(mid)
        if abs(fm) < 1e-6:
            return mid, {"converged": True, "iterations": i + 1}
        if f_lo * fm > 0:
            lo, f_lo = mid, fm
        else:
            hi, f_hi = mid, fm
        if hi - lo < 1e-6:
            return 0.5 * (hi + lo), {"converged": True, "iterations": i + 1}
    return 0.5 * (hi + lo), {"converged": False, "iterations": 100}


def compute_atm_iv(ce_mid: float | None, pe_mid: float | None, S: float, K_atm: float,
                   tau_y: float, r: float, q: float, F: float | None = None) -> Tuple[float, Dict]:
    """Compute ATM implied volatility from call/put mids.

    IVs are solved separately for the call and put.  When both are available
    and within 15 volatility points the average is returned.  Outliers are
    discarded.  Forward-based pricing is used throughout.
    """
    F = compute_forward(S, F, r, q, tau_y)
    diag: Dict[str, object] = {"F": F, "K": K_atm, "iv_ce": math.nan, "iv_pe": math.nan}
    ivs = []
    if ce_mid and ce_mid == ce_mid:
        iv_ce, d_ce = implied_vol_bs(ce_mid, S, K_atm, tau_y, r, q, "C")
        diag["iv_ce"], diag["ce_iters"] = iv_ce, d_ce.get("iterations")
        if iv_ce == iv_ce and iv_ce > 0:
            ivs.append(iv_ce)
    if pe_mid and pe_mid == pe_mid:
        iv_pe, d_pe = implied_vol_bs(pe_mid, S, K_atm, tau_y, r, q, "P")
        diag["iv_pe"], diag["pe_iters"] = iv_pe, d_pe.get("iterations")
        if iv_pe == iv_pe and iv_pe > 0:
            ivs.append(iv_pe)
    if len(ivs) == 2 and abs(ivs[0] - ivs[1]) <= 0.15:
        atm_iv = sum(ivs) / 2.0
    elif ivs:
        atm_iv = ivs[0] if diag["iv_ce"] == diag["iv_ce"] else diag["iv_pe"]
    else:
        atm_iv = float("nan")
    diag["atm_iv"] = atm_iv
    return atm_iv, diag


def compute_iv_percentile(series: Sequence[float], current: float) -> Tuple[float, float]:
    """Return (percentile, rank) for ``current`` within ``series``.

    Percentile counts how many historic values fall strictly below ``current``.
    Rank measures the position within the min/max range.
    """
    vals = [v for v in series if v == v]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    pct = 100.0 * sum(1 for v in vals if v < current) / n
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return pct, float("nan")
    rank = 100.0 * (current - lo) / (hi - lo)
    return pct, rank


def compute_pcr(oi_put: Mapping[float, float], oi_call: Mapping[float, float],
                strikes: Sequence[float], K_atm: float, step: int, m: int = 6) -> Dict[str, float]:
    """Compute put/call ratios over the entire chain and an active band.

    ``oi_put`` and ``oi_call`` map strikes to open interest.  ``strikes`` should
    contain the union of available strikes.  ``K_atm`` is the selected ATM
    strike, ``step`` the strike spacing and ``m`` the half-width of the band in
    steps (default six).
    """
    tot_p = tot_c = band_p = band_c = 0.0
    lo = K_atm - m * step
    hi = K_atm + m * step
    band_count = 0
    for k in strikes:
        p = float(oi_put.get(k, 0))
        c = float(oi_call.get(k, 0))
        tot_p += p
        tot_c += c
        if lo <= k <= hi:
            band_p += p
            band_c += c
            band_count += 1
    res = {
        "PCR_OI_total": (tot_p / tot_c) if tot_c > 0 else float("nan"),
        "PCR_OI_band": (band_p / band_c) if band_c > 0 else float("nan"),
        "band_lo": lo,
        "band_hi": hi,
        "band_count": band_count,
    }
    return res


__all__ = [
    "detect_strike_step",
    "pick_expiry",
    "compute_forward",
    "pick_atm_strike",
    "implied_vol_bs",
    "compute_atm_iv",
    "compute_iv_percentile",
    "compute_pcr",
]
