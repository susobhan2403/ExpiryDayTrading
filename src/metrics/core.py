from __future__ import annotations
"""Core option metrics and gating utilities.

This module implements robust, production quality calculations for
Indian index options following strict conventions.
All functions are deterministic and never return ``NaN``; invalid
results are represented by ``None`` accompanied by human readable
reasons in a diagnostics dictionary.
"""

import math
import datetime as dt
from collections import Counter
from typing import Dict, Iterable, Mapping, Sequence, Tuple, Literal, Optional, List


def infer_strike_step(strikes: Sequence[float]) -> int:
    """Infer strike spacing from a list of strikes.

    The step is the statistical mode of positive adjacent differences
    rounded to the nearest integer.  If less than two strikes are
    supplied the function returns ``0``.
    """
    arr = sorted(float(k) for k in strikes)
    if len(arr) < 2:
        return 0
    diffs = [int(round(b - a)) for a, b in zip(arr, arr[1:]) if b > a]
    if not diffs:
        return 0
    step, _ = Counter(diffs).most_common(1)[0]
    return int(step)


def choose_expiry(now_ist: dt.datetime, expiries: Sequence[dt.datetime], min_tau_h: float) -> dt.datetime:
    """Select the nearest non-expired expiry with a minimum time-to-expiry.

    Parameters
    ----------
    now_ist: timezone aware ``datetime`` in IST.
    expiries: iterable of expiry datetimes (also IST aware).
    min_tau_h: minimum permissible hours to expiry.  If the nearest expiry
        falls short the next one is chosen.  A ``ValueError`` is raised
        when no valid expiries remain.
    """
    future = [e for e in sorted(expiries) if e > now_ist]
    if not future:
        raise ValueError("no valid expiries")
    first = future[0]
    tau_h = (first - now_ist).total_seconds() / 3600.0
    if tau_h < min_tau_h and len(future) > 1:
        return future[1]
    return first


def compute_forward(spot: float, fut_mid: Optional[float], r: float, q: float, tau_years: float) -> float:
    """Return the forward price.

    ``fut_mid`` is preferred when provided and positive.  Otherwise the
    cost-of-carry relation ``F = S * exp((r-q)*tau)`` is used.
    """
    if fut_mid is not None and fut_mid > 0:
        return float(fut_mid)
    return float(spot * math.exp((r - q) * tau_years))


def _bs_price_forward(F: float, K: float, tau: float, sigma: float, opt_type: Literal["C", "P"]) -> float:
    """Black–Scholes price in forward measure without discounting."""
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
    """Select the ATM strike against forward ``F``.

    The nearest strike to ``F`` is chosen.  When equidistant, a tie break is
    performed using straddle mids compared with a theoretical ATM straddle
    priced with ``σ=20%`` and ``τ=1/52``.  Remaining ties favour the higher
    strike.  Diagnostics include tie break information and the forward snap
    grid.
    """
    ks = sorted(float(k) for k in strikes)
    diag: Dict[str, object] = {"F": F, "candidates": []}
    if not ks:
        diag["notes"] = ["no strikes"]
        return 0.0, diag
    snapped = round(F / step) * step if step > 0 else F
    lower = max([k for k in ks if k <= F], default=None)
    upper = min([k for k in ks if k >= F], default=None)
    if lower is None and upper is None:
        return 0.0, diag
    if lower is None or (upper is not None and abs(upper - F) < abs(F - lower)):
        diag["candidates"].append(upper)
        return upper, diag
    if upper is None or abs(F - lower) < abs(upper - F):
        diag["candidates"].append(lower)
        return lower, diag
    # tie
    theo = 2.0 * _bs_price_forward(F, F, 1/52, 0.2, "C")
    m_lower = ce_mid.get(lower, math.nan) + pe_mid.get(lower, math.nan)
    m_upper = ce_mid.get(upper, math.nan) + pe_mid.get(upper, math.nan)
    diff_lower = abs(m_lower - theo)
    diff_upper = abs(m_upper - theo)
    diag.update({"candidates": [lower, upper],
                 "theo_straddle": theo,
                 "straddle_mid": {lower: m_lower, upper: m_upper}})
    if diff_lower < diff_upper:
        diag["tie_break"] = "lower"
        return lower, diag
    if diff_upper < diff_lower:
        diag["tie_break"] = "upper"
        return upper, diag
    diag["tie_break"] = "higher_preferred"
    return max(lower, upper), diag


def implied_vol(price: float, F: float, K: float, tau_years: float, r: float,
                 opt_type: Literal["C", "P"],
                 tol: float = 1e-6, max_iter: int = 100) -> Tuple[Optional[float], Dict]:
    """Solve for implied volatility using a Brent/Bisection hybrid.

    The function prices options in the forward measure and discounts using
    ``r``.  Bounds are ``σ∈[1e-4, 5.0]``.  ``None`` is returned on failure
    together with diagnostics ``{"converged": False, "reason": ...}``.
    """
    if price is None or price <= 0 or F <= 0 or K <= 0 or tau_years <= 0:
        return None, {"converged": False, "reason": "invalid_inputs"}
    df = math.exp(-r * tau_years)
    target = price / df
    def f(sig: float) -> float:
        return _bs_price_forward(F, K, tau_years, sig, opt_type) - target
    lo, hi = 1e-4, 5.0
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return None, {"converged": False, "reason": "bad_bracket"}
    a, b = lo, hi
    fa, fb = f_lo, f_hi
    for i in range(max_iter):
        # Brent step
        if fb != fa:
            c = b - fb * (b - a) / (fb - fa)
        else:
            c = 0.5 * (a + b)
        fc = f(c)
        if abs(fc) < tol:
            return c, {"converged": True, "iterations": i + 1}
        if fa * fc > 0:
            a, fa = c, fc
        else:
            b, fb = c, fc
        if b - a < tol:
            return 0.5 * (a + b), {"converged": True, "iterations": i + 1}
    return 0.5 * (a + b), {"converged": False, "reason": "max_iter"}


def compute_atm_iv(ce_mid: Optional[float], pe_mid: Optional[float], F: float,
                   K_atm: float, tau_years: float, r: float) -> Tuple[Optional[float], Dict]:
    """Compute ATM IV using call and put mids.

    IVs are solved separately.  If both succeed and differ by no more than
    0.15 volatility points the average is returned.  Otherwise the valid leg
    is used.  When neither leg is usable ``None`` is returned with reasons.
    """
    diag: Dict[str, object] = {"iv_ce": None, "iv_pe": None}
    ivs: List[float] = []
    if ce_mid is not None:
        iv_ce, d = implied_vol(ce_mid, F, K_atm, tau_years, r, "C")
        diag["iv_ce"], diag["ce_diag"] = iv_ce, d
        if iv_ce is not None:
            ivs.append(iv_ce)
    if pe_mid is not None:
        iv_pe, d = implied_vol(pe_mid, F, K_atm, tau_years, r, "P")
        diag["iv_pe"], diag["pe_diag"] = iv_pe, d
        if iv_pe is not None:
            ivs.append(iv_pe)
    if len(ivs) == 2 and abs(ivs[0] - ivs[1]) <= 0.15:
        atm_iv = sum(ivs) / 2.0
    elif ivs:
        atm_iv = ivs[0]
    else:
        diag["reason"] = "no_valid_legs"
        return None, diag
    diag["atm_iv"] = atm_iv
    return atm_iv, diag


def compute_iv_stats(history: Sequence[float], current: Optional[float]) -> Dict:
    """Return IV percentile and rank without ``NaN``.

    ``history`` should contain past ATM IV observations.  ``current`` may be
    ``None``.  The function returns ``{"percentile": float|None, "iv_rank": float|None,
    "reasons": [...]}``.
    """
    vals = [v for v in history if v is not None and v == v]
    out: Dict[str, object] = {"percentile": None, "iv_rank": None, "reasons": []}
    if current is None or current != current:
        out["reasons"].append("current_missing")
        return out
    n = len(vals)
    if n == 0:
        out["reasons"].append("no_history")
        return out
    out["percentile"] = 100.0 * sum(1 for v in vals if v < current) / n
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        out["reasons"].append("zero-range")
        return out
    out["iv_rank"] = 100.0 * (current - lo) / (hi - lo)
    return out


def compute_pcr(oi_put: Mapping[float, int], oi_call: Mapping[float, int],
                strikes: Sequence[float], K_atm: float, step: int, m: int = 6) -> Dict:
    """Compute put–call ratios for the full chain and a band around ATM.

    Parameters
    ----------
    oi_put, oi_call: mappings of strike to open interest.
    strikes: iterable of available strikes.
    K_atm: chosen ATM strike.
    step: strike step size.
    m: half-width of band in steps (default six).
    """
    res: Dict[str, object] = {"PCR_OI_total": None, "PCR_OI_band": None,
                              "band_lo": K_atm - m * step,
                              "band_hi": K_atm + m * step,
                              "band_count": 0, "reasons": []}
    tot_p = tot_c = band_p = band_c = 0.0
    lo, hi = res["band_lo"], res["band_hi"]
    for k in strikes:
        p = float(oi_put.get(k, 0) or 0)
        c = float(oi_call.get(k, 0) or 0)
        if p <= 0 and c <= 0:
            continue  # ghost strike
        if p > 0:
            tot_p += p
        if c > 0:
            tot_c += c
        if lo <= k <= hi and p > 0 and c > 0:
            band_p += p
            band_c += c
            res["band_count"] += 1
    if tot_c > 0:
        res["PCR_OI_total"] = tot_p / tot_c
    else:
        res["reasons"].append("no_calls_total")
    if band_c > 0 and res["band_count"] >= 2:
        res["PCR_OI_band"] = band_p / band_c
    else:
        res["reasons"].append("insufficient_band_data")
    return res


# ---- gating utilities -----------------------------------------------------

class GateDecision:
    """Simple container for gate outcomes."""
    def __init__(self, muted: bool, size_factor: float = 1.0, override: bool = False, reason: str | None = None):
        self.muted = muted
        self.size_factor = size_factor
        self.override = override
        self.reason = reason

    def as_dict(self) -> Dict[str, object]:
        return {"muted": self.muted, "size_factor": self.size_factor,
                "override": self.override, "reason": self.reason}


def apply_gates(signals: Sequence[str], spike_label: str, confirming_bars: int,
                min_confirm: int = 2) -> GateDecision:
    """Apply decision gates with override rules.

    Parameters
    ----------
    signals: list of independent signal names that agree on direction.
    spike_label: output of spike classifier ``{MICROSTRUCTURE, NEWS_SHOCK, VALID_IMPULSE}``.
    confirming_bars: number of consecutive confirming bars.
    min_confirm: required confirmations when override is not triggered.

    Rules
    -----
    - If ``spike_label`` is ``MICROSTRUCTURE`` the trade is muted.
    - If at least three independent signals align, the gate is overridden
      unless the spike is ``MICROSTRUCTURE``.  ``NEWS_SHOCK`` halves size but
      still allows trading.
    - Otherwise at least ``min_confirm`` confirming bars are required.
    """
    if spike_label == "MICROSTRUCTURE":
        return GateDecision(True, reason="microstructure")
    if len(signals) >= 3:
        factor = 0.5 if spike_label == "NEWS_SHOCK" else 1.0
        return GateDecision(False, size_factor=factor, override=True)
    if confirming_bars >= min_confirm:
        return GateDecision(False)
    return GateDecision(True, reason="insufficient_confirmation")


__all__ = [
    "infer_strike_step",
    "choose_expiry",
    "compute_forward",
    "pick_atm_strike",
    "implied_vol",
    "compute_atm_iv",
    "compute_iv_stats",
    "compute_pcr",
    "apply_gates",
    "GateDecision",
]
