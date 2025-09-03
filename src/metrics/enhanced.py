"""Enhanced option metrics with India-specific conventions and robust error handling.

This module provides production-grade implementations that:
- Follow Indian index options conventions
- Use forward-based ATM selection
- Provide robust IV solving with proper brackets
- Include tenor-filtered IV percentiles
- Return diagnostics instead of NaN
- Support deterministic behavior for backtesting
"""

from __future__ import annotations

import math
import datetime as dt
from collections import Counter
from typing import Dict, Iterable, Mapping, Sequence, Tuple, Literal, Optional, List

import pytz

IST = pytz.timezone("Asia/Kolkata")


def validate_inputs(*args, **kwargs) -> Tuple[bool, str]:
    """Validate numeric inputs for calculations."""
    for arg in args:
        if arg is None or (isinstance(arg, float) and (math.isnan(arg) or math.isinf(arg))):
            return False, f"invalid_input: {arg}"
        if isinstance(arg, (int, float)) and arg < 0:
            return False, f"negative_input: {arg}"
    return True, "valid"


def infer_strike_step_enhanced(strikes: Sequence[float]) -> Tuple[int, Dict]:
    """Enhanced strike step inference with diagnostics.
    
    Returns the most common positive difference between adjacent strikes.
    For Indian markets, typical steps are: NIFTY=50, BANKNIFTY=100, SENSEX=100.
    """
    diag = {"method": "modal_difference", "candidates": [], "confidence": 0.0}
    
    arr = sorted(float(k) for k in strikes if isinstance(k, (int, float)) and not math.isnan(k))
    if len(arr) < 2:
        diag["reason"] = "insufficient_strikes"
        return 0, diag
    
    diffs = []
    for a, b in zip(arr, arr[1:]):
        diff = int(round(b - a))
        if diff > 0:
            diffs.append(diff)
            
    if not diffs:
        diag["reason"] = "no_positive_differences"
        return 0, diag
    
    counter = Counter(diffs)
    diag["candidates"] = list(counter.most_common())
    
    step, count = counter.most_common(1)[0]
    diag["confidence"] = count / len(diffs)
    diag["selected_step"] = step
    
    return int(step), diag


def compute_forward_enhanced(spot: float, fut_mid: Optional[float], r: float, q: float, tau_years: float) -> Tuple[float, Dict]:
    """Enhanced forward price computation with Indian market conventions."""
    diag = {"method": "unknown", "inputs": {"spot": spot, "fut_mid": fut_mid, "r": r, "q": q, "tau_years": tau_years}}
    
    valid, reason = validate_inputs(spot, r, q, tau_years)
    if not valid:
        diag["reason"] = reason
        return 0.0, diag
    
    # Prefer futures price when available (common in Indian markets)
    if fut_mid is not None and fut_mid > 0:
        diag["method"] = "futures_mid"
        diag["forward"] = float(fut_mid)
        return float(fut_mid), diag
    
    # Use cost-of-carry model
    try:
        forward = float(spot * math.exp((r - q) * tau_years))
        diag["method"] = "cost_of_carry"
        diag["forward"] = forward
        return forward, diag
    except (OverflowError, ValueError) as e:
        diag["reason"] = f"calculation_error: {e}"
        return 0.0, diag


def pick_atm_strike_enhanced(F: float, strikes: Sequence[float], step: int,
                           ce_mid: Mapping[float, float], pe_mid: Mapping[float, float]) -> Tuple[float, Dict]:
    """Enhanced ATM strike selection using forward price with Indian market tie-breaking."""
    diag = {"F": F, "method": "forward_based", "candidates": [], "tie_break": None}
    
    valid, reason = validate_inputs(F)
    if not valid:
        diag["reason"] = reason
        return 0.0, diag
    
    ks = sorted(float(k) for k in strikes if isinstance(k, (int, float)) and not math.isnan(k))
    if not ks:
        diag["reason"] = "no_valid_strikes"
        return 0.0, diag
    
    # Find strikes closest to forward
    lower = max([k for k in ks if k <= F], default=None)
    upper = min([k for k in ks if k >= F], default=None)
    
    diag["lower_candidate"] = lower
    diag["upper_candidate"] = upper
    
    if lower is None and upper is None:
        diag["reason"] = "no_bracketing_strikes"
        return 0.0, diag
    
    if lower is None:
        diag["candidates"] = [upper]
        return upper, diag
    
    if upper is None:
        diag["candidates"] = [lower]
        return lower, diag
    
    # Check if one is clearly closer
    diff_lower = abs(F - lower)
    diff_upper = abs(upper - F)
    
    if diff_lower < diff_upper:
        diag["candidates"] = [lower]
        diag["selection_reason"] = "closer_to_forward"
        return lower, diag
    
    if diff_upper < diff_lower:
        diag["candidates"] = [upper]
        diag["selection_reason"] = "closer_to_forward"
        return upper, diag
    
    # Tie-breaking using straddle comparison (Indian market convention)
    diag["candidates"] = [lower, upper]
    diag["tie_detected"] = True
    
    # Use theoretical ATM straddle with 20% vol and 1-week expiry for comparison
    tau_ref = 1.0 / 52.0  # 1 week reference
    sig_ref = 0.20  # 20% reference volatility
    
    def theo_straddle(S: float, K: float) -> float:
        if tau_ref <= 0 or sig_ref <= 0:
            return 0.0
        sqrtT = math.sqrt(tau_ref)
        d1 = (math.log(S / K) + 0.5 * sig_ref * sig_ref * tau_ref) / (sig_ref * sqrtT)
        d2 = d1 - sig_ref * sqrtT
        N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        call = S * N(d1) - K * N(-d2)
        put = K * N(-d2) - S * N(-d1)
        return call + put
    
    theo_lower = theo_straddle(F, lower)
    theo_upper = theo_straddle(F, upper)
    
    market_lower = ce_mid.get(lower, 0.0) + pe_mid.get(lower, 0.0)
    market_upper = ce_mid.get(upper, 0.0) + pe_mid.get(upper, 0.0)
    
    diag["theo_straddles"] = {"lower": theo_lower, "upper": theo_upper}
    diag["market_straddles"] = {"lower": market_lower, "upper": market_upper}
    
    if market_lower > 0 and market_upper > 0:
        diff_lower_theo = abs(market_lower - theo_lower)
        diff_upper_theo = abs(market_upper - theo_upper)
        
        if diff_lower_theo < diff_upper_theo:
            diag["tie_break"] = "better_straddle_fit_lower"
            return lower, diag
        elif diff_upper_theo < diff_lower_theo:
            diag["tie_break"] = "better_straddle_fit_upper"
            return upper, diag
    
    # Final tie-break: prefer higher strike (Indian convention for weekly expiries)
    diag["tie_break"] = "higher_strike_preferred"
    return max(lower, upper), diag


def compute_atm_iv_enhanced(ce_mid: Optional[float], pe_mid: Optional[float], F: float,
                          K_atm: float, tau_years: float, r: float) -> Tuple[Optional[float], Dict]:
    """Enhanced ATM IV computation with robust solving and diagnostics."""
    diag = {
        "method": "dual_leg_average",
        "inputs": {"ce_mid": ce_mid, "pe_mid": pe_mid, "F": F, "K_atm": K_atm, "tau_years": tau_years, "r": r},
        "iv_call": None,
        "iv_put": None,
        "call_diag": None,
        "put_diag": None,
        "spread_tolerance": 0.15  # 15 vol points max spread
    }
    
    valid, reason = validate_inputs(F, K_atm, tau_years, r)
    if not valid:
        diag["reason"] = reason
        return None, diag
    
    ivs = []
    
    # Solve call IV
    if ce_mid is not None and ce_mid > 0:
        iv_call, call_diag = implied_vol_enhanced(ce_mid, F, K_atm, tau_years, r, "C")
        diag["iv_call"] = iv_call
        diag["call_diag"] = call_diag
        if iv_call is not None:
            ivs.append(("call", iv_call))
    
    # Solve put IV
    if pe_mid is not None and pe_mid > 0:
        iv_put, put_diag = implied_vol_enhanced(pe_mid, F, K_atm, tau_years, r, "P")
        diag["iv_put"] = iv_put
        diag["put_diag"] = put_diag
        if iv_put is not None:
            ivs.append(("put", iv_put))
    
    if not ivs:
        diag["reason"] = "no_valid_legs"
        return None, diag
    
    if len(ivs) == 1:
        leg_type, iv = ivs[0]
        diag["method"] = f"single_leg_{leg_type}"
        diag["final_iv"] = iv
        return iv, diag
    
    # Both legs available - check spread
    call_iv = next(iv for leg, iv in ivs if leg == "call")
    put_iv = next(iv for leg, iv in ivs if leg == "put")
    spread = abs(call_iv - put_iv)
    
    diag["iv_spread"] = spread
    
    if spread <= diag["spread_tolerance"]:
        final_iv = (call_iv + put_iv) / 2.0
        diag["method"] = "averaged_legs"
        diag["final_iv"] = final_iv
        return final_iv, diag
    else:
        # Use the leg with lower IV (more conservative for Indian markets)
        chosen_iv = min(call_iv, put_iv)
        chosen_leg = "call" if call_iv < put_iv else "put"
        diag["method"] = f"conservative_leg_{chosen_leg}"
        diag["reason"] = f"spread_too_wide: {spread:.4f}"
        diag["final_iv"] = chosen_iv
        return chosen_iv, diag


def implied_vol_enhanced(price: float, F: float, K: float, tau_years: float, r: float,
                        opt_type: Literal["C", "P"], tol: float = 1e-6, max_iter: int = 100) -> Tuple[Optional[float], Dict]:
    """Enhanced implied volatility solver with robust bracketing."""
    diag = {
        "method": "brent_bisection_hybrid",
        "inputs": {"price": price, "F": F, "K": K, "tau_years": tau_years, "r": r, "opt_type": opt_type},
        "bounds": [1e-4, 5.0],
        "iterations": 0,
        "converged": False
    }
    
    valid, reason = validate_inputs(price, F, K, tau_years, r)
    if not valid or price <= 0:
        diag["reason"] = reason if not valid else "non_positive_price"
        return None, diag
    
    # Discount factor
    df = math.exp(-r * tau_years)
    target = price / df
    
    def bs_price_forward(F: float, K: float, tau: float, sigma: float, opt_type: str) -> float:
        """Black-Scholes price in forward measure."""
        if tau <= 0 or sigma <= 0 or F <= 0 or K <= 0:
            return 0.0
        
        sqrtT = math.sqrt(tau)
        d1 = (math.log(F / K) + 0.5 * sigma * sigma * tau) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        
        if opt_type == "C":
            return F * N(d1) - K * N(d2)
        else:  # opt_type == "P"
            return K * N(-d2) - F * N(-d1)
    
    def objective(sig: float) -> float:
        return bs_price_forward(F, K, tau_years, sig, opt_type) - target
    
    # Enhanced bracketing for Indian market conditions
    lo, hi = diag["bounds"]
    
    # Adaptive bounds based on market conditions
    if F > 0 and K > 0:
        moneyness = K / F
        if moneyness > 1.2 or moneyness < 0.8:  # Deep OTM
            hi = min(hi, 2.0)  # Cap volatility for deep OTM
        if tau_years < 7.0 / 365.0:  # Less than 1 week
            hi = min(hi, 1.0)  # Cap for very short expiry
    
    f_lo, f_hi = objective(lo), objective(hi)
    diag["initial_bracket"] = {"lo": lo, "hi": hi, "f_lo": f_lo, "f_hi": f_hi}
    
    if f_lo * f_hi > 0:
        diag["reason"] = "no_root_in_bracket"
        return None, diag
    
    # Brent's method with bisection fallback
    a, b = lo, hi
    fa, fb = f_lo, f_hi
    c, fc = a, fa
    mflag = True
    
    for i in range(max_iter):
        diag["iterations"] = i + 1
        
        if abs(fb) < tol:
            diag["converged"] = True
            diag["final_value"] = b
            return b, diag
        
        if abs(b - a) < tol:
            diag["converged"] = True
            diag["final_value"] = 0.5 * (a + b)
            return 0.5 * (a + b), diag
        
        # Brent step
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + \
                (b * fa * fc) / ((fb - fa) * (fb - fc)) + \
                (c * fa * fb) / ((fc - fa) * (fc - fb))
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)
        
        # Check if bisection is needed
        use_bisection = (
            not ((3 * a + b) / 4 <= s <= b) or
            (mflag and abs(s - b) >= abs(b - c) / 2) or
            (not mflag and abs(s - b) >= abs(c - d) / 2) or  # d from previous iteration
            (mflag and abs(b - c) < tol) or
            (not mflag and abs(c - d) < tol)
        )
        
        if use_bisection:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False
        
        d = c  # Store for next iteration
        c = b
        fc = fb
        
        fs = objective(s)
        b = s
        fb = fs
        
        if fa * fb > 0:
            a, fa = c, fc
        else:
            c, fc = a, fa
    
    diag["reason"] = "max_iterations_reached"
    diag["final_value"] = 0.5 * (a + b)
    return 0.5 * (a + b), diag


def compute_iv_percentile_enhanced(history: Sequence[object], current: Optional[float],
                                 current_tau: Optional[float] = None, tau_tol: float = 1/365) -> Tuple[Optional[float], Optional[float], Dict]:
    """Enhanced IV percentile with tenor filtering for Indian markets."""
    diag = {
        "method": "tenor_filtered_percentile",
        "history_count": len(history),
        "filtered_count": 0,
        "current": current,
        "current_tau": current_tau,
        "tau_tolerance": tau_tol
    }
    
    if current is None or math.isnan(current):
        diag["reason"] = "invalid_current_iv"
        return None, None, diag
    
    filtered_ivs = []
    
    # Process history based on format
    for item in history:
        if isinstance(item, (int, float)) and not math.isnan(item):
            if current_tau is None:
                # Assume same tenor
                filtered_ivs.append(float(item))
            # Skip if we need tenor filtering but don't have it
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            tau, iv = item[0], item[1]
            if isinstance(iv, (int, float)) and not math.isnan(iv):
                if current_tau is None or abs(float(tau) - current_tau) <= tau_tol:
                    filtered_ivs.append(float(iv))
    
    diag["filtered_count"] = len(filtered_ivs)
    
    if not filtered_ivs:
        diag["reason"] = "no_valid_history"
        return None, None, diag
    
    # Calculate percentile (percentage below current)
    below_count = sum(1 for iv in filtered_ivs if iv < current)
    percentile = 100.0 * below_count / len(filtered_ivs)
    
    # Calculate IV rank (normalized position in range)
    min_iv, max_iv = min(filtered_ivs), max(filtered_ivs)
    if max_iv - min_iv < 1e-12:
        iv_rank = 50.0  # Middle rank for constant series
        diag["constant_series"] = True
    else:
        iv_rank = 100.0 * (current - min_iv) / (max_iv - min_iv)
    
    diag.update({
        "percentile": percentile,
        "iv_rank": iv_rank,
        "min_iv": min_iv,
        "max_iv": max_iv,
        "range": max_iv - min_iv
    })
    
    return percentile, iv_rank, diag


def compute_pcr_enhanced(oi_put: Mapping[float, int], oi_call: Mapping[float, int],
                        strikes: Sequence[float], K_atm: float, step: int, m: int = 6) -> Tuple[Dict, Dict]:
    """Enhanced PCR computation with Indian market conventions."""
    diag = {
        "method": "oi_based_pcr",
        "K_atm": K_atm,
        "step": step,
        "band_width": m,
        "band_strikes": [],
        "total_strikes": len(strikes),
        "valid_strikes": 0
    }
    
    # Calculate band boundaries
    band_lo = K_atm - m * step
    band_hi = K_atm + m * step
    
    results = {
        "PCR_OI_total": None,
        "PCR_OI_band": None,
        "band_lo": band_lo,
        "band_hi": band_hi,
        "band_strikes_count": 0,
        "total_put_oi": 0,
        "total_call_oi": 0,
        "band_put_oi": 0,
        "band_call_oi": 0
    }
    
    total_put_oi = total_call_oi = 0
    band_put_oi = band_call_oi = 0
    valid_strikes = 0
    band_strikes = []
    
    for strike in strikes:
        strike = float(strike)
        put_oi = float(oi_put.get(strike, 0) or 0)
        call_oi = float(oi_call.get(strike, 0) or 0)
        
        # Skip strikes with no open interest
        if put_oi <= 0 and call_oi <= 0:
            continue
        
        valid_strikes += 1
        
        # Add to totals
        if put_oi > 0:
            total_put_oi += put_oi
        if call_oi > 0:
            total_call_oi += call_oi
        
        # Check if in ATM band
        if band_lo <= strike <= band_hi:
            band_strikes.append(strike)
            if put_oi > 0:
                band_put_oi += put_oi
            if call_oi > 0:
                band_call_oi += call_oi
    
    diag.update({
        "valid_strikes": valid_strikes,
        "band_strikes": sorted(band_strikes),
        "band_strikes_count": len(band_strikes)
    })
    
    results.update({
        "total_put_oi": total_put_oi,
        "total_call_oi": total_call_oi,
        "band_put_oi": band_put_oi,
        "band_call_oi": band_call_oi,
        "band_strikes_count": len(band_strikes)
    })
    
    # Calculate total PCR
    if total_call_oi > 0:
        results["PCR_OI_total"] = total_put_oi / total_call_oi
        diag["total_pcr_calculated"] = True
    else:
        diag["total_pcr_reason"] = "no_call_oi"
    
    # Calculate band PCR (require at least 3 strikes for reliability)
    if band_call_oi > 0 and len(band_strikes) >= 3:
        results["PCR_OI_band"] = band_put_oi / band_call_oi
        diag["band_pcr_calculated"] = True
    else:
        if band_call_oi <= 0:
            diag["band_pcr_reason"] = "no_call_oi_in_band"
        else:
            diag["band_pcr_reason"] = f"insufficient_strikes_in_band: {len(band_strikes)}"
    
    return results, diag


__all__ = [
    "infer_strike_step_enhanced",
    "compute_forward_enhanced", 
    "pick_atm_strike_enhanced",
    "implied_vol_enhanced",
    "compute_atm_iv_enhanced",
    "compute_iv_percentile_enhanced",
    "compute_pcr_enhanced",
    "validate_inputs",
]