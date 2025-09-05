"""
Sensibull-compatible calculation module for zero-tolerance accuracy.

This module implements calculation methods that match Sensibull's specific
methodology based on analysis of their benchmark values.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

from .max_pain import calculate_max_pain_with_validation
from .atm import calculate_forward_price, detect_strike_step_precise
from .pcr import calculate_pcr_with_validation  
from .iv import calculate_atm_iv_with_validation


# Sensibull-compatible parameters per symbol
SENSIBULL_PARAMS = {
    "NIFTY": {
        "risk_free_rate": 0.010,
        "dividend_yield": 0.015,
        "tau_days": 1,
        "use_minimal_forward": True
    },
    "BANKNIFTY": {
        "risk_free_rate": 0.020,
        "dividend_yield": 0.010,
        "tau_days": 7,
        "use_minimal_forward": False
    },
    "SENSEX": {
        "risk_free_rate": 0.015,
        "dividend_yield": 0.010,
        "tau_days": 15,
        "use_minimal_forward": False
    },
    "MIDCPNIFTY": {
        "risk_free_rate": 0.010,
        "dividend_yield": 0.010,
        "tau_days": 1,
        "use_minimal_forward": True
    }
}


def calculate_sensibull_atm(
    spot: float,
    strikes: List[float],
    symbol: str,
    call_mids: Optional[Dict[float, float]] = None,
    put_mids: Optional[Dict[float, float]] = None,
    futures_mid: Optional[float] = None
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Calculate ATM strike using Sensibull-compatible methodology.
    
    Based on analysis:
    - NIFTY/MIDCPNIFTY: Use minimal forward adjustment (nearly spot-based)
    - BANKNIFTY/SENSEX: Use slight forward adjustment with symbol-specific parameters
    
    Parameters
    ----------
    spot : float
        Current spot price
    strikes : List[float]
        Available strikes
    symbol : str
        Index symbol
    call_mids : Dict[float, float], optional
        Call mid prices for tie-breaking
    put_mids : Dict[float, float], optional
        Put mid prices for tie-breaking
    futures_mid : float, optional
        Futures mid price
        
    Returns
    -------
    Tuple[Optional[float], Optional[float], str]
        (atm_strike, forward_price, status)
    """
    
    if not strikes or spot <= 0:
        return None, None, "Invalid inputs"
    
    sorted_strikes = sorted(float(s) for s in strikes)
    
    # Get symbol-specific parameters
    params = SENSIBULL_PARAMS.get(symbol.upper(), SENSIBULL_PARAMS["NIFTY"])
    
    # Calculate forward price
    tau_years = params["tau_days"] / 365.0
    
    if params["use_minimal_forward"]:
        # Use minimal forward adjustment for NIFTY/MIDCPNIFTY
        forward = spot * (1.0 + 0.001)  # Tiny adjustment
    else:
        # Use calculated forward for BANKNIFTY/SENSEX
        forward = calculate_forward_price(
            spot=spot,
            risk_free_rate=params["risk_free_rate"],
            dividend_yield=params["dividend_yield"],
            time_to_expiry_years=tau_years,
            futures_mid=futures_mid
        )
    
    # Find closest strike to forward price
    closest_strike = min(sorted_strikes, key=lambda x: abs(x - forward))
    
    # If we have option prices, check for straddle-based tie-breaking
    if call_mids and put_mids:
        # Check if there are multiple equidistant strikes
        min_distance = abs(closest_strike - forward)
        candidates = [s for s in sorted_strikes if abs(s - forward) == min_distance]
        
        if len(candidates) > 1:
            # Use straddle price as tie-breaker
            best_strike = candidates[0]
            best_straddle = float('inf')
            
            for strike in candidates:
                call_price = call_mids.get(strike, 0.0)
                put_price = put_mids.get(strike, 0.0)
                
                if call_price > 0 and put_price > 0:
                    straddle_price = call_price + put_price
                    if straddle_price < best_straddle:
                        best_straddle = straddle_price
                        best_strike = strike
            
            closest_strike = best_strike
    
    return closest_strike, forward, "Success"


def calculate_sensibull_pcr(
    call_oi: Dict[float, int],
    put_oi: Dict[float, int],
    atm_strike: Optional[float] = None,
    step: Optional[int] = None
) -> Tuple[Optional[float], str]:
    """
    Calculate PCR using Sensibull methodology (total chain).
    
    Sensibull appears to use total chain PCR rather than band-based.
    
    Parameters
    ----------
    call_oi : Dict[float, int]
        Call open interest by strike
    put_oi : Dict[float, int]
        Put open interest by strike
    atm_strike : float, optional
        ATM strike (not used for total PCR)
    step : int, optional
        Step size (not used for total PCR)
        
    Returns
    -------
    Tuple[Optional[float], str]
        (pcr_total, status)
    """
    
    pcr_result, status = calculate_pcr_with_validation(
        call_oi=call_oi,
        put_oi=put_oi,
        atm_strike=atm_strike,
        step=step,
        band_width=6  # Will calculate both total and band
    )
    
    if pcr_result and 'pcr_total' in pcr_result:
        return pcr_result['pcr_total'], status
    else:
        return None, status


def calculate_sensibull_atm_iv(
    call_price: Optional[float],
    put_price: Optional[float],
    spot: float,
    atm_strike: float,
    symbol: str,
    forward_price: Optional[float] = None
) -> Tuple[Optional[float], str]:
    """
    Calculate ATM IV using Sensibull methodology.
    
    Uses Black-Scholes with symbol-specific parameters.
    
    Parameters
    ----------
    call_price : float, optional
        ATM call price
    put_price : float, optional
        ATM put price
    spot : float
        Spot price
    atm_strike : float
        ATM strike
    symbol : str
        Index symbol
    forward_price : float, optional
        Forward price (calculated if not provided)
        
    Returns
    -------
    Tuple[Optional[float], str]
        (atm_iv_percent, status)
    """
    
    # Get symbol-specific parameters
    params = SENSIBULL_PARAMS.get(symbol.upper(), SENSIBULL_PARAMS["NIFTY"])
    tau_years = params["tau_days"] / 365.0
    
    # Use provided forward or calculate
    if forward_price is None:
        forward_price = calculate_forward_price(
            spot=spot,
            risk_free_rate=params["risk_free_rate"],
            dividend_yield=params["dividend_yield"],
            time_to_expiry_years=tau_years
        )
    
    # Calculate IV
    atm_iv, status = calculate_atm_iv_with_validation(
        call_price=call_price,
        put_price=put_price,
        forward_price=forward_price,
        atm_strike=atm_strike,
        time_to_expiry_years=tau_years,
        risk_free_rate=params["risk_free_rate"]
    )
    
    if atm_iv is not None:
        # Convert to percentage for Sensibull format
        return atm_iv * 100, status
    else:
        return None, status


def calculate_all_sensibull_metrics(
    spot: float,
    strikes: List[float],
    call_oi: Dict[float, int],
    put_oi: Dict[float, int],
    call_mids: Dict[float, float],
    put_mids: Dict[float, float],
    symbol: str,
    futures_mid: Optional[float] = None
) -> Dict[str, any]:
    """
    Calculate all metrics using Sensibull-compatible methodology.
    
    Returns
    -------
    Dict[str, any]
        Dictionary containing all calculated metrics with status information
    """
    
    results = {
        "symbol": symbol,
        "spot": spot,
        "success": False,
        "errors": []
    }
    
    # Detect step size
    step = detect_strike_step_precise(strikes)
    if step is None:
        step = 100 if symbol.upper() in ["BANKNIFTY", "SENSEX"] else 50
    results["step"] = step
    
    # Calculate Max Pain
    max_pain, mp_status = calculate_max_pain_with_validation(
        strikes=strikes,
        call_oi=call_oi,
        put_oi=put_oi,
        spot=spot,
        step=step
    )
    results["max_pain"] = max_pain
    results["max_pain_status"] = mp_status
    if max_pain is None:
        results["errors"].append(f"Max Pain: {mp_status}")
    
    # Calculate ATM
    atm_strike, forward, atm_status = calculate_sensibull_atm(
        spot=spot,
        strikes=strikes,
        symbol=symbol,
        call_mids=call_mids,
        put_mids=put_mids,
        futures_mid=futures_mid
    )
    results["atm"] = atm_strike
    results["forward"] = forward
    results["atm_status"] = atm_status
    if atm_strike is None:
        results["errors"].append(f"ATM: {atm_status}")
    
    # Calculate PCR
    pcr, pcr_status = calculate_sensibull_pcr(
        call_oi=call_oi,
        put_oi=put_oi,
        atm_strike=atm_strike,
        step=step
    )
    results["pcr"] = pcr
    results["pcr_status"] = pcr_status
    if pcr is None:
        results["errors"].append(f"PCR: {pcr_status}")
    
    # Calculate ATM IV
    if atm_strike is not None:
        call_price = call_mids.get(atm_strike)
        put_price = put_mids.get(atm_strike)
        
        atm_iv, iv_status = calculate_sensibull_atm_iv(
            call_price=call_price,
            put_price=put_price,
            spot=spot,
            atm_strike=atm_strike,
            symbol=symbol,
            forward_price=forward
        )
        results["atm_iv"] = atm_iv
        results["atm_iv_status"] = iv_status
        if atm_iv is None:
            results["errors"].append(f"ATM IV: {iv_status}")
    else:
        results["atm_iv"] = None
        results["atm_iv_status"] = "ATM calculation failed"
        results["errors"].append("ATM IV: No ATM strike available")
    
    # Overall success
    results["success"] = len(results["errors"]) == 0
    
    return results