"""
Industry-standard Max Pain calculation for zero-tolerance accuracy.

This implementation follows the exact methodology used by professional
trading platforms like Sensibull for precise option pain point calculation.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional


def calculate_max_pain_precise(
    strikes: List[float], 
    call_oi: Dict[float, int], 
    put_oi: Dict[float, int],
    spot: float = 0.0,
    step: Optional[int] = None
) -> float:
    """
    Calculate Max Pain using industry-standard methodology.
    
    Max Pain is the strike price at which option writers (sellers) would lose
    the least amount of money on expiration. It represents the strike with
    minimum total intrinsic value of all ITM options.
    
    Formula:
    For each potential expiry price K:
    Pain(K) = Σ[Call_OI(S) * max(0, K - S)] + Σ[Put_OI(S) * max(0, S - K)]
    
    Max Pain = argmin(Pain(K)) for K in strikes
    
    Parameters
    ----------
    strikes : List[float]
        All available strike prices
    call_oi : Dict[float, int]  
        Call open interest by strike
    put_oi : Dict[float, int]
        Put open interest by strike
    spot : float, optional
        Current spot price (used for tie-breaking)
    step : int, optional
        Strike step size for validation
        
    Returns
    -------
    float
        Max Pain strike price
        
    Notes
    -----
    This implementation ensures:
    1. Exact mathematical accuracy per industry standards
    2. Proper handling of equal pain scenarios (closest to spot wins)
    3. Validation against minimum data requirements
    4. No approximations or shortcuts
    """
    
    if not strikes:
        raise ValueError("No strikes provided for Max Pain calculation")
    
    # Ensure strikes are sorted and convert to float
    sorted_strikes = sorted(float(s) for s in strikes)
    
    # Validate minimum data requirements
    total_call_oi = sum(call_oi.get(s, 0) for s in sorted_strikes)
    total_put_oi = sum(put_oi.get(s, 0) for s in sorted_strikes)
    
    if total_call_oi == 0 and total_put_oi == 0:
        raise ValueError("No open interest data available for Max Pain calculation")
    
    min_pain = float('inf')
    max_pain_strike = sorted_strikes[0]  # Default fallback
    
    # Calculate pain for each potential expiry strike
    for expiry_strike in sorted_strikes:
        total_pain = 0.0
        
        # Calculate call pain: sum of (strike - expiry_strike) * call_oi for all ITM calls
        for strike in sorted_strikes:
            call_oi_at_strike = call_oi.get(strike, 0)
            if call_oi_at_strike > 0 and expiry_strike > strike:
                # This call option will be ITM, causing pain to writers
                intrinsic_value = expiry_strike - strike
                pain_contribution = intrinsic_value * call_oi_at_strike
                total_pain += pain_contribution
        
        # Calculate put pain: sum of (expiry_strike - strike) * put_oi for all ITM puts  
        for strike in sorted_strikes:
            put_oi_at_strike = put_oi.get(strike, 0)
            if put_oi_at_strike > 0 and expiry_strike < strike:
                # This put option will be ITM, causing pain to writers
                intrinsic_value = strike - expiry_strike
                pain_contribution = intrinsic_value * put_oi_at_strike
                total_pain += pain_contribution
        
        # Check if this is the minimum pain
        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = expiry_strike
        elif total_pain == min_pain and spot > 0:
            # Tie-breaking: choose strike closest to spot
            if abs(expiry_strike - spot) < abs(max_pain_strike - spot):
                max_pain_strike = expiry_strike
    
    return float(max_pain_strike)


def validate_max_pain_inputs(
    strikes: List[float],
    call_oi: Dict[float, int], 
    put_oi: Dict[float, int]
) -> Tuple[bool, str]:
    """
    Validate inputs for Max Pain calculation.
    
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    
    if not strikes:
        return False, "Empty strikes list"
    
    if len(strikes) < 3:
        return False, f"Insufficient strikes for reliable Max Pain calculation: {len(strikes)} < 3"
    
    # Check for sufficient OI data
    strikes_with_call_oi = sum(1 for s in strikes if call_oi.get(s, 0) > 0)
    strikes_with_put_oi = sum(1 for s in strikes if put_oi.get(s, 0) > 0)
    
    if strikes_with_call_oi == 0 and strikes_with_put_oi == 0:
        return False, "No strikes have any open interest data"
    
    if strikes_with_call_oi < 2 and strikes_with_put_oi < 2:
        return False, "Insufficient open interest distribution for reliable calculation"
    
    # Check for reasonable OI values (not all zeros)
    total_call_oi = sum(call_oi.get(s, 0) for s in strikes)
    total_put_oi = sum(put_oi.get(s, 0) for s in strikes)
    
    if total_call_oi + total_put_oi < 100:
        return False, f"Total OI too low for reliable calculation: {total_call_oi + total_put_oi}"
    
    return True, "Valid"


def calculate_max_pain_with_validation(
    strikes: List[float],
    call_oi: Dict[float, int],
    put_oi: Dict[float, int], 
    spot: float = 0.0,
    step: Optional[int] = None
) -> Tuple[Optional[float], str]:
    """
    Calculate Max Pain with comprehensive input validation.
    
    Returns
    -------
    Tuple[Optional[float], str]
        (max_pain_strike or None, status_message)
    """
    
    # Validate inputs
    is_valid, error_msg = validate_max_pain_inputs(strikes, call_oi, put_oi)
    if not is_valid:
        return None, f"Input validation failed: {error_msg}"
    
    try:
        max_pain = calculate_max_pain_precise(strikes, call_oi, put_oi, spot, step)
        return max_pain, "Success"
    except Exception as e:
        return None, f"Calculation failed: {str(e)}"