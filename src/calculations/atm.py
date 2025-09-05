"""
Industry-standard ATM strike calculation for zero-tolerance accuracy.

This implementation follows professional trading platform methodology
for precise at-the-money strike selection using forward pricing.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional


def calculate_forward_price(
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
    time_to_expiry_years: float,
    futures_mid: Optional[float] = None
) -> float:
    """
    Calculate forward price using standard financial mathematics.
    
    If futures price is available and reliable, use it directly.
    Otherwise, calculate using cost-of-carry model:
    F = S * exp((r - q) * T)
    
    Parameters
    ----------
    spot : float
        Current spot price
    risk_free_rate : float
        Risk-free rate (annualized decimal, e.g., 0.06 for 6%)
    dividend_yield : float  
        Dividend yield (annualized decimal)
    time_to_expiry_years : float
        Time to expiry in years
    futures_mid : float, optional
        Current futures mid price if available
        
    Returns
    -------
    float
        Forward price
    """
    
    if spot <= 0:
        raise ValueError(f"Invalid spot price: {spot}")
    
    if time_to_expiry_years <= 0:
        raise ValueError(f"Invalid time to expiry: {time_to_expiry_years}")
    
    # If futures price is available and reasonable, use it
    if futures_mid is not None and futures_mid > 0:
        # Validate futures price is reasonable (within ±10% of theoretical forward)
        theoretical_forward = spot * math.exp((risk_free_rate - dividend_yield) * time_to_expiry_years)
        futures_deviation = abs(futures_mid - theoretical_forward) / theoretical_forward
        
        if futures_deviation <= 0.10:  # Within 10% tolerance
            return futures_mid
        else:
            # Futures price seems unreliable, use theoretical calculation
            pass
    
    # Calculate theoretical forward using cost-of-carry
    forward = spot * math.exp((risk_free_rate - dividend_yield) * time_to_expiry_years)
    return forward


def calculate_atm_strike_precise(
    forward_price: float,
    strikes: List[float],
    step: int,
    call_mids: Optional[Dict[float, float]] = None,
    put_mids: Optional[Dict[float, float]] = None,
    spot: float = 0.0
) -> float:
    """
    Calculate ATM strike using forward price with straddle tie-breaking.
    
    Method:
    1. Find the strike closest to forward price
    2. If tie (two equidistant strikes), use straddle pricing as tie-breaker
    3. If no option prices available, use simple closest-to-forward
    
    Parameters
    ----------
    forward_price : float
        Forward price calculated from spot + carry
    strikes : List[float]
        Available strike prices
    step : int
        Strike step size
    call_mids : Dict[float, float], optional
        Call mid prices by strike for tie-breaking
    put_mids : Dict[float, float], optional  
        Put mid prices by strike for tie-breaking
    spot : float, optional
        Spot price for additional validation
        
    Returns
    -------
    float
        ATM strike price
    """
    
    if not strikes:
        raise ValueError("No strikes provided for ATM calculation")
    
    if forward_price <= 0:
        raise ValueError(f"Invalid forward price: {forward_price}")
    
    if step <= 0:
        raise ValueError(f"Invalid step size: {step}")
    
    # Sort strikes
    sorted_strikes = sorted(float(s) for s in strikes)
    
    # Find closest strike(s) to forward price
    min_distance = float('inf')
    closest_strikes = []
    
    for strike in sorted_strikes:
        distance = abs(strike - forward_price)
        if distance < min_distance:
            min_distance = distance
            closest_strikes = [strike]
        elif distance == min_distance:
            closest_strikes.append(strike)
    
    # If only one closest strike, return it
    if len(closest_strikes) == 1:
        return closest_strikes[0]
    
    # Multiple equidistant strikes - use straddle tie-breaking if data available
    if call_mids and put_mids and len(closest_strikes) >= 2:
        best_strike = closest_strikes[0]
        best_straddle_price = float('inf')
        
        for strike in closest_strikes:
            call_price = call_mids.get(strike, 0.0)
            put_price = put_mids.get(strike, 0.0)
            
            # Only use strikes with valid option prices
            if call_price > 0 and put_price > 0:
                straddle_price = call_price + put_price
                if straddle_price < best_straddle_price:
                    best_straddle_price = straddle_price
                    best_strike = strike
        
        # If we found a valid straddle-based winner, return it
        if best_straddle_price < float('inf'):
            return best_strike
    
    # Fallback: return the first closest strike
    return closest_strikes[0]


def detect_strike_step_precise(strikes: List[float]) -> Optional[int]:
    """
    Detect strike step with high accuracy.
    
    Parameters
    ----------
    strikes : List[float]
        Available strikes
        
    Returns
    -------
    Optional[int]
        Strike step if detectable, None otherwise
    """
    
    if len(strikes) < 2:
        return None
    
    sorted_strikes = sorted(float(s) for s in strikes)
    
    # Calculate all gaps
    gaps = []
    for i in range(len(sorted_strikes) - 1):
        gap = sorted_strikes[i + 1] - sorted_strikes[i]
        if gap > 0:
            gaps.append(int(round(gap)))
    
    if not gaps:
        return None
    
    # Find most common gap (mode)
    gap_counts = {}
    for gap in gaps:
        gap_counts[gap] = gap_counts.get(gap, 0) + 1
    
    # Get the most frequent gap
    most_common_gap = max(gap_counts, key=gap_counts.get)
    
    # Verify it represents at least 50% of gaps for reliability
    if gap_counts[most_common_gap] >= len(gaps) * 0.5:
        return most_common_gap
    
    return None


def calculate_atm_with_validation(
    spot: float,
    strikes: List[float],
    risk_free_rate: float,
    dividend_yield: float,
    time_to_expiry_years: float,
    futures_mid: Optional[float] = None,
    call_mids: Optional[Dict[float, float]] = None,
    put_mids: Optional[Dict[float, float]] = None,
    symbol: str = ""
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Calculate ATM strike with comprehensive validation.
    
    Returns
    -------
    Tuple[Optional[float], Optional[float], str]
        (atm_strike or None, forward_price or None, status_message)
    """
    
    try:
        # Input validation
        if spot <= 0:
            return None, None, f"Invalid spot price: {spot}"
        
        if time_to_expiry_years <= 0:
            return None, None, f"Invalid time to expiry: {time_to_expiry_years}"
        
        if not strikes or len(strikes) < 2:
            return None, None, f"Insufficient strikes: {len(strikes) if strikes else 0}"
        
        # Calculate forward price
        forward_price = calculate_forward_price(
            spot, risk_free_rate, dividend_yield, time_to_expiry_years, futures_mid
        )
        
        # Detect step size
        step = detect_strike_step_precise(strikes)
        if step is None:
            # Fallback to symbol-based step mapping
            symbol_steps = {
                "NIFTY": 50,
                "BANKNIFTY": 100, 
                "SENSEX": 100,
                "MIDCPNIFTY": 50,
                "FINNIFTY": 50
            }
            step = symbol_steps.get(symbol.upper(), 50)
        
        # Calculate ATM strike
        atm_strike = calculate_atm_strike_precise(
            forward_price, strikes, step, call_mids, put_mids, spot
        )
        
        return atm_strike, forward_price, "Success"
        
    except Exception as e:
        return None, None, f"Calculation failed: {str(e)}"