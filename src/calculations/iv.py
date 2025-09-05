"""
Industry-standard ATM IV calculation for zero-tolerance accuracy.

This implementation follows Black-Scholes methodology used by professional
trading platforms for precise implied volatility calculation.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
from scipy.stats import norm


def black_scholes_price(
    S: float,
    K: float, 
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True
) -> float:
    """
    Calculate Black-Scholes option price.
    
    Parameters
    ----------
    S : float
        Current underlying price (forward price for European options)
    K : float
        Strike price
    T : float
        Time to expiry in years
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    is_call : bool, default True
        True for call, False for put
        
    Returns
    -------
    float
        Option price
    """
    
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if is_call:
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return max(0.0, price)


def black_scholes_vega(
    S: float,
    K: float,
    T: float, 
    r: float,
    sigma: float
) -> float:
    """
    Calculate vega (derivative of price w.r.t. volatility).
    
    Returns
    -------
    float
        Vega value
    """
    
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    vega = S * math.sqrt(T) * norm.pdf(d1)
    
    return vega


def newton_raphson_iv(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool = True,
    initial_guess: float = 0.25,
    max_iterations: int = 50,
    tolerance: float = 1e-6
) -> Optional[float]:
    """
    Calculate implied volatility using Newton-Raphson method.
    
    This is more accurate than bisection for well-behaved cases.
    
    Parameters
    ----------
    option_price : float
        Market option price
    S : float  
        Current underlying price (forward for European)
    K : float
        Strike price
    T : float
        Time to expiry in years
    r : float
        Risk-free rate
    is_call : bool, default True
        True for call, False for put
    initial_guess : float, default 0.25
        Starting volatility guess (25%)
    max_iterations : int, default 50
        Maximum iterations
    tolerance : float, default 1e-6
        Convergence tolerance
        
    Returns
    -------
    Optional[float]
        Implied volatility or None if no convergence
    """
    
    if option_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    
    sigma = initial_guess
    
    for _ in range(max_iterations):
        # Calculate price and vega at current sigma
        price = black_scholes_price(S, K, T, r, sigma, is_call)
        vega = black_scholes_vega(S, K, T, r, sigma)
        
        # Check for convergence
        price_diff = price - option_price
        if abs(price_diff) < tolerance:
            return sigma
        
        # Check for invalid vega (prevents division by zero)
        if vega < 1e-10:
            return None
        
        # Newton-Raphson update
        sigma_new = sigma - price_diff / vega
        
        # Keep sigma positive and reasonable
        sigma_new = max(0.001, min(5.0, sigma_new))
        
        # Check for convergence in sigma
        if abs(sigma_new - sigma) < tolerance:
            return sigma_new
        
        sigma = sigma_new
    
    return None  # No convergence


def bisection_iv(
    option_price: float,
    S: float,
    K: float,
    T: float, 
    r: float,
    is_call: bool = True,
    low: float = 0.001,
    high: float = 5.0,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Optional[float]:
    """
    Calculate implied volatility using bisection method.
    
    More robust than Newton-Raphson for edge cases.
    
    Parameters
    ----------
    option_price : float
        Market option price
    S : float
        Current underlying price (forward for European)
    K : float
        Strike price
    T : float
        Time to expiry in years
    r : float
        Risk-free rate
    is_call : bool, default True
        True for call, False for put
    low : float, default 0.001
        Lower bound for volatility search
    high : float, default 5.0
        Upper bound for volatility search
    max_iterations : int, default 100
        Maximum iterations
    tolerance : float, default 1e-6
        Convergence tolerance
        
    Returns
    -------
    Optional[float]
        Implied volatility or None if no solution found
    """
    
    if option_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    
    # Check bounds
    price_low = black_scholes_price(S, K, T, r, low, is_call)
    price_high = black_scholes_price(S, K, T, r, high, is_call)
    
    if option_price < price_low or option_price > price_high:
        return None  # Solution outside bounds
    
    for _ in range(max_iterations):
        mid = (low + high) / 2
        price_mid = black_scholes_price(S, K, T, r, mid, is_call)
        
        if abs(price_mid - option_price) < tolerance:
            return mid
        
        if price_mid < option_price:
            low = mid
        else:
            high = mid
        
        if high - low < tolerance:
            return (low + high) / 2
    
    return None


def calculate_atm_iv_precise(
    call_price: Optional[float],
    put_price: Optional[float],
    forward_price: float,
    atm_strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float
) -> Optional[float]:
    """
    Calculate ATM implied volatility using straddle approach.
    
    Method:
    1. If both call and put prices available, use average of their IVs
    2. If only one available, use that one
    3. Use Newton-Raphson first, fallback to bisection if needed
    
    Parameters
    ----------
    call_price : float, optional
        ATM call option price
    put_price : float, optional
        ATM put option price
    forward_price : float
        Forward price of underlying
    atm_strike : float
        ATM strike price
    time_to_expiry_years : float
        Time to expiry in years
    risk_free_rate : float
        Risk-free rate (annualized)
        
    Returns
    -------
    Optional[float]
        ATM implied volatility or None if calculation fails
    """
    
    if time_to_expiry_years <= 0:
        return None
    
    if forward_price <= 0 or atm_strike <= 0:
        return None
    
    ivs = []
    
    # Calculate call IV if available
    if call_price is not None and call_price > 0:
        # Try Newton-Raphson first
        call_iv = newton_raphson_iv(
            call_price, forward_price, atm_strike, time_to_expiry_years, 
            risk_free_rate, is_call=True
        )
        
        # Fallback to bisection if Newton-Raphson fails
        if call_iv is None:
            call_iv = bisection_iv(
                call_price, forward_price, atm_strike, time_to_expiry_years,
                risk_free_rate, is_call=True
            )
        
        if call_iv is not None:
            ivs.append(call_iv)
    
    # Calculate put IV if available
    if put_price is not None and put_price > 0:
        # Try Newton-Raphson first
        put_iv = newton_raphson_iv(
            put_price, forward_price, atm_strike, time_to_expiry_years,
            risk_free_rate, is_call=False
        )
        
        # Fallback to bisection if Newton-Raphson fails
        if put_iv is None:
            put_iv = bisection_iv(
                put_price, forward_price, atm_strike, time_to_expiry_years,
                risk_free_rate, is_call=False
            )
        
        if put_iv is not None:
            ivs.append(put_iv)
    
    # Return average if multiple IVs available
    if ivs:
        return sum(ivs) / len(ivs)
    
    return None


def calculate_atm_iv_with_validation(
    call_price: Optional[float],
    put_price: Optional[float], 
    forward_price: float,
    atm_strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float
) -> Tuple[Optional[float], str]:
    """
    Calculate ATM IV with comprehensive validation.
    
    Returns
    -------
    Tuple[Optional[float], str]
        (atm_iv or None, status_message)
    """
    
    try:
        # Input validation
        if time_to_expiry_years <= 0:
            return None, f"Invalid time to expiry: {time_to_expiry_years}"
        
        if forward_price <= 0:
            return None, f"Invalid forward price: {forward_price}"
        
        if atm_strike <= 0:
            return None, f"Invalid ATM strike: {atm_strike}"
        
        if call_price is None and put_price is None:
            return None, "No option prices available"
        
        if call_price is not None and call_price <= 0:
            call_price = None
        
        if put_price is not None and put_price <= 0:
            put_price = None
        
        if call_price is None and put_price is None:
            return None, "No valid option prices available"
        
        # Calculate IV
        atm_iv = calculate_atm_iv_precise(
            call_price, put_price, forward_price, atm_strike,
            time_to_expiry_years, risk_free_rate
        )
        
        if atm_iv is None:
            return None, "IV calculation failed to converge"
        
        # Sanity check on result
        if atm_iv < 0.001 or atm_iv > 5.0:
            return None, f"IV result outside reasonable range: {atm_iv:.4f}"
        
        return atm_iv, "Success"
        
    except Exception as e:
        return None, f"IV calculation error: {str(e)}"