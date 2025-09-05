"""
Industry-standard PCR (Put-Call Ratio) calculation for zero-tolerance accuracy.

This implementation follows professional trading platform methodology
for precise PCR calculation using multiple methodologies.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional


def calculate_pcr_total(
    call_oi: Dict[float, int],
    put_oi: Dict[float, int],
    strikes: Optional[List[float]] = None
) -> float:
    """
    Calculate total PCR across all strikes.
    
    PCR_Total = Total_Put_OI / Total_Call_OI
    
    Parameters
    ----------
    call_oi : Dict[float, int]
        Call open interest by strike
    put_oi : Dict[float, int]
        Put open interest by strike
    strikes : List[float], optional
        Strikes to include (if None, uses all available)
        
    Returns
    -------
    float
        Total PCR value
    """
    
    if strikes is None:
        # Use all available strikes from both calls and puts
        all_strikes = set(call_oi.keys()) | set(put_oi.keys())
    else:
        all_strikes = set(strikes)
    
    total_call_oi = sum(call_oi.get(strike, 0) for strike in all_strikes)
    total_put_oi = sum(put_oi.get(strike, 0) for strike in all_strikes)
    
    if total_call_oi == 0:
        raise ValueError("Total call OI is zero - cannot calculate PCR")
    
    return total_put_oi / total_call_oi


def calculate_pcr_band(
    call_oi: Dict[float, int],
    put_oi: Dict[float, int],
    atm_strike: float,
    step: int,
    band_width: int = 6
) -> float:
    """
    Calculate PCR for a band around ATM strike.
    
    PCR_Band = Sum(Put_OI[ATM±band]) / Sum(Call_OI[ATM±band])
    
    Parameters
    ----------
    call_oi : Dict[float, int]
        Call open interest by strike
    put_oi : Dict[float, int]
        Put open interest by strike
    atm_strike : float
        ATM strike price
    step : int
        Strike step size
    band_width : int, default 6
        Number of strikes on each side of ATM (±6 = 13 total strikes)
        
    Returns
    -------
    float
        Band PCR value
    """
    
    if step <= 0:
        raise ValueError(f"Invalid step size: {step}")
    
    if band_width <= 0:
        raise ValueError(f"Invalid band width: {band_width}")
    
    # Generate band strikes
    band_strikes = []
    for i in range(-band_width, band_width + 1):
        strike = atm_strike + (i * step)
        band_strikes.append(strike)
    
    total_call_oi = sum(call_oi.get(strike, 0) for strike in band_strikes)
    total_put_oi = sum(put_oi.get(strike, 0) for strike in band_strikes)
    
    if total_call_oi == 0:
        raise ValueError("Total call OI in band is zero - cannot calculate band PCR")
    
    return total_put_oi / total_call_oi


def calculate_pcr_volume(
    call_volumes: Dict[float, int],
    put_volumes: Dict[float, int],
    strikes: Optional[List[float]] = None
) -> float:
    """
    Calculate PCR based on trading volumes.
    
    PCR_Volume = Total_Put_Volume / Total_Call_Volume
    
    Parameters
    ----------
    call_volumes : Dict[float, int]
        Call trading volumes by strike
    put_volumes : Dict[float, int]
        Put trading volumes by strike
    strikes : List[float], optional
        Strikes to include
        
    Returns
    -------
    float
        Volume-based PCR
    """
    
    if strikes is None:
        all_strikes = set(call_volumes.keys()) | set(put_volumes.keys())
    else:
        all_strikes = set(strikes)
    
    total_call_volume = sum(call_volumes.get(strike, 0) for strike in all_strikes)
    total_put_volume = sum(put_volumes.get(strike, 0) for strike in all_strikes)
    
    if total_call_volume == 0:
        raise ValueError("Total call volume is zero - cannot calculate volume PCR")
    
    return total_put_volume / total_call_volume


def validate_pcr_inputs(
    call_oi: Dict[float, int],
    put_oi: Dict[float, int],
    atm_strike: Optional[float] = None,
    step: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Validate inputs for PCR calculation.
    
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    
    if not call_oi and not put_oi:
        return False, "Both call and put OI dictionaries are empty"
    
    # Check for non-negative OI values
    for strike, oi in call_oi.items():
        if oi < 0:
            return False, f"Negative call OI at strike {strike}: {oi}"
    
    for strike, oi in put_oi.items():
        if oi < 0:
            return False, f"Negative put OI at strike {strike}: {oi}"
    
    # Check total OI
    total_call_oi = sum(call_oi.values())
    total_put_oi = sum(put_oi.values())
    
    if total_call_oi == 0:
        return False, "Total call OI is zero"
    
    if total_put_oi == 0:
        return False, "Total put OI is zero"
    
    # Minimum threshold check
    if total_call_oi + total_put_oi < 100:
        return False, f"Total OI too low: {total_call_oi + total_put_oi}"
    
    # ATM-specific validation for band calculations
    if atm_strike is not None and step is not None:
        if atm_strike <= 0:
            return False, f"Invalid ATM strike: {atm_strike}"
        if step <= 0:
            return False, f"Invalid step size: {step}"
    
    return True, "Valid"


def calculate_pcr_comprehensive(
    call_oi: Dict[float, int],
    put_oi: Dict[float, int],
    atm_strike: Optional[float] = None,
    step: Optional[int] = None,
    band_width: int = 6,
    call_volumes: Optional[Dict[float, int]] = None,
    put_volumes: Optional[Dict[float, int]] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive PCR metrics with validation.
    
    Parameters
    ----------
    call_oi : Dict[float, int]
        Call open interest by strike
    put_oi : Dict[float, int]
        Put open interest by strike
    atm_strike : float, optional
        ATM strike for band calculation
    step : int, optional
        Strike step size for band calculation
    band_width : int, default 6
        Band width for ATM-centered calculation
    call_volumes : Dict[float, int], optional
        Call volumes for volume-based PCR
    put_volumes : Dict[float, int], optional
        Put volumes for volume-based PCR
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'pcr_total': Total PCR across all strikes
        - 'pcr_band': PCR for ATM±band (if ATM/step provided)
        - 'pcr_volume': Volume-based PCR (if volumes provided)
        
    Raises
    ------
    ValueError
        If input validation fails or calculations cannot be performed
    """
    
    # Validate inputs
    is_valid, error_msg = validate_pcr_inputs(call_oi, put_oi, atm_strike, step)
    if not is_valid:
        raise ValueError(f"PCR input validation failed: {error_msg}")
    
    results = {}
    
    # Calculate total PCR
    try:
        results['pcr_total'] = calculate_pcr_total(call_oi, put_oi)
    except Exception as e:
        raise ValueError(f"Total PCR calculation failed: {str(e)}")
    
    # Calculate band PCR if ATM and step are provided
    if atm_strike is not None and step is not None:
        try:
            results['pcr_band'] = calculate_pcr_band(
                call_oi, put_oi, atm_strike, step, band_width
            )
        except Exception as e:
            # Band PCR failure is not critical, continue without it
            results['pcr_band'] = None
    
    # Calculate volume PCR if volumes are provided
    if call_volumes is not None and put_volumes is not None:
        try:
            results['pcr_volume'] = calculate_pcr_volume(call_volumes, put_volumes)
        except Exception as e:
            # Volume PCR failure is not critical
            results['pcr_volume'] = None
    
    return results


def calculate_pcr_with_validation(
    call_oi: Dict[float, int],
    put_oi: Dict[float, int],
    atm_strike: Optional[float] = None,
    step: Optional[int] = None,
    band_width: int = 6
) -> Tuple[Optional[Dict[str, float]], str]:
    """
    Calculate PCR with comprehensive error handling.
    
    Returns
    -------
    Tuple[Optional[Dict[str, float]], str]
        (pcr_results or None, status_message)
    """
    
    try:
        results = calculate_pcr_comprehensive(
            call_oi, put_oi, atm_strike, step, band_width
        )
        return results, "Success"
    except Exception as e:
        return None, f"PCR calculation failed: {str(e)}"