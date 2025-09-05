#!/usr/bin/env python3
"""
Test Sensibull-compatible calculations with mock data that should produce
results close to the benchmarks.
"""

import sys
import math

sys.path.append('/home/runner/work/ExpiryDayTrading/ExpiryDayTrading')

from src.calculations.sensibull_compat import calculate_all_sensibull_metrics

# Create mock data that should produce results close to Sensibull benchmarks
def create_sensibull_mock_data(symbol: str):
    """
    Create mock data designed to produce results close to Sensibull benchmarks.
    """
    
    benchmarks = {
        "NIFTY": {"spot": 24741, "max_pain": 24750, "atm": 24750, "pcr": 0.77, "atm_iv": 7.9},
        "BANKNIFTY": {"spot": 54114.55, "max_pain": 54600, "atm": 54300, "pcr": 0.87, "atm_iv": 10.2},
        "SENSEX": {"spot": 80710.76, "max_pain": 83500, "atm": 80800, "pcr": 1.37, "atm_iv": 9.0},
        "MIDCPNIFTY": {"spot": 12778.15, "max_pain": 12800, "atm": 12825, "pcr": 1.17, "atm_iv": 15.4}
    }
    
    benchmark = benchmarks[symbol]
    spot = benchmark["spot"]
    target_max_pain = benchmark["max_pain"]
    target_atm = benchmark["atm"]
    target_pcr = benchmark["pcr"]
    target_iv = benchmark["atm_iv"] / 100.0  # Convert to decimal
    
    # Determine step size
    step = 100 if symbol in ["BANKNIFTY", "SENSEX"] else 50
    
    # Create strike range
    base_strike = int(spot / step) * step
    strikes = list(range(base_strike - 500, base_strike + 500, step))
    
    # Design OI distribution to produce target Max Pain
    call_oi = {}
    put_oi = {}
    
    total_call_oi = 100000  # Base total
    total_put_oi = int(total_call_oi * target_pcr)  # Target PCR
    
    # Distribute OI to create target Max Pain
    for strike in strikes:
        distance_from_spot = abs(strike - spot)
        
        # Base OI decreases with distance
        base_oi = max(1000, 15000 - distance_from_spot * 5)
        
        # Adjust distribution to push Max Pain toward target
        if symbol == "NIFTY":
            # Balanced distribution for NIFTY (Max Pain ≈ ATM)
            call_mult = 1.0 - 0.1 * min(1.0, distance_from_spot / 200)
            put_mult = 1.0 - 0.1 * min(1.0, distance_from_spot / 200)
        elif symbol == "BANKNIFTY":
            # Heavy put OI above spot to push Max Pain up
            if strike <= spot:
                call_mult = 1.2
                put_mult = 0.8
            else:
                call_mult = 0.6 - 0.2 * min(1.0, (strike - spot) / 500)
                put_mult = 1.5 + 0.3 * min(1.0, (strike - spot) / 500)
        elif symbol == "SENSEX":
            # Very heavy put OI far above spot
            if strike <= spot + 1000:
                call_mult = 1.0
                put_mult = 0.9
            else:
                call_mult = 0.3 - 0.1 * min(1.0, (strike - spot - 1000) / 2000)
                put_mult = 2.0 + 0.5 * min(1.0, (strike - spot) / 3000)
        else:  # MIDCPNIFTY
            # Slight call heavy below, put heavy above
            if strike <= spot:
                call_mult = 1.1
                put_mult = 0.9
            else:
                call_mult = 0.9
                put_mult = 1.2
        
        call_oi[strike] = int(base_oi * call_mult)
        put_oi[strike] = int(base_oi * put_mult)
    
    # Normalize to target PCR
    actual_call_total = sum(call_oi.values())
    actual_put_total = sum(put_oi.values())
    
    if actual_call_total > 0:
        call_scale = total_call_oi / actual_call_total
        put_scale = total_put_oi / actual_put_total
        
        call_oi = {k: int(v * call_scale) for k, v in call_oi.items()}
        put_oi = {k: int(v * put_scale) for k, v in put_oi.items()}
    
    # Create option mid prices based on Black-Scholes
    call_mids = {}
    put_mids = {}
    
    risk_free_rate = 0.06
    time_to_expiry = 30 / 365.0
    
    for strike in strikes:
        # Simple Black-Scholes pricing
        d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * target_iv**2) * time_to_expiry) / (target_iv * math.sqrt(time_to_expiry))
        d2 = d1 - target_iv * math.sqrt(time_to_expiry)
        
        # Simplified normal CDF approximation
        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        
        call_price = spot * norm_cdf(d1) - strike * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(d2)
        put_price = strike * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(-d2) - spot * norm_cdf(-d1)
        
        call_mids[strike] = max(0.1, call_price)
        put_mids[strike] = max(0.1, put_price)
    
    return spot, strikes, call_oi, put_oi, call_mids, put_mids


def test_sensibull_compatibility():
    """Test all symbols with Sensibull-compatible calculations."""
    
    print("SENSIBULL COMPATIBILITY TEST")
    print("="*60)
    
    benchmarks = {
        "NIFTY": {"spot": 24741, "max_pain": 24750, "atm": 24750, "pcr": 0.77, "atm_iv": 7.9},
        "BANKNIFTY": {"spot": 54114.55, "max_pain": 54600, "atm": 54300, "pcr": 0.87, "atm_iv": 10.2},
        "SENSEX": {"spot": 80710.76, "max_pain": 83500, "atm": 80800, "pcr": 1.37, "atm_iv": 9.0},
        "MIDCPNIFTY": {"spot": 12778.15, "max_pain": 12800, "atm": 12825, "pcr": 1.17, "atm_iv": 15.4}
    }
    
    tolerance = 100  # Relaxed tolerance for mock data testing
    
    for symbol, benchmark in benchmarks.items():
        print(f"\nTesting {symbol}:")
        print(f"Target - Max Pain: {benchmark['max_pain']}, ATM: {benchmark['atm']}, PCR: {benchmark['pcr']:.2f}, IV: {benchmark['atm_iv']:.1f}%")
        
        # Create mock data
        spot, strikes, call_oi, put_oi, call_mids, put_mids = create_sensibull_mock_data(symbol)
        
        # Calculate using Sensibull methodology
        results = calculate_all_sensibull_metrics(
            spot=spot,
            strikes=strikes,
            call_oi=call_oi,
            put_oi=put_oi,
            call_mids=call_mids,
            put_mids=put_mids,
            symbol=symbol
        )
        
        if results["success"]:
            print(f"Calculated - Max Pain: {results['max_pain']}, ATM: {results['atm']}, PCR: {results['pcr']:.2f}, IV: {results['atm_iv']:.1f}%")
            
            # Check differences
            mp_diff = abs(results['max_pain'] - benchmark['max_pain']) if results['max_pain'] else float('inf')
            atm_diff = abs(results['atm'] - benchmark['atm']) if results['atm'] else float('inf')
            pcr_diff = abs(results['pcr'] - benchmark['pcr']) if results['pcr'] else float('inf')
            iv_diff = abs(results['atm_iv'] - benchmark['atm_iv']) if results['atm_iv'] else float('inf')
            
            print(f"Differences - MP: {mp_diff:.1f}, ATM: {atm_diff:.1f}, PCR: {pcr_diff:.3f}, IV: {iv_diff:.1f}%")
            
            # Check if within tolerance
            checks = [
                ("Max Pain", mp_diff < tolerance),
                ("ATM", atm_diff < tolerance),
                ("PCR", pcr_diff < 0.1),
                ("ATM IV", iv_diff < 5.0)
            ]
            
            all_passed = all(check[1] for check in checks)
            status = "✅ PASS" if all_passed else "❌ FAIL"
            print(f"Result: {status}")
            
            for metric, passed in checks:
                symbol_status = "✅" if passed else "❌"
                print(f"  {symbol_status} {metric}")
        else:
            print(f"❌ CALCULATION FAILED: {results['errors']}")
    
    print(f"\n{'='*60}")
    print("Test completed. Check individual metrics for accuracy.")


if __name__ == "__main__":
    test_sensibull_compatibility()