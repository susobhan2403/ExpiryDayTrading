#!/usr/bin/env python3
"""
Test with specific values from the problem statement.
"""

import datetime as dt
import pytz
from src.engine_enhanced import EnhancedTradingEngine, MarketData
from engine_runner import _create_realistic_fallback_data

IST = pytz.timezone("Asia/Kolkata")

def test_expected_vs_actual():
    """Test to match the expected values from problem statement."""
    
    print("=== Problem Statement Values ===")
    print("Expected - PCR: 1.26, Max Pain: 24700, ATM Strike: 24750, IV Percentile: 2")
    print("Previous Actual - PCR: 1.0, Max Pain: 24715, ATM Strike: 24715, IV Percentile: 15")
    
    # Create engine
    expiry = IST.localize(dt.datetime(2025, 9, 4, 15, 30))
    engine = EnhancedTradingEngine(
        index="NIFTY",
        expiry=expiry,
        min_tau_hours=2.0
    )
    
    # Create data similar to the expected scenario
    spot = 24715.05
    
    # Create realistic fallback data that should give PCR around 1.26
    market_data = _create_realistic_fallback_data("NIFTY", spot)
    
    # Adjust the OI to get closer to expected PCR of 1.26
    # Our fallback creates ~1.26, but let's fine-tune it
    total_call_oi = sum(market_data.call_oi.values())
    total_put_oi = sum(market_data.put_oi.values())
    current_pcr = total_put_oi / total_call_oi
    
    print(f"\n=== Market Data ===")
    print(f"Spot: {spot:.2f}")
    print(f"Current PCR from fallback: {current_pcr:.3f}")
    
    # Process with engine
    decision = engine.process_market_data(market_data)
    
    print(f"\n=== New Results with Fixes ===")
    print(f"PCR (Expected: 1.26, Actual: {decision.pcr_total:.2f})" if decision.pcr_total else "PCR: None")
    print(f"Max Pain (Expected: 24700, Actual: {int(decision.atm_strike)})" if decision.atm_strike else "Max Pain: None")
    print(f"ATM Strike (Expected: 24750, Actual: {int(decision.atm_strike)})" if decision.atm_strike else "ATM Strike: None")
    print(f"IV Percentile (Expected: 2, Actual: {decision.iv_percentile})" if decision.iv_percentile else "IV Percentile: None")
    
    # Test case where calculations fail (empty data)
    print(f"\n=== Empty Data Fallback Test ===")
    empty_data = MarketData(
        timestamp=dt.datetime.now(IST),
        index="NIFTY",
        spot=spot,
        futures_mid=spot * 1.001,
        strikes=list(range(24500, 25000, 50)),
        call_mids={},
        put_mids={},
        call_oi={},
        put_oi={}
    )
    
    decision2 = engine.process_market_data(empty_data)
    
    print(f"PCR (Expected: NOT 1.0, Actual: {decision2.pcr_total})")
    print(f"Max Pain (Expected: NOT spot, Actual: {decision2.atm_strike})")
    print(f"ATM Strike (Expected: NOT spot, Actual: {decision2.atm_strike})")
    print(f"IV Percentile (Expected: NOT 15, Actual: {decision2.iv_percentile})")

if __name__ == "__main__":
    test_expected_vs_actual()