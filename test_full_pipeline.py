#!/usr/bin/env python3
"""
Test the full pipeline with realistic fallback data to verify fixes.
"""

import datetime as dt
import pytz
from src.engine_enhanced import EnhancedTradingEngine, MarketData
from src.output.logging_formatter import DualOutputFormatter
from engine_runner import _create_realistic_fallback_data

IST = pytz.timezone("Asia/Kolkata")

def test_full_pipeline():
    """Test the full pipeline including logging formatter."""
    
    print("=== Testing Full Pipeline with Realistic Fallback Data ===")
    
    # Create engine
    expiry = IST.localize(dt.datetime(2025, 9, 4, 15, 30))
    engine = EnhancedTradingEngine(
        index="NIFTY",
        expiry=expiry,
        min_tau_hours=2.0
    )
    
    # Test 1: With realistic fallback data (simulating provider failure)
    print("\n--- Test 1: Realistic Fallback Data ---")
    spot = 24715.05
    market_data = _create_realistic_fallback_data("NIFTY", spot)
    
    print(f"Spot: {spot:.2f}")
    print(f"Strikes: {len(market_data.strikes)} from {min(market_data.strikes)} to {max(market_data.strikes)}")
    print(f"Call mids: {len(market_data.call_mids)} entries")
    print(f"Put mids: {len(market_data.put_mids)} entries")
    print(f"Call OI: {len(market_data.call_oi)} entries")
    print(f"Put OI: {len(market_data.put_oi)} entries")
    
    # Calculate expected values
    total_call_oi = sum(market_data.call_oi.values())
    total_put_oi = sum(market_data.put_oi.values())
    expected_pcr = total_put_oi / total_call_oi
    expected_atm = min(market_data.strikes, key=lambda k: abs(k - spot))
    
    print(f"Expected PCR: {expected_pcr:.3f}")
    print(f"Expected ATM Strike: {expected_atm}")
    
    # Process with engine
    decision = engine.process_market_data(market_data)
    
    print(f"\nEngine Results:")
    print(f"Action: {decision.action}")
    print(f"ATM Strike: {decision.atm_strike}")
    print(f"PCR Total: {decision.pcr_total:.3f}" if decision.pcr_total else "PCR Total: None")
    print(f"PCR Band: {decision.pcr_band:.3f}" if decision.pcr_band else "PCR Band: None")
    print(f"ATM IV: {decision.atm_iv:.4f}" if decision.atm_iv else "ATM IV: None")
    print(f"IV Percentile: {decision.iv_percentile:.1f}" if decision.iv_percentile else "IV Percentile: None")
    print(f"Forward: {decision.forward:.2f}")
    
    # Test logging formatter
    formatter = DualOutputFormatter()
    console_output, file_output = formatter.format_decision_output(market_data, decision, 1)
    
    print(f"\n--- Formatted Output (File Version) ---")
    for line in file_output.split('\n'):
        if line.strip():
            print(line)
    
    print(f"\n--- Test 2: Empty Options Data ---")
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
    console_output2, file_output2 = formatter.format_decision_output(empty_data, decision2, 2)
    
    print(f"Engine Results with Empty Data:")
    print(f"ATM Strike: {decision2.atm_strike}")
    print(f"PCR Total: {decision2.pcr_total}")
    print(f"ATM IV: {decision2.atm_iv}")
    
    print(f"\n--- Formatted Output for Empty Data ---")
    for line in file_output2.split('\n'):
        if line.strip() and ('PCR' in line or 'ATM' in line or 'IV' in line):
            print(line)

if __name__ == "__main__":
    test_full_pipeline()