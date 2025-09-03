#!/usr/bin/env python3
"""
Test the log output format to see actual vs expected values.
"""

import datetime as dt
import pytz
import logging
from src.engine_enhanced import EnhancedTradingEngine, MarketData
from src.output.logging_formatter import DualOutputFormatter
from engine_runner import _create_realistic_fallback_data

IST = pytz.timezone("Asia/Kolkata")

def test_log_output():
    """Test the actual log output to verify fixes."""
    
    # Setup logging
    logger = logging.getLogger("test_engine")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d INFO: %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Create engine
    expiry = IST.localize(dt.datetime(2025, 9, 4, 15, 30))
    engine = EnhancedTradingEngine(
        index="NIFTY",
        expiry=expiry,
        min_tau_hours=2.0
    )
    
    print("=== Testing Log Output Format ===")
    
    # Test 1: With realistic data (simulating successful chain loading with fixes)
    print("\n--- Test 1: With Realistic Data (Post-Fix) ---")
    spot = 24715.05
    market_data = _create_realistic_fallback_data("NIFTY", spot)
    
    decision = engine.process_market_data(market_data)
    
    # Create formatted output
    formatter_obj = DualOutputFormatter()
    console_output, file_output = formatter_obj.format_decision_output(market_data, decision, 1)
    
    # Extract key values for comparison
    lines = file_output.split('\n')
    
    print("Problem Statement Expected:")
    print("  PCR: 1.26, Max Pain: 24700, ATM Strike: 24750, IV Percentile: 2")
    print("  Previous (broken): PCR: 1.0, Max Pain: 24715, ATM Strike: 24715, IV Percentile: 15")
    
    print("\nActual Log Output (Post-Fix):")
    for line in lines:
        if 'PCR' in line or 'ATM' in line or 'expiry=' in line:
            print(f"  {line.strip()}")
    
    print(f"\nDirect Engine Values:")
    print(f"  PCR Total: {decision.pcr_total:.3f}" if decision.pcr_total else "  PCR Total: None")
    print(f"  ATM Strike: {decision.atm_strike}" if decision.atm_strike else "  ATM Strike: None") 
    print(f"  IV Percentile: {decision.iv_percentile}" if decision.iv_percentile else "  IV Percentile: None")
    print(f"  ATM IV: {decision.atm_iv:.4f}" if decision.atm_iv else "  ATM IV: None")
    
    # Test 2: With empty data (simulating provider failure)
    print("\n--- Test 2: With Empty Data (Provider Failure) ---")
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
    console_output2, file_output2 = formatter_obj.format_decision_output(empty_data, decision2, 2)
    
    lines2 = file_output2.split('\n')
    
    print("Expected: Should NOT show hardcoded 1.0, 15.0%, etc.")
    print("Actual Log Output:")
    for line in lines2:
        if 'PCR' in line or 'ATM' in line:
            print(f"  {line.strip()}")
    
    print(f"\nDirect Engine Values (Empty Data):")
    print(f"  PCR Total: {decision2.pcr_total}")
    print(f"  ATM Strike: {decision2.atm_strike}")
    print(f"  IV Percentile: {decision2.iv_percentile}")
    print(f"  ATM IV: {decision2.atm_iv}")

if __name__ == "__main__":
    test_log_output()