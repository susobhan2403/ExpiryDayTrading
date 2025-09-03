#!/usr/bin/env python3
"""
Test script to reproduce the empty options data issue.
"""

import datetime as dt
import pytz
from src.engine_enhanced import EnhancedTradingEngine, MarketData

IST = pytz.timezone("Asia/Kolkata")

def test_empty_options_data():
    """Test with empty options data to reproduce the issue."""
    
    # Create an engine for NIFTY
    expiry = IST.localize(dt.datetime(2025, 9, 4, 15, 30))
    engine = EnhancedTradingEngine(
        index="NIFTY",
        expiry=expiry,
        min_tau_hours=2.0
    )
    
    # Create market data with empty options (simulating failed options chain)
    spot = 24715.05
    strikes = list(range(24500, 25000, 50))  # Strikes exist but no pricing/OI data
    
    market_data = MarketData(
        timestamp=dt.datetime.now(IST),
        index="NIFTY",
        spot=spot,
        futures_mid=spot * 1.001,
        strikes=strikes,
        call_mids={},  # Empty - this is the problem
        put_mids={},   # Empty - this is the problem
        call_oi={},    # Empty - this is the problem
        put_oi={},     # Empty - this is the problem
        adx=25.0,
        volume_ratio=1.5,
        spread_bps=12.0,
        momentum_score=0.1
    )
    
    print(f"=== Testing Empty Options Data ===")
    print(f"Spot: {spot:.2f}")
    print(f"Strikes: {len(strikes)} from {min(strikes)} to {max(strikes)}")
    print(f"Call mids: {len(market_data.call_mids)} entries")
    print(f"Put mids: {len(market_data.put_mids)} entries")
    print(f"Call OI: {len(market_data.call_oi)} entries")
    print(f"Put OI: {len(market_data.put_oi)} entries")
    
    # Process with engine
    decision = engine.process_market_data(market_data)
    
    print(f"\n=== Results with Empty Data ===")
    print(f"Action: {decision.action}")
    print(f"ATM Strike: {decision.atm_strike}")
    print(f"PCR Total: {decision.pcr_total}")
    print(f"PCR Band: {decision.pcr_band}")
    print(f"ATM IV: {decision.atm_iv}")
    print(f"IV Percentile: {decision.iv_percentile}")
    print(f"Forward: {decision.forward}")
    print(f"Processing Time: {decision.processing_time_ms:.2f}ms")
    
    print(f"\n=== Expected Issues ===")
    print(f"PCR should be None (no OI data), but logging formatter defaults to 1.0")
    print(f"ATM IV should be None (no price data)")
    print(f"ATM Strike might be 0 (calculation fails)")

if __name__ == "__main__":
    test_empty_options_data()