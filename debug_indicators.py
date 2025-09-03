#!/usr/bin/env python3
"""
Debug script to test technical indicator calculations.
"""

import datetime as dt
import pytz
from src.engine_enhanced import EnhancedTradingEngine, MarketData

IST = pytz.timezone("Asia/Kolkata")

def test_technical_indicators():
    """Test technical indicators with sample data to reproduce the issue."""
    
    # Create an engine for NIFTY
    expiry = IST.localize(dt.datetime(2025, 9, 4, 15, 30))
    engine = EnhancedTradingEngine(
        index="NIFTY",
        expiry=expiry,
        min_tau_hours=2.0
    )
    
    # Create market data with comprehensive options chain
    spot = 24715.05
    strikes = list(range(24500, 25000, 50))  # 24500, 24550, ..., 24950
    
    # Create realistic call and put prices
    call_mids = {}
    put_mids = {}
    call_oi = {}
    put_oi = {}
    
    for strike in strikes:
        # Simple ATM pricing model
        intrinsic_call = max(0, spot - strike)
        intrinsic_put = max(0, strike - spot)
        
        # Add some time value
        time_value = 50 + abs(strike - spot) * 0.1
        
        call_mids[strike] = intrinsic_call + time_value
        put_mids[strike] = intrinsic_put + time_value
        
        # OI distribution - higher near ATM
        distance_from_atm = abs(strike - spot)
        base_oi = max(1000, 10000 - distance_from_atm * 20)
        
        call_oi[strike] = int(base_oi * (0.8 + 0.4 * (distance_from_atm / 500)))
        put_oi[strike] = int(base_oi * (1.2 + 0.6 * (distance_from_atm / 500)))
    
    # Create market data
    market_data = MarketData(
        timestamp=dt.datetime.now(IST),
        index="NIFTY",
        spot=spot,
        futures_mid=spot * 1.001,
        strikes=strikes,
        call_mids=call_mids,
        put_mids=put_mids,
        call_oi=call_oi,
        put_oi=put_oi,
        adx=25.0,
        volume_ratio=1.5,
        spread_bps=12.0,
        momentum_score=0.1
    )
    
    print(f"=== Testing NIFTY Technical Indicators ===")
    print(f"Spot: {spot:.2f}")
    print(f"Strikes: {len(strikes)} from {min(strikes)} to {max(strikes)}")
    
    # Calculate expected values manually
    total_call_oi = sum(call_oi.values())
    total_put_oi = sum(put_oi.values())
    expected_pcr = total_put_oi / total_call_oi
    print(f"Expected PCR: {expected_pcr:.3f}")
    
    # Find expected ATM strike (closest to spot)
    expected_atm = min(strikes, key=lambda k: abs(k - spot))
    print(f"Expected ATM Strike: {expected_atm}")
    
    # Process with engine
    decision = engine.process_market_data(market_data)
    
    print(f"\n=== Results ===")
    print(f"Action: {decision.action}")
    print(f"ATM Strike: {decision.atm_strike}")
    print(f"PCR Total: {decision.pcr_total}")
    print(f"PCR Band: {decision.pcr_band}")
    print(f"ATM IV: {decision.atm_iv}")
    print(f"IV Percentile: {decision.iv_percentile}")
    print(f"Forward: {decision.forward}")
    print(f"Processing Time: {decision.processing_time_ms:.2f}ms")
    
    print(f"\n=== Comparison ===")
    print(f"PCR - Expected: {expected_pcr:.3f}, Actual: {decision.pcr_total}")
    print(f"ATM Strike - Expected: {expected_atm}, Actual: {decision.atm_strike}")

if __name__ == "__main__":
    test_technical_indicators()