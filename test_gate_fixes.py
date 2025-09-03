#!/usr/bin/env python3
"""
Test the enhanced gate fixes to ensure valid trades aren't being suppressed.
"""

import datetime as dt
import pytz
from src.engine_enhanced import EnhancedTradingEngine, MarketData

def create_strong_signal_data(symbol: str, scenario: str) -> MarketData:
    """Create market data with strong signals that should trigger trades."""
    
    params = {
        "NIFTY": {"spot": 25000.0, "step": 50},
        "BANKNIFTY": {"spot": 52000.0, "step": 100},
    }
    
    spot = params[symbol]["spot"]
    step = params[symbol]["step"]
    
    now = dt.datetime.now(pytz.timezone("Asia/Kolkata"))
    
    if scenario == "strong_bullish":
        # Create conditions that should definitely trigger a LONG trade
        market_data = MarketData(
            timestamp=now,
            index=symbol,
            spot=spot * 1.01,  # 1% above base
            futures_mid=spot * 1.012,  # Strong basis
            strikes=[spot + (i - 10) * step for i in range(21)],
            adx=25.0,  # Strong trend
            volume_ratio=1.6,  # High volume - should trigger volume signal
            spread_bps=12.0,  # Good liquidity
            momentum_score=0.7  # Strong bullish momentum - should trigger price action signal
        )
    elif scenario == "strong_bearish":
        # Create conditions that should definitely trigger a SHORT trade
        market_data = MarketData(
            timestamp=now,
            index=symbol,
            spot=spot * 0.99,  # 1% below base
            futures_mid=spot * 0.988,  # Negative basis
            strikes=[spot + (i - 10) * step for i in range(21)],
            adx=22.0,  # Strong trend
            volume_ratio=1.4,  # High volume
            spread_bps=15.0,  # Good liquidity
            momentum_score=-0.6  # Strong bearish momentum
        )
    elif scenario == "moderate_signals":
        # Create moderate signals that should pass with reduced thresholds
        market_data = MarketData(
            timestamp=now,
            index=symbol,
            spot=spot * 1.005,  # Small move
            futures_mid=spot * 1.007,  # Moderate basis
            strikes=[spot + (i - 10) * step for i in range(21)],
            adx=18.0,  # Moderate trend
            volume_ratio=1.3,  # Moderate volume
            spread_bps=18.0,  # Fair liquidity
            momentum_score=0.25  # Moderate momentum
        )
    else:  # weak_signals
        # Create weak signals that should be blocked
        market_data = MarketData(
            timestamp=now,
            index=symbol,
            spot=spot,  # No move
            futures_mid=spot * 1.001,  # Minimal basis
            strikes=[spot + (i - 10) * step for i in range(21)],
            adx=8.0,  # Very weak trend
            volume_ratio=0.9,  # Low volume
            spread_bps=35.0,  # Poor liquidity
            momentum_score=0.05  # Very weak momentum
        )
    
    # Create realistic option chain
    for strike in market_data.strikes:
        moneyness = (market_data.spot - strike) / market_data.spot
        
        # Simple option pricing
        call_price = max(5, market_data.spot * max(0, moneyness) + market_data.spot * 0.02)
        put_price = max(5, market_data.spot * max(0, -moneyness) + market_data.spot * 0.02)
        
        market_data.call_mids[strike] = call_price
        market_data.put_mids[strike] = put_price
        
        # Create OI distribution that supports the scenario
        base_oi = 1500
        distance_factor = max(0.1, 1 - abs(moneyness) * 2)
        
        if scenario == "strong_bullish":
            # Low PCR for bullish signal (fewer puts relative to calls)
            call_oi = int(base_oi * distance_factor * 1.4)  # Higher call OI
            put_oi = int(base_oi * distance_factor * 0.8)   # Lower put OI
        elif scenario == "strong_bearish":
            # High PCR for bearish signal (more puts relative to calls)
            call_oi = int(base_oi * distance_factor * 0.7)  # Lower call OI  
            put_oi = int(base_oi * distance_factor * 1.6)   # Higher put OI
        else:
            # Balanced OI
            call_oi = int(base_oi * distance_factor)
            put_oi = int(base_oi * distance_factor)
        
        market_data.call_oi[strike] = call_oi
        market_data.put_oi[strike] = put_oi
        
        # Volume
        market_data.call_volumes[strike] = int(call_oi * 0.12)
        market_data.put_volumes[strike] = int(put_oi * 0.12)
    
    return market_data

def test_gate_improvements():
    """Test that gate improvements allow valid trades while blocking invalid ones."""
    
    print("Testing Enhanced Gate Improvements")
    print("=" * 50)
    
    test_cases = [
        ("NIFTY", "strong_bullish", True),
        ("NIFTY", "strong_bearish", True),
        ("NIFTY", "moderate_signals", True),
        ("NIFTY", "weak_signals", False),
        ("BANKNIFTY", "strong_bullish", True),
        ("BANKNIFTY", "moderate_signals", True),
        ("BANKNIFTY", "weak_signals", False),
    ]
    
    results = []
    
    for symbol, scenario, should_trade in test_cases:
        print(f"\n--- Testing {symbol} with {scenario} ---")
        
        # Create engine
        expiry = dt.datetime.now(pytz.timezone("Asia/Kolkata")) + dt.timedelta(hours=4)
        engine = EnhancedTradingEngine(symbol, expiry)
        
        # Create market data
        market_data = create_strong_signal_data(symbol, scenario)
        
        # Process decision with some trend score and ORB signal for stronger signal alignment
        if scenario in ["strong_bullish", "moderate_signals"]:
            trend_score = 0.6
            orb_signal = "LONG"
            orb_strength = 0.7
        elif scenario == "strong_bearish":
            trend_score = -0.6
            orb_signal = "SHORT"
            orb_strength = 0.7
        else:  # weak_signals
            trend_score = 0.1
            orb_signal = None
            orb_strength = 0.0
        
        decision = engine.process_market_data(
            market_data, 
            trend_score=trend_score,
            orb_signal=orb_signal,
            orb_strength=orb_strength,
            orb_breakout_size=50.0
        )
        
        # Check result
        is_trading = decision.action != "NO_TRADE" and not (decision.gate_decision and decision.gate_decision.muted)
        
        print(f"✓ Spot: {market_data.spot:.2f}")
        print(f"✓ Volume Ratio: {market_data.volume_ratio:.2f}")
        print(f"✓ Momentum Score: {market_data.momentum_score:.2f}")
        print(f"✓ ADX: {market_data.adx:.1f}")
        print(f"✓ PCR: {decision.pcr_total:.2f}" if decision.pcr_total else "✓ PCR: N/A")
        print(f"✓ Decision: {decision.action}")
        print(f"✓ Direction: {decision.direction}")
        print(f"✓ Confidence: {decision.confidence:.2f}")
        print(f"✓ Muted: {decision.gate_decision.muted if decision.gate_decision else 'N/A'}")
        if decision.gate_decision:
            print(f"✓ Gate Reason: {decision.gate_decision.primary_reason}")
            if decision.gate_decision.supporting_factors:
                print(f"✓ Supporting Factors: {decision.gate_decision.supporting_factors}")
        
        # Determine if test passed
        test_passed = is_trading == should_trade
        status = "✅ PASS" if test_passed else "❌ FAIL"
        expected = "should trade" if should_trade else "should NOT trade"
        actual = "is trading" if is_trading else "is NOT trading"
        
        print(f"✓ Result: {status} - {expected}, {actual}")
        
        results.append((symbol, scenario, should_trade, is_trading, test_passed))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, _, _, _, test_passed in results if test_passed)
    total = len(results)
    
    for symbol, scenario, should_trade, is_trading, test_passed in results:
        status = "✅" if test_passed else "❌"
        print(f"{status} {symbol} {scenario}: Expected {'TRADE' if should_trade else 'NO-TRADE'}, Got {'TRADE' if is_trading else 'NO-TRADE'}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Gate fixes are working correctly.")
    else:
        print("⚠️  Some tests failed. Gate logic may need further adjustment.")
    
    return passed == total

if __name__ == "__main__":
    test_gate_improvements()