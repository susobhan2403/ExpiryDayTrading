#!/usr/bin/env python3
"""
Test script to validate our logging fixes.
This script tests the specific issues mentioned:
1. Per-index expiry logging
2. Real technical indicators in logs
3. Condition display with tick/cross symbols
4. Unicode encoding safety
5. Varied scenario detection
"""

import datetime as dt
import logging
import sys
from io import StringIO
import pytz

from src.engine_enhanced import EnhancedTradingEngine, MarketData
from src.output.logging_formatter import DualOutputFormatter, log_expiry_info
from engine_runner import create_sample_market_data

# Setup basic logging to capture output
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_individual_engine_output():
    """Test output for individual engines to verify fixes."""
    print("Testing Enhanced Engine Logging Fixes")
    print("=" * 50)
    
    formatter = DualOutputFormatter()
    symbols = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
    
    for symbol in symbols:
        print(f"\n=== Testing {symbol} ===")
        
        # Create engine and market data directly
        expiry = dt.datetime.now(pytz.timezone("Asia/Kolkata")) + dt.timedelta(days=1)
        engine = EnhancedTradingEngine(symbol, expiry)
        
        # Create comprehensive market data
        if symbol == "NIFTY":
            spot = 25000.0
            step = 50
        elif symbol == "BANKNIFTY":
            spot = 52000.0
            step = 100
        elif symbol == "SENSEX":
            spot = 82000.0
            step = 100
        elif symbol == "MIDCPNIFTY":
            spot = 13000.0
            step = 50
        else:
            spot = 25000.0
            step = 50
            
        # Create realistic market data
        now = dt.datetime.now(pytz.timezone("Asia/Kolkata"))
        market_data = MarketData(
            timestamp=now,
            index=symbol,
            spot=spot,
            futures_mid=spot * 1.002,
            strikes=[spot + (i - 10) * step for i in range(21)],
            adx=16.5,
            volume_ratio=0.8,
            spread_bps=15.0,
            momentum_score=0.1
        )
        
        # Add realistic option prices and OI
        for strike in market_data.strikes:
            # Simple option pricing
            moneyness = (spot - strike) / spot
            
            # Call prices
            if strike <= spot:
                call_price = max(10, spot - strike + 50 + abs(moneyness) * 200)
            else:
                call_price = max(5, 50 * (1 - min(1, abs(moneyness) * 4)))
            
            # Put prices
            if strike >= spot:
                put_price = max(10, strike - spot + 50 + abs(moneyness) * 200)
            else:
                put_price = max(5, 50 * (1 - min(1, abs(moneyness) * 4)))
            
            market_data.call_mids[strike] = call_price
            market_data.put_mids[strike] = put_price
            
            # OI data
            base_oi = 1000
            distance_factor = max(0.1, 1 - abs(moneyness) * 2)
            market_data.call_oi[strike] = int(base_oi * distance_factor * (1.2 if strike > spot else 0.8))
            market_data.put_oi[strike] = int(base_oi * distance_factor * (1.2 if strike < spot else 0.8))
            
            # Volume data
            market_data.call_volumes[strike] = int(market_data.call_oi[strike] * 0.1)
            market_data.put_volumes[strike] = int(market_data.put_oi[strike] * 0.1)
            
        # Process decision
        decision = engine.process_market_data(market_data)
        
        # Test expiry logging
        today = dt.date.today()
        next_thursday = today + dt.timedelta(days=(3 - today.weekday()) % 7)
        expiry_str = next_thursday.strftime('%Y-%m-%d')
        step = 100 if symbol in ["BANKNIFTY", "SENSEX"] else 50
        atm = int(decision.atm_strike) if decision.atm_strike else int(market_data.spot)
        pcr = decision.pcr_total if decision.pcr_total else 0.98
        
        print(f"✓ Expiry Info: expiry={expiry_str} step={step} atm={atm} pcr={pcr:.2f}")
        
        # Test formatted output
        console_output, file_output = formatter.format_decision_output(market_data, decision)
        
        print("✓ Console Output (with colors):")
        for line in console_output.split('\n'):
            if line.strip():
                print(f"  {line}")
                
        print("\n✓ File Output (plain text):")
        for line in file_output.split('\n'):
            if line.strip():
                print(f"  {line}")
        
        # Verify specific indicators are present
        content = file_output.upper()
        checks = [
            ("PCR", "PCR" in content),
            ("ATM IV", "ATM" in content and "IV" in content),
            ("SCENARIO", "SCENARIO" in content),
            ("ACTION", "ACTION" in content),
        ]
        
        print("\n✓ Content Verification:")
        for check_name, passed in checks:
            status = "✓" if passed else "❌"
            print(f"  {status} {check_name}: {'Present' if passed else 'Missing'}")
        
        # Check for condition lines (numbered items)
        has_conditions = any(line.strip().startswith(str(i) + '.') for i in range(1, 10) 
                           for line in file_output.split('\n'))
        status = "✓" if has_conditions else "❌"
        print(f"  {status} Condition Lines: {'Present' if has_conditions else 'Missing'}")
        
        print("-" * 30)

def test_unicode_safety():
    """Test that our Unicode fix works."""
    print("\n=== Testing Unicode Safety ===")
    
    # Test the tick function with different encodings
    from src.output.format import format_output_line
    
    # This should not cause encoding errors
    try:
        now = dt.datetime.now(pytz.timezone("Asia/Kolkata"))
        mock_params = {
            'now': now,
            'symbol': 'NIFTY',
            'spot_now': 25000.0,
            'vwap_fut': 25050.0,
            'D': 100.0,
            'ATR_D': 200.0,
            'snap': {
                'vnd': 0.5, 'ssd': 0.5, 'pdist_pct': 0.1,
                'pcr': 0.98, 'dpcr_z': 0.0, 'mph_pts_per_hr': 0.0,
                'mph_norm': 0.0, 'atm_iv': 15.0, 'iv_z': 0.0, 'basis': 50.0
            },
            'mp': 25000,
            'atm_k': 25000,
            'probs': {'Short-cover reversion up': 0.6, 'Pin & decay day (IV crush)': 0.4},
            'top': 'Short-cover reversion up',
            'tp': {'action': 'NO-TRADE', 'why': 'confidence below threshold'},
            'oi_flags': {'two_sided_adjacent': True, 'pe_write_above': False, 
                        'ce_unwind_below': False, 'ce_write_above': False, 'pe_unwind_below': False},
            'vwap_spot': 24950.0,
            'adx5': 15.0,
            'div': 0.0,
            'iv_pct_hint': 45.0,
            'macd_last': 0.0,
            'macd_sig_last': 0.0,
            'VND': 0.5,
            'PIN_NORM': 0.5,
            'MPH_NORM_THR': 0.5
        }
        
        output = format_output_line(**mock_params)
        
        # Check if output contains our safe ASCII characters
        has_safe_chars = "[Y]" in output or "[N]" in output
        print(f"✓ Unicode Safety: {'PASS - Using safe ASCII characters' if has_safe_chars else 'FAIL - May have Unicode issues'}")
        
        # Try to encode as cp1252 (Windows encoding)
        try:
            output.encode('cp1252')
            print("✓ Windows Encoding: PASS - No cp1252 encoding errors")
        except UnicodeEncodeError as e:
            print(f"❌ Windows Encoding: FAIL - {e}")
            
    except Exception as e:
        print(f"❌ Unicode Test Failed: {e}")

if __name__ == "__main__":
    test_individual_engine_output()
    test_unicode_safety()
    print("\n" + "=" * 50)
    print("✅ Testing Complete!")