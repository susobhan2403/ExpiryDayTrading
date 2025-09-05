#!/usr/bin/env python3
"""
Zero-tolerance accuracy validation against Sensibull benchmarks.

This script validates our precise calculation implementations against
the exact values provided by the user for zero-tolerance compliance.

Target Values:
NIFTY:      Spot 24741, Max Pain 24750, ATM 24750, PCR 0.77, ATM IV 7.9
BANKNIFTY:  Spot 54114.55, Max Pain 54600, ATM 54300, PCR 0.87, ATM IV 10.2
SENSEX:     Spot 80710.76, Max Pain 83500, ATM 80800, PCR 1.37, ATM IV 9
MIDCPNIFTY: Spot 12778.15, Max Pain 12800, ATM 12825, PCR 1.17, ATM IV 15.4
"""

import sys
import datetime as dt
import logging
from typing import Dict, Any
import pytz

# Setup paths
sys.path.append('/home/runner/work/ExpiryDayTrading/ExpiryDayTrading')

from src.engine_enhanced import EnhancedTradingEngine, MarketData
from src.provider.kite import KiteProvider
from src.calculations.max_pain import calculate_max_pain_with_validation
from src.calculations.atm import calculate_atm_with_validation
from src.calculations.pcr import calculate_pcr_with_validation
from src.calculations.iv import calculate_atm_iv_with_validation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

# Sensibull benchmark values
BENCHMARKS = {
    "NIFTY": {
        "spot": 24741,
        "max_pain": 24750,
        "atm": 24750,
        "pcr": 0.77,
        "atm_iv": 7.9
    },
    "BANKNIFTY": {
        "spot": 54114.55,
        "max_pain": 54600,
        "atm": 54300,
        "pcr": 0.87,
        "atm_iv": 10.2
    },
    "SENSEX": {
        "spot": 80710.76,
        "max_pain": 83500,
        "atm": 80800,
        "pcr": 1.37,
        "atm_iv": 9.0
    },
    "MIDCPNIFTY": {
        "spot": 12778.15,
        "max_pain": 12800,
        "atm": 12825,
        "pcr": 1.17,
        "atm_iv": 15.4
    }
}

TOLERANCE = 0.01  # Maximum allowed error per R5 requirement


def validate_calculation(calculated: float, benchmark: float, metric_name: str, symbol: str) -> bool:
    """
    Validate calculated value against benchmark with zero tolerance.
    
    Returns True if within tolerance, False otherwise.
    """
    if calculated is None:
        logger.error(f"{symbol} {metric_name}: FAILED - Calculation returned None")
        return False
    
    error = abs(calculated - benchmark)
    error_pct = (error / benchmark) * 100 if benchmark != 0 else 0
    
    if error <= TOLERANCE:
        logger.info(f"{symbol} {metric_name}: PASSED - Calculated: {calculated:.4f}, Benchmark: {benchmark:.4f}, Error: {error:.6f}")
        return True
    else:
        logger.error(f"{symbol} {metric_name}: FAILED - Calculated: {calculated:.4f}, Benchmark: {benchmark:.4f}, Error: {error:.6f} (>{TOLERANCE})")
        return False


def create_test_market_data(symbol: str, provider: KiteProvider) -> MarketData:
    """
    Create market data for testing using real Kite Connect data.
    
    R3 COMPLIANCE: Only real data, no synthetic fallbacks.
    """
    try:
        # Get real spot price
        quotes = provider.get_indices_snapshot([symbol])
        if not quotes or symbol not in quotes:
            raise ValueError(f"Failed to get spot price for {symbol}")
        
        spot = quotes[symbol]
        logger.info(f"{symbol} real spot price: {spot}")
        
        # Get next expiry
        from src.features.options import nearest_weekly_expiry
        expiry_iso = nearest_weekly_expiry(dt.datetime.now(IST), symbol)
        
        # Get real options chain data
        chain = provider.get_option_chain(symbol, expiry_iso)
        logger.info(f"{symbol} got options chain with {len(chain.get('strikes', []))} strikes")
        
        # Extract data
        strikes = chain.get('strikes', [])
        if not strikes:
            raise ValueError(f"No strikes available for {symbol}")
        
        call_mids = {}
        put_mids = {}
        call_oi = {}
        put_oi = {}
        
        # Process calls
        for strike, data in chain.get('calls', {}).items():
            if isinstance(data, dict):
                bid = data.get('bid', 0.0)
                ask = data.get('ask', 0.0)
                ltp = data.get('ltp', 0.0)
                
                mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else ltp
                call_mids[strike] = mid if mid > 0 else 0.0
                call_oi[strike] = data.get('oi', 0)
        
        # Process puts
        for strike, data in chain.get('puts', {}).items():
            if isinstance(data, dict):
                bid = data.get('bid', 0.0)
                ask = data.get('ask', 0.0)
                ltp = data.get('ltp', 0.0)
                
                mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else ltp
                put_mids[strike] = mid if mid > 0 else 0.0
                put_oi[strike] = data.get('oi', 0)
        
        # Calculate total OI for validation
        total_call_oi = sum(call_oi.values())
        total_put_oi = sum(put_oi.values())
        logger.info(f"{symbol} Total OI - Calls: {total_call_oi:,}, Puts: {total_put_oi:,}")
        
        if total_call_oi == 0 or total_put_oi == 0:
            raise ValueError(f"Insufficient OI data for {symbol}")
        
        # Create futures price (realistic carry calculation)
        time_to_expiry = (dt.date.fromisoformat(expiry_iso) - dt.date.today()).days / 365.0
        futures_mid = spot * (1.0 + 0.06 * time_to_expiry)  # 6% carry rate
        
        return MarketData(
            timestamp=dt.datetime.now(IST),
            index=symbol,
            spot=spot,
            futures_mid=futures_mid,
            strikes=strikes,
            call_mids=call_mids,
            put_mids=put_mids,
            call_oi=call_oi,
            put_oi=put_oi
        )
    
    except Exception as e:
        logger.error(f"Failed to create market data for {symbol}: {e}")
        raise


def test_individual_calculations(symbol: str, market_data: MarketData) -> Dict[str, float]:
    """
    Test individual calculation modules directly.
    """
    benchmark = BENCHMARKS[symbol]
    results = {}
    
    # Strike step detection
    from src.calculations.atm import detect_strike_step_precise
    step = detect_strike_step_precise(market_data.strikes)
    if step is None:
        step = 100 if symbol in ["BANKNIFTY", "SENSEX"] else 50
    logger.info(f"{symbol} detected step: {step}")
    
    # Test Max Pain calculation
    max_pain_result, max_pain_status = calculate_max_pain_with_validation(
        strikes=market_data.strikes,
        call_oi=market_data.call_oi,
        put_oi=market_data.put_oi,
        spot=market_data.spot,
        step=step
    )
    results['max_pain'] = max_pain_result
    logger.info(f"{symbol} Max Pain: {max_pain_result} (Status: {max_pain_status})")
    
    # Test ATM calculation  
    time_to_expiry = 30 / 365.0  # Assume 30 days for testing
    atm_result, forward_price, atm_status = calculate_atm_with_validation(
        spot=market_data.spot,
        strikes=market_data.strikes,
        risk_free_rate=0.06,
        dividend_yield=0.01,
        time_to_expiry_years=time_to_expiry,
        futures_mid=market_data.futures_mid,
        call_mids=market_data.call_mids,
        put_mids=market_data.put_mids,
        symbol=symbol
    )
    results['atm'] = atm_result
    results['forward'] = forward_price
    logger.info(f"{symbol} ATM: {atm_result}, Forward: {forward_price} (Status: {atm_status})")
    
    # Test PCR calculation
    pcr_result, pcr_status = calculate_pcr_with_validation(
        call_oi=market_data.call_oi,
        put_oi=market_data.put_oi,
        atm_strike=atm_result,
        step=step,
        band_width=6
    )
    if pcr_result:
        results['pcr'] = pcr_result.get('pcr_total')
        logger.info(f"{symbol} PCR: {results['pcr']} (Status: {pcr_status})")
    
    # Test ATM IV calculation
    if atm_result and forward_price:
        call_price = market_data.call_mids.get(atm_result)
        put_price = market_data.put_mids.get(atm_result)
        
        atm_iv_result, iv_status = calculate_atm_iv_with_validation(
            call_price=call_price,
            put_price=put_price,
            forward_price=forward_price,
            atm_strike=atm_result,
            time_to_expiry_years=time_to_expiry,
            risk_free_rate=0.06
        )
        if atm_iv_result:
            results['atm_iv'] = atm_iv_result * 100  # Convert to percentage
            logger.info(f"{symbol} ATM IV: {results['atm_iv']:.2f}% (Status: {iv_status})")
    
    return results


def main():
    """
    Main validation function.
    """
    logger.info("Starting zero-tolerance accuracy validation against Sensibull benchmarks")
    logger.info(f"Error tolerance: {TOLERANCE}")
    
    # Initialize provider
    try:
        provider = KiteProvider()
        logger.info("Kite provider initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Kite provider: {e}")
        logger.error("Ensure .kite_session.json exists and KITE_API_KEY is set")
        return False
    
    overall_results = {}
    all_tests_passed = True
    
    for symbol in BENCHMARKS.keys():
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {symbol}")
        logger.info(f"{'='*60}")
        
        try:
            # Get real market data (R3 compliance - no synthetic data)
            market_data = create_test_market_data(symbol, provider)
            
            # Test individual calculations
            calculated_values = test_individual_calculations(symbol, market_data)
            
            # Validate against benchmarks
            benchmark = BENCHMARKS[symbol]
            symbol_passed = True
            
            for metric in ['max_pain', 'atm', 'pcr', 'atm_iv']:
                if metric in calculated_values and calculated_values[metric] is not None:
                    benchmark_value = benchmark[metric]
                    calculated_value = calculated_values[metric]
                    
                    test_passed = validate_calculation(
                        calculated_value, benchmark_value, metric.upper(), symbol
                    )
                    
                    if not test_passed:
                        symbol_passed = False
                        all_tests_passed = False
                else:
                    logger.error(f"{symbol} {metric.upper()}: FAILED - No calculated value")
                    symbol_passed = False
                    all_tests_passed = False
            
            overall_results[symbol] = {
                'passed': symbol_passed,
                'calculated': calculated_values,
                'benchmark': benchmark
            }
            
            logger.info(f"\n{symbol} Overall: {'PASSED' if symbol_passed else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"Critical error testing {symbol}: {e}")
            overall_results[symbol] = {'passed': False, 'error': str(e)}
            all_tests_passed = False
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("FINAL VALIDATION SUMMARY")
    logger.info(f"{'='*80}")
    
    for symbol, result in overall_results.items():
        status = "PASSED" if result.get('passed', False) else "FAILED"
        logger.info(f"{symbol:12}: {status}")
    
    logger.info(f"\nOverall Result: {'ALL TESTS PASSED' if all_tests_passed else 'SOME TESTS FAILED'}")
    
    if all_tests_passed:
        logger.info("✅ Zero-tolerance accuracy requirement satisfied!")
        logger.info("All calculations match Sensibull benchmarks within 0.01 tolerance")
    else:
        logger.error("❌ Zero-tolerance accuracy requirement NOT satisfied")
        logger.error("Some calculations deviate from Sensibull benchmarks")
    
    return all_tests_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)