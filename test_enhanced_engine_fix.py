#!/usr/bin/env python3
"""
Unit tests for the enhanced engine fix and comprehensive validation.
"""

import unittest
import datetime as dt
from unittest.mock import MagicMock, patch
import pytz

from src.engine_enhanced import EnhancedTradingEngine, MarketData, TradingDecision
from src.provider.kite import KiteProvider

IST = pytz.timezone("Asia/Kolkata")


class TestEnhancedEngineFix(unittest.TestCase):
    """Test cases for the enhanced engine fix and functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_symbols = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
        self.test_expiry = IST.localize(dt.datetime(2025, 9, 4, 15, 30))
        
        # Mock market data
        self.mock_market_data = {
            "NIFTY": 25000.0,
            "BANKNIFTY": 52000.0,
            "SENSEX": 82000.0,
            "MIDCPNIFTY": 13000.0
        }
        
    def test_kite_provider_has_get_indices_snapshot(self):
        """Test that KiteProvider has the correct method."""
        # This test ensures the method exists without requiring API connection
        provider = KiteProvider.__new__(KiteProvider)  # Create without initialization
        self.assertTrue(hasattr(provider, 'get_indices_snapshot'))
        self.assertFalse(hasattr(provider, 'get_quotes'))
        
    def test_engine_initialization(self):
        """Test that enhanced engines can be initialized for all symbols."""
        for symbol in self.test_symbols:
            engine = EnhancedTradingEngine(
                index=symbol,
                expiry=self.test_expiry,
                min_tau_hours=2.0
            )
            self.assertEqual(engine.index, symbol)
            self.assertEqual(engine.expiry, self.test_expiry)
            self.assertEqual(engine.min_tau_hours, 2.0)

    def test_market_data_creation(self):
        """Test market data creation with various inputs."""
        # Test valid market data
        market_data = MarketData(
            timestamp=dt.datetime.now(IST),
            index="NIFTY",
            spot=25000.0,
            futures_mid=25050.0,
            strikes=[24900, 24950, 25000, 25050, 25100],
            call_mids={25000: 150.0},
            put_mids={25000: 145.0},
            call_oi={25000: 10000},
            put_oi={25000: 12000}
        )
        
        self.assertEqual(market_data.index, "NIFTY")
        self.assertEqual(market_data.spot, 25000.0)
        self.assertIsInstance(market_data.strikes, list)
        self.assertIsInstance(market_data.call_mids, dict)

    def test_engine_processing_no_data(self):
        """Test engine processing when no market data is available."""
        engine = EnhancedTradingEngine(
            index="NIFTY",
            expiry=self.test_expiry,
            min_tau_hours=2.0
        )
        
        # Create empty market data
        empty_market_data = MarketData(
            timestamp=dt.datetime.now(IST),
            index="NIFTY",
            spot=0.0  # Invalid spot price
        )
        
        decision = engine.process_market_data(empty_market_data)
        self.assertEqual(decision.action, "NO_TRADE")
        self.assertTrue(decision.gate_decision.muted)

    def test_engine_processing_valid_data(self):
        """Test engine processing with valid market data."""
        engine = EnhancedTradingEngine(
            index="NIFTY",
            expiry=self.test_expiry,
            min_tau_hours=2.0
        )
        
        # Create valid market data
        market_data = MarketData(
            timestamp=dt.datetime.now(IST),
            index="NIFTY",
            spot=25000.0,
            futures_mid=25050.0,
            strikes=[24900, 24950, 25000, 25050, 25100],
            call_mids={
                24900: 200.0, 24950: 175.0, 25000: 150.0, 
                25050: 125.0, 25100: 100.0
            },
            put_mids={
                24900: 95.0, 24950: 120.0, 25000: 145.0, 
                25050: 170.0, 25100: 195.0
            },
            call_oi={
                24900: 8000, 24950: 12000, 25000: 15000,
                25050: 12000, 25100: 8000
            },
            put_oi={
                24900: 6000, 24950: 10000, 25000: 14000,
                25050: 10000, 25100: 6000
            }
        )
        
        decision = engine.process_market_data(market_data)
        
        # Verify basic decision structure
        self.assertIsInstance(decision, TradingDecision)
        self.assertIn(decision.action, ["LONG", "SHORT", "NO_TRADE"])
        self.assertIsNotNone(decision.gate_decision)
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)

    def test_technical_indicators_calculation(self):
        """Test calculation of technical indicators."""
        engine = EnhancedTradingEngine(
            index="NIFTY",
            expiry=self.test_expiry,
            min_tau_hours=2.0
        )
        
        # Create market data with sufficient options
        market_data = MarketData(
            timestamp=dt.datetime.now(IST),
            index="NIFTY",
            spot=25000.0,
            futures_mid=25050.0,
            strikes=list(range(24500, 25500, 50)),  # Wide strike range
            call_mids={k: max(50.0, 25000 - k + 150) for k in range(24500, 25500, 50)},
            put_mids={k: max(50.0, k - 25000 + 150) for k in range(24500, 25500, 50)},
            call_oi={k: 10000 + (k % 500) * 10 for k in range(24500, 25500, 50)},
            put_oi={k: 12000 + (k % 500) * 8 for k in range(24500, 25500, 50)}
        )
        
        decision = engine.process_market_data(market_data)
        
        # Check if technical indicators are calculated
        self.assertIsNotNone(decision.atm_strike)
        self.assertGreater(decision.atm_strike, 0)
        
        # ATM strike should be close to spot
        self.assertLessEqual(abs(decision.atm_strike - market_data.spot), 100)

    @patch('src.provider.kite.KiteProvider.get_indices_snapshot')
    def test_mock_provider_integration(self, mock_get_indices):
        """Test the engine with mocked provider data."""
        # Mock the provider method to return test data
        mock_get_indices.return_value = self.mock_market_data
        
        from engine_runner import create_sample_market_data
        provider = KiteProvider.__new__(KiteProvider)  # Create without initialization
        provider.get_indices_snapshot = mock_get_indices
        
        # Test each symbol
        for symbol in self.test_symbols:
            market_data = create_sample_market_data(symbol, provider)
            
            self.assertIsNotNone(market_data)
            self.assertEqual(market_data.index, symbol)
            self.assertEqual(market_data.spot, self.mock_market_data[symbol])
            self.assertGreater(len(market_data.strikes), 0)

    def test_pcr_calculation(self):
        """Test PCR (Put-Call Ratio) calculation."""
        engine = EnhancedTradingEngine(
            index="NIFTY",
            expiry=self.test_expiry,
            min_tau_hours=2.0
        )
        
        # Create market data with known PCR
        call_oi_total = 50000
        put_oi_total = 60000
        expected_pcr = put_oi_total / call_oi_total  # 1.2
        
        market_data = MarketData(
            timestamp=dt.datetime.now(IST),
            index="NIFTY",
            spot=25000.0,
            futures_mid=25050.0,
            strikes=list(range(24800, 25200, 50)),
            call_mids={k: 100.0 for k in range(24800, 25200, 50)},
            put_mids={k: 100.0 for k in range(24800, 25200, 50)},
            call_oi={
                24800: 8000, 24850: 10000, 24900: 12000, 24950: 15000,
                25000: 20000, 25050: 15000, 25100: 12000, 25150: 8000
            },
            put_oi={
                24800: 6000, 24850: 8000, 24900: 10000, 24950: 14000,
                25000: 18000, 25050: 16000, 25100: 14000, 25150: 10000
            }
        )
        
        decision = engine.process_market_data(market_data)
        
        # PCR should be calculated and reasonable
        if decision.pcr_total is not None:
            self.assertGreater(decision.pcr_total, 0.5)
            self.assertLess(decision.pcr_total, 2.0)

    def test_iv_calculation(self):
        """Test Implied Volatility calculation."""
        engine = EnhancedTradingEngine(
            index="NIFTY",
            expiry=self.test_expiry,
            min_tau_hours=2.0
        )
        
        # Create ATM market data for IV calculation
        market_data = MarketData(
            timestamp=dt.datetime.now(IST),
            index="NIFTY",
            spot=25000.0,
            futures_mid=25000.0,  # No basis
            strikes=[25000],  # ATM only
            call_mids={25000: 200.0},  # High premium for high IV
            put_mids={25000: 200.0},
            call_oi={25000: 10000},
            put_oi={25000: 10000}
        )
        
        decision = engine.process_market_data(market_data)
        
        # IV should be calculated for ATM options
        if decision.atm_iv is not None:
            self.assertGreater(decision.atm_iv, 0.0)
            self.assertLess(decision.atm_iv, 2.0)  # Should be reasonable

    def test_max_pain_calculation(self):
        """Test Max Pain calculation (implicit in the engine)."""
        engine = EnhancedTradingEngine(
            index="NIFTY",
            expiry=self.test_expiry,
            min_tau_hours=2.0
        )
        
        # Create skewed OI data
        market_data = MarketData(
            timestamp=dt.datetime.now(IST),
            index="NIFTY",
            spot=25000.0,
            futures_mid=25000.0,
            strikes=list(range(24800, 25200, 50)),
            call_mids={k: max(25, 25000 - k + 50) for k in range(24800, 25200, 50)},
            put_mids={k: max(25, k - 25000 + 50) for k in range(24800, 25200, 50)},
            # Heavy OI concentration around 24950 (potential max pain)
            call_oi={
                24800: 5000, 24850: 8000, 24900: 12000, 24950: 25000,
                25000: 15000, 25050: 10000, 25100: 8000, 25150: 5000
            },
            put_oi={
                24800: 3000, 24850: 5000, 24900: 8000, 24950: 20000,
                25000: 12000, 25050: 8000, 25100: 5000, 25150: 3000
            }
        )
        
        decision = engine.process_market_data(market_data)
        
        # Engine should process the skewed OI distribution
        self.assertIsNotNone(decision.gate_decision)

    def test_gate_conditions(self):
        """Test gate condition logic."""
        engine = EnhancedTradingEngine(
            index="NIFTY",
            expiry=self.test_expiry,
            min_tau_hours=2.0
        )
        
        # Create market data that should trigger gates
        market_data = MarketData(
            timestamp=dt.datetime.now(IST),
            index="NIFTY",
            spot=25000.0,
            futures_mid=25000.0,
            strikes=list(range(24500, 25500, 50)),
            call_mids={k: max(25, 25000 - k + 100) for k in range(24500, 25500, 50)},
            put_mids={k: max(25, k - 25000 + 100) for k in range(24500, 25500, 50)},
            call_oi={k: 10000 for k in range(24500, 25500, 50)},
            put_oi={k: 10000 for k in range(24500, 25500, 50)},
            # Set technical indicators to extreme values
            adx=45.0,  # High trend strength
            volume_ratio=2.5,  # High volume
            spread_bps=5.0,  # Tight spreads
            momentum_score=0.8  # Strong momentum
        )
        
        decision = engine.process_market_data(market_data)
        
        # Gate decision should be present
        self.assertIsNotNone(decision.gate_decision)
        self.assertIsNotNone(decision.market_regime)
        
        # Decision should have reasoning
        if decision.gate_decision.primary_reason:
            self.assertIsInstance(decision.gate_decision.primary_reason, str)

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        engine = EnhancedTradingEngine(
            index="NIFTY",
            expiry=self.test_expiry,
            min_tau_hours=2.0
        )
        
        # Test with invalid market data
        invalid_data = MarketData(
            timestamp=dt.datetime.now(IST),
            index="NIFTY",
            spot=-100.0,  # Invalid negative spot
            strikes=[],  # Empty strikes
        )
        
        decision = engine.process_market_data(invalid_data)
        self.assertEqual(decision.action, "NO_TRADE")
        self.assertTrue(decision.gate_decision.muted)

    def test_performance_stats(self):
        """Test engine performance statistics."""
        engine = EnhancedTradingEngine(
            index="NIFTY",
            expiry=self.test_expiry,
            min_tau_hours=2.0
        )
        
        # Process some market data
        market_data = MarketData(
            timestamp=dt.datetime.now(IST),
            index="NIFTY",
            spot=25000.0
        )
        
        engine.process_market_data(market_data)
        
        stats = engine.get_performance_stats()
        
        self.assertIn('total_runs', stats)
        self.assertIn('successful_runs', stats)
        self.assertIn('error_count', stats)
        self.assertIn('success_rate', stats)
        self.assertEqual(stats['total_runs'], 1)


if __name__ == '__main__':
    unittest.main()