"""Integration test for the complete enhanced trading engine."""

import pytest
import datetime as dt
import pytz
from pathlib import Path

from src.engine_enhanced import EnhancedTradingEngine, MarketData, TradingDecision


class TestEngineIntegration:
    """Test complete engine integration."""
    
    @pytest.fixture
    def ist_timezone(self):
        """IST timezone fixture."""
        return pytz.timezone("Asia/Kolkata")
    
    @pytest.fixture
    def sample_market_data(self, ist_timezone):
        """Sample market data for testing."""
        timestamp = ist_timezone.localize(dt.datetime(2024, 1, 11, 14, 30))
        
        # NIFTY market data
        strikes = [18900, 18950, 19000, 19050, 19100, 19150, 19200]
        
        return MarketData(
            timestamp=timestamp,
            index="NIFTY",
            spot=19025.0,
            futures_mid=19035.0,
            strikes=strikes,
            call_mids={
                18900: 130.0, 18950: 85.0, 19000: 50.0, 19050: 25.0,
                19100: 12.0, 19150: 6.0, 19200: 3.0
            },
            put_mids={
                18900: 5.0, 18950: 10.0, 19000: 20.0, 19050: 40.0,
                19100: 75.0, 19150: 120.0, 19200: 175.0
            },
            call_oi={
                18900: 1000, 18950: 1500, 19000: 2500, 19050: 3000,
                19100: 2000, 19150: 1200, 19200: 800
            },
            put_oi={
                18900: 800, 18950: 1200, 19000: 2000, 19050: 2800,
                19100: 3500, 19150: 2500, 19200: 1500
            },
            call_volumes={
                18900: 50, 18950: 80, 19000: 150, 19050: 200,
                19100: 120, 19150: 60, 19200: 30
            },
            put_volumes={
                18900: 30, 18950: 60, 19000: 120, 19050: 180,
                19100: 220, 19150: 150, 19200: 80
            },
            adx=25.0,
            volume_ratio=1.5,
            spread_bps=8.0,
            momentum_score=0.6
        )
    
    @pytest.fixture
    def engine(self, ist_timezone):
        """Enhanced trading engine instance."""
        expiry = ist_timezone.localize(dt.datetime(2024, 1, 11, 15, 29, 59))
        return EnhancedTradingEngine(
            index="NIFTY",
            expiry=expiry,
            min_tau_hours=1.0,
            risk_free_rate=0.066,
            dividend_yield=0.01
        )
    
    def test_complete_engine_flow(self, engine, sample_market_data):
        """Test complete engine processing flow."""
        
        # Process market data with bullish signals
        decision = engine.process_market_data(
            market_data=sample_market_data,
            trend_score=0.7,  # Bullish trend
            orb_signal="LONG",
            orb_strength=0.8,
            orb_breakout_size=25.0
        )
        
        # Validate decision structure
        assert isinstance(decision, TradingDecision)
        assert decision.action in ["LONG", "SHORT", "NO_TRADE"]
        assert 0.0 <= decision.confidence <= 1.0
        assert 0.0 <= decision.size_factor <= 1.0
        assert decision.tau_hours > 0
        assert decision.processing_time_ms > 0
        
        # Validate metrics computation
        assert decision.forward > 0
        assert decision.atm_strike in sample_market_data.strikes
        
        # For good quality data and signals, should not be muted
        if decision.action != "NO_TRADE":
            assert decision.gate_decision is not None
            assert not decision.gate_decision.muted
            assert decision.direction in ["LONG", "SHORT"]
    
    def test_engine_performance_tracking(self, engine, sample_market_data):
        """Test engine performance tracking."""
        
        # Process multiple times
        for i in range(5):
            decision = engine.process_market_data(
                sample_market_data,
                trend_score=0.5,
                orb_signal="LONG" if i % 2 == 0 else "SHORT",
                orb_strength=0.7
            )
        
        # Check performance stats
        stats = engine.get_performance_stats()
        assert stats["total_runs"] == 5
        assert stats["successful_runs"] <= stats["total_runs"]
        assert stats["success_rate"] <= 1.0
        assert stats["last_processing_time_ms"] > 0
    
    def test_explain_json_generation(self, engine, sample_market_data):
        """Test comprehensive explain JSON generation."""
        
        decision = engine.process_market_data(
            sample_market_data,
            trend_score=0.6,
            orb_signal="LONG",
            orb_strength=0.8
        )
        
        # Create signals for explain
        from src.strategy.enhanced_gates import MultiFactorSignal
        signals = MultiFactorSignal(
            orb_signal="LONG",
            orb_strength=0.8,
            volume_signal="LONG",
            volume_strength=0.7
        )
        
        explain_json = engine.emit_explain_json(decision, sample_market_data, signals)
        
        # Validate JSON structure
        import json
        explain_data = json.loads(explain_json)
        
        assert "metadata" in explain_data
        assert "market_data" in explain_data
        assert "metrics" in explain_data
        assert "regime" in explain_data
        assert "signals" in explain_data
        assert "gates" in explain_data
        assert "decision" in explain_data
        assert "data_quality" in explain_data
        assert "insights" in explain_data
        
        # Validate specific fields
        assert explain_data["metadata"]["index"] == "NIFTY"
        assert explain_data["decision"]["action"] in ["LONG", "SHORT", "NO_TRADE"]
        assert "trade_viability" in explain_data["insights"]
    
    def test_edge_case_handling(self, engine, ist_timezone):
        """Test handling of edge cases."""
        
        # Empty market data
        empty_data = MarketData(
            timestamp=ist_timezone.localize(dt.datetime(2024, 1, 11, 14, 30)),
            index="NIFTY",
            spot=19000.0
        )
        
        decision = engine.process_market_data(empty_data)
        assert decision.action == "NO_TRADE"
        assert decision.gate_decision.muted
        
        # Very close to expiry
        near_expiry_data = MarketData(
            timestamp=ist_timezone.localize(dt.datetime(2024, 1, 11, 15, 29, 58)),
            index="NIFTY",
            spot=19000.0,
            strikes=[19000],
            call_mids={19000: 0.05},
            put_mids={19000: 0.05}
        )
        
        decision = engine.process_market_data(near_expiry_data)
        assert decision.action == "NO_TRADE"
        assert "expiry" in decision.gate_decision.primary_reason.lower()
    
    def test_data_quality_assessment(self, engine, sample_market_data):
        """Test data quality assessment."""
        
        # Modify data to create quality issues
        poor_quality_data = sample_market_data
        poor_quality_data.spread_bps = 100.0  # Very wide spreads
        poor_quality_data.volume_ratio = 0.2  # Very low volume
        poor_quality_data.strikes = [19000]  # Only one strike
        
        decision = engine.process_market_data(poor_quality_data)
        
        # Should reflect data quality issues
        assert decision.data_quality_score < 0.8
    
    def test_regime_adaptation(self, engine, sample_market_data):
        """Test adaptation to different market regimes."""
        
        # High volatility, trending regime
        high_vol_decision = engine.process_market_data(
            sample_market_data,
            trend_score=0.8  # Strong trend
        )
        
        # Flat, low volatility regime
        sample_market_data.adx = 8.0  # Low directional movement
        sample_market_data.volume_ratio = 0.6  # Low volume
        sample_market_data.spread_bps = 30.0  # Wide spreads
        
        flat_decision = engine.process_market_data(
            sample_market_data,
            trend_score=0.1  # Flat
        )
        
        # High vol regime should be more tradeable
        if high_vol_decision.action != "NO_TRADE" and flat_decision.action != "NO_TRADE":
            assert high_vol_decision.confidence >= flat_decision.confidence
    
    def test_processing_time_constraint(self, engine, sample_market_data):
        """Test that processing stays within time constraints."""
        
        decision = engine.process_market_data(sample_market_data)
        
        # Should be well under 100ms constraint
        assert decision.processing_time_ms < 100.0
        
        # Run multiple times to check consistency
        times = []
        for _ in range(10):
            decision = engine.process_market_data(sample_market_data)
            times.append(decision.processing_time_ms)
        
        # Average should be well under constraint
        avg_time = sum(times) / len(times)
        assert avg_time < 50.0  # Even more conservative
    
    def test_deterministic_behavior(self, engine, sample_market_data):
        """Test deterministic behavior with same inputs."""
        
        # Run same inputs multiple times
        decisions = []
        for _ in range(3):
            decision = engine.process_market_data(
                sample_market_data,
                trend_score=0.6,
                orb_signal="LONG",
                orb_strength=0.7
            )
            decisions.append(decision)
        
        # Should get same decision (allowing for floating point precision)
        first_decision = decisions[0]
        for decision in decisions[1:]:
            assert decision.action == first_decision.action
            assert decision.direction == first_decision.direction
            assert abs(decision.confidence - first_decision.confidence) < 1e-6
            assert abs(decision.size_factor - first_decision.size_factor) < 1e-6
    
    def test_iv_history_management(self, engine, sample_market_data):
        """Test IV history management."""
        
        # Process data multiple times to build history
        for i in range(10):
            # Vary the IV slightly
            for strike in sample_market_data.call_mids:
                sample_market_data.call_mids[strike] *= (1.0 + i * 0.01)
                sample_market_data.put_mids[strike] *= (1.0 + i * 0.01)
            
            decision = engine.process_market_data(sample_market_data)
        
        # Should have built IV history
        assert len(engine.iv_history) > 0
        
        # IV percentile should be calculated after some history
        if len(engine.iv_history) >= 5:
            decision = engine.process_market_data(sample_market_data)
            if decision.iv_percentile is not None:
                assert 0.0 <= decision.iv_percentile <= 100.0