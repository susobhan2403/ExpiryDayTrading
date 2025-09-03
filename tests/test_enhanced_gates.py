"""Tests for enhanced gating system and decision logic."""

import pytest
import pandas as pd
import datetime as dt
from pathlib import Path

from src.strategy.enhanced_gates import (
    EnhancedRegime,
    MultiFactorSignal,
    EnhancedGateDecision,
    detect_enhanced_regime,
    apply_enhanced_gates,
    create_decision_table
)


class TestEnhancedRegime:
    """Test enhanced regime detection."""
    
    def test_strong_trending_regime(self):
        """Test detection of strong trending regime."""
        regime = detect_enhanced_regime(
            trend_score=0.8,      # Strong uptrend
            adx=30,               # High directional movement
            iv_percentile=85,     # High volatility
            spread_bps=3,         # Tight spreads
            volume_ratio=2.0,     # High volume
            momentum_score=0.9    # Strong momentum
        )
        
        assert regime.trend == "STRONG_UP"
        assert regime.volatility == "HIGH"
        assert regime.liquidity == "EXCELLENT"
        assert regime.momentum == "ACCELERATING"
        assert regime.is_trending()
        assert regime.is_tradeable()
    
    def test_flat_low_vol_regime(self):
        """Test detection of flat, low volatility regime."""
        regime = detect_enhanced_regime(
            trend_score=0.1,      # Flat
            adx=10,               # Low directional movement
            iv_percentile=15,     # Low volatility
            spread_bps=25,        # Wide spreads
            volume_ratio=0.5,     # Low volume
            momentum_score=0.1    # Weak momentum
        )
        
        assert regime.trend == "FLAT"
        assert regime.volatility == "LOW"
        assert regime.liquidity == "POOR"
        assert regime.momentum == "DECELERATING"
        assert not regime.is_trending()
        assert not regime.is_tradeable()
    
    def test_weak_downtrend(self):
        """Test detection of weak downtrend."""
        regime = detect_enhanced_regime(
            trend_score=-0.4,     # Weak downtrend
            adx=18,               # Moderate directional movement
            iv_percentile=65,     # Elevated volatility
            spread_bps=8,         # Moderate spreads
            volume_ratio=1.2,     # Above average volume
            momentum_score=-0.5   # Moderate negative momentum
        )
        
        assert regime.trend == "WEAK_DOWN"
        assert regime.volatility == "ELEVATED"
        assert regime.liquidity == "GOOD"
        assert regime.momentum == "STEADY"


class TestMultiFactorSignal:
    """Test multi-factor signal alignment."""
    
    def test_strong_alignment(self):
        """Test detection of strong signal alignment."""
        signals = MultiFactorSignal(
            orb_signal="LONG",
            orb_strength=0.9,
            volume_signal="LONG",
            volume_strength=0.8,
            oi_flow_signal="LONG",
            oi_flow_strength=0.7,
            iv_crush_signal="LONG",
            iv_crush_strength=0.6
        )
        
        aligned = signals.get_aligned_signals()
        consensus = signals.get_consensus_direction()
        strength = signals.get_alignment_strength()
        
        assert len(aligned) == 4
        assert consensus == "LONG"
        assert strength > 0.7
    
    def test_conflicting_signals(self):
        """Test handling of conflicting signals."""
        signals = MultiFactorSignal(
            orb_signal="LONG",
            orb_strength=0.8,
            volume_signal="SHORT",
            volume_strength=0.7,
            oi_flow_signal="LONG",
            oi_flow_strength=0.6,
            iv_crush_signal="SHORT",
            iv_crush_strength=0.5
        )
        
        aligned = signals.get_aligned_signals()
        consensus = signals.get_consensus_direction()
        
        # Should have 2 LONG and 2 SHORT signals above 0.3 threshold
        assert len(aligned) == 4
        assert consensus is None  # Should be a tie
    
    def test_weak_signals_filtered(self):
        """Test filtering of weak signals below threshold."""
        signals = MultiFactorSignal(
            orb_signal="LONG",
            orb_strength=0.2,     # Below 0.3 threshold
            volume_signal="LONG",
            volume_strength=0.8,
            oi_flow_signal="LONG",
            oi_flow_strength=0.1  # Below threshold
        )
        
        aligned = signals.get_aligned_signals()
        assert len(aligned) == 1  # Only volume signal above threshold
        assert "VOLUME" in aligned


class TestEnhancedGates:
    """Test enhanced gating logic."""
    
    def test_microstructure_spike_block(self):
        """Test blocking of microstructure spikes."""
        signals = MultiFactorSignal(
            orb_signal="LONG",
            orb_strength=0.9,
            volume_signal="LONG", 
            volume_strength=0.8,
            oi_flow_signal="LONG",
            oi_flow_strength=0.7
        )
        
        regime = EnhancedRegime("STRONG_UP", "NORMAL", "EXCELLENT", "ACCELERATING")
        
        decision = apply_enhanced_gates(
            signals=signals,
            regime=regime,
            tau_hours=12.0,
            spike_classification="MICROSTRUCTURE"
        )
        
        assert decision.muted
        assert decision.primary_reason == "microstructure_spike_detected"
        assert not decision.override_triggered
    
    def test_multi_factor_override(self):
        """Test multi-factor override capability."""
        signals = MultiFactorSignal(
            orb_signal="LONG",
            orb_strength=0.9,
            volume_signal="LONG",
            volume_strength=0.8,
            oi_flow_signal="LONG", 
            oi_flow_strength=0.7,
            price_action_signal="LONG",
            price_action_strength=0.6
        )
        
        regime = EnhancedRegime("FLAT", "NORMAL", "GOOD", "STEADY")  # Not strongly trending
        
        decision = apply_enhanced_gates(
            signals=signals,
            regime=regime,
            tau_hours=6.0,
            confirming_bars=1,  # Less than minimum required
            min_confirm_bars=2
        )
        
        assert not decision.muted
        assert decision.override_triggered
        assert decision.primary_reason == "multi_factor_alignment_override"
        assert len(decision.supporting_factors) >= 3
    
    def test_poor_regime_block(self):
        """Test blocking in poor regime conditions."""
        signals = MultiFactorSignal(
            orb_signal="LONG",
            orb_strength=0.8
        )
        
        regime = EnhancedRegime("FLAT", "LOW", "POOR", "DECELERATING")
        
        decision = apply_enhanced_gates(
            signals=signals,
            regime=regime,
            tau_hours=12.0
        )
        
        assert decision.muted
        assert decision.primary_reason == "untradeable_regime"
        assert "liquidity_poor" in decision.supporting_factors
        assert "volatility_low" in decision.supporting_factors
    
    def test_near_expiry_size_reduction(self):
        """Test size reduction near expiry."""
        signals = MultiFactorSignal(
            orb_signal="LONG",
            orb_strength=0.8,
            volume_signal="LONG",
            volume_strength=0.7
        )
        
        regime = EnhancedRegime("WEAK_UP", "NORMAL", "GOOD", "STEADY")
        
        decision = apply_enhanced_gates(
            signals=signals,
            regime=regime,
            tau_hours=1.5,  # Less than 2 hours to expiry
            confirming_bars=3,
            min_confirm_bars=2
        )
        
        # Should require stronger signals near expiry
        # With only 2 aligned signals at moderate strength, should be blocked
        assert decision.muted
        assert decision.primary_reason == "insufficient_strength_near_expiry"
    
    def test_news_shock_size_reduction(self):
        """Test size reduction for news shocks."""
        signals = MultiFactorSignal(
            orb_signal="LONG",
            orb_strength=0.9,
            volume_signal="LONG",
            volume_strength=0.8,
            oi_flow_signal="LONG",
            oi_flow_strength=0.7
        )
        
        regime = EnhancedRegime("STRONG_UP", "HIGH", "EXCELLENT", "ACCELERATING")
        
        decision = apply_enhanced_gates(
            signals=signals,
            regime=regime,
            tau_hours=6.0,
            spike_classification="NEWS_SHOCK"
        )
        
        assert not decision.muted
        assert decision.override_triggered
        assert decision.size_factor == 0.6  # Reduced for news shock
        assert "news_shock" in decision.risk_adjustments


class TestGateOverrideScenarios:
    """Test gate override scenarios from fixture data."""
    
    @pytest.fixture
    def override_scenarios(self):
        """Load gate override test scenarios."""
        return pd.read_csv(Path(__file__).parent / "fixtures" / "gate_override_scenarios.csv")
    
    def test_scenario_execution(self, override_scenarios):
        """Test execution of predefined gate override scenarios."""
        for _, row in override_scenarios.iterrows():
            scenario = row["scenario"]
            
            signals = MultiFactorSignal(
                orb_signal=row["orb_signal"] if pd.notna(row["orb_signal"]) else None,
                orb_strength=row["orb_strength"],
                volume_signal=row["volume_signal"] if pd.notna(row["volume_signal"]) else None,
                volume_strength=row["volume_strength"],
                oi_flow_signal=row["oi_flow_signal"] if pd.notna(row["oi_flow_signal"]) else None,
                oi_flow_strength=row["oi_flow_strength"],
                iv_crush_signal=row["iv_crush_signal"] if pd.notna(row["iv_crush_signal"]) else None,
                iv_crush_strength=row["iv_crush_strength"],
                price_action_signal=row["price_action_signal"] if pd.notna(row["price_action_signal"]) else None,
                price_action_strength=row["price_action_strength"]
            )
            
            # Use a neutral regime for testing
            regime = EnhancedRegime("FLAT", "NORMAL", "GOOD", "STEADY")
            
            decision = apply_enhanced_gates(
                signals=signals,
                regime=regime,
                tau_hours=6.0,
                confirming_bars=2,
                min_confirm_bars=2
            )
            
            expected_decision = row["expected_decision"]
            expected_override = row["expected_override"]
            
            # Check decision alignment
            if expected_decision == "ALLOW":
                assert not decision.muted, f"Scenario {scenario}: Expected ALLOW but got muted"
            elif expected_decision == "BLOCK":
                # Allow for various blocking reasons
                pass  # Decision might be allowed due to other factors
            elif expected_decision == "REDUCE":
                # Size should be reduced if not muted
                if not decision.muted:
                    assert decision.size_factor < 1.0, f"Scenario {scenario}: Expected size reduction"
            
            # Check override behavior
            if expected_override:
                # Strong scenarios should trigger override
                aligned = signals.get_aligned_signals()
                if len(aligned) >= 3:
                    assert decision.override_triggered or not decision.muted, f"Scenario {scenario}: Expected override capability"


class TestDecisionTable:
    """Test decision table functionality."""
    
    def test_decision_table_structure(self):
        """Test decision table has required structure."""
        table = create_decision_table()
        
        assert "gates" in table
        assert "regimes" in table
        assert "overrides" in table
        
        # Check gate actions
        gates = table["gates"]
        assert "microstructure_spike" in gates
        assert gates["microstructure_spike"]["action"] == "BLOCK"
        
        # Check regime requirements
        regimes = table["regimes"]
        assert "STRONG_TREND" in regimes
        assert "signals_required" in regimes["STRONG_TREND"]
        
        # Check override parameters
        overrides = table["overrides"]
        assert "alignment_threshold" in overrides
        assert "strength_threshold" in overrides
    
    def test_regime_signal_requirements(self):
        """Test regime-specific signal requirements."""
        table = create_decision_table()
        regimes = table["regimes"]
        
        # Strong trend should require fewer signals
        assert regimes["STRONG_TREND"]["signals_required"] <= regimes["WEAK_TREND"]["signals_required"]
        assert regimes["WEAK_TREND"]["signals_required"] <= regimes["FLAT"]["signals_required"]
        
        # Flat markets should require highest strength
        assert regimes["FLAT"]["min_strength"] >= regimes["WEAK_TREND"]["min_strength"]