"""Enhanced gating system with multi-factor alignment and regime detection.

This module provides sophisticated gating logic for Indian options trading:
- Multi-factor signal alignment (ORB+volume+dOI/dt+IV crush)
- Enhanced regime detection (trend/vol/liquidity)
- Override mechanisms for strong multi-factor signals
- Deterministic decision table with clear reasoning
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple
from collections import deque
import statistics

from ..metrics.core import GateDecision


@dataclass(frozen=True)
class EnhancedRegime:
    """Comprehensive market regime classification."""
    
    trend: Literal["STRONG_UP", "WEAK_UP", "FLAT", "WEAK_DOWN", "STRONG_DOWN"]
    volatility: Literal["HIGH", "ELEVATED", "NORMAL", "LOW"]
    liquidity: Literal["EXCELLENT", "GOOD", "FAIR", "POOR"]
    momentum: Literal["ACCELERATING", "STEADY", "DECELERATING"]
    
    def is_trending(self) -> bool:
        """Check if regime shows clear directional bias."""
        return self.trend in ["STRONG_UP", "STRONG_DOWN"]
    
    def is_tradeable(self) -> bool:
        """Check if regime supports active trading."""
        # Made less restrictive - allow FAIR liquidity and LOW volatility in some cases
        liquidity_ok = self.liquidity in ["EXCELLENT", "GOOD", "FAIR"]
        volatility_ok = self.volatility != "LOW" or self.is_trending()  # Allow low vol if trending
        return liquidity_ok and volatility_ok


@dataclass
class MultiFactorSignal:
    """Container for multi-factor signal alignment."""
    
    orb_signal: Optional[Literal["LONG", "SHORT"]] = None
    volume_signal: Optional[Literal["LONG", "SHORT"]] = None  
    oi_flow_signal: Optional[Literal["LONG", "SHORT"]] = None
    iv_crush_signal: Optional[Literal["LONG", "SHORT"]] = None
    price_action_signal: Optional[Literal["LONG", "SHORT"]] = None
    
    # Signal strengths (0.0 to 1.0)
    orb_strength: float = 0.0
    volume_strength: float = 0.0
    oi_flow_strength: float = 0.0
    iv_crush_strength: float = 0.0
    price_action_strength: float = 0.0
    
    # Supporting metrics
    volume_ratio: float = 1.0  # Current volume / avg volume
    oi_delta_rate: float = 0.0  # Rate of OI change
    iv_percentile: float = 50.0  # IV percentile
    orb_breakout_size: float = 0.0  # Size of ORB breakout
    
    def get_aligned_signals(self) -> List[str]:
        """Return list of signals that agree on direction."""
        signals = {}
        
        # Lowered thresholds from 0.3 to 0.2 to allow more signals to qualify
        if self.orb_signal and self.orb_strength > 0.2:
            signals["ORB"] = self.orb_signal
        if self.volume_signal and self.volume_strength > 0.2:
            signals["VOLUME"] = self.volume_signal
        if self.oi_flow_signal and self.oi_flow_strength > 0.2:
            signals["OI_FLOW"] = self.oi_flow_signal
        if self.iv_crush_signal and self.iv_crush_strength > 0.2:
            signals["IV_CRUSH"] = self.iv_crush_signal
        if self.price_action_signal and self.price_action_strength > 0.2:
            signals["PRICE_ACTION"] = self.price_action_signal
        
        # Find consensus direction
        long_signals = [name for name, direction in signals.items() if direction == "LONG"]
        short_signals = [name for name, direction in signals.items() if direction == "SHORT"]
        
        if len(long_signals) >= len(short_signals):
            return long_signals
        else:
            return short_signals
            
    def get_all_active_signals(self) -> List[str]:
        """Return all signals above strength threshold, regardless of direction."""
        signals = []
        
        # Lowered thresholds from 0.3 to 0.2 to allow more signals to qualify
        if self.orb_signal and self.orb_strength > 0.2:
            signals.append("ORB")
        if self.volume_signal and self.volume_strength > 0.2:
            signals.append("VOLUME")
        if self.oi_flow_signal and self.oi_flow_strength > 0.2:
            signals.append("OI_FLOW")
        if self.iv_crush_signal and self.iv_crush_strength > 0.2:
            signals.append("IV_CRUSH")
        if self.price_action_signal and self.price_action_strength > 0.2:
            signals.append("PRICE_ACTION")
        
        return signals
    
    def get_consensus_direction(self) -> Optional[Literal["LONG", "SHORT"]]:
        """Get consensus direction from aligned signals."""
        # Get all signals above threshold and check for ties
        signals = {}
        
        if self.orb_signal and self.orb_strength > 0.3:
            signals["ORB"] = self.orb_signal
        if self.volume_signal and self.volume_strength > 0.3:
            signals["VOLUME"] = self.volume_signal
        if self.oi_flow_signal and self.oi_flow_strength > 0.3:
            signals["OI_FLOW"] = self.oi_flow_signal
        if self.iv_crush_signal and self.iv_crush_strength > 0.3:
            signals["IV_CRUSH"] = self.iv_crush_signal
        if self.price_action_signal and self.price_action_strength > 0.3:
            signals["PRICE_ACTION"] = self.price_action_signal
        
        if not signals:
            return None
        
        # Count directions
        long_count = sum(1 for direction in signals.values() if direction == "LONG")
        short_count = sum(1 for direction in signals.values() if direction == "SHORT")
        
        if long_count > short_count:
            return "LONG"
        elif short_count > long_count:
            return "SHORT"
        else:
            return None  # Tie
    
    def get_alignment_strength(self) -> float:
        """Calculate overall alignment strength (0.0 to 1.0)."""
        aligned = self.get_aligned_signals()
        if len(aligned) < 2:
            return 0.0
        
        # Weight by number of aligned signals and their individual strengths
        total_strength = 0.0
        for signal_name in aligned:
            if signal_name == "ORB":
                total_strength += self.orb_strength
            elif signal_name == "VOLUME":
                total_strength += self.volume_strength
            elif signal_name == "OI_FLOW":
                total_strength += self.oi_flow_strength
            elif signal_name == "IV_CRUSH":
                total_strength += self.iv_crush_strength
            elif signal_name == "PRICE_ACTION":
                total_strength += self.price_action_strength
        
        return min(1.0, total_strength / max(1, len(aligned)))


def detect_enhanced_regime(
    trend_score: float,
    adx: float,
    iv_percentile: float,
    spread_bps: float,
    volume_ratio: float,
    momentum_score: float
) -> EnhancedRegime:
    """Enhanced regime detection with multiple factors."""
    
    # Trend classification with ADX confirmation
    if trend_score > 0.7 and adx > 25:
        trend = "STRONG_UP"
    elif trend_score > 0.3 and adx > 15:
        trend = "WEAK_UP"
    elif trend_score < -0.7 and adx > 25:
        trend = "STRONG_DOWN"
    elif trend_score < -0.3 and adx > 15:
        trend = "WEAK_DOWN"
    else:
        trend = "FLAT"
    
    # Volatility regime based on IV percentile
    iv_percentile = iv_percentile or 50.0  # Default to 50 if None
    if iv_percentile > 80:
        volatility = "HIGH"
    elif iv_percentile > 60:
        volatility = "ELEVATED"
    elif iv_percentile > 20:
        volatility = "NORMAL"
    else:
        volatility = "LOW"
    
    # Liquidity assessment
    if spread_bps <= 5 and volume_ratio >= 1.5:
        liquidity = "EXCELLENT"
    elif spread_bps <= 10 and volume_ratio >= 1.0:
        liquidity = "GOOD"
    elif spread_bps <= 20 and volume_ratio >= 0.7:
        liquidity = "FAIR"
    else:
        liquidity = "POOR"
    
    # Momentum assessment
    if abs(momentum_score) > 0.8:
        momentum = "ACCELERATING"
    elif abs(momentum_score) > 0.3:
        momentum = "STEADY"
    else:
        momentum = "DECELERATING"
    
    return EnhancedRegime(trend, volatility, liquidity, momentum)


@dataclass
class EnhancedGateDecision:
    """Enhanced gate decision with detailed reasoning."""
    
    muted: bool
    direction: Optional[Literal["LONG", "SHORT"]]
    size_factor: float = 1.0
    confidence: float = 0.0
    override_triggered: bool = False
    primary_reason: str = ""
    supporting_factors: List[str] = field(default_factory=list)
    risk_adjustments: Dict[str, float] = field(default_factory=dict)
    
    def as_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "muted": self.muted,
            "direction": self.direction,
            "size_factor": self.size_factor,
            "confidence": self.confidence,
            "override_triggered": self.override_triggered,
            "primary_reason": self.primary_reason,
            "supporting_factors": list(self.supporting_factors),
            "risk_adjustments": dict(self.risk_adjustments)
        }


def apply_enhanced_gates(
    signals: MultiFactorSignal,
    regime: EnhancedRegime,
    tau_hours: float,
    confirming_bars: int = 0,
    min_confirm_bars: int = 2,
    spike_classification: str = "UNKNOWN"
) -> EnhancedGateDecision:
    """Apply enhanced gating logic with multi-factor override capability.
    
    Parameters
    ----------
    signals: MultiFactorSignal
        Multi-factor signal alignment data
    regime: EnhancedRegime
        Current market regime
    tau_hours: float
        Hours to expiry
    confirming_bars: int
        Number of consecutive confirming bars
    min_confirm_bars: int
        Minimum bars required without override
    spike_classification: str
        Spike type: MICROSTRUCTURE, NEWS_SHOCK, VALID_IMPULSE, UNKNOWN
    """
    
    aligned_signals = signals.get_aligned_signals()
    consensus_direction = signals.get_consensus_direction()
    alignment_strength = signals.get_alignment_strength()
    
    decision = EnhancedGateDecision(
        muted=True,
        direction=consensus_direction,
        confidence=alignment_strength
    )
    
    # Rule 1: Block microstructure spikes
    if spike_classification == "MICROSTRUCTURE":
        decision.primary_reason = "microstructure_spike_detected"
        decision.supporting_factors.append("high_frequency_noise")
        return decision
    
    # Rule 2: Block if regime is not tradeable
    if not regime.is_tradeable():
        decision.primary_reason = "untradeable_regime"
        decision.supporting_factors.extend([
            f"liquidity_{regime.liquidity.lower()}",
            f"volatility_{regime.volatility.lower()}"
        ])
        return decision
    
    # Rule 3: Multi-factor override (reduce threshold to allow more valid trades)
    if len(aligned_signals) >= 2 and alignment_strength >= 0.4:  # Lowered from 3 signals and 0.6 strength
        decision.muted = False
        decision.override_triggered = True
        decision.primary_reason = "multi_factor_alignment_override"
        decision.supporting_factors = aligned_signals.copy()
        
        # Size adjustments for overrides
        if spike_classification == "NEWS_SHOCK":
            decision.size_factor = 0.6  # Reduce size for news-driven moves
            decision.risk_adjustments["news_shock"] = -0.4
        elif regime.volatility == "HIGH":
            decision.size_factor = 0.8  # Reduce size in high vol
            decision.risk_adjustments["high_vol"] = -0.2
        elif tau_hours < 0.4:  # Near expiry - reduce size even for overrides
            if tau_hours < 0.15:  # Critical expiry
                decision.size_factor = 0.5
                decision.risk_adjustments["critical_expiry"] = -0.5
            else:  # Near expiry
                decision.size_factor = 0.7
                decision.risk_adjustments["near_expiry"] = -0.3
        else:
            decision.size_factor = 1.0
        
        # Boost confidence for strong alignment
        decision.confidence = min(1.0, alignment_strength + 0.2)
        
        return decision
    
    # Rule 4: Strong single-factor signals with regime support (reduce thresholds)
    if alignment_strength >= 0.6 and len(aligned_signals) >= 1:  # Lowered from 0.8 and 2 signals
        if regime.is_trending() and consensus_direction:
            # Check if direction aligns with regime trend
            trend_aligned = (
                (consensus_direction == "LONG" and regime.trend in ["STRONG_UP", "WEAK_UP"]) or
                (consensus_direction == "SHORT" and regime.trend in ["STRONG_DOWN", "WEAK_DOWN"])
            )
            
            if trend_aligned:
                decision.muted = False
                decision.primary_reason = "strong_signals_with_trend_alignment"
                decision.supporting_factors = aligned_signals.copy()
                decision.size_factor = 0.9
                decision.confidence = alignment_strength
                return decision
    
    # Rule 5: Standard confirmation requirement
    if confirming_bars >= min_confirm_bars and consensus_direction:
        # Additional checks for expiry proximity
        if tau_hours < 0.4:  # Less than 24 minutes to expiry
            # Require stronger signals near expiry - need 3+ signals OR very high strength
            if (alignment_strength >= 0.8 and len(aligned_signals) >= 3) or alignment_strength >= 0.9:
                decision.muted = False
                decision.primary_reason = "confirmed_signals_near_expiry"
                # More restrictive size reduction for very close expiry
                if tau_hours < 0.15:  # Less than 9 minutes
                    decision.size_factor = 0.5
                    decision.risk_adjustments["critical_expiry"] = -0.5
                else:  # 9-24 minutes
                    decision.size_factor = 0.7
                    decision.risk_adjustments["near_expiry"] = -0.3
            else:
                decision.primary_reason = "insufficient_strength_near_expiry"
                decision.supporting_factors.append(f"tau_hours_{tau_hours:.1f}")
                decision.supporting_factors.append(f"need_3plus_signals_or_90pct_strength")
        else:
            decision.muted = False
            decision.primary_reason = "standard_confirmation"
            
            # Size adjustments based on signal count and strength
            if len(aligned_signals) == 2 and alignment_strength >= 0.8:
                # Strong two-factor signals - reduce size slightly
                decision.size_factor = 0.8
                decision.risk_adjustments["two_factor_strong"] = -0.2
            else:
                decision.size_factor = 1.0
        
        decision.supporting_factors.extend(aligned_signals)
        decision.confidence = alignment_strength
        return decision
    
    # Rule 6: Insufficient confirmation
    decision.primary_reason = "insufficient_confirmation"
    decision.supporting_factors.extend([
        f"confirming_bars_{confirming_bars}_lt_{min_confirm_bars}",
        f"aligned_signals_{len(aligned_signals)}",
        f"alignment_strength_{alignment_strength:.2f}"
    ])
    
    return decision


def create_decision_table() -> Dict:
    """Create comprehensive decision table for trade logic."""
    return {
        "gates": {
            "microstructure_spike": {
                "action": "BLOCK",
                "reason": "High frequency noise detected"
            },
            "poor_liquidity": {
                "action": "BLOCK", 
                "reason": "Insufficient market liquidity"
            },
            "low_volatility": {
                "action": "REDUCE_SIZE",
                "size_factor": 0.5,
                "reason": "Low volatility environment"
            },
            "multi_factor_override": {
                "action": "ALLOW",
                "size_factor": 1.0,
                "reason": "Strong multi-factor signal alignment"
            },
            "near_expiry": {
                "action": "REDUCE_SIZE",
                "size_factor": 0.7,
                "reason": "Risk reduction near expiry"
            }
        },
        "regimes": {
            "STRONG_TREND": {
                "signals_required": 2,
                "min_strength": 0.6,
                "size_factor": 1.0
            },
            "WEAK_TREND": {
                "signals_required": 3,
                "min_strength": 0.7,
                "size_factor": 0.8
            },
            "FLAT": {
                "signals_required": 3,
                "min_strength": 0.8,
                "size_factor": 0.6
            }
        },
        "overrides": {
            "alignment_threshold": 3,  # Minimum aligned signals for override
            "strength_threshold": 0.6,  # Minimum alignment strength
            "confirmation_bypass": True  # Allow bypass of confirmation bars
        }
    }


__all__ = [
    "EnhancedRegime",
    "MultiFactorSignal", 
    "EnhancedGateDecision",
    "detect_enhanced_regime",
    "apply_enhanced_gates",
    "create_decision_table"
]