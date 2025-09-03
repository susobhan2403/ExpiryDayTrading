"""Enhanced observability with comprehensive explain JSON for trade decisions.

This module provides detailed, structured explanations of every decision made
by the trading engine, including metrics, signals, gates, and final outcomes.
"""

from __future__ import annotations

import json
import datetime as dt
from typing import Any, Dict, List, Optional, Sequence, Union
from dataclasses import asdict

from ..strategy.enhanced_gates import EnhancedRegime, MultiFactorSignal, EnhancedGateDecision


def emit_comprehensive_explain(
    # Core identifiers
    index: str,
    expiry: dt.datetime,
    timestamp: dt.datetime,
    tau_hours: float,
    step: int,
    
    # Market data
    spot: float,
    forward: float,
    K_atm: float,
    strikes_analyzed: List[float],
    
    # Derived metrics
    iv_metrics: Dict[str, Any],
    pcr_metrics: Dict[str, Any],
    regime: EnhancedRegime,
    
    # Signals and alignment
    signals: MultiFactorSignal,
    aligned_signals: List[str],
    
    # Gating and decision
    gate_decision: EnhancedGateDecision,
    final_decision: str,
    
    # Data quality
    data_quality_flags: List[str],
    
    # Performance metrics
    processing_time_ms: float,
    
    # Optional diagnostics
    diagnostics: Optional[Dict[str, Any]] = None
    
) -> str:
    """Generate comprehensive explain JSON for a single trading decision.
    
    This function creates a complete audit trail of the decision-making process,
    suitable for debugging, backtesting analysis, and regulatory compliance.
    """
    
    # Core metadata
    explain_data = {
        "metadata": {
            "index": index,
            "expiry": expiry.isoformat() if expiry.tzinfo else expiry.replace(tzinfo=dt.timezone.utc).isoformat(),
            "timestamp": timestamp.isoformat() if timestamp.tzinfo else timestamp.replace(tzinfo=dt.timezone.utc).isoformat(),
            "tau_hours": round(tau_hours, 4),
            "strike_step": step,
            "processing_time_ms": round(processing_time_ms, 2)
        },
        
        # Market snapshot
        "market_data": {
            "spot_price": round(spot, 2),
            "forward_price": round(forward, 2),
            "atm_strike": K_atm,
            "strikes_analyzed": sorted(strikes_analyzed),
            "forward_premium": round((forward - spot) / spot * 100, 4) if spot > 0 else None
        },
        
        # Options metrics
        "metrics": {
            "implied_volatility": _format_iv_metrics(iv_metrics),
            "put_call_ratio": _format_pcr_metrics(pcr_metrics),
            "moneyness": round(K_atm / forward, 4) if forward > 0 else None
        },
        
        # Market regime
        "regime": {
            "trend": regime.trend,
            "volatility": regime.volatility,
            "liquidity": regime.liquidity,
            "momentum": regime.momentum,
            "is_trending": regime.is_trending(),
            "is_tradeable": regime.is_tradeable()
        },
        
        # Signal analysis
        "signals": _format_signal_analysis(signals, aligned_signals),
        
        # Gating logic
        "gates": gate_decision.as_dict(),
        
        # Final decision
        "decision": {
            "action": final_decision,
            "muted": gate_decision.muted,
            "direction": gate_decision.direction,
            "size_factor": round(gate_decision.size_factor, 3),
            "confidence": round(gate_decision.confidence, 3),
            "override_applied": gate_decision.override_triggered
        },
        
        # Data quality assessment
        "data_quality": {
            "flags": data_quality_flags,
            "score": _calculate_dq_score(data_quality_flags),
            "sufficient_for_trading": len(data_quality_flags) == 0
        }
    }
    
    # Add diagnostics if provided
    if diagnostics:
        explain_data["diagnostics"] = diagnostics
    
    # Add computed insights
    explain_data["insights"] = _generate_insights(explain_data)
    
    return json.dumps(explain_data, default=str, indent=2)


def _format_iv_metrics(iv_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Format IV metrics for explain output."""
    formatted = {}
    
    if "atm_iv" in iv_metrics:
        formatted["atm_iv"] = round(iv_metrics["atm_iv"], 4) if iv_metrics["atm_iv"] else None
    
    if "percentile" in iv_metrics:
        formatted["percentile"] = round(iv_metrics["percentile"], 1) if iv_metrics["percentile"] else None
    
    if "iv_rank" in iv_metrics:
        formatted["iv_rank"] = round(iv_metrics["iv_rank"], 1) if iv_metrics["iv_rank"] else None
    
    if "call_iv" in iv_metrics:
        formatted["call_iv"] = round(iv_metrics["call_iv"], 4) if iv_metrics["call_iv"] else None
    
    if "put_iv" in iv_metrics:
        formatted["put_iv"] = round(iv_metrics["put_iv"], 4) if iv_metrics["put_iv"] else None
    
    if "iv_spread" in iv_metrics:
        formatted["call_put_spread"] = round(iv_metrics["iv_spread"], 4) if iv_metrics["iv_spread"] else None
    
    # Classification
    if formatted.get("percentile"):
        if formatted["percentile"] > 80:
            formatted["iv_regime"] = "HIGH"
        elif formatted["percentile"] > 60:
            formatted["iv_regime"] = "ELEVATED"
        elif formatted["percentile"] > 40:
            formatted["iv_regime"] = "NORMAL"
        elif formatted["percentile"] > 20:
            formatted["iv_regime"] = "SUBDUED"
        else:
            formatted["iv_regime"] = "LOW"
    
    return formatted


def _format_pcr_metrics(pcr_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Format PCR metrics for explain output."""
    formatted = {}
    
    if "PCR_OI_total" in pcr_metrics:
        formatted["total"] = round(pcr_metrics["PCR_OI_total"], 3) if pcr_metrics["PCR_OI_total"] else None
    
    if "PCR_OI_band" in pcr_metrics:
        formatted["atm_band"] = round(pcr_metrics["PCR_OI_band"], 3) if pcr_metrics["PCR_OI_band"] else None
    
    if "band_lo" in pcr_metrics and "band_hi" in pcr_metrics:
        formatted["band_range"] = [pcr_metrics["band_lo"], pcr_metrics["band_hi"]]
    
    if "band_strikes_count" in pcr_metrics:
        formatted["band_strikes_count"] = pcr_metrics["band_strikes_count"]
    
    # OI distribution
    oi_data = {}
    for key in ["total_put_oi", "total_call_oi", "band_put_oi", "band_call_oi"]:
        if key in pcr_metrics:
            oi_data[key] = pcr_metrics[key]
    
    if oi_data:
        formatted["open_interest"] = oi_data
    
    # Classification
    if formatted.get("total"):
        pcr_total = formatted["total"]
        if pcr_total > 1.5:
            formatted["sentiment"] = "BEARISH"
        elif pcr_total > 1.2:
            formatted["sentiment"] = "SLIGHTLY_BEARISH"
        elif pcr_total > 0.8:
            formatted["sentiment"] = "NEUTRAL"
        elif pcr_total > 0.6:
            formatted["sentiment"] = "SLIGHTLY_BULLISH"
        else:
            formatted["sentiment"] = "BULLISH"
    
    return formatted


def _format_signal_analysis(signals: MultiFactorSignal, aligned_signals: List[str]) -> Dict[str, Any]:
    """Format signal analysis for explain output."""
    
    # Individual signals
    individual = {}
    signal_map = {
        "orb": ("orb_signal", "orb_strength"),
        "volume": ("volume_signal", "volume_strength"),
        "oi_flow": ("oi_flow_signal", "oi_flow_strength"),
        "iv_crush": ("iv_crush_signal", "iv_crush_strength"),
        "price_action": ("price_action_signal", "price_action_strength")
    }
    
    for signal_name, (signal_attr, strength_attr) in signal_map.items():
        signal_value = getattr(signals, signal_attr, None)
        strength_value = getattr(signals, strength_attr, 0.0)
        
        if signal_value or strength_value > 0:
            individual[signal_name] = {
                "direction": signal_value,
                "strength": round(strength_value, 3),
                "active": strength_value > 0.3
            }
    
    # Supporting metrics
    supporting = {
        "volume_ratio": round(signals.volume_ratio, 2),
        "oi_delta_rate": round(signals.oi_delta_rate, 4),
        "iv_percentile": round(signals.iv_percentile, 1),
        "orb_breakout_size": round(signals.orb_breakout_size, 2)
    }
    
    # Alignment analysis
    consensus_direction = signals.get_consensus_direction()
    alignment_strength = signals.get_alignment_strength()
    
    alignment = {
        "aligned_signals": aligned_signals,
        "consensus_direction": consensus_direction,
        "alignment_strength": round(alignment_strength, 3),
        "signal_count": len(aligned_signals),
        "agreement_level": _classify_agreement(len(aligned_signals), alignment_strength)
    }
    
    return {
        "individual": individual,
        "supporting_metrics": supporting,
        "alignment": alignment
    }


def _classify_agreement(signal_count: int, strength: float) -> str:
    """Classify the level of signal agreement."""
    if signal_count >= 4 and strength >= 0.8:
        return "VERY_STRONG"
    elif signal_count >= 3 and strength >= 0.7:
        return "STRONG"
    elif signal_count >= 2 and strength >= 0.6:
        return "MODERATE"
    elif signal_count >= 2 and strength >= 0.4:
        return "WEAK"
    else:
        return "INSUFFICIENT"


def _calculate_dq_score(flags: List[str]) -> float:
    """Calculate data quality score (0.0 to 1.0)."""
    if not flags:
        return 1.0
    
    # Weight different types of issues
    weights = {
        "missing_oi": 0.3,
        "wide_spreads": 0.2,
        "stale_data": 0.4,
        "insufficient_volume": 0.2,
        "missing_strikes": 0.3,
        "invalid_iv": 0.4,
        "expiry_too_close": 0.5
    }
    
    total_penalty = sum(weights.get(flag, 0.2) for flag in flags)
    return max(0.0, 1.0 - min(1.0, total_penalty))


def _generate_insights(explain_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate high-level insights from the explain data."""
    insights = {}
    
    # Market condition assessment
    regime = explain_data["regime"]
    signals = explain_data["signals"]
    decision = explain_data["decision"]
    
    # Overall market assessment
    if regime["is_trending"] and regime["is_tradeable"]:
        market_condition = "FAVORABLE"
    elif regime["is_tradeable"]:
        market_condition = "MIXED"
    else:
        market_condition = "CHALLENGING"
    
    insights["market_condition"] = market_condition
    
    # Signal quality
    alignment = signals["alignment"]
    if alignment["agreement_level"] in ["VERY_STRONG", "STRONG"]:
        signal_quality = "HIGH"
    elif alignment["agreement_level"] == "MODERATE":
        signal_quality = "MEDIUM"
    else:
        signal_quality = "LOW"
    
    insights["signal_quality"] = signal_quality
    
    # Decision confidence
    confidence = decision["confidence"]
    if confidence >= 0.8:
        decision_confidence = "HIGH"
    elif confidence >= 0.6:
        decision_confidence = "MEDIUM"
    elif confidence >= 0.4:
        decision_confidence = "LOW"
    else:
        decision_confidence = "VERY_LOW"
    
    insights["decision_confidence"] = decision_confidence
    
    # Risk level
    tau_hours = explain_data["metadata"]["tau_hours"]
    size_factor = decision["size_factor"]
    
    risk_factors = []
    if tau_hours < 2:
        risk_factors.append("near_expiry")
    if size_factor < 0.8:
        risk_factors.append("reduced_sizing")
    if explain_data["data_quality"]["score"] < 0.8:
        risk_factors.append("data_quality_issues")
    
    if len(risk_factors) >= 2:
        risk_level = "HIGH"
    elif len(risk_factors) == 1:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    insights["risk_level"] = risk_level
    insights["risk_factors"] = risk_factors
    
    # Trade viability
    if decision["muted"]:
        trade_viability = "BLOCKED"
    elif decision_confidence == "HIGH" and market_condition == "FAVORABLE":
        trade_viability = "EXCELLENT"
    elif decision_confidence in ["HIGH", "MEDIUM"] and market_condition in ["FAVORABLE", "MIXED"]:
        trade_viability = "GOOD"
    elif decision_confidence == "MEDIUM":
        trade_viability = "FAIR"
    else:
        trade_viability = "POOR"
    
    insights["trade_viability"] = trade_viability
    
    return insights


def emit_simple_explain(
    index: str,
    expiry: dt.datetime,
    tau_hours: float,
    step: int,
    forward: float,
    K_atm: float,
    ivs: Dict[str, Any],
    pcr: Dict[str, Any],
    signals: List[str],
    gates: Dict[str, Any],
    decision: str,
    dq_flags: List[str]
) -> str:
    """Emit a simplified explain JSON for backward compatibility."""
    
    payload = {
        "index": index,
        "expiry": expiry.isoformat() if expiry.tzinfo else expiry.replace(tzinfo=dt.timezone.utc).isoformat(),
        "tau_hours": round(tau_hours, 4),
        "step": step,
        "forward": round(forward, 2),
        "K_atm": K_atm,
        "ivs": ivs,
        "pcr": pcr,
        "signals": signals,
        "gates": gates,
        "decision": decision,
        "data_quality_flags": dq_flags
    }
    
    return json.dumps(payload, default=str)


__all__ = [
    "emit_comprehensive_explain",
    "emit_simple_explain"
]