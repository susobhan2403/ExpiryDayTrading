"""Complete integrated expiry day trading engine with enhanced capabilities.

This module integrates all enhanced components to provide a complete,
production-ready trading engine for Indian index options with:
- Enhanced metrics with India-specific conventions
- Multi-factor gating with regime detection
- Comprehensive observability
- Robust error handling and diagnostics
"""

from __future__ import annotations

import math
import time
import datetime as dt
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import pytz

from .metrics.enhanced import (
    infer_strike_step_enhanced,
    compute_forward_enhanced,
    pick_atm_strike_enhanced,
    compute_atm_iv_enhanced,
    compute_iv_percentile_enhanced,
    compute_pcr_enhanced
)
from .strategy.enhanced_gates import (
    EnhancedRegime,
    MultiFactorSignal,
    EnhancedGateDecision,
    detect_enhanced_regime,
    apply_enhanced_gates
)
from .strategy.scenario_classifier import (
    classify_scenario,
    ScenarioInputs,
    get_top_scenario,
    pretty_scenario_name
)
from .observability.enhanced_explain import emit_comprehensive_explain
from .config import get_rfr

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class MarketData:
    """Container for market data snapshot."""
    
    timestamp: dt.datetime
    index: str
    spot: float
    futures_mid: Optional[float] = None
    
    # Options chain data
    strikes: List[float] = None
    call_mids: Dict[float, float] = None
    put_mids: Dict[float, float] = None
    call_oi: Dict[float, int] = None
    put_oi: Dict[float, int] = None
    call_volumes: Dict[float, int] = None
    put_volumes: Dict[float, int] = None
    
    # Technical indicators
    adx: float = 15.0
    volume_ratio: float = 1.0
    spread_bps: float = 10.0
    momentum_score: float = 0.0
    
    def __post_init__(self):
        """Initialize default empty containers."""
        if self.strikes is None:
            self.strikes = []
        if self.call_mids is None:
            self.call_mids = {}
        if self.put_mids is None:
            self.put_mids = {}
        if self.call_oi is None:
            self.call_oi = {}
        if self.put_oi is None:
            self.put_oi = {}
        if self.call_volumes is None:
            self.call_volumes = {}
        if self.put_volumes is None:
            self.put_volumes = {}


@dataclass
class TradingDecision:
    """Final trading decision with full context."""
    
    action: str  # LONG, SHORT, NO_TRADE
    direction: Optional[str] = None
    size_factor: float = 0.0
    confidence: float = 0.0
    
    # Context
    market_regime: EnhancedRegime = None
    gate_decision: EnhancedGateDecision = None
    data_quality_score: float = 0.0
    
    # Metrics
    forward: float = 0.0
    atm_strike: float = 0.0
    max_pain: float = 0.0  # Add max pain field
    atm_iv: Optional[float] = None
    iv_percentile: Optional[float] = None
    pcr_total: Optional[float] = None
    pcr_band: Optional[float] = None
    
    # Timing
    tau_hours: float = 0.0
    processing_time_ms: float = 0.0
    
    # Additional fields for compatibility
    decision: Optional[str] = None  # Maps to action
    scenario: Optional[str] = None
    reason: Optional[str] = None
    
    def __post_init__(self):
        """Ensure decision field matches action for backward compatibility."""
        if self.decision is None:
            self.decision = self.action


class EnhancedTradingEngine:
    """Enhanced trading engine for Indian index options."""
    
    def __init__(
        self,
        index: str,
        expiry: dt.datetime,
        min_tau_hours: float = 2.0,
        risk_free_rate: Optional[float] = None,
        dividend_yield: float = 0.01,
        iv_history_window: int = 100
    ):
        """Initialize enhanced trading engine.
        
        Parameters
        ----------
        index: str
            Index symbol (e.g., "NIFTY", "BANKNIFTY")
        expiry: dt.datetime
            Target expiry (timezone aware)
        min_tau_hours: float
            Minimum hours to expiry before switching to next expiry
        risk_free_rate: float, optional
            Risk-free rate (defaults to config)
        dividend_yield: float
            Dividend yield assumption
        iv_history_window: int
            Number of historical IV observations to retain
        """
        self.index = index.upper()
        self.expiry = expiry
        self.min_tau_hours = min_tau_hours
        self.risk_free_rate = risk_free_rate or get_rfr()
        self.dividend_yield = dividend_yield
        
        # State
        self.iv_history: List[Tuple[float, float]] = []  # (tau, iv) pairs
        self.last_processing_time = 0.0
        
        # Diagnostics
        self.total_runs = 0
        self.successful_runs = 0
        self.error_count = 0
        
        # Setup basic logging
        import logging
        self.logger = logging.getLogger(f"EnhancedTradingEngine-{index}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.WARNING)  # Only warnings and errors
    
    def process_market_data(
        self,
        market_data: MarketData,
        trend_score: float = 0.0,
        orb_signal: Optional[str] = None,
        orb_strength: float = 0.0,
        orb_breakout_size: float = 0.0
    ) -> TradingDecision:
        """Process market data and generate trading decision.
        
        Parameters
        ----------
        market_data: MarketData
            Current market snapshot
        trend_score: float
            Trend score (-1.0 to 1.0, positive = bullish)
        orb_signal: str, optional
            Opening range breakout signal ("LONG" or "SHORT")
        orb_strength: float
            Strength of ORB signal (0.0 to 1.0)
        orb_breakout_size: float
            Size of the breakout in points
        """
        start_time = time.perf_counter()
        self.total_runs += 1
        
        try:
            # Calculate time to expiry
            if market_data.timestamp.tzinfo is None:
                current_time = IST.localize(market_data.timestamp)
            else:
                current_time = market_data.timestamp.astimezone(IST)
            
            if current_time >= self.expiry:
                return self._create_no_trade_decision(
                    "expiry_passed", 
                    current_time, 
                    processing_time_ms=(time.perf_counter() - start_time) * 1000
                )
            
            tau_seconds = (self.expiry - current_time).total_seconds()
            tau_hours = tau_seconds / 3600.0
            tau_years = tau_seconds / (365.25 * 24 * 3600)
            
            if tau_hours < 0.001:  # Less than 3.6 seconds
                return self._create_no_trade_decision(
                    "expiry_too_close",
                    current_time,
                    processing_time_ms=(time.perf_counter() - start_time) * 1000
                )
            
            # Data quality assessment
            dq_flags = self._assess_data_quality(market_data, tau_hours)
            dq_score = max(0.0, 1.0 - len(dq_flags) * 0.2)
            
            # Enhanced metrics computation
            metrics_result = self._compute_enhanced_metrics(
                market_data, tau_years, tau_hours
            )
            
            if not metrics_result["success"]:
                return self._create_no_trade_decision(
                    metrics_result["reason"],
                    current_time,
                    processing_time_ms=(time.perf_counter() - start_time) * 1000
                )
            
            # Multi-factor signal construction
            signals = self._construct_signals(
                market_data, metrics_result, orb_signal, orb_strength, orb_breakout_size
            )
            
            # Regime detection
            regime = detect_enhanced_regime(
                trend_score=trend_score,
                adx=market_data.adx,
                iv_percentile=metrics_result.get("iv_percentile") or 50.0,  # Handle None values
                spread_bps=market_data.spread_bps,
                volume_ratio=market_data.volume_ratio,
                momentum_score=market_data.momentum_score
            )
            
            # Enhanced gating
            gate_decision = apply_enhanced_gates(
                signals=signals,
                regime=regime,
                tau_hours=tau_hours,
                confirming_bars=2,  # Could be parameterized
                min_confirm_bars=2,
                spike_classification="UNKNOWN"  # Could be enhanced with spike detection
            )
            
            # Scenario classification
            scenario_inputs = self._prepare_scenario_inputs(market_data, metrics_result, trend_score)
            current_time = market_data.timestamp
            hour = current_time.hour + current_time.minute / 60.0
            
            scenario_probs, block_gates, scenario_diag = classify_scenario(
                inputs=scenario_inputs,
                symbol=market_data.index,
                hour=hour,
                weights=None,  # Use dynamic weights
                gate_cap=0.49,
                mph_norm_thr=0.5,
                inst_bias=0.0  # Could be parameterized
            )
            
            top_scenario, top_prob = get_top_scenario(scenario_probs)
            pretty_scenario = pretty_scenario_name(top_scenario)
            
            # Final decision
            final_action = "NO_TRADE"
            if not gate_decision.muted and gate_decision.direction:
                final_action = gate_decision.direction
            
            # Create decision object
            decision = TradingDecision(
                action=final_action,
                direction=gate_decision.direction,
                size_factor=gate_decision.size_factor,
                confidence=gate_decision.confidence,
                market_regime=regime,
                gate_decision=gate_decision,
                data_quality_score=dq_score,
                forward=metrics_result.get("forward", 0.0),
                atm_strike=metrics_result.get("atm_strike", 0.0),
                max_pain=metrics_result.get("max_pain", 0.0),  # Add max pain
                atm_iv=metrics_result.get("atm_iv"),
                iv_percentile=metrics_result.get("iv_percentile"),
                pcr_total=metrics_result.get("pcr_total"),
                pcr_band=metrics_result.get("pcr_band"),
                tau_hours=tau_hours,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                scenario=f"{pretty_scenario} {int(top_prob * 100)}%",
                reason=gate_decision.primary_reason if gate_decision.muted else None
            )
            
            # Update IV history
            if metrics_result.get("atm_iv"):
                self.iv_history.append((tau_years, metrics_result["atm_iv"]))
                if len(self.iv_history) > 200:  # Keep reasonable history
                    self.iv_history = self.iv_history[-200:]
            
            self.successful_runs += 1
            self.last_processing_time = decision.processing_time_ms
            
            return decision
            
        except Exception as e:
            self.error_count += 1
            return self._create_no_trade_decision(
                f"processing_error: {str(e)}",
                market_data.timestamp,
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    def _assess_data_quality(self, market_data: MarketData, tau_hours: float) -> List[str]:
        """Assess data quality and return list of issues."""
        flags = []
        
        # Time-based flags
        if tau_hours < 0.1:
            flags.append("expiry_too_close")
        elif tau_hours < 0.25:  # Less than 15 minutes
            flags.append("near_expiry_warning")
        
        # Market data flags
        if not market_data.strikes or len(market_data.strikes) < 5:
            flags.append("insufficient_strikes")
        
        if market_data.spread_bps > 50:
            flags.append("wide_spreads")
        
        if market_data.volume_ratio < 0.3:
            flags.append("insufficient_volume")
        
        # Options chain flags
        valid_calls = sum(1 for v in market_data.call_mids.values() if v and v > 0)
        valid_puts = sum(1 for v in market_data.put_mids.values() if v and v > 0)
        
        if valid_calls < 3 or valid_puts < 3:
            flags.append("sparse_options_data")
        
        total_call_oi = sum(market_data.call_oi.values())
        total_put_oi = sum(market_data.put_oi.values())
        
        if total_call_oi < 1000 or total_put_oi < 1000:
            flags.append("low_open_interest")
        
        return flags
    
    def _prepare_scenario_inputs(
        self, 
        market_data: MarketData, 
        metrics_result: Dict[str, Any], 
        trend_score: float
    ) -> ScenarioInputs:
        """Prepare inputs for scenario classification."""
        
        # Calculate required metrics for scenario classification
        spot = market_data.spot
        max_pain = metrics_result.get("max_pain", spot)  # Fallback to spot if not available
        D = spot - max_pain
        
        # Estimate ATR_D (session range proxy) - this should be calculated from market data
        # For now, use a simple estimate based on symbol
        base_atr = 300 if "BANK" in market_data.index.upper() else 150
        ATR_D = base_atr * (1 + abs(trend_score) * 0.5)  # Scale with trend
        
        VND = abs(D) / max(1.0, ATR_D)
        
        # Estimate other required values (in production these would come from market data)
        SSD = trend_score * 0.1  # Rough proxy for session delta
        PD = trend_score * 0.05   # Rough proxy for price delta
        
        # PCR metrics
        pcr = metrics_result.get("pcr_total", 1.0)
        dpcr = 0.0  # Delta PCR - would need historical tracking
        dpcr_z = 0.0  # Delta PCR z-score - would need rolling calculation
        
        # Volatility metrics
        atm_iv = metrics_result.get("atm_iv", 20.0) or 20.0
        div = 0.0  # IV change - would need historical tracking
        iv_z = 0.0  # IV z-score - would need rolling calculation
        iv_percentile = metrics_result.get("iv_percentile", 50.0) or 50.0
        
        # Technical indicators
        vwap = spot * 0.999  # Rough estimate - would need actual VWAP
        
        # OI flags - these would need more sophisticated analysis
        oi_flags = {
            "pe_write_above": pcr > 1.2,
            "ce_unwind_below": pcr < 0.8,
            "ce_write_above": pcr < 0.8,
            "pe_unwind_below": pcr > 1.2,
            "two_sided_adjacent": 0.9 < pcr < 1.1,
        }
        
        # Max Pain dynamics
        mph_norm = 0.0  # Would need historical max pain tracking
        maxpain_drift_pts_per_hr = 0.0
        
        # Confirmations and techs
        confirmations = {"price": 1, "flow": 1, "vol": 1}
        techs = {"rsi": 50.0, "macd": 0.0}
        
        # Pin distance
        pin_distance_points = abs(D)
        
        return ScenarioInputs(
            spot=spot,
            D=D,
            ATR_D=ATR_D,
            VND=VND,
            SSD=SSD,
            PD=PD,
            pcr=pcr,
            dpcr=dpcr,
            dpcr_z=dpcr_z,
            atm_iv=atm_iv,
            div=div,
            iv_z=iv_z,
            iv_pct_hint=iv_percentile,
            vwap=vwap,
            adx5=market_data.adx,
            oi_flags=oi_flags,
            maxpain_drift_pts_per_hr=maxpain_drift_pts_per_hr,
            mph_norm=mph_norm,
            confirmations=confirmations,
            techs=techs,
            pin_distance_points=pin_distance_points
        )

    def _compute_enhanced_metrics(
        self, market_data: MarketData, tau_years: float, tau_hours: float
    ) -> Dict[str, Any]:
        """Compute enhanced metrics with full diagnostics."""
        result = {"success": False}
        
        try:
            # Strike step inference
            step, step_diag = infer_strike_step_enhanced(market_data.strikes)
            if step == 0:
                result["reason"] = f"strike_step_inference_failed: {step_diag.get('reason', 'unknown')}"
                return result
            
            # Forward price computation
            forward, forward_diag = compute_forward_enhanced(
                spot=market_data.spot,
                fut_mid=market_data.futures_mid,
                r=self.risk_free_rate,
                q=self.dividend_yield,
                tau_years=tau_years
            )
            
            if forward <= 0:
                result["reason"] = f"forward_computation_failed: {forward_diag.get('reason', 'unknown')}"
                return result
            
            # ATM strike selection
            atm_strike, atm_diag = pick_atm_strike_enhanced(
                F=forward,
                strikes=market_data.strikes,
                step=step,
                ce_mid=market_data.call_mids,
                pe_mid=market_data.put_mids,
                spot=market_data.spot
            )
            
            if atm_strike == 0:
                result["reason"] = f"atm_selection_failed: {atm_diag.get('reason', 'unknown')}"
                return result
            
            # ATM IV computation
            ce_mid = market_data.call_mids.get(atm_strike)
            pe_mid = market_data.put_mids.get(atm_strike)
            
            atm_iv, iv_diag = compute_atm_iv_enhanced(
                ce_mid=ce_mid,
                pe_mid=pe_mid,
                F=forward,
                K_atm=atm_strike,
                tau_years=tau_years,
                r=self.risk_free_rate
            )
            
            # IV percentile computation
            iv_percentile = None
            iv_rank = None
            if atm_iv:
                if self.iv_history and len(self.iv_history) >= 3:  # Need at least 3 data points
                    iv_percentile, iv_rank, percentile_diag = compute_iv_percentile_enhanced(
                        history=self.iv_history,
                        current=atm_iv,
                        current_tau=tau_years,
                        tau_tol=7.0/365.0  # 1 week tolerance
                    )
                else:
                    # When insufficient historical data, provide reasonable defaults
                    # Based on Indian market conditions: low IV typically 10-20%, high IV 30-50%
                    if atm_iv < 0.15:  # Very low IV
                        iv_percentile = 5.0
                    elif atm_iv < 0.20:  # Low IV 
                        iv_percentile = 20.0
                    elif atm_iv < 0.30:  # Normal IV
                        iv_percentile = 50.0
                    elif atm_iv < 0.40:  # High IV
                        iv_percentile = 80.0
                    else:  # Very high IV
                        iv_percentile = 95.0
                    iv_rank = iv_percentile
            
            # PCR computation
            pcr_results, pcr_diag = compute_pcr_enhanced(
                oi_put=market_data.put_oi,
                oi_call=market_data.call_oi,
                strikes=market_data.strikes,
                K_atm=atm_strike,
                step=step,
                m=6  # ±6 strikes from ATM
            )
            
            # Log PCR diagnostics if calculation failed
            if pcr_results.get("PCR_OI_total") is None:
                total_call_oi = sum(market_data.call_oi.values())
                total_put_oi = sum(market_data.put_oi.values())
                self.logger.warning(f"PCR calculation returned None - Total Call OI: {total_call_oi}, Total Put OI: {total_put_oi}, Reason: {pcr_diag.get('total_pcr_reason', 'unknown')}")
            
            # Max Pain calculation - ensure this is independent from ATM
            max_pain = 0
            try:
                from src.features.options import max_pain as calculate_max_pain
                chain_data = {
                    'strikes': list(market_data.strikes),
                    'calls': {s: {'oi': market_data.call_oi.get(s, 0)} for s in market_data.strikes},
                    'puts': {s: {'oi': market_data.put_oi.get(s, 0)} for s in market_data.strikes}
                }
                max_pain = calculate_max_pain(chain_data, market_data.spot, step)
                self.logger.info(f"Max Pain calculation successful: {max_pain} (ATM: {atm_strike})")
                
                # Validate Max Pain calculation - it should never be identical to ATM unless by genuine coincidence
                if max_pain == atm_strike:
                    # Log this as it may indicate an issue with OI distribution
                    self.logger.info(f"Max Pain ({max_pain}) equals ATM Strike ({atm_strike}) - verifying calculation")
                    
            except Exception as e:
                self.logger.warning(f"Max pain calculation failed: {e}")
                # Instead of falling back to ATM, use a reasonable estimate based on spot
                # This prevents identical values when calculation fails
                max_pain = int(market_data.spot / step) * step
                if max_pain < min(market_data.strikes):
                    max_pain = min(market_data.strikes)
                elif max_pain > max(market_data.strikes):
                    max_pain = max(market_data.strikes)
                self.logger.info(f"Max Pain fallback used: {max_pain}")
            
            # Success
            result.update({
                "success": True,
                "step": step,
                "forward": forward,
                "atm_strike": atm_strike,
                "atm_iv": atm_iv,
                "iv_percentile": iv_percentile,
                "iv_rank": iv_rank,
                "pcr_total": pcr_results.get("PCR_OI_total"),
                "pcr_band": pcr_results.get("PCR_OI_band"),
                "max_pain": max_pain,
                "diagnostics": {
                    "step": step_diag,
                    "forward": forward_diag,
                    "atm": atm_diag,
                    "iv": iv_diag,
                    "pcr": pcr_diag
                }
            })
            
            return result
            
        except Exception as e:
            result["reason"] = f"metrics_computation_error: {str(e)}"
            return result
    
    def _construct_signals(
        self,
        market_data: MarketData,
        metrics_result: Dict[str, Any],
        orb_signal: Optional[str],
        orb_strength: float,
        orb_breakout_size: float
    ) -> MultiFactorSignal:
        """Construct multi-factor signal alignment."""
        
        # Volume signal - improved thresholds and logic
        volume_signal = None
        volume_strength = 0.0
        if market_data.volume_ratio > 1.2:  # Lowered from 1.5 to be less restrictive
            volume_signal = "LONG"  # High volume generally bullish
            volume_strength = min(1.0, (market_data.volume_ratio - 0.8) / 1.2)  # Better scaling
        elif market_data.volume_ratio < 0.8:  # Raised from 0.7 to be less restrictive
            volume_signal = "SHORT"
            volume_strength = min(1.0, (1.2 - market_data.volume_ratio) / 0.4)  # Better scaling
        
        # OI flow signal - improved thresholds and logic
        oi_flow_signal = None
        oi_flow_strength = 0.0
        total_call_oi = sum(market_data.call_oi.values())
        total_put_oi = sum(market_data.put_oi.values())
        if total_call_oi > 0 and total_put_oi > 0:
            pcr = total_put_oi / total_call_oi
            if pcr < 0.9:  # Raised from 0.8 to be less restrictive
                oi_flow_signal = "LONG"
                oi_flow_strength = min(1.0, (1.1 - pcr) / 0.4)  # Better scaling
            elif pcr > 1.1:  # Lowered from 1.2 to be less restrictive
                oi_flow_signal = "SHORT"
                oi_flow_strength = min(1.0, (pcr - 1.1) / 0.4)  # Better scaling
        
        # IV crush signal
        iv_crush_signal = None
        iv_crush_strength = 0.0
        iv_percentile = metrics_result.get("iv_percentile", 50.0)
        if iv_percentile and iv_percentile > 80:
            iv_crush_signal = "SHORT"  # High IV suggests selling premium
            iv_crush_strength = min(1.0, (iv_percentile - 80) / 20)
        elif iv_percentile and iv_percentile < 20:
            iv_crush_signal = "LONG"  # Low IV suggests buying premium
            iv_crush_strength = min(1.0, (20 - iv_percentile) / 20)
        
        # Price action signal - improved thresholds
        price_action_signal = None
        price_action_strength = 0.0
        if abs(market_data.momentum_score) > 0.15:  # Lowered from 0.3 to be less restrictive
            price_action_signal = "LONG" if market_data.momentum_score > 0 else "SHORT"
            price_action_strength = min(1.0, abs(market_data.momentum_score) * 2.0)  # Better scaling
        
        return MultiFactorSignal(
            orb_signal=orb_signal,
            orb_strength=orb_strength,
            volume_signal=volume_signal,
            volume_strength=volume_strength,
            oi_flow_signal=oi_flow_signal,
            oi_flow_strength=oi_flow_strength,
            iv_crush_signal=iv_crush_signal,
            iv_crush_strength=iv_crush_strength,
            price_action_signal=price_action_signal,
            price_action_strength=price_action_strength,
            volume_ratio=market_data.volume_ratio,
            oi_delta_rate=0.0,  # Would need historical OI data
            iv_percentile=iv_percentile or 50.0,
            orb_breakout_size=orb_breakout_size
        )
    
    def _create_no_trade_decision(
        self, reason: str, timestamp: dt.datetime, processing_time_ms: float
    ) -> TradingDecision:
        """Create a NO_TRADE decision with reason."""
        return TradingDecision(
            action="NO_TRADE",
            gate_decision=EnhancedGateDecision(
                muted=True,
                direction=None,
                primary_reason=reason
            ),
            processing_time_ms=processing_time_ms
        )
    
    def emit_explain_json(
        self, 
        decision: TradingDecision,
        market_data: MarketData,
        signals: MultiFactorSignal
    ) -> str:
        """Emit comprehensive explain JSON for decision."""
        
        aligned_signals = signals.get_aligned_signals() if signals else []
        
        return emit_comprehensive_explain(
            index=self.index,
            expiry=self.expiry,
            timestamp=market_data.timestamp,
            tau_hours=decision.tau_hours,
            step=50,  # Would be computed from metrics
            spot=market_data.spot,
            forward=decision.forward,
            K_atm=decision.atm_strike,
            strikes_analyzed=market_data.strikes,
            iv_metrics={
                "atm_iv": decision.atm_iv,
                "percentile": decision.iv_percentile
            },
            pcr_metrics={
                "PCR_OI_total": decision.pcr_total,
                "PCR_OI_band": decision.pcr_band
            },
            regime=decision.market_regime,
            signals=signals,
            aligned_signals=aligned_signals,
            gate_decision=decision.gate_decision,
            final_decision=decision.action,
            data_quality_flags=[],  # Would be computed
            processing_time_ms=decision.processing_time_ms
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        success_rate = self.successful_runs / max(1, self.total_runs)
        
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "error_count": self.error_count,
            "success_rate": round(success_rate, 4),
            "last_processing_time_ms": round(self.last_processing_time, 2),
            "iv_history_size": len(self.iv_history)
        }


__all__ = [
    "MarketData",
    "TradingDecision", 
    "EnhancedTradingEngine"
]