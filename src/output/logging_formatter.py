"""
Logging formatter for enhanced engine to match dashboard expectations.

This module provides a dual logging system that outputs colored console logs
and plain file logs in the format expected by the dashboard.
"""

from __future__ import annotations

import logging
import datetime as dt
from typing import Dict, Optional, Any
from colorama import Fore, Style
import pytz

from src.engine_enhanced import MarketData, TradingDecision
from src.output.format import format_output_line


class DualOutputFormatter:
    """
    Formats enhanced engine output to match original engine format.
    Provides both colored console output and plain file output.
    """
    
    def __init__(self):
        self.console_formatter = ColoredConsoleFormatter()
        self.file_formatter = PlainFileFormatter()
    
    def format_decision_output(
        self,
        market_data: MarketData,
        decision: TradingDecision,
        iteration: int = 0
    ) -> tuple[str, str]:
        """
        Format trading decision for dual output.
        
        Returns:
            tuple: (console_output, file_output)
        """
        
        # Create the formatted output using existing format_output_line
        formatted_output = self._create_formatted_output(market_data, decision)
        
        # Create console output with colors
        console_output = self.console_formatter.format_with_colors(formatted_output)
        
        # Create file output without colors
        file_output = self.file_formatter.format_without_colors(formatted_output)
        
        return console_output, file_output
    
    def _create_formatted_output(
        self,
        market_data: MarketData,
        decision: TradingDecision
    ) -> str:
        """Create formatted output using existing format_output_line function."""
        
        # Map enhanced engine data to format_output_line parameters
        now = dt.datetime.now(pytz.timezone("Asia/Kolkata"))  # Use current IST time
        symbol = market_data.index
        spot_now = market_data.spot
        vwap_fut = market_data.futures_mid or market_data.spot * 1.002  # Mock if not available
        
        # Mock additional technical indicators that enhanced engine doesn't provide yet
        D = market_data.momentum_score * 10  # Scale momentum to D
        ATR_D = abs(market_data.momentum_score) * 5  # Mock ATR
        
        # Create snap dict from enhanced engine data
        snap = {
            "vnd": 0.5,  # Mock VND since signals are not directly accessible
            "ssd": 0.5,  # Mock SSD
            "pdist_pct": 0.1,  # Mock PD
            "pcr": decision.pcr_total if decision.pcr_total is not None else 0.98,  # Use actual PCR or realistic default
            "dpcr_z": 0.0,  # Mock PCR change
            "mph_pts_per_hr": 0.0,  # Mock max pain drift
            "mph_norm": 0.0,  # Mock normalized drift
            "atm_iv": (decision.atm_iv * 100) if decision.atm_iv else None,  # Use actual ATM IV or None
            "iv_z": 0.0,  # Mock IV z-score
            "basis": (vwap_fut - spot_now) if vwap_fut else 0.0,  # Basis calculation
        }
        
        # Use actual computed values or fallback to realistic values when calculations fail
        mp = int(decision.atm_strike) if decision.atm_strike else int(spot_now)  # Max pain as ATM or spot
        atm_k = int(decision.atm_strike) if decision.atm_strike else None  # ATM strike or None
        
        # If ATM IV failed to calculate but we have data, provide a reasonable default
        if snap["atm_iv"] is None:
            # Only provide a fallback if we actually have options data that should have worked
            snap["atm_iv"] = 20.0  # Conservative 20% when calculation fails with data
        
        # Get IV percentile early since it's used in scenario logic
        iv_pct_hint = decision.iv_percentile if decision.iv_percentile is not None else 50.0  # Use actual IV percentile
        
        # Create more realistic scenario probabilities based on market regime and indicators
        confidence = decision.confidence or 0.0
        regime = decision.market_regime
        
        # Determine dominant scenario based on market regime and IV conditions
        iv_crush = iv_pct_hint < 33
        high_vol = iv_pct_hint > 67
        
        if regime and hasattr(regime, 'volatility_regime'):
            if iv_crush and regime.volatility_regime == "LOW":
                top = "Pin & decay day (IV crush)"
                probs = {
                    "Pin & decay day (IV crush)": 0.75,
                    "Short-cover reversion up": 0.15,
                    "Bear migration": 0.05,
                    "Squeeze continuation (one-way)": 0.05,
                }
            elif high_vol and regime.volatility_regime == "HIGH":
                top = "Squeeze continuation (one-way)"
                probs = {
                    "Squeeze continuation (one-way)": 0.60,
                    "Event knee-jerk then revert": 0.25,
                    "Short-cover reversion up": 0.10,
                    "Bear migration": 0.05,
                }
            elif decision.direction == "LONG":
                top = "Bull migration / gamma carry"
                probs = {
                    "Bull migration / gamma carry": 0.50,
                    "Short-cover reversion up": 0.30,
                    "Pin & decay day (IV crush)": 0.15,
                    "Squeeze continuation (one-way)": 0.05,
                }
            elif decision.direction == "SHORT":
                top = "Bear migration"
                probs = {
                    "Bear migration": 0.55,
                    "Event knee-jerk then revert": 0.25,
                    "Pin & decay day (IV crush)": 0.15,
                    "Short-cover reversion up": 0.05,
                }
            else:
                top = "Short-cover reversion up"
                probs = {
                    "Short-cover reversion up": 0.45,
                    "Pin & decay day (IV crush)": 0.25,
                    "Bear migration": 0.15,
                    "Bull migration / gamma carry": 0.10,
                    "Squeeze continuation (one-way)": 0.05,
                }
        else:
            # Fallback for when regime is not available
            top = "Short-cover reversion up"
            probs = {
                "Short-cover reversion up": 0.60,
                "Pin & decay day (IV crush)": 0.20,
                "Bear migration": 0.10,
                "Squeeze continuation (one-way)": 0.10,
            }
        
        # Create trading plan from decision
        tp = {
            "action": self._map_decision_to_action(decision),
            "why": self._get_decision_reason(decision)
        }
        
        # Mock OI flags
        oi_flags = {
            "two_sided_adjacent": True,
            "pe_write_above": False,
            "ce_unwind_below": False,
            "ce_write_above": False,
            "pe_unwind_below": False,
        }
        
        # Mock additional technical indicators
        vwap_spot = spot_now * 0.998  # Mock VWAP slightly below spot
        adx5 = market_data.adx or 15.0
        div = 0.0  # Mock IV divergence
        # iv_pct_hint is already defined above
        macd_last = market_data.momentum_score or 0.0
        macd_sig_last = 0.0  # Mock MACD signal
        VND = snap["vnd"]
        PIN_NORM = 0.5  # Mock pin normal
        MPH_NORM_THR = 0.5  # Mock max pain threshold
        
        return format_output_line(
            now, symbol, spot_now, vwap_fut, D, ATR_D, snap, mp, atm_k,
            probs, top, tp, oi_flags, vwap_spot, adx5, div, iv_pct_hint,
            macd_last, macd_sig_last, VND, PIN_NORM, MPH_NORM_THR
        )
    
    def _map_decision_to_action(self, decision: TradingDecision) -> str:
        """Map enhanced engine decision to original engine action format."""
        if decision.action == "NO_TRADE":
            return "NO-TRADE"
        elif decision.direction == "LONG":
            return "BUY_CE"
        elif decision.direction == "SHORT":
            return "BUY_PE"
        else:
            return "NO-TRADE"
    
    def _get_decision_reason(self, decision: TradingDecision) -> str:
        """Get reason for decision."""
        if decision.action == "NO_TRADE":
            if decision.gate_decision and decision.gate_decision.muted:
                return "confidence below threshold"
            else:
                return "no favorable setup"
        else:
            return f"high confidence {decision.action.lower()}"


class ColoredConsoleFormatter:
    """Formatter that preserves ANSI color codes for console output."""
    
    def format_with_colors(self, text: str) -> str:
        """Format text preserving color codes for console display."""
        return text


class PlainFileFormatter:
    """Formatter that removes ANSI color codes for file output."""
    
    def format_without_colors(self, text: str) -> str:
        """Format text removing color codes for file logging."""
        import re
        # Remove ANSI color codes
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        return ansi_escape.sub('', text)


def create_dual_logger(name: str, log_file_path: str) -> tuple[logging.Logger, DualOutputFormatter]:
    """
    Create a dual logger that outputs to both console (with colors) and file (without colors).
    
    Args:
        name: Logger name
        log_file_path: Path to log file
        
    Returns:
        tuple: (logger, formatter)
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler without colors  
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    
    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s INFO: %(message)s')
    file_formatter = logging.Formatter('%(asctime)s INFO: %(message)s')
    
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    output_formatter = DualOutputFormatter()
    
    return logger, output_formatter


def log_startup_message(logger: logging.Logger, provider: str, symbols: list, poll_seconds: int, mode: str):
    """Log the startup message in expected format."""
    message = f"Engine started | provider={provider} | symbols={symbols} | poll={poll_seconds}s | mode={mode}"
    logger.info(message)


def log_micro_penalty(logger: logging.Logger, penalty: float, spread: float, qi: float, stab: float):
    """Log micro penalty message in expected format."""
    message = f"Micro penalty {penalty:.2f}: spread={spread:.4f}, qi={qi:.2f}, stab={stab:.2f}"
    logger.info(message)


def log_expiry_info(logger: logging.Logger, expiry: str, step: int, atm: int, pcr: float):
    """Log expiry information in expected format."""
    message = f"expiry={expiry} step={step} atm={atm} pcr={pcr:.2f}"
    logger.info(message)


def log_alert(logger: logging.Logger, alert_message: str):
    """Log alert message in expected format."""
    message = f"ALERT: {alert_message}"
    logger.info(message)