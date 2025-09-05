"""Enhanced scenario classification engine for expiry day trading.

This module implements the deterministic scenario classification logic
extracted and enhanced from the legacy engine, following Indian options
market conventions and providing explainable decision making.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Scenario definitions
SCENARIOS = [
    "Short-cover reversion up",
    "Bear migration", 
    "Bull migration / gamma carry",
    "Pin & decay day (IV crush)",
    "Squeeze continuation (one-way)",
    "Event knee-jerk then revert",
]

# Default weights following India-specific conventions
DEFAULT_WEIGHTS = {
    "price_trend": 0.35,
    "options_flow": 0.45, 
    "volatility": 0.20
}

@dataclass
class ScenarioInputs:
    """Inputs for scenario classification."""
    
    # Price metrics
    spot: float
    D: float  # Spot - MaxPain
    ATR_D: float  # Session range proxy
    VND: float  # |D| / ATR_D
    SSD: float  # Spot/Session Delta
    PD: float   # Price Delta
    
    # Options flow metrics
    pcr: float
    dpcr: float
    dpcr_z: float
    
    # Volatility metrics
    atm_iv: float
    div: float  # IV change
    iv_z: float
    iv_pct_hint: float
    
    # Technical indicators
    vwap: float
    adx5: float
    
    # Open Interest flags
    oi_flags: Dict[str, bool]
    
    # Max Pain dynamics
    maxpain_drift_pts_per_hr: float
    mph_norm: float
    
    # Technical confirmations
    confirmations: Dict[str, int]
    techs: Dict[str, float]
    
    # Pin distance
    pin_distance_points: int

def softmax(values: List[float]) -> List[float]:
    """Compute softmax probabilities."""
    if not values:
        return []
    
    # Prevent overflow
    max_val = max(values)
    exp_values = [math.exp(v - max_val) for v in values]
    sum_exp = sum(exp_values)
    
    if sum_exp == 0:
        return [1.0 / len(values)] * len(values)
    
    return [v / sum_exp for v in exp_values]

def dynamic_weight_adjustment(
    symbol: str, 
    atr_d: float, 
    hour: float, 
    inst_bias: float = 0.0
) -> Dict[str, float]:
    """Adjust scenario weights based on market conditions."""
    weights = DEFAULT_WEIGHTS.copy()
    
    # Base range for volatility regime
    base_range = 300 if "BANK" in symbol.upper() else 150
    
    # Volatility regime adjustment
    vol_factor = max(-0.1, min(0.1, (atr_d - base_range) / max(1.0, base_range)))
    weights["volatility"] += 0.2 * vol_factor
    weights["price_trend"] -= 0.1 * vol_factor
    weights["options_flow"] -= 0.1 * vol_factor
    
    # Time of day adjustment - midday relies more on options flow
    if 10 <= hour <= 14.5:
        weights["options_flow"] += 0.05
        weights["price_trend"] -= 0.025
        weights["volatility"] -= 0.025
    else:
        weights["volatility"] += 0.05
        weights["price_trend"] -= 0.025
        weights["options_flow"] -= 0.025
    
    # Institutional bias adjustment
    if inst_bias > 0:
        weights["price_trend"] += 0.05 * inst_bias
        weights["options_flow"] -= 0.025 * inst_bias
        weights["volatility"] -= 0.025 * inst_bias
    
    # Normalize weights
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    
    return weights

def build_evidence_blocks(inputs: ScenarioInputs) -> Tuple[Dict, Dict, Dict]:
    """Build evidence blocks for price, options flow, and volatility."""
    
    # Price/Trend block
    price_block = {
        "reversion": (
            inputs.VND > 1.0 and  # Outside normal range
            inputs.SSD < -0.3 and  # Session delta negative
            inputs.adx5 < 25  # Not in strong trend
        ),
        "bear": (
            inputs.D < -50 and  # Below max pain
            inputs.spot < inputs.vwap and  # Below VWAP
            inputs.adx5 >= 18  # Trending
        ),
        "bull": (
            inputs.D > 50 and  # Above max pain
            inputs.spot > inputs.vwap and  # Above VWAP
            inputs.adx5 >= 18  # Trending
        ),
        "pin": (
            abs(inputs.D) < 50 and  # Near max pain
            inputs.VND < 0.5 and  # Low normalized distance
            inputs.adx5 < 18  # Choppy
        ),
        "squeeze": (
            abs(inputs.D) > 100 and  # Far from max pain
            inputs.adx5 >= 25 and  # Strong trend
            abs(inputs.mph_norm) > 0.5  # Max pain drifting
        ),
        "event": (
            abs(inputs.dpcr_z) > 2.0 or  # PCR spike
            abs(inputs.iv_z) > 2.0  # IV spike
        )
    }
    
    # Options Flow block
    flow_block = {
        "reversion": (
            inputs.dpcr_z > 1.0 and  # PCR momentum up
            inputs.oi_flags.get("pe_write_above", False)  # PE writing above
        ),
        "bear": (
            inputs.dpcr_z < -1.0 and  # PCR momentum down
            inputs.oi_flags.get("ce_write_above", False) and  # CE writing
            inputs.oi_flags.get("pe_unwind_below", False)  # PE unwinding
        ),
        "bull": (
            inputs.dpcr_z > 1.0 and  # PCR momentum up
            inputs.oi_flags.get("pe_write_above", False) and  # PE writing
            inputs.oi_flags.get("ce_unwind_below", False)  # CE unwinding
        ),
        "pin": (
            abs(inputs.dpcr_z) < 0.5 and  # PCR stable
            inputs.oi_flags.get("two_sided_adjacent", False)  # Two-sided OI
        ),
        "squeeze": (
            abs(inputs.dpcr_z) > 1.5 and  # PCR momentum
            (inputs.oi_flags.get("ce_write_above", False) or 
             inputs.oi_flags.get("pe_write_above", False))  # Directional writing
        ),
        "event": (
            abs(inputs.dpcr_z) > 2.5 or  # Extreme PCR
            (inputs.oi_flags.get("ce_write_above", False) and 
             inputs.oi_flags.get("pe_write_above", False))  # Both sides writing
        )
    }
    
    # Volatility block
    iv_crush = inputs.div <= 0 and inputs.iv_pct_hint < 33
    
    vol_block = {
        "reversion": (
            inputs.iv_z < -1.0 and  # IV falling
            iv_crush  # IV crush conditions
        ),
        "bear": (
            inputs.iv_z > 0.5 and  # IV rising
            inputs.atm_iv > 15  # Elevated IV
        ),
        "bull": (
            inputs.iv_z > 0.5 and  # IV rising
            inputs.atm_iv > 15  # Elevated IV
        ),
        "pin": (
            abs(inputs.iv_z) < 1.0 and  # IV stable
            iv_crush  # IV crush
        ),
        "squeeze": (
            inputs.iv_z > 1.5 and  # IV spiking
            inputs.atm_iv > 20  # High IV
        ),
        "event": (
            abs(inputs.iv_z) > 2.0 and  # IV volatility
            inputs.atm_iv > 25  # Very high IV
        )
    }
    
    return price_block, flow_block, vol_block

def classify_scenario(
    inputs: ScenarioInputs,
    symbol: str,
    hour: float,
    weights: Optional[Dict[str, float]] = None,
    gate_cap: float = 0.49,
    mph_norm_thr: float = 0.5,
    inst_bias: float = 0.0
) -> Tuple[Dict[str, float], Dict[str, bool], Dict]:
    """
    Classify market scenario with probabilities.
    
    Returns:
        (scenario_probabilities, block_gates_ok, diagnostics)
    """
    
    # Adjust weights dynamically
    if weights is None:
        weights = dynamic_weight_adjustment(symbol, inputs.ATR_D, hour, inst_bias)
    
    # Build evidence blocks
    price_block, flow_block, vol_block = build_evidence_blocks(inputs)
    
    # Calculate scores for each scenario
    scores = {
        "Short-cover reversion up": 
            weights["price_trend"] * float(price_block["reversion"]) +
            weights["options_flow"] * float(flow_block["reversion"]) +
            weights["volatility"] * float(vol_block["reversion"]),
            
        "Bear migration":
            weights["price_trend"] * float(price_block["bear"]) +
            weights["options_flow"] * float(flow_block["bear"]) +
            weights["volatility"] * float(vol_block["bear"]),
            
        "Bull migration / gamma carry":
            weights["price_trend"] * float(price_block["bull"]) +
            weights["options_flow"] * float(flow_block["bull"]) +
            weights["volatility"] * float(vol_block["bull"]),
            
        "Pin & decay day (IV crush)":
            weights["price_trend"] * float(price_block["pin"]) +
            weights["options_flow"] * float(flow_block["pin"]) +
            weights["volatility"] * float(vol_block["pin"]),
            
        "Squeeze continuation (one-way)":
            weights["price_trend"] * float(price_block["squeeze"]) +
            weights["options_flow"] * float(flow_block["squeeze"]) +
            weights["volatility"] * float(vol_block["squeeze"]),
            
        "Event knee-jerk then revert":
            weights["price_trend"] * float(price_block["event"]) +
            weights["options_flow"] * float(flow_block["event"]) +
            weights["volatility"] * float(vol_block["event"]),
    }
    
    # Apply block gating - require signals from all blocks
    block_ok = {}
    for name in SCENARIOS:
        if name == "Short-cover reversion up":
            ok = price_block["reversion"] and flow_block["reversion"] and vol_block["reversion"]
        elif name == "Bear migration":
            ok = price_block["bear"] and flow_block["bear"] and vol_block["bear"]
        elif name == "Bull migration / gamma carry":
            ok = price_block["bull"] and flow_block["bull"] and vol_block["bull"]
        elif name == "Pin & decay day (IV crush)":
            ok = price_block["pin"] and flow_block["pin"] and vol_block["pin"]
        elif name == "Squeeze continuation (one-way)":
            ok = price_block["squeeze"] and flow_block["squeeze"] and vol_block["squeeze"]
        else:  # Event knee-jerk then revert
            ok = price_block["event"] and flow_block["event"] and vol_block["event"]
        
        block_ok[name] = ok
        
        # Cap scenarios that don't meet all block requirements
        if not ok:
            scores[name] = min(scores[name], gate_cap)
    
    # Convert to probabilities using softmax
    probs = softmax(list(scores.values()))
    scenario_probs = {k: round(v, 3) for k, v in zip(scores.keys(), probs)}
    
    # Diagnostics
    diagnostics = {
        "weights_used": weights,
        "price_block": price_block,
        "flow_block": flow_block,
        "vol_block": vol_block,
        "raw_scores": scores,
        "block_gates": block_ok,
        "gate_cap_applied": {k: not v for k, v in block_ok.items()}
    }
    
    return scenario_probs, block_ok, diagnostics

def get_top_scenario(scenario_probs: Dict[str, float]) -> Tuple[str, float]:
    """Get the top scenario and its probability."""
    if not scenario_probs:
        return "Unknown", 0.0
    
    top_scenario = max(scenario_probs.items(), key=lambda x: x[1])
    return top_scenario[0], top_scenario[1]

def pretty_scenario_name(name: str) -> str:
    """Format scenario name for display."""
    mapping = {
        "Short-cover reversion up": "Short-Cover Reversion Up",
        "Bear migration": "Bear Migration",
        "Bull migration / gamma carry": "Bull Migration",
        "Pin & decay day (IV crush)": "Pin and Decay (IV crush)",
        "Squeeze continuation (one-way)": "Squeeze Continuation (One-way)",
        "Event knee-jerk then revert": "Event Knee-jerk then Revert",
    }
    return mapping.get(name, name)