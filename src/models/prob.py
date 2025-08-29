from __future__ import annotations
import math
from typing import Dict

def sigmoid(x: float) -> float:
    try:
        return 1.0/(1.0+math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def scp_probability(feat: Dict[str, float]) -> float:
    """Short-Covering Probability (SCP) logistic model.
    Inputs (expected keys, z-scored or naturally scaled):
      dbasis, voi_ce_unw_atm (negative), voi_pe_add_otm (positive), rsi_slope, macd_hist_slope,
      price_above_avwap (0/1), d_gex_toward_zero, adx
    Returns probability in [0,1].
    """
    x = 0.0
    x += 1.2 * (feat.get('dbasis', 0.0))
    x += 0.8 * (-feat.get('voi_ce_atm', 0.0))
    x += 0.8 * (feat.get('voi_pe_atm', 0.0))
    x += 0.7 * (feat.get('rsi_slope', 0.0))
    x += 0.6 * (feat.get('macd_hist_slope', 0.0))
    x += 0.9 * (1.0 if feat.get('price_above_avwap', False) else 0.0)
    x += 0.5 * (feat.get('d_gex_toward_zero', 0.0))
    x += 0.2 * max(0.0, (feat.get('adx', 0.0) - 18.0)/22.0)
    x += feat.get('bias', 0.0)
    return max(0.0, min(1.0, sigmoid(x)))

def erp_probability(feat: Dict[str, float]) -> float:
    """Expiry Reversion Probability (ERP) logistic model.
    Inputs:
      pin_density, dist_to_pin (negative helpful), iv_percentile_low (1/0), gex_mag_small (1/0), oi_conc_at_atm
    """
    x = 0.0
    x += 1.2 * feat.get('pin_density', 0.0)
    x += 0.8 * (-feat.get('dist_to_pin', 0.0))
    x += 0.6 * (1.0 if feat.get('ivp_low', False) else 0.0)
    x += 0.5 * (1.0 if feat.get('gex_small', False) else 0.0)
    x += 0.7 * feat.get('oi_conc_at_atm', 0.0)
    x += feat.get('bias', 0.0)
    return max(0.0, min(1.0, sigmoid(x)))

