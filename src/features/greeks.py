from __future__ import annotations
import math
from typing import Dict, Tuple, Optional

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)

def bs_gamma(S: float, K: float, r: float, T: float, sigma: float) -> float:
    if S<=0 or K<=0 or T<=0 or sigma<=0:
        return 0.0
    d1 = (math.log(S/K) + (r+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    return _norm_pdf(d1)/(S*sigma*math.sqrt(T))

def gex_from_chain(chain: Dict, spot: float, minutes_to_exp: float, r: float, atm_iv: float, contract_mult: int = 1) -> float:
    """Approximate dealer gamma exposure (GEX): sum of gamma * OI * contract_mult * sign.
    Assumes net dealer short options stance → sign = -1 for both calls and puts.
    Uses a flat IV (atm_iv) for tractability.
    """
    if not chain or not chain.get('strikes'):
        return 0.0
    T = max(1e-9, minutes_to_exp)/(365*24*60)
    iv = max(1e-6, atm_iv)
    gex = 0.0
    for k in chain['strikes']:
        try:
            K = float(k)
        except Exception:
            continue
        gamma = bs_gamma(spot, K, r, T, iv)
        ce_oi = int(chain['calls'].get(k, {}).get('oi', 0))
        pe_oi = int(chain['puts'].get(k, {}).get('oi', 0))
        # Sign negative for dealer short
        gex += -gamma * (ce_oi + pe_oi) * contract_mult
    return float(gex)

def zero_gamma_level(chain: Dict, spot: float, minutes_to_exp: float, r: float, atm_iv: float, step: int) -> float:
    """Brute scan for zero-gamma crossing near spot using a small window of strikes."""
    if not chain or not chain.get('strikes'):
        return float('nan')
    strikes = sorted(chain['strikes'])
    lo = max(min(strikes), int(spot - 10*step))
    hi = min(max(strikes), int(spot + 10*step))
    best = float('nan'); best_abs = float('inf')
    for S in range(int(lo), int(hi)+1, step):
        g = gex_from_chain(chain, float(S), minutes_to_exp, r, atm_iv)
        if abs(g) < best_abs:
            best_abs = abs(g); best = float(S)
    return best

