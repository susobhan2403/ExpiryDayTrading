from __future__ import annotations
import math
import datetime as dt
from typing import Dict, List

from src.config import STEP_MAP, get_rfr
from .robust_metrics import (
    detect_strike_step,
    compute_forward,
    pick_atm_strike,
    compute_atm_iv,
    compute_pcr,
)

import pytz

IST = pytz.timezone("Asia/Kolkata")


def _mid_price(node: Dict) -> float:
    """Return a simple mid price for tie-breaking and IV calculations."""
    if not node:
        return float("nan")
    bid = float(node.get("bid", 0.0))
    ask = float(node.get("ask", 0.0))
    if bid > 0 and ask > 0:
        return 0.5 * (bid + ask)
    ltp = float(node.get("ltp", 0.0))
    return ltp if ltp > 0 else float("nan")


def bs_price(S, K, r, T, sig, call=True):
    """Black–Scholes price for a European option.

    Parameters
    ----------
    r : float
        Annualised risk-free rate as a decimal (e.g., ``0.066`` for 6.6%).
    """
    if T <= 0 or sig <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return (S * N(d1) - K * math.exp(-r * T) * N(d2)) if call else (
        K * math.exp(-r * T) * N(-d2) - S * N(-d1)
    )

def implied_vol(
    price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    call: bool = True,
    hi: float = 3.0,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """Solve Black–Scholes IV using bisection with wide bounds.

    Parameters
    ----------
    r : float
        Annualised risk-free rate expressed as a decimal (e.g., ``0.066``).

    Notes
    -----
    The previous implementation fixed the search window to ``[0.03, 1.5]`` and
    60 iterations which frequently failed in high volatility regimes (e.g.
    BANKNIFTY spikes) and returned biased or ``NaN`` values. The new solver
    expands the upper bound and exposes tolerance/iteration controls so callers
    can tune accuracy.
    """
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")
    lo = 1e-3
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pm = bs_price(S, K, r, T, mid, call)
        if abs(pm - price) < tol:
            return mid
        if pm > price:
            hi = mid
        else:
            lo = mid
    return mid

def bs_delta(S: float, K: float, r: float, T: float, sig: float, call: bool = True) -> float:
    """Black–Scholes delta.

    Parameters
    ----------
    r : float
        Annualised risk-free rate as a decimal.
    """
    if S <= 0 or K <= 0 or T <= 0 or sig <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    if call:
        # N(d1)
        return 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    else:
        # N(d1)-1
        return 0.5 * (1 + math.erf(d1 / math.sqrt(2))) - 1.0

def bs_gamma(S: float, K: float, r: float, T: float, sig: float) -> float:
    """Black–Scholes gamma.

    Parameters
    ----------
    r : float
        Annualised risk-free rate as a decimal.
    """
    if S <= 0 or K <= 0 or T <= 0 or sig <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return math.exp(-0.5 * d1 * d1) / (S * sig * math.sqrt(2 * math.pi * T))

def risk_reversal_25(
    chain: Dict, spot: float, minutes_to_exp: float, r: float, atm_iv: float
) -> Dict[str, float]:
    """
    Approximate 25-delta risk reversal: ``RR = IV_call(25d) - IV_put(25d)``.

    Parameters
    ----------
    r : float
        Annualised risk-free rate as a decimal (e.g., ``0.066``).

    Returns
    -------
    dict
        ``{"rr": float, "k_call": int, "k_put": int, "iv_call": float, "iv_put": float}``
    """
    out = {"rr": float('nan'), "k_call": 0, "k_put": 0, "iv_call": float('nan'), "iv_put": float('nan')}
    strikes = sorted(chain.get('strikes') or [])
    if not strikes:
        return out
    T = max(1e-9, minutes_to_exp)/(365*24*60)
    sig = atm_iv if atm_iv==atm_iv and atm_iv>0 else 0.2
    # select strikes by delta closeness
    best_call = None; best_call_diff = 9e9
    best_put = None; best_put_diff = 9e9
    for K in strikes:
        dc = abs(bs_delta(spot, K, r, T, sig, call=True) - 0.25)
        if dc < best_call_diff:
            best_call_diff = dc; best_call = K
        dp = abs(abs(bs_delta(spot, K, r, T, sig, call=False)) - 0.25)
        if dp < best_put_diff:
            best_put_diff = dp; best_put = K
    if best_call is None or best_put is None:
        return out
    # compute IVs from market mids
    def mid_price(node, K):
        bid = node.get('bid',0.0); ask = node.get('ask',0.0); ltp = node.get('ltp',0.0)
        if bid>0 and ask>0 and (ask-bid)/max(1.0,K) <= 0.03:
            return 0.5*(bid+ask)
        return ltp if ltp>0 else None
    call_node = chain['calls'].get(best_call, {})
    put_node = chain['puts'].get(best_put, {})
    m_c = mid_price(call_node, best_call)
    m_p = mid_price(put_node, best_put)
    iv_c = implied_vol(m_c, spot, best_call, r, T, call=True) if m_c else float('nan')
    iv_p = implied_vol(m_p, spot, best_put, r, T, call=False) if m_p else float('nan')
    if iv_c==iv_c and iv_p==iv_p and iv_c>0 and iv_p>0:
        out.update({"rr": float(iv_c - iv_p), "k_call": int(best_call), "k_put": int(best_put), "iv_call": float(iv_c), "iv_put": float(iv_p)})
    return out

def nearest_weekly_expiry(now_ist: dt.datetime, symbol: str) -> str:
    """
    Dynamic weekly expiry by symbol with rule changes from Sep 2025:
    - NIFTY: Tuesday from 2025-09-01 (was Thursday before)
    - SENSEX: Thursday from 2025-09-01 (was Tuesday before)
    - Others: default Thursday unless specialized later
    """
    d = now_ist.date()
    sym = (symbol or "").upper()
    eff = dt.date(2025, 9, 1)
    # default Thursday
    weekday_target = 3
    # Weekly mapping by index
    if sym == "NIFTY":
        weekday_target = 1 if d >= eff else 3  # Tue after eff else Thu
    elif sym == "BANKNIFTY":
        weekday_target = 1  # Tue
    elif sym in ("FINNIFTY", "MIDCPNIFTY"):
        weekday_target = 1  # Tue
    elif sym == "SENSEX":
        weekday_target = 3 if d >= eff else 1  # Thu after eff else Tue
    # else: keep default Thursday
    days_ahead = (weekday_target - d.weekday()) % 7
    if days_ahead==0 and now_ist.time() > dt.time(15,30):
        days_ahead = 7
    return (d + dt.timedelta(days=days_ahead)).isoformat()

def minutes_to_expiry(expiry_iso: str) -> float:
    d = dt.date.fromisoformat(expiry_iso)
    expiry_dt = IST.localize(dt.datetime(d.year,d.month,d.day,15,30))
    now = dt.datetime.now(IST)
    return max(0.0, (expiry_dt - now).total_seconds()/60.0)

def atm_strike(
    spot: float,
    strikes: List[int],
    symbol: str = "",
    chain: Dict | None = None,
    fut_mid: float | None = None,
    minutes_to_exp: float | None = None,
    r: float | None = None,
    q: float = 0.0,
) -> int:
    """Return the forward-based at-the-money strike.

    Parameters
    ----------
    spot: float
        Current underlying spot price.
    strikes: list[int]
        Available strikes from the option chain.
    symbol: str, optional
        Underlying symbol used for strike-step fallback when ``strikes`` are
        insufficient to infer the spacing.
    chain: dict, optional
        Full option chain; when supplied call/put mids are used for the
        straddle-based tie-breaker.
    fut_mid: float, optional
        Mid-price of the matching futures contract.  When omitted the forward
        is derived from ``spot`` and ``r``/``q``.
    minutes_to_exp: float, optional
        Minutes remaining to expiry used for forward calculation.
    r: float, optional
        Annualised risk-free rate.  Defaults to :func:`get_rfr`.
    q: float, default 0.0
        Dividend yield estimate.
    """
    if not strikes:
        return 0
    strikes = sorted(int(k) for k in strikes)
    step = detect_strike_step(strikes) or STEP_MAP.get(symbol.upper(), 50)
    tau_y = (minutes_to_exp or 0.0) / (365 * 24 * 60)
    r = r if r is not None else get_rfr()
    F = compute_forward(spot, fut_mid, r, q, tau_y)
    if step > 0:
        if F > strikes[-1]:
            return int(math.ceil(F / step) * step)
        if F < strikes[0]:
            return int(strikes[0])
    ce_mid: Dict[float, float] = {}
    pe_mid: Dict[float, float] = {}
    if chain:
        for k in strikes:
            ce_mid[k] = _mid_price(chain.get("calls", {}).get(k, {}))
            pe_mid[k] = _mid_price(chain.get("puts", {}).get(k, {}))
    K, _ = pick_atm_strike(F, strikes, step, ce_mid, pe_mid)
    return int(K)


# Backwards compatibility – alias old name
def atm_strike_with_tie_high(spot: float, strikes: List[int]) -> int:
    return atm_strike(spot, strikes)

def pcr_from_chain(
    chain: Dict,
    spot: float | None = None,
    symbol: str = "",
    band_steps: int = 6,
) -> Dict[str, float]:
    """Return put/call ratios over total OI and an ATM-centred band."""

    strikes = chain.get("strikes") or []
    if not strikes:
        return {
            "PCR_OI_total": float("nan"),
            "PCR_OI_band": float("nan"),
            "band_lo": float("nan"),
            "band_hi": float("nan"),
            "band_count": 0,
        }
    strikes = sorted(int(k) for k in strikes)
    step = detect_strike_step(strikes) or STEP_MAP.get(symbol.upper(), 50)
    oi_put = {k: float(chain["puts"].get(k, {}).get("oi", 0)) for k in strikes}
    oi_call = {k: float(chain["calls"].get(k, {}).get("oi", 0)) for k in strikes}
    if spot is None:
        tot_p = sum(oi_put.values())
        tot_c = sum(oi_call.values())
        return {
            "PCR_OI_total": (tot_p / tot_c) if tot_c > 0 else float("nan"),
            "PCR_OI_band": float("nan"),
            "band_lo": float("nan"),
            "band_hi": float("nan"),
            "band_count": 0,
        }
    atm = atm_strike(spot, strikes, symbol, chain=chain)
    res = compute_pcr(oi_put, oi_call, strikes, atm, step, m=band_steps)
    return res

def gamma_exposure(
    chain: Dict,
    spot: float,
    minutes_to_exp: float,
    atm_iv: float,
    r: float = get_rfr(),
) -> tuple[float, Dict[int, float]]:
    """Compute dealer gamma exposure and zero-gamma level.

    Parameters
    ----------
    r : float, default ``get_rfr()``
        Annualised risk-free rate as a decimal.
    """
    strikes = chain.get("strikes") or []
    if not strikes:
        return float("nan"), {}
    T = max(1e-9, minutes_to_exp) / (365 * 24 * 60)
    sig = atm_iv if atm_iv == atm_iv and atm_iv > 0 else 0.2
    gex_map: Dict[int, float] = {}
    for k in strikes:
        ce = chain["calls"].get(k, {})
        pe = chain["puts"].get(k, {})
        oi_tot = float(ce.get("oi", 0) + pe.get("oi", 0))
        skew = 1.0 if ce.get("oi", 0) >= pe.get("oi", 0) else -1.0
        gamma = bs_gamma(spot, k, r, T, sig)
        gex_map[int(k)] = oi_tot * gamma * skew
    zg = float("nan")
    if gex_map:
        ks = sorted(gex_map.keys())
        cum = 0.0
        for i, k in enumerate(ks[:-1]):
            cum += gex_map[k]
            if cum <= 0 and cum + gex_map[ks[i + 1]] >= 0:
                zg = float(ks[i + 1])
                break
    return zg, gex_map

def max_pain(chain: Dict, spot: float = 0.0, step: int | None = None) -> int:
    """Return the strike where option writers experience minimal loss."""

    strikes = sorted(int(k) for k in chain.get("strikes", []))
    if not strikes:
        return 0
    ce_oi = {k: float(chain.get("calls", {}).get(k, {}).get("oi", 0)) for k in strikes}
    pe_oi = {k: float(chain.get("puts", {}).get(k, {}).get("oi", 0)) for k in strikes}
    bestK, bestPain = None, float("inf")
    for K in strikes:
        pain = sum(ce_oi[s] * max(0, K - s) for s in strikes) + \
               sum(pe_oi[s] * max(0, s - K) for s in strikes)
        if pain < bestPain or (pain == bestPain and bestK is not None and abs(K - spot) < abs(bestK - spot)):
            bestPain, bestK = pain, K
    if bestK is None:
        return 0
    if step is None or step <= 0:
        step = detect_strike_step(strikes) or 50
    while spot > bestK and (bestK + step) not in strikes:
        bestK += step
    return int(bestK)

def atm_iv_from_chain(
    chain: Dict,
    spot: float,
    minutes_to_exp: float,
    risk_free_rate: float,
    symbol: str = "",
    fut_mid: float | None = None,
    dividend_yield: float = 0.0,
) -> float:
    """Infer ATM implied volatility from an option chain using forward pricing."""

    strikes = chain.get("strikes") or []
    if not strikes:
        return float("nan")
    strikes = sorted(int(k) for k in strikes)
    step = detect_strike_step(strikes) or STEP_MAP.get(symbol.upper(), 50)
    tau_y = max(1e-9, minutes_to_exp) / (365 * 24 * 60)
    ce_mid = {k: _mid_price(chain.get("calls", {}).get(k, {})) for k in strikes}
    pe_mid = {k: _mid_price(chain.get("puts", {}).get(k, {})) for k in strikes}
    F = compute_forward(spot, fut_mid, risk_free_rate, dividend_yield, tau_y)
    K, _ = pick_atm_strike(F, strikes, step, ce_mid, pe_mid)
    atm_iv, _ = compute_atm_iv(
        ce_mid.get(K),
        pe_mid.get(K),
        spot,
        K,
        tau_y,
        risk_free_rate,
        dividend_yield,
        F,
    )
    return float(atm_iv) if atm_iv is not None else float("nan")
