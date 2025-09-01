from __future__ import annotations
import math
import datetime as dt
from typing import Dict, List

import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")

def bs_price(S,K,r,T,sig,call=True):
    if T<=0 or sig<=0: return 0.0
    d1 = (math.log(S/K) + (r+0.5*sig*sig)*T)/(sig*math.sqrt(T))
    d2 = d1 - sig*math.sqrt(T)
    N = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
    return (S*N(d1) - K*math.exp(-r*T)*N(d2)) if call else (K*math.exp(-r*T)*N(-d2) - S*N(-d1))

def implied_vol(price,S,K,r,T,call=True):
    if price<=0 or S<=0 or K<=0 or T<=0: return float('nan')
    lo, hi = 0.03, 1.5
    for _ in range(60):
        mid = 0.5*(lo+hi)
        pm = bs_price(S,K,r,T,mid,call)
        if abs(pm-price) < 1e-3: return mid
        if pm > price: hi = mid
        else: lo = mid
    return mid

def bs_delta(S: float, K: float, r: float, T: float, sig: float, call: bool = True) -> float:
    if S<=0 or K<=0 or T<=0 or sig<=0:
        return 0.0
    d1 = (math.log(S/K) + (r+0.5*sig*sig)*T)/(sig*math.sqrt(T))
    if call:
        # N(d1)
        return 0.5*(1+math.erf(d1/math.sqrt(2)))
    else:
        # N(d1)-1
        return 0.5*(1+math.erf(d1/math.sqrt(2))) - 1.0

def bs_gamma(S: float, K: float, r: float, T: float, sig: float) -> float:
    if S<=0 or K<=0 or T<=0 or sig<=0:
        return 0.0
    d1 = (math.log(S/K) + (r+0.5*sig*sig)*T) / (sig*math.sqrt(T))
    return math.exp(-0.5*d1*d1) / (S*sig*math.sqrt(2*math.pi*T))

def risk_reversal_25(chain: Dict, spot: float, minutes_to_exp: float, r: float, atm_iv: float) -> Dict[str, float]:
    """
    Approximate 25-delta risk reversal: RR = IV_call(25d) - IV_put(25d)
    Uses ATM IV to select strikes with delta≈±0.25, then backs out IV from market prices.
    Returns { 'rr': float, 'k_call': int, 'k_put': int, 'iv_call': float, 'iv_put': float }
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

def atm_strike_with_tie_high(spot: float, strikes: List[int]) -> int:
    """Return the smallest strike that is >= ``spot``.

    The option-chain data returned by brokers occasionally omits strikes that
    are far from the money.  When the underlying rallies beyond the highest
    strike in the chain our previous implementation simply returned that last
    available value, leading to large ATM mismatches (e.g. engine reporting
    53800 when the underlying was trading above 54100).  To stay aligned with
    vendors such as Sensibull we always *round up* to the next strike using the
    observed strike spacing when we run out of data.

    Parameters
    ----------
    spot: float
        Current underlying price.
    strikes: list[int]
        Strike prices available in the option chain.  The function infers the
        strike spacing from this list and will extrapolate beyond its bounds if
        required.
    """
    if not strikes:
        return 0

    strikes = sorted(int(k) for k in strikes)
    higher = [k for k in strikes if k >= spot]
    if higher:
        return higher[0]

    # Spot is above the highest available strike.  Infer the step size and
    # project the next valid strike rather than returning the truncated value.
    if len(strikes) >= 2:
        step = min(b - a for a, b in zip(strikes, strikes[1:])) or 50
    else:
        step = 50
    return int(math.ceil(spot / step) * step)

def pcr_from_chain(chain: Dict) -> float:
    ce = sum(v['oi'] for v in chain['calls'].values())
    pe = sum(v['oi'] for v in chain['puts'].values())
    return (pe/ce) if ce>0 else float('nan')

def gamma_exposure(chain: Dict, spot: float, minutes_to_exp: float, atm_iv: float, r: float = 0.0) -> tuple[float, Dict[int,float]]:
    strikes = chain.get('strikes') or []
    if not strikes:
        return float('nan'), {}
    T = max(1e-9, minutes_to_exp) / (365*24*60)
    sig = atm_iv if atm_iv==atm_iv and atm_iv>0 else 0.2
    gex_map: Dict[int,float] = {}
    for k in strikes:
        ce = chain['calls'].get(k, {})
        pe = chain['puts'].get(k, {})
        oi_tot = float(ce.get('oi',0) + pe.get('oi',0))
        skew = 1.0 if ce.get('oi',0) >= pe.get('oi',0) else -1.0
        gamma = bs_gamma(spot, k, r, T, sig)
        gex_map[int(k)] = oi_tot * gamma * skew
    zg = float('nan')
    if gex_map:
        ks = sorted(gex_map.keys())
        cum = 0.0
        for i,k in enumerate(ks[:-1]):
            cum += gex_map[k]
            if cum <= 0 and cum + gex_map[ks[i+1]] >= 0:
                zg = float(ks[i+1])
                break
    return zg, gex_map

def max_pain(chain: Dict) -> int:
    strikes = sorted(chain['strikes'])
    ce_oi = {k: chain['calls'][k]['oi'] for k in strikes}
    pe_oi = {k: chain['puts'][k]['oi'] for k in strikes}
    bestK, bestPain = None, float('inf')
    for K in strikes:
        pain = (
            sum(ce_oi[s] * max(0, K - s) for s in strikes) +
            sum(pe_oi[s] * max(0, s - K) for s in strikes)
        )
        if pain < bestPain: bestPain, bestK = pain, K
    return int(bestK)

def atm_iv_from_chain(chain: Dict, spot: float, minutes_to_exp: float, risk_free_rate: float) -> float:
    strikes = chain['strikes']
    if not strikes: return float('nan')
    K = atm_strike_with_tie_high(spot, strikes)
    ce = chain['calls'].get(K); pe = chain['puts'].get(K)
    T = max(1e-9, minutes_to_exp) / (365*24*60)
    ivs=[]
    now = dt.datetime.now(IST)
    for row, is_call in [(ce, True), (pe, False)]:
        if not row:
            continue
        bid = row.get("bid", 0.0); ask = row.get("ask", 0.0)
        bq = row.get("bid_qty", 0.0); aq = row.get("ask_qty", 0.0)
        ltp = row.get("ltp", 0.0)
        ts_str = row.get("ltp_ts")
        ltt = pd.to_datetime(ts_str) if ts_str else None
        price = None
        if bid>0 and ask>0 and bq>0 and aq>0:
            price = (ask*bq + bid*aq) / (bq + aq)
        elif bid>0 and ask>0 and (ask-bid)/max(1,K) <= 0.006 and bq>0 and aq>0:
            price = 0.5*(bid+ask)
        elif ltp>0 and ltt is not None and abs((now - ltt).total_seconds()) <= 60:
            price = ltp
        if price:
            ivs.append(implied_vol(price, spot, K, risk_free_rate, T, call=is_call))
    ivs=[v for v in ivs if v==v and v>0]
    if not ivs:
        return float('nan')
    # Use the average of call/put IVs when both sides are available.  The
    # previous implementation returned the minimum which consistently
    # under-reported volatility when one side of the book was stale or
    # mispriced (e.g. wide put quotes).  Sensibull and other vendors average
    # the two, yielding figures that more closely reflect the true ATM level.
    return sum(ivs)/len(ivs)
