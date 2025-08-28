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
        weekday_target = 1 #if d >= eff else 3  # Tue after eff else Thu
    elif sym == "BANKNIFTY":
        weekday_target = 1  # Tue
    elif sym in ("FINNIFTY", "MIDCPNIFTY"):
        weekday_target = 1  # Tue
    elif sym == "SENSEX":
        weekday_target = 3 #if d >= eff else 1  # Thu after eff else Tue
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
    return min(strikes, key=lambda k: (abs(spot - k), -k))

def pcr_from_chain(chain: Dict) -> float:
    ce = sum(v['oi'] for v in chain['calls'].values())
    pe = sum(v['oi'] for v in chain['puts'].values())
    return (pe/ce) if ce>0 else float('nan')

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
    for row, is_call in [(ce, True), (pe, False)]:
        if not row: continue
        bid, ask, ltp = row.get("bid",0), row.get("ask",0), row.get("ltp",0)
        mid = 0.5*(bid+ask) if (bid>0 and ask>0 and (ask-bid)/max(1,K) <= 0.006) else (ltp if ltp>0 else None)
        if mid:
            ivs.append(implied_vol(mid, spot, K, risk_free_rate, T, call=is_call))
    ivs=[v for v in ivs if v==v and v>0]
    if not ivs: return float('nan')
    return min(ivs) if len(ivs)==2 else ivs[0]
