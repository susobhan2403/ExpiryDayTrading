#!/usr/bin/env python3
"""
Expiry Day Engine (Kite provider + Normalized Framework)
- Uses Zerodha Kite data with SAME auth flow: reads .kite_session.json made by your existing get_access_token.py
- Env: KITE_API_KEY required here (KITE_API_SECRET only used by your auth script)
- Implements normalization:
  * D = Spot - MaxPain
  * ATR_D (session range proxy) and VND = |D| / ATR_D
  * SSD, PD
  * MaxPain drift speed & MPH_norm = (pts/hr)/ATR_D
  * ΔPCR z-score (rolling 20 snaps)
  * IV spike z-score (rolling 20 snaps)
  * OI writing/unwinding via MAD(ΔOI) with liquidity guard (bid/ask present)
- Guardrails: expiry sanity, ATM tie->higher strike, liquidity filter, block confirmations,
  two-snapshot confirmation for OI/drift, scenario flip gate (>=0.15 score delta)
- Index-agnostic scenarios with weights: Price/Trend 0.35, Options Flow 0.45, Vol 0.20
- Persists: ./out/snapshots/*.json, ./out/*_rollup.csv, logs/engine.log
"""

from __future__ import annotations
import os, sys, json, time, math, argparse, logging, pathlib, itertools, statistics
import datetime as dt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import pytz
import requests
from logging.handlers import RotatingFileHandler
from colorama import init as colorama_init, Fore, Style

# ---------- paths & logging ----------
IST = pytz.timezone("Asia/Kolkata")
ROOT = pathlib.Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"
SNAP_DIR = OUT_DIR / "snapshots"
LOG_DIR = ROOT / "logs"
for p in (OUT_DIR, SNAP_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)

colorama_init(autoreset=True)

# Ensure stdout/stderr can handle Unicode characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


logger = logging.getLogger("engine")
logger.setLevel(logging.INFO)

fh = RotatingFileHandler(LOG_DIR / "engine.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(ColorFormatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S"))
logger.addHandler(sh)

# ---------- cli defaults / env ----------
DEFAULT_SYMBOLS = ["BANKNIFTY"]
DEFAULT_PROVIDER = "KITE"   # ← use Kite by default per your request
DEFAULT_POLL_SECS = 240
RISK_FREE = float(os.getenv("RISK_FREE_RATE", "0.066"))

# Scenario weighting and gates
WEIGHTS = {"price_trend": 0.35, "options_flow": 0.45, "volatility": 0.20}
GATE_MIN_BLOCKS = {"price_trend": 1, "options_flow": 1, "volatility": 1}
SCENARIO_FLIP_DELTA = 0.15
CONFIRM_SNAPSHOTS = 2         # require consecutive confirmations for OI/drift regimes
MAD_K = float(os.getenv("OI_DELTA_K", "1.5"))

ADX_LOW = int(os.getenv("ADX_THRESHOLD_LOW", "15"))
PIN_NORM = float(os.getenv("PIN_DISTANCE_NORM", "0.4"))     # VND < 0.4
REV_NORM = float(os.getenv("REVERSION_NORM", "1.0"))        # VND >= 1.0
SQUEEZE_NORM = float(os.getenv("SQUEEZE_DISTANCE_NORM", "1.5"))
MPH_NORM_THR = float(os.getenv("MAXPAIN_DRIFT_NORM_THR", "0.6"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK", "")

# ---------- indicators ----------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    rs = up.ewm(alpha=1/length, adjust=False).mean() / dn.ewm(alpha=1/length, adjust=False).mean().replace(0,np.nan)
    return 100 - 100/(1+rs)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    line = ema(series, fast) - ema(series, slow)
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def adx(df: pd.DataFrame, length=14) -> pd.Series:
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,np.nan)
    return dx.ewm(alpha=1/length, adjust=False).mean()

def session_vwap(df: pd.DataFrame) -> float:
    tp = (df['high']+df['low']+df['close'])/3.0
    v = df['volume'].replace(0, np.nan)
    vwap = (tp*v).cumsum() / v.cumsum()
    return float(vwap.iloc[-1]) if len(vwap) else float('nan')

def volume_profile(df: pd.DataFrame, bins: int = 50) -> Dict[str,float]:
    if df.empty: return {"POC": float('nan'), "VAL": float('nan'), "VAH": float('nan')}
    lo, hi = df['low'].min(), df['high'].max()
    edges = np.linspace(lo, hi, bins+1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    idx = np.clip(np.digitize(df['close'], edges) - 1, 0, bins-1)
    vol_bins = np.zeros(bins)
    for i, vol in zip(idx, df['volume'].values):
        vol_bins[int(i)] += vol
    poc_idx = int(vol_bins.argmax())
    total = vol_bins.sum()
    target = 0.7 * total
    cum = vol_bins[poc_idx]; left = right = poc_idx
    while cum < target and (left>0 or right<bins-1):
        left_cand = max(0, left-1); right_cand = min(bins-1, right+1)
        if left==0: right = right_cand; cum += vol_bins[right]
        elif right==bins-1: left = left_cand; cum += vol_bins[left]
        else:
            if vol_bins[left_cand] >= vol_bins[right_cand]:
                left = left_cand; cum += vol_bins[left]
            else:
                right = right_cand; cum += vol_bins[right]
    return {"POC": float(centers[poc_idx]), "VAL": float(centers[left]), "VAH": float(centers[right])}

# ---------- options/math ----------
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

def nearest_weekly_expiry(now_ist: dt.datetime) -> str:
    d = now_ist.date()
    days_ahead = (3 - d.weekday()) % 7  # Thursday
    if days_ahead==0 and now_ist.time() > dt.time(15,30): days_ahead = 7
    return (d + dt.timedelta(days=days_ahead)).isoformat()

def minutes_to_expiry(expiry_iso: str) -> float:
    d = dt.date.fromisoformat(expiry_iso)
    expiry_dt = IST.localize(dt.datetime(d.year,d.month,d.day,15,30))
    now = dt.datetime.now(IST)
    return max(0.0, (expiry_dt - now).total_seconds()/60.0)

def atm_strike_with_tie_high(spot: float, strikes: List[int]) -> int:
    # tie → higher strike
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
        # Option value at settlement K for strike s:
        #   Calls -> max(0, K - s)
        #   Puts  -> max(0, s - K)
        pain = (
            sum(ce_oi[s] * max(0, K - s) for s in strikes) +
            sum(pe_oi[s] * max(0, s - K) for s in strikes)
        )
        if pain < bestPain: bestPain, bestK = pain, K
    return int(bestK)

def atm_iv_from_chain(chain: Dict, spot: float, minutes_to_exp: float) -> float:
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
            ivs.append(implied_vol(mid, spot, K, RISK_FREE, T, call=is_call))
    ivs=[v for v in ivs if v==v and v>0]
    if not ivs: return float('nan')
    return min(ivs) if len(ivs)==2 else ivs[0]

# ---------- providers ----------
class MarketDataProvider:
    def get_spot_ohlcv(self, symbol: str, interval: str, lookback_minutes: int) -> pd.DataFrame: ...
    def get_futures_ohlcv(self, symbol: str, interval: str, expiry: str, lookback_minutes: int) -> pd.DataFrame: ...
    def get_option_chain(self, symbol: str, expiry: str) -> Dict: ...
    def get_indices_snapshot(self, symbols: List[str]) -> Dict[str, float]: ...

class KiteProvider(MarketDataProvider):
    """Zerodha Kite provider using existing auth: .kite_session.json + KITE_API_KEY."""
    def __init__(self):
        try:
            from kiteconnect import KiteConnect
        except Exception:
            logger.error("kiteconnect not installed. pip install kiteconnect")
            raise
        if not os.path.exists(".kite_session.json"):
            raise SystemExit("Run get_access_token.py first (creates .kite_session.json).")
        sess = json.loads(open(".kite_session.json","r").read())
        api_key = os.getenv("KITE_API_KEY")
        if not api_key:
            raise SystemExit("Set KITE_API_KEY env var.")
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(sess["access_token"])
        self._nfo = None
        self._nse = None

    def _instruments(self, exch: str) -> pd.DataFrame:
        if exch=="NFO":
            if self._nfo is None:
                self._nfo = pd.DataFrame(self.kite.instruments("NFO"))
            return self._nfo
        else:
            if self._nse is None:
                self._nse = pd.DataFrame(self.kite.instruments("NSE"))
            return self._nse

    @staticmethod
    def _index_ltp_key(symbol: str) -> str:
        m = symbol.upper()
        if m=="BANKNIFTY": return "NSE:NIFTY BANK"
        if m=="NIFTY": return "NSE:NIFTY 50"
        if m=="FINNIFTY": return "NSE:NIFTY FIN SERVICE"  # fallback; if fails we’ll proxy via fut
        if m=="MIDCPNIFTY": return "NSE:NIFTY MID SELECT" # fallback
        return "NSE:NIFTY 50"

    def _resolve_index_token(self, symbol: str) -> Optional[int]:
        # try to find index token in NSE instruments
        df = self._instruments("NSE")
        # common tokens: NIFTY 50, NIFTY BANK
        name_map = {"NIFTY":"NIFTY 50", "BANKNIFTY":"NIFTY BANK",
                    "FINNIFTY":"NIFTY FIN SERVICE", "MIDCPNIFTY":"NIFTY MID SELECT"}
        ts = name_map.get(symbol.upper(), "NIFTY 50")
        m = df[df["tradingsymbol"].str.upper()==ts.upper()]
        if not m.empty:
            return int(m.iloc[0]["instrument_token"])
        return None

    def _nearest_weekly_chain_df(self, symbol: str) -> Tuple[pd.DataFrame, str]:
        nfo = self._instruments("NFO").copy()
        df = nfo[(nfo["name"]==symbol.upper()) & (nfo["segment"]=="NFO-OPT")].copy()
        if df.empty:
            raise RuntimeError(f"No NFO options for {symbol}")
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
        today = dt.datetime.now(IST).date()
        exps = sorted([e for e in df["expiry"].unique() if e >= today])
        if not exps: raise RuntimeError("No future expiries")
        expiry = exps[0]
        chain = df[df["expiry"]==expiry].copy()
        chain["strike"] = chain["strike"].astype(int)
        return chain, expiry.isoformat()

    def _nearest_future_row(self, symbol: str) -> pd.Series:
        nfo = self._instruments("NFO").copy()
        futs = nfo[(nfo["name"]==symbol.upper()) & (nfo["instrument_type"]=="FUT")].copy()
        futs["expiry"] = pd.to_datetime(futs["expiry"]).dt.date
        today = dt.datetime.now(IST).date()
        futs = futs[futs["expiry"]>=today].sort_values("expiry")
        if futs.empty: raise RuntimeError(f"No futures for {symbol}")
        return futs.iloc[0]

    def _hist(self, token: int, interval: str, lookback_minutes: int, is_future: bool = False) -> pd.DataFrame:
        # Kite expects naive datetimes
        now = dt.datetime.now().replace(tzinfo=None)
        start = now - dt.timedelta(minutes=max(lookback_minutes + 15, 240))
        data = []
        try:
            data = self.kite.historical_data(
                token, start, now, "minute",
                continuous=False,  # continuous minute candles unsupported
                oi=False
            )
        except Exception as e:
            logger.warning(f"[hist] token={token} error: {e}")
            data = []

        if not data:
            # Retry with a wider window (2 days) — sometimes minute API can be sparse around session boundaries
            try:
                alt_start = now - dt.timedelta(days=2)
                data = self.kite.historical_data(
                    token, alt_start, now, "minute",
                    continuous=False,
                    oi=False
                )
            except Exception as e:
                logger.warning(f"[hist-retry] token={token} error: {e}")
                data = []

        if not data:
            return pd.DataFrame(columns=["open","high","low","close","volume"])

        df = pd.DataFrame(data)

        # Handle tz-awareness safely (Kite may already return tz-aware datetimes)
        dates = pd.to_datetime(df["date"])
        try:
            if dates.dt.tz is None:
                dates = dates.dt.tz_localize(dt.timezone.utc).dt.tz_convert(IST)
            else:
                dates = dates.dt.tz_convert(IST)
        except Exception:
            # last resort: assume UTC then convert
            dates = pd.to_datetime(df["date"]).tz_localize(dt.timezone.utc).tz_convert(IST)

        df = df.set_index(dates)[["open","high","low","close","volume"]]
        df.index.name = "date"

        if interval == "5m":
            df = df.resample("5min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(how="any")

        return df.tail(lookback_minutes)


    def get_spot_ohlcv(self, symbol: str, interval: str, lookback_minutes: int) -> pd.DataFrame:
        tok = self._resolve_index_token(symbol)
        df = pd.DataFrame()
        if tok:
            df = self._hist(tok, interval, lookback_minutes, is_future=False)
        if df.empty:
            # Fallback to futures for spot technicals if index candles aren’t available
            fut = self._nearest_future_row(symbol)
            df = self._hist(int(fut["instrument_token"]), interval, lookback_minutes, is_future=True)
            if df.empty:
                logger.warning(f"{symbol}: spot proxy via futures also empty (fut token {int(fut['instrument_token'])}).")
        return df

    def get_futures_ohlcv(self, symbol: str, interval: str, expiry: str, lookback_minutes: int) -> pd.DataFrame:
        fut = self._nearest_future_row(symbol)
        return self._hist(int(fut["instrument_token"]), interval, lookback_minutes, is_future=True)


    def get_option_chain(self, symbol: str, expiry: str) -> Dict:
        chain_df, exp_sel = self._nearest_weekly_chain_df(symbol)
        # expiry sanity guard
        if exp_sel != expiry:
            raise RuntimeError(f"Expiry mismatch chain={exp_sel} selected={expiry}")
        # batch quote
        syms = [f"NFO:{ts}" for ts in chain_df["tradingsymbol"].tolist()]
        chain = {"symbol": symbol, "expiry": expiry, "strikes": sorted(chain_df["strike"].unique()),
                 "calls": {}, "puts": {}}
        for i in range(0, len(syms), 450):
            q = self.kite.quote(syms[i:i+450])
            for v in q.values():
                tok = v["instrument_token"]
                row = chain_df[chain_df["instrument_token"]==tok]
                if row.empty: continue
                strike = int(row.iloc[0]["strike"])
                typ = row.iloc[0]["instrument_type"]
                bid=ask=ltp=0.0
                dep = v.get("depth")
                if dep and dep.get("buy") and dep.get("sell"):
                    try:
                        bid = float(dep["buy"][0]["price"]); ask = float(dep["sell"][0]["price"])
                    except: pass
                ltp = float(v.get("last_price") or 0.0)
                oi  = int(v.get("oi") or v.get("open_interest") or 0)
                node = {"oi": oi, "ltp": ltp, "bid": bid, "ask": ask}
                if typ=="CE": chain["calls"][strike]=node
                else: chain["puts"][strike]=node
        # ensure all strikes have nodes
        for k in chain["strikes"]:
            chain["calls"].setdefault(k, {"oi":0,"ltp":0.0,"bid":0.0,"ask":0.0})
            chain["puts"].setdefault(k,  {"oi":0,"ltp":0.0,"bid":0.0,"ask":0.0})
        return chain

    def get_indices_snapshot(self, symbols: List[str]) -> Dict[str, float]:
        keys = []
        map_key = {}
        for s in symbols:
            k = self._index_ltp_key(s)
            keys.append(k); map_key[k]=s
        out = {}
        try:
            q = self.kite.ltp(keys)
            for k,v in q.items():
                out[map_key[k]] = float(v["last_price"])
        except Exception as e:
            logger.warning(f"ltp error: {e}")
        # fallback via futures if ltp failed
        for s in symbols:
            if s not in out:
                try:
                    fut = self._nearest_future_row(s)
                    q = self.kite.quote([f"NFO:{fut['tradingsymbol']}"])
                    out[s] = float(list(q.values())[0]["last_price"])
                except: out[s] = float('nan')
        return out

# ---------- state & helpers ----------
@dataclass
class Snapshot:
    ts: str
    symbol: str
    spot: float
    vwap_spot: float
    vwap_fut: float
    rsi5: float
    macd: float
    macd_signal: float
    adx5: float
    pcr: float
    dpcr: float
    dpcr_z: float
    maxpain: int
    dist_to_mp: float
    atr_d: float
    vnd: float
    mph_pts_per_hr: float
    mph_norm: float
    atm: int
    atm_iv: float
    div: float
    iv_z: float
    iv_pct_hint: float
    basis: float
    ssd: float
    pdist_pct: float
    scen_probs: Dict[str,float]
    scen_top: str
    trade: Dict
    eq_line: str

def market_open_now() -> bool:
    now = dt.datetime.now(IST)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    return dt.time(9,15) <= now.time() <= dt.time(15,30)

def softmax(x: List[float]) -> List[float]:
    m = max(x); ex = [math.exp(v-m) for v in x]; s=sum(ex) or 1.0
    return [v/s for v in ex]

# ---------- normalized evidence & scoring ----------
SCENARIOS = [
    "Short-cover reversion up",
    "Bear migration",
    "Bull migration / gamma carry",
    "Pin & decay day (IV crush)",
    "Squeeze continuation (one-way)",
    "Event knee-jerk then revert",
]

def score_normalized(
    symbol: str,
    D: float, ATR_D: float, VND: float, SSD: float, PD: float,
    pcr: float, dpcr: float, dpcr_z: float,
    maxpain_drift_pts_per_hr: float, mph_norm: float,
    atm_iv: float, div: float, iv_z: float, iv_pct_hint: float,
    spot: float, vwap: float, adx5: float,
    oi_flags: Dict[str,bool],
    confirmations: Dict[str,int],
) -> Tuple[Dict[str,float], Dict[str,Dict[str,bool]]]:
    """
    Returns: (probs_by_scenario, block_flags_by_scenario)
    block flags ensure at least one signal in each: price_trend, options_flow, volatility
    """
    above_vwap = spot > vwap if (spot==spot and vwap==vwap) else False
    below_vwap = spot < vwap if (spot==spot and vwap==vwap) else False
    macd_up = True  # placeholder: MACD cross assessed in caller if needed
    price_block = {}
    flow_block = {}
    vol_block = {}

    # OI regime confirmations (two snapshots)
    def confirmed(flag_name): return confirmations.get(flag_name,0) >= CONFIRM_SNAPSHOTS

    ce_write_abv = confirmed("ce_write_above")
    pe_unw_bel   = confirmed("pe_unwind_below")
    pe_write_abv = confirmed("pe_write_above")
    ce_unw_bel   = confirmed("ce_unwind_below")

    # Price/Trend flags
    price_block["reversion"] = (VND >= REV_NORM and D < 0 and (above_vwap or macd_up))
    price_block["bear"]      = (below_vwap and adx5 >= 18)
    price_block["bull"]      = (above_vwap and adx5 >= 18)
    price_block["pin"]       = (VND < PIN_NORM and adx5 < ADX_LOW)
    price_block["squeeze"]   = (VND >= SQUEEZE_NORM and adx5 >= 20 and (above_vwap or below_vwap))
    price_block["event"]     = (abs(div) > 0 and abs(iv_z) >= 2.0)

    # Options Flow flags
    flow_block["reversion"] = (ce_unw_bel and pe_write_abv)
    flow_block["bear"]      = (ce_write_abv and pe_unw_bel and mph_norm <= -MPH_NORM_THR)
    flow_block["bull"]      = (pe_write_abv and ce_unw_bel and mph_norm >= MPH_NORM_THR)
    flow_block["pin"]       = (oi_flags.get("two_sided_adjacent", False))
    flow_block["squeeze"]   = (oi_flags.get("same_side_add", False) and not oi_flags.get("unwind_present", False))
    flow_block["event"]     = ( (pe_write_abv and ce_unw_bel) or (ce_write_abv and pe_unw_bel) ) and (abs(dpcr_z) < 0.5)

    # Volatility flags
    vol_block["reversion"] = (div <= 0 or (iv_pct_hint < 50))
    vol_block["bear"]      = (dpcr_z <= -1.0)
    vol_block["bull"]      = (dpcr_z >= +1.0)
    vol_block["pin"]       = (iv_pct_hint < 25 and div <= 0)
    vol_block["squeeze"]   = (iv_z >= +0.5)   # IV rising
    vol_block["event"]     = (iv_z >= +2.0 and div < 0)  # spike then cools (neg div)

    # scoring per scenario (sum normalized evidences within blocks)
    def block_score(flags: Dict[str,bool], keys: List[str]) -> float:
        return sum(1.0 for k in keys if flags.get(k, False)) / max(1,len(keys))

    scores = {
        "Short-cover reversion up": WEIGHTS["price_trend"]*float(price_block["reversion"])
                                    + WEIGHTS["options_flow"]*float(flow_block["reversion"])
                                    + WEIGHTS["volatility"]*float(vol_block["reversion"]),
        "Bear migration": WEIGHTS["price_trend"]*float(price_block["bear"])
                                    + WEIGHTS["options_flow"]*float(flow_block["bear"])
                                    + WEIGHTS["volatility"]*float(vol_block["bear"]),
        "Bull migration / gamma carry": WEIGHTS["price_trend"]*float(price_block["bull"])
                                    + WEIGHTS["options_flow"]*float(flow_block["bull"])
                                    + WEIGHTS["volatility"]*float(vol_block["bull"]),
        "Pin & decay day (IV crush)": WEIGHTS["price_trend"]*float(price_block["pin"])
                                    + WEIGHTS["options_flow"]*float(flow_block["pin"])
                                    + WEIGHTS["volatility"]*float(vol_block["pin"]),
        "Squeeze continuation (one-way)": WEIGHTS["price_trend"]*float(price_block["squeeze"])
                                    + WEIGHTS["options_flow"]*float(flow_block["squeeze"])
                                    + WEIGHTS["volatility"]*float(vol_block["squeeze"]),
        "Event knee-jerk then revert": WEIGHTS["price_trend"]*float(price_block["event"])
                                    + WEIGHTS["options_flow"]*float(flow_block["event"])
                                    + WEIGHTS["volatility"]*float(vol_block["event"]),
    }

    # block gating: require at least one signal from each block → else cap at 0.49
    block_ok = {}
    for name in SCENARIOS:
        if name=="Short-cover reversion up":
            ok = price_block["reversion"] and flow_block["reversion"] and vol_block["reversion"]
        elif name=="Bear migration":
            ok = price_block["bear"] and flow_block["bear"] and vol_block["bear"]
        elif name=="Bull migration / gamma carry":
            ok = price_block["bull"] and flow_block["bull"] and vol_block["bull"]
        elif name=="Pin & decay day (IV crush)":
            ok = price_block["pin"] and flow_block["pin"] and vol_block["pin"]
        elif name=="Squeeze continuation (one-way)":
            ok = price_block["squeeze"] and flow_block["squeeze"] and vol_block["squeeze"]
        else:
            ok = price_block["event"] and flow_block["event"] and vol_block["event"]
        block_ok[name] = ok
        if not ok:
            scores[name] = min(scores[name], 0.49)

    probs = softmax(list(scores.values()))
    return ({k: round(v,3) for k,v in zip(scores.keys(), probs)}, block_ok)

# ---------- main cycle ----------
def run_once(provider: MarketDataProvider, symbol: str, poll_secs: int, use_telegram: bool, slack_webhook: str) -> Optional[Snapshot]:
    now = dt.datetime.now(IST)
    expiry = nearest_weekly_expiry(now)

    # OHLCV
    lookback = 420
    spot_1m = provider.get_spot_ohlcv(symbol, "1m", lookback)
    fut_1m  = provider.get_futures_ohlcv(symbol, "1m", expiry, lookback)

    # Substitute if one side is missing so we can still compute indicators/VWAP
    if spot_1m.empty and not fut_1m.empty:
        logger.warning(f"{symbol}: spot OHLCV empty, substituting futures OHLCV for spot technicals.")
        spot_1m = fut_1m.copy()
    if fut_1m.empty and not spot_1m.empty:
        logger.warning(f"{symbol}: futures OHLCV empty, substituting spot OHLCV for futures technicals.")
        fut_1m = spot_1m.copy()

    if spot_1m.empty and fut_1m.empty:
        logger.warning(f"{symbol}: both spot and futures OHLCV empty. Check Kite Historical addon and access token.")
        return None
    spot_5m = spot_1m.resample("5min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    fut_5m  = fut_1m.resample("5min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

    # Technicals
    rsi5 = float(rsi(spot_5m["close"], 14).iloc[-1])
    macd_line, macd_sig, _ = macd(spot_5m["close"]); macd_last, macd_sig_last = float(macd_line.iloc[-1]), float(macd_sig.iloc[-1])
    adx5 = float(adx(spot_5m, 14).iloc[-1])
    vwap_spot = session_vwap(spot_1m); vwap_fut = session_vwap(fut_1m)
    vp = volume_profile(spot_1m); poc = vp["POC"]

    # Snapshot index & chain
    idx_snap = provider.get_indices_snapshot([symbol]); spot_now = float(idx_snap[symbol])
    chain = provider.get_option_chain(symbol, expiry)

    # Expiry sanity
    if chain["expiry"] != expiry:
        logger.warning(f"{symbol}: chain expiry mismatch ({chain['expiry']} != {expiry}) → NO-TRADE this tick.")
    # ATM with tie-high rule
    atm_k = atm_strike_with_tie_high(spot_now, chain["strikes"]) if chain["strikes"] else None

    # Core metrics
    pcr = pcr_from_chain(chain)
    mp  = max_pain(chain)
    D   = float(spot_now - mp)
    session_hi, session_lo = float(spot_1m["high"].max()), float(spot_1m["low"].min())
    ATR_D = max(1.0, session_hi - session_lo)  # session-range proxy (robust intraday)
    VND = abs(D) / ATR_D
    step = 100 if "BANK" in symbol.upper() else 50
    SSD  = abs(D) / step
    PD   = 100.0 * abs(D) / max(1.0, spot_now)

    # Basis & VWAP relation
    basis = float(fut_1m["close"].iloc[-1] - spot_1m["close"].iloc[-1])

    # Max Pain drift per hour & MPH_norm (keep last 20 in file)
    drift_file = OUT_DIR / f"maxpain_hist_{symbol}.json"
    mp_hist = []
    if drift_file.exists():
        try: mp_hist = json.loads(drift_file.read_text())
        except: mp_hist=[]
    mp_hist.append({"ts": now.isoformat(), "maxpain": mp})
    mp_hist = mp_hist[-20:]
    if len(mp_hist) >= 4:
        y = np.array([x["maxpain"] for x in mp_hist], dtype=float)
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]  # pts per snapshot
        snaps_per_hr = max(1, int(3600/poll_secs))
        mph_pts_hr = float(slope * snaps_per_hr)
    else:
        mph_pts_hr = float('nan')
    mph_norm = (mph_pts_hr / ATR_D) if (mph_pts_hr==mph_pts_hr) else float('nan')

    # ATM IV + deltas for z-scores
    mins_to_exp = minutes_to_expiry(expiry)
    atm_iv = atm_iv_from_chain(chain, spot_now, mins_to_exp)  # fraction (e.g., 0.12)
    # Rolling z-scores (20)
    state_file = OUT_DIR / f"state_{symbol}.json"
    state = {}
    if state_file.exists():
        try: state = json.loads(state_file.read_text())
        except: state = {}
    prev_pcr = state.get("pcr", float('nan'))
    dpcr = (pcr - prev_pcr) if (pcr==pcr and prev_pcr==prev_pcr) else float('nan')
    dpcr_series = state.get("dpcr_series", [])
    if dpcr==dpcr: dpcr_series.append(dpcr)
    dpcr_series = dpcr_series[-20:]
    if len(dpcr_series)>=5 and statistics.pstdev(dpcr_series)>0:
        dpcr_z = (dpcr - statistics.mean(dpcr_series))/ (statistics.pstdev(dpcr_series)+1e-9) if dpcr==dpcr else 0.0
    else:
        dpcr_z = 0.0

    prev_iv = state.get("atm_iv", float('nan'))
    div = (atm_iv - prev_iv) if (atm_iv==atm_iv and prev_iv==prev_iv) else float('nan')
    div_series = state.get("div_series", [])
    if div==div: div_series.append(div)
    div_series = div_series[-20:]
    if len(div_series)>=5 and statistics.pstdev(div_series)>0:
        iv_z = (div - statistics.mean(div_series))/ (statistics.pstdev(div_series)+1e-9) if div==div else 0.0
    else:
        iv_z = 0.0

    # IV percentile hint (intraday estimate)
    iv_hist_file = OUT_DIR / "iv_intraday.json"
    iv_hist = {}
    if iv_hist_file.exists():
        try: iv_hist = json.loads(iv_hist_file.read_text())
        except: iv_hist = {}
    arr = iv_hist.get(symbol, [])
    if atm_iv==atm_iv: 
        arr.append(atm_iv); iv_hist[symbol]=arr[-1500:]
        iv_hist_file.write_text(json.dumps(iv_hist))
    vals = sorted([x for x in arr if x==x and x>0])
    iv_pct_hint = round(100.0 * (sum(1 for x in vals if x<=atm_iv)/len(vals)),1) if (vals and atm_iv==atm_iv) else float('nan')

    # OI regime flags via MAD of ΔOI (liquidity: bid & ask must be >0)
    prev_chain = state.get("chain") or {"calls":{}, "puts":{}}
    def mad(arr):
        if not arr:
            return 0.0
        m = statistics.median(arr)
        return statistics.median([abs(x - m) for x in arr])
    d_oi_ce=[]; d_oi_pe=[]
    ce_write_above=False; pe_unwind_below=False; pe_write_above=False; ce_unwind_below=False
    same_side_add=False; unwind_present=False; two_sided_adjacent=False
    if prev_chain and chain["strikes"]:
        atm = atm_k
        above = set([k for k in chain["strikes"] if k>atm and k<=atm+3*(100 if "BANK" in symbol.upper() else 50)])
        below = set([k for k in chain["strikes"] if k<atm and k>=atm-2*(100 if "BANK" in symbol.upper() else 50)])
        # collect deltas where quotes liquid
        ce_deltas={}; pe_deltas={}
        for k in chain["strikes"]:
            c_now, p_now = chain["calls"][k], chain["puts"][k]
            c_prev = prev_chain.get("calls",{}).get(str(k), prev_chain.get("calls",{}).get(k, {"oi":0}))
            p_prev = prev_chain.get("puts",{}).get(str(k),  prev_chain.get("puts",{}).get(k, {"oi":0}))
            # liquidity: have bid & ask (both >0)
            if c_now.get("bid",0)>0 and c_now.get("ask",0)>0:
                ce_deltas[k] = c_now["oi"] - int(c_prev.get("oi",0))
                d_oi_ce.append(ce_deltas[k])
            if p_now.get("bid",0)>0 and p_now.get("ask",0)>0:
                pe_deltas[k] = p_now["oi"] - int(p_prev.get("oi",0))
                d_oi_pe.append(pe_deltas[k])
        ce_m = mad(d_oi_ce); pe_m = mad(d_oi_pe)
        # flags
        ce_write_above = any( (k in above) and (ce_deltas.get(k,0) > MAD_K*max(1,ce_m)) for k in above )
        pe_unwind_below = any( (k in below) and (pe_deltas.get(k,0) < -MAD_K*max(1,pe_m)) for k in below )
        pe_write_above = any( (k in above) and (pe_deltas.get(k,0) > MAD_K*max(1,pe_m)) for k in above )
        ce_unwind_below = any( (k in below) and (ce_deltas.get(k,0) < -MAD_K*max(1,ce_m)) for k in below )
        same_side_add = ( (spot_now>vwap_spot and any(ce_deltas.get(k,0)>MAD_K*max(1,ce_m) for k in above)) or
                          (spot_now<vwap_spot and any(pe_deltas.get(k,0)>MAD_K*max(1,pe_m) for k in below)) )
        unwind_present = any( x < -MAD_K*max(1,ce_m) for x in ce_deltas.values() ) or any( x < -MAD_K*max(1,pe_m) for x in pe_deltas.values() )
        # adjacent two-sided
        if atm in chain["strikes"]:
            step = 100 if "BANK" in symbol.upper() else 50
            s1, s2 = atm-step, atm+step
            two_sided_adjacent = ( (ce_deltas.get(s2,0)>MAD_K*max(1,ce_m)) and (pe_deltas.get(s1,0)>MAD_K*max(1,pe_m)) )

    # consecutive confirmations
    conf = state.get("confirmations", {})
    def bump(flag, val):
        c = conf.get(flag,0)
        conf[flag] = c+1 if val else 0
    bump("ce_write_above", ce_write_above)
    bump("pe_unwind_below", pe_unwind_below)
    bump("pe_write_above", pe_write_above)
    bump("ce_unwind_below", ce_unwind_below)

    oi_flags = {
        "ce_write_above": ce_write_above,
        "pe_unwind_below": pe_unwind_below,
        "pe_write_above": pe_write_above,
        "ce_unwind_below": ce_unwind_below,
        "same_side_add": same_side_add,
        "unwind_present": unwind_present,
        "two_sided_adjacent": two_sided_adjacent,
    }

    # Normalized scoring
    probs, blocks_ok = score_normalized(
        symbol, D, ATR_D, VND, SSD, PD, pcr, dpcr, dpcr_z,
        mph_pts_hr, mph_norm, atm_iv, div, iv_z, iv_pct_hint,
        spot_now, vwap_spot, adx5, oi_flags, conf
    )
    # Scenario flip gating
    last_probs = state.get("last_probs", {})
    last_top = state.get("last_top", "")
    top = max(probs, key=probs.get)
    if last_top and top != last_top:
        gain = probs[top] - last_probs.get(last_top, 0.0)
        if gain < SCENARIO_FLIP_DELTA:
            # keep old
            top = last_top

    # Trade plan (mechanical)
    def trade_plan(top: str) -> Dict:
        if top == "Bear migration":
            return {"action":"BUY_PE","instrument":f"PE {atm_k-200}",
                    "entry":"VWAP fail + 5m close below VWAP",
                    "sl":"Above VWAP or last LH","targets":[f"{poc:.0f}", f"{mp-100}"],
                    "invalidate":"VWAP reclaim + ΔPCR z ≥ +0.5"}
        if top == "Bull migration / gamma carry":
            return {"action":"BUY_CE","instrument":f"CE {atm_k+200}",
                    "entry":"VWAP reclaim + MACD(5m) cross up",
                    "sl":"Below VWAP or last HL","targets":[f"{poc:.0f}", f"{mp+100}"],
                    "invalidate":"VWAP loss + ΔPCR z ≤ -0.5"}
        if top == "Short-cover reversion up":
            return {"action":"BUY_CE","instrument":f"CE {atm_k+100}",
                    "entry":"VWAP reclaim + RSI(5m) > 50",
                    "sl":"VWAP - 0.15×ATR_D","targets":[f"{poc:.0f}", f"{mp-50}"],
                    "invalidate":"VWAP loss + CE write above spot"}
        if top == "Squeeze continuation (one-way)":
            side = "CE" if spot_now>vwap_spot else "PE"; k = atm_k+100 if side=="CE" else atm_k-100
            return {"action":f"BUY_{side}","instrument":f"{side} {k}",
                    "entry":"Pullback to 9-EMA(5m) with trend intact",
                    "sl":"21-EMA break or opposite VWAP","targets":[f"{poc:.0f}", f"{(spot_now + (D>0)*100 - (D<0)*100):.0f}"],
                    "invalidate":"ADX flattens + opposite OI flip"}
        if top == "Pin & decay day (IV crush)":
            return {"action":"NO-TRADE","why":"Pin & decay: prefer credits / tiny scalps"}
        if top == "Event knee-jerk then revert":
            side="PE" if spot_now>mp else "CE"; k = atm_k-100 if side=="PE" else atm_k+100
            return {"action":f"BUY_{side}","instrument":f"{side} {k}",
                    "entry":"After 15–30m OI flip + IV cools (div<0)",
                    "sl":"Beyond spike high/low","targets":[f"{mp}","VWAP"],
                    "invalidate":"IV stays elevated + same-side OI persists"}
        return {"action":"NO-TRADE","why":"signals conflicted"}

    tp = trade_plan(top)

    # Equivalence line (only if we have both NIFTY and BANKNIFTY ATRs saved)
    atr_map_file = OUT_DIR / "atr_map.json"
    atr_map = {}
    if atr_map_file.exists():
        try: atr_map = json.loads(atr_map_file.read_text())
        except: atr_map={}
    atr_map[symbol] = ATR_D
    atr_map_file.write_text(json.dumps(atr_map))
    eq_line = ""
    nf = atr_map.get("NIFTY"); bn = atr_map.get("BANKNIFTY")
    if nf and bn:
        eq_line = f"Equivalence: 200 NIFTY ≈ {int(round(200*(bn/nf)))} BANKNIFTY (ATR_NF={int(nf)}, ATR_BN={int(bn)})"

    # Snapshot + persist
    snap = Snapshot(
        ts=now.isoformat(), symbol=symbol, spot=float(spot_now),
        vwap_spot=float(vwap_spot), vwap_fut=float(vwap_fut),
        rsi5=float(rsi5), macd=float(macd_last), macd_signal=float(macd_sig_last), adx5=float(adx5),
        pcr=float(round(pcr,3)) if pcr==pcr else float('nan'),
        dpcr=float(round(dpcr,3)) if dpcr==dpcr else float('nan'),
        dpcr_z=float(round(dpcr_z,2)),
        maxpain=int(mp), dist_to_mp=float(round(D,1)), atr_d=float(round(ATR_D,1)), vnd=float(round(VND,2)),
        mph_pts_per_hr=float(round(mph_pts_hr,1)) if mph_pts_hr==mph_pts_hr else float('nan'),
        mph_norm=float(round(mph_norm,2)) if mph_norm==mph_norm else float('nan'),
        atm=int(atm_k) if atm_k else 0,
        atm_iv=float(round(atm_iv*100,2)) if atm_iv==atm_iv else float('nan'),
        div=float(round(div*100,2)) if div==div else float('nan'),
        iv_z=float(round(iv_z,2)),
        iv_pct_hint=float(iv_pct_hint if iv_pct_hint==iv_pct_hint else float('nan')),
        basis=float(round(basis,1)) if basis==basis else float('nan'),
        ssd=float(round(SSD,2)), pdist_pct=float(round(PD,2)),
        scen_probs=probs, scen_top=top, trade=tp, eq_line=eq_line
    )

    ts_tag = now.strftime("%Y%m%d_%H%M%S")
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    (SNAP_DIR / f"{symbol}_{ts_tag}.json").write_text(json.dumps(asdict(snap), indent=2))

    csv_file = OUT_DIR / f"{symbol}_rollup.csv"
    header = not csv_file.exists()
    with open(csv_file, "a", newline="") as f:
        cols = ["ts","spot","vwap_spot","pcr","maxpain","dist_to_mp","atr_d","vnd","mph_pts_per_hr","mph_norm","atm","atm_iv","dpcr_z","iv_z","scenario"]
        row  = [snap.ts, snap.spot, snap.vwap_spot, snap.pcr, snap.maxpain, snap.dist_to_mp, snap.atr_d, snap.vnd, snap.mph_pts_per_hr, snap.mph_norm, snap.atm, snap.atm_iv, snap.dpcr_z, snap.iv_z, snap.scen_top]
        if header: f.write(",".join(cols)+"\n")
        f.write(",".join(map(str,row))+"\n")

    # Save state
    state = {
        "pcr": pcr, "atm_iv": atm_iv, "dpcr_series": dpcr_series, "div_series": div_series,
        "chain": {"calls": chain["calls"], "puts": chain["puts"]},
        "last_probs": probs, "last_top": top, "confirmations": conf
    }
    state_file.write_text(json.dumps(state))
    drift_file.write_text(json.dumps(mp_hist))

    # Print/alerts
    probs_sorted = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    alt = f"{probs_sorted[1][0]} {int(probs_sorted[1][1]*100)}%" if len(probs_sorted)>1 else ""
    spot_print = int(spot_now) if not math.isnan(spot_now) else "NA"
    vwap_fut_print = int(vwap_fut) if not math.isnan(vwap_fut) else "NA"
    line = (f"{now.strftime('%H:%M')} IST | {symbol} {spot_print} | VWAP(fut) {vwap_fut_print}\n"
            f"D={int(D)} | ATR_D={int(ATR_D)} | VND={snap.vnd} | SSD={snap.ssd} | PD={snap.pdist_pct}%\n"
            f"PCR {snap.pcr} (Δz={snap.dpcr_z}) | MaxPain {mp} | Drift {snap.mph_pts_per_hr}/hr (norm {snap.mph_norm})\n"
            f"ATM {atm_k} IV {snap.atm_iv}% (ΔIV_z={snap.iv_z}) | Basis {snap.basis}\n"
            f"Scenario: {top} {int(probs[top]*100)}% (alt: {alt})\n"
            f"Action: {tp.get('action')} {tp.get('instrument','')} | Entry {tp.get('entry','-')} | SL {tp.get('sl','-')} | "
            f"Targets {', '.join(tp.get('targets',[])) if tp.get('targets') else '-'} | Invalidate {tp.get('invalidate','-')}\n"
            f"{snap.eq_line if snap.eq_line else ''}")
    logger.info(line)

    if use_telegram and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                         params={"chat_id": TELEGRAM_CHAT_ID, "text": line})
        except Exception as e:
            logger.warning(f"Telegram error: {e}")
    if slack_webhook:
        try:
            requests.post(slack_webhook, json={"text": line}, timeout=3)
        except Exception as e:
            logger.warning(f"Slack error: {e}")

    return snap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="Comma-separated, e.g., BANKNIFTY,NIFTY")
    ap.add_argument("--provider", default=DEFAULT_PROVIDER, help="KITE (default)")
    ap.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECS)
    ap.add_argument("--run-once", action="store_true")
    ap.add_argument("--use-telegram", action="store_true")
    ap.add_argument("--slack-webhook", default=os.getenv("SLACK_WEBHOOK",""))
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.provider.upper()!="KITE":
        logger.error("Only KITE provider is wired in this build. Use --provider KITE.")
        sys.exit(2)

    provider = KiteProvider()
    logger.info(f"Engine started | provider=KITE | symbols={symbols} | poll={args.poll_seconds}s")
    if args.run_once:
        for sym in symbols:
            run_once(provider, sym, args.poll_seconds, args.use_telegram, args.slack_webhook)
        return
    while True:
        start = time.time()
        if market_open_now():
            for sym in symbols:
                try:
                    run_once(provider, sym, args.poll_seconds, args.use_telegram, args.slack_webhook)
                except Exception as e:
                    logger.exception(f"Run error for {sym}: {e}")
        else:
            logger.info("Market closed (IST). Sleeping until next poll...")
        slp = max(10, args.poll_seconds - int(time.time()-start))
        time.sleep(slp)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
