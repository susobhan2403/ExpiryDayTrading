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
import os, sys, json, time, math, argparse, logging, pathlib, itertools, statistics, glob
import datetime as dt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import pytz
import requests
from logging.handlers import RotatingFileHandler
from colorama import init as colorama_init, Fore, Style
import src.provider.kite as provider_mod
import src.features.technicals as tech
import src.features.options as opt
from src.config import load_settings, save_settings, compute_dynamic_bands
from src.ai.ensemble import ai_predict_probs, blend_probs
from prometheus_client import Gauge, start_http_server
from src.strategy.trend_consensus import TrendConsensus, TrendResult

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

# ---------- helpers ----------
def to_native(obj):
    """Recursively convert numpy types and non-serializable keys/values to
    builtin Python types so they can be JSON serialized."""
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_native(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

# ---------- cli defaults / env ----------
DEFAULT_SYMBOLS = ["BANKNIFTY"]
DEFAULT_PROVIDER = "KITE"   # ← use Kite by default per your request
DEFAULT_POLL_SECS = 240
RISK_FREE = float(os.getenv("RISK_FREE_RATE", "0.066"))

# Scenario weighting and gates
# Allow override via settings.json -> {"WEIGHTS": {"price_trend": 0.35, ...}}
DEFAULT_WEIGHTS = {"price_trend": 0.35, "options_flow": 0.45, "volatility": 0.20}

def load_weights() -> Dict[str, float]:
    """Load scenario weights from settings.json with defaults."""
    try:
        cfg = load_settings()
        w = (cfg or {}).get("WEIGHTS") if isinstance(cfg, dict) else None
        if isinstance(w, dict):
            return {
                "price_trend": float(w.get("price_trend", DEFAULT_WEIGHTS["price_trend"])),
                "options_flow": float(w.get("options_flow", DEFAULT_WEIGHTS["options_flow"])),
                "volatility": float(w.get("volatility", DEFAULT_WEIGHTS["volatility"]))
            }
    except Exception:
        pass
    return DEFAULT_WEIGHTS.copy()

WEIGHTS = load_weights()
GATE_MIN_BLOCKS = {"price_trend": 1, "options_flow": 1, "volatility": 1}
SCENARIO_FLIP_DELTA = 0.15
CONFIRM_SNAPSHOTS = 2         # require consecutive confirmations for OI/drift regimes
MAD_K = float(os.getenv("OI_DELTA_K", "1.5"))
TURNOVER_MIN = float(os.getenv("TURNOVER_MIN", "100000"))

ADX_LOW = int(os.getenv("ADX_THRESHOLD_LOW", "15"))
# Min ADX increase to qualify as rising trend for breakout conditions (can be overridden via settings.json)
ADX_RISE_MIN = float(os.getenv("ADX_RISE_MIN", "2.0"))
PIN_NORM = float(os.getenv("PIN_DISTANCE_NORM", "0.4"))     # VND < 0.4
REV_NORM = float(os.getenv("REVERSION_NORM", "1.0"))        # VND >= 1.0
SQUEEZE_NORM = float(os.getenv("SQUEEZE_DISTANCE_NORM", "1.5"))
MPH_NORM_THR = float(os.getenv("MAXPAIN_DRIFT_NORM_THR", "0.6"))
DECISION_CONFIDENCE_THRESHOLD = float(os.getenv("DECISION_CONFIDENCE_THR", "0.6"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK", "")

# ---------- prometheus metrics ----------
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
start_http_server(METRICS_PORT)

VND_GAUGE = Gauge("vnd", "Volatility normalized distance")
MPH_NORM_GAUGE = Gauge("mph_norm", "MaxPain drift normalized")
IV_Z_GAUGE = Gauge("iv_z", "IV z-score")
PCR_Z_GAUGE = Gauge("pcr_z", "PCR z-score")
SPREAD_PCT_GAUGE = Gauge("spread_pct", "Bid-ask spread percentage")
DEPTH_STAB_GAUGE = Gauge("depth_stability", "Quote depth stability")
SCENARIO_GAUGE = Gauge("scenario_probability", "Scenario probability", ["scenario"])

# Multi-timeframe trend engines per symbol
TREND_ENGINES: Dict[str, TrendConsensus] = {}

# ---------- dynamic weighting & thresholds ----------
DEFAULT_TF_WEIGHTS = {1: 0.2, 5: 0.3, 10: 0.25, 15: 0.25}
DEFAULT_TF_BIAS = {
    "FII": {10: 0.05, 15: 0.05},
    "DII": {1: 0.05, 5: 0.05}
}

def load_tf_config() -> Tuple[Dict[int, float], Dict[str, Dict[int, float]]]:
    """Load base timeframe weights and bias adjustments from settings.json."""
    try:
        cfg = load_settings() or {}
        base_cfg = cfg.get("TF_WEIGHTS", {})
        base = {**DEFAULT_TF_WEIGHTS,
                **{int(k): float(v) for k, v in base_cfg.items() if str(k).isdigit()}}
        bias_cfg = cfg.get("TF_BIAS", {})
        bias = {
            "FII": {**DEFAULT_TF_BIAS["FII"],
                    **{int(k): float(v) for k, v in bias_cfg.get("FII", {}).items() if str(k).isdigit()}},
            "DII": {**DEFAULT_TF_BIAS["DII"],
                    **{int(k): float(v) for k, v in bias_cfg.get("DII", {}).items() if str(k).isdigit()}},
        }
        return base, bias
    except Exception:
        return DEFAULT_TF_WEIGHTS.copy(), DEFAULT_TF_BIAS.copy()

TF_BASE_WEIGHTS, TF_BIAS = load_tf_config()

def dynamic_weight_gate(symbol: str, atr_d: float, now: dt.datetime, inst_bias: float = 0.0):
    """Return (scenario_weights, gate_cap, tf_weights) adjusted for volatility,
    time of day and institutional bias."""
    weights = WEIGHTS.copy()
    tf_weights = TF_BASE_WEIGHTS.copy()
    base_range = 300 if "BANK" in symbol.upper() else 150
    # Volatility regime – emphasise volatility block when range expands
    vol_factor = max(-0.1, min(0.1, (atr_d - base_range) / max(1.0, base_range)))
    weights["volatility"] += 0.2 * vol_factor
    weights["price_trend"] -= 0.1 * vol_factor
    weights["options_flow"] -= 0.1 * vol_factor
    # Time of day – midday relies more on options flow, edges on volatility
    hr = now.astimezone(IST).hour + now.astimezone(IST).minute/60.0
    if 10 <= hr <= 14.5:
        weights["options_flow"] += 0.05
        weights["price_trend"] -= 0.025
        weights["volatility"] -= 0.025
    else:
        weights["volatility"] += 0.05
        weights["price_trend"] -= 0.025
        weights["options_flow"] -= 0.025
    # Institutional bias tilts towards trend weight and timeframe emphasis
    if inst_bias > 0:
        weights["price_trend"] += 0.05
        weights["volatility"] -= 0.05
        for tf, delta in TF_BIAS.get("FII", {}).items():
            tf_weights[tf] = tf_weights.get(tf, 0) + delta
    elif inst_bias < 0:
        weights["price_trend"] += 0.05
        weights["volatility"] -= 0.05
        for tf, delta in TF_BIAS.get("DII", {}).items():
            tf_weights[tf] = tf_weights.get(tf, 0) + delta
    # Normalize weights
    tot = sum(max(v, 0.0) for v in weights.values()) or 1.0
    weights = {k: max(v, 0.0) / tot for k, v in weights.items()}
    tf_tot = sum(max(v, 0.0) for v in tf_weights.values()) or 1.0
    tf_weights = {k: max(v, 0.0) / tf_tot for k, v in tf_weights.items()}
    gate = 0.49 + 0.1 * min(1.0, max(0.0, atr_d / (base_range * 1.5)))
    return weights, gate, tf_weights

def dynamic_thresholds(symbol: str, atr_d: float, avg_oi: float) -> Tuple[float, float]:
    """Return (MAD multiplier, mph_norm threshold) adjusted for liquidity and
    session range."""
    base_liq = 20000 if "BANK" in symbol.upper() else 10000
    liq_factor = 1.0
    if avg_oi < 0.5 * base_liq:
        liq_factor = 1.3
    elif avg_oi > 1.5 * base_liq:
        liq_factor = 0.8
    mad_k = MAD_K * liq_factor
    base_range = 300 if "BANK" in symbol.upper() else 150
    mph_thr = MPH_NORM_THR * max(0.5, min(1.5, atr_d / max(1.0, base_range)))
    return mad_k, mph_thr

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

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def bollinger(series: pd.Series, length: int = 20, numsd: float = 2.0):
    mid = series.rolling(length).mean()
    std = series.rolling(length).std()
    upper = mid + numsd * std
    lower = mid - numsd * std
    return upper, mid, lower

def fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    diff = high - low
    return {
        "236": high - 0.236 * diff,
        "382": high - 0.382 * diff,
        "618": high - 0.618 * diff,
    }

def ichimoku(df: pd.DataFrame):
    high = df['high']; low = df['low']; close = df['close']
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = close.shift(-26)
    return tenkan, kijun, span_a, span_b, chikou

def pivot_points(df: pd.DataFrame):
    daily = df.resample('1D').agg({'high':'max','low':'min','close':'last'})
    if daily.shape[0] < 2:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    prev = daily.iloc[-2]
    pivot = (prev['high'] + prev['low'] + prev['close']) / 3.0
    r1 = 2 * pivot - prev['low']
    s1 = 2 * pivot - prev['high']
    r2 = pivot + (prev['high'] - prev['low'])
    s2 = pivot - (prev['high'] - prev['low'])
    return float(pivot), float(r1), float(s1), float(r2), float(s2)

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

# ---------- settings helpers ----------
def _load_settings() -> Dict:
    path = ROOT / "settings.json"
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}

def _save_settings(cfg: Dict) -> None:
    path = ROOT / "settings.json"
    try:
        path.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass

def compute_dynamic_bands(symbol: str, expiry_today: bool, ATR_D: float, adx5: float, VND: float, D: float, step: int) -> Tuple[int,int,int,int]:
    """
    Returns: (max_above_steps, max_below_steps, far_otm_points, pin_distance_points)
    Tuned by index type, expiry vs non-expiry, and current intraday conditions.
    """
    sym = symbol.upper()
    is_bank = "BANK" in sym
    # Base values
    base_above = 3
    base_below = 2
    base_far = 1200 if is_bank else 600
    base_pin = 200 if is_bank else 100

    # Regime flags
    range_bound = (VND < PIN_NORM and adx5 < ADX_LOW and abs(D) <= base_pin)
    trending = (adx5 >= 25) or (VND >= SQUEEZE_NORM) or (ATR_D >= (300 if is_bank else 150))

    above = base_above
    below = base_below
    far_pts = base_far
    pin_pts = base_pin

    if expiry_today:
        # Tighter bands around ATM on expiry unless strong trend
        if trending:
            above += 1; below += 1
            far_pts += (400 if is_bank else 200)
            pin_pts = int(pin_pts * 1.25)
        else:
            above = max(2, above); below = max(2, below)
            far_pts = max(base_far - (400 if is_bank else 200), base_far//2)
            pin_pts = max(int(pin_pts * 0.7), step)
    else:
        # Non-expiry: widen for trend days, narrow for range days
        if trending:
            above += 1; below += 1
            far_pts += (400 if is_bank else 200)
            pin_pts = int(pin_pts * 1.5)
        if range_bound:
            above = min(above, 2)
            below = min(below, 2)
            far_pts = max(base_far - (400 if is_bank else 200), base_far//2)
            pin_pts = max(int(pin_pts * 0.7), step)

    # Clamp sensible bounds
    above = int(max(1, min(6, above)))
    below = int(max(1, min(6, below)))
    far_pts = int(max(6*step, min(30*step, far_pts)))
    pin_pts = int(max(1*step, min(6*step, pin_pts)))
    return above, below, far_pts, pin_pts

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
    # Weekly index expiries (NIFTY/BANKNIFTY) are on Thursday => weekday 3
    days_ahead = (1 - d.weekday()) % 7
    # After market close on expiry day, roll to next week
    if days_ahead == 0 and now_ist.time() > dt.time(15, 30):
        days_ahead = 7
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
        df = nfo[(nfo["name"] == symbol.upper()) & (nfo["segment"] == "NFO-OPT")].copy()
        if df.empty:
            raise RuntimeError(f"No NFO options for {symbol}")
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
        now_ist = dt.datetime.now(IST)
        today = now_ist.date()
        # If we are past the market close on expiry day, skip today's expiry
        if now_ist.time() > dt.time(15, 30):
            exps = sorted([e for e in df["expiry"].unique() if e > today])
        else:
            exps = sorted([e for e in df["expiry"].unique() if e >= today])
        if not exps:
            raise RuntimeError("No future expiries")
        expiry = exps[0]
        chain = df[df["expiry"] == expiry].copy()
        chain["strike"] = chain["strike"].astype(int)
        return chain, expiry.isoformat()

    def _nearest_future_row(self, symbol: str) -> pd.Series:
        nfo = self._instruments("NFO").copy()
        futs = nfo[(nfo["name"] == symbol.upper()) & (nfo["instrument_type"] == "FUT")].copy()
        futs["expiry"] = pd.to_datetime(futs["expiry"]).dt.date
        now_ist = dt.datetime.now(IST)
        today = now_ist.date()
        # After market close on expiry day, skip today's futures expiry
        if now_ist.time() > dt.time(15, 30):
            futs = futs[futs["expiry"] > today]
        else:
            futs = futs[futs["expiry"] >= today]
        futs = futs.sort_values("expiry")
        if futs.empty:
            raise RuntimeError(f"No futures for {symbol}")
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
    basis_slope: float
    ssd: float
    pdist_pct: float
    sma20: float
    ema20: float
    bb_pos: float
    ichimoku_trend: float
    pivot: float
    fib382: float
    fib618: float
    bullish_signals: int
    bearish_signals: int
    ce_oi_shift: float
    pe_oi_shift: float
    zero_gamma: float
    trend_direction: str
    trend_score: float
    trend_confidence: float
    scen_probs: Dict[str,float]
    scen_top: str
    trade: Dict
    eq_line: str


@dataclass
class DecisionResult:
    trend_score: float
    trend_direction: str
    confidence: float
    last_change_ts: Optional[str]


def update_metrics(snap: Snapshot, spread_pct: float, depth_stab: float) -> None:
    """Push values from the latest snapshot to Prometheus gauges."""
    try:
        if snap.vnd == snap.vnd:
            VND_GAUGE.set(snap.vnd)
        if snap.mph_norm == snap.mph_norm:
            MPH_NORM_GAUGE.set(snap.mph_norm)
        if snap.iv_z == snap.iv_z:
            IV_Z_GAUGE.set(snap.iv_z)
        if snap.dpcr_z == snap.dpcr_z:
            PCR_Z_GAUGE.set(snap.dpcr_z)
        SPREAD_PCT_GAUGE.set(spread_pct)
        DEPTH_STAB_GAUGE.set(depth_stab)
        for scen, prob in (snap.scen_probs or {}).items():
            SCENARIO_GAUGE.labels(scenario=scen).set(prob)
    except Exception:
        # metrics should never interrupt engine flow
        pass

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
    techs: Dict[str,float],
    pin_distance_points: int,
    weights: Optional[Dict[str,float]] = None,
    gate_cap: float = 0.49,
    mph_norm_thr: float = MPH_NORM_THR,
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
    if weights is None:
        weights = WEIGHTS

    # OI regime confirmations (two snapshots)
    def confirmed(flag_name): return confirmations.get(flag_name,0) >= CONFIRM_SNAPSHOTS

    ce_write_abv = confirmed("ce_write_above")
    pe_unw_bel   = confirmed("pe_unwind_below")
    pe_write_abv = confirmed("pe_write_above")
    ce_unw_bel   = confirmed("ce_unwind_below")
    ce_oi_down   = oi_flags.get("ce_oi_down", False)
    pe_oi_down   = oi_flags.get("pe_oi_down", False)
    short_cover  = oi_flags.get("short_covering", False)
    fut_up       = techs.get("fut_move_up", False)
    bullish_bias = techs.get("bullish",0) > techs.get("bearish",0)
    bearish_bias = techs.get("bearish",0) > techs.get("bullish",0)
    bb_pos       = techs.get("bb_pos",0.0)
    vol_pressure = techs.get("vol_pressure", 0.0)
    far_bull     = bool(techs.get("far_bull", False))
    far_bear     = bool(techs.get("far_bear", False))
    term_iv_slope= float(techs.get("term_iv_slope", 0.0))

    # Price/Trend flags
    cpr_tc = techs.get("cpr_tc", float('nan'))
    cpr_bc = techs.get("cpr_bc", float('nan'))
    cpr_bull = (spot > cpr_tc) if (cpr_tc==cpr_tc) else False
    cpr_bear = (spot < cpr_bc) if (cpr_bc==cpr_bc) else False
    price_block["reversion"] = (VND >= REV_NORM and D < 0 and (above_vwap or macd_up)) or abs(bb_pos) >= 1.0
    price_block["bear"]      = (below_vwap and adx5 >= 18) or bearish_bias or (not fut_up) or cpr_bear \
                               or techs.get("micro_bear", False) or techs.get("orb_down", False) \
                               or techs.get("inst_bear", False) or (vol_pressure < -0.5)
    price_block["bull"]      = (above_vwap and adx5 >= 18) or bullish_bias or fut_up or cpr_bull \
                               or techs.get("micro_bull", False) or techs.get("orb_up", False) \
                               or techs.get("inst_bull", False) or (vol_pressure > 0.5)
    price_block["pin"]       = (VND < PIN_NORM and adx5 < ADX_LOW and abs(D) <= pin_distance_points)
    price_block["squeeze"]   = (VND >= SQUEEZE_NORM and adx5 >= 20 and (above_vwap or below_vwap))
    price_block["event"]     = (abs(div) > 0 and abs(iv_z) >= 2.0)

    # Options Flow flags
    flow_block["reversion"] = (ce_unw_bel and pe_write_abv) or short_cover
    flow_block["bear"]      = ((ce_write_abv and pe_unw_bel and mph_norm <= -mph_norm_thr) or (pe_oi_down and not above_vwap) or far_bear)
    flow_block["bull"]      = ((pe_write_abv and ce_unw_bel and mph_norm >= mph_norm_thr) or (ce_oi_down and above_vwap) or far_bull)
    flow_block["pin"]       = (oi_flags.get("two_sided_adjacent", False))
    flow_block["squeeze"]   = (oi_flags.get("same_side_add", False) and not oi_flags.get("unwind_present", False))
    flow_block["event"]     = ( (pe_write_abv and ce_unw_bel) or (ce_write_abv and pe_unw_bel) ) and (abs(dpcr_z) < 0.5)

    # Volatility flags
    vol_block["reversion"] = (div <= 0 or (iv_pct_hint < 50))
    vol_block["bear"]      = (dpcr_z <= -1.0)
    vol_block["bull"]      = (dpcr_z >= +1.0)
    vol_block["pin"]       = (iv_pct_hint < 25 and div <= 0)
    vol_block["squeeze"]   = (iv_z >= +0.5)   # IV rising
    vol_block["event"]     = ( (iv_z >= +2.0 and div < 0) or abs(term_iv_slope) > 0.03 )

    # scoring per scenario (sum normalized evidences within blocks)
    def block_score(flags: Dict[str,bool], keys: List[str]) -> float:
        return sum(1.0 for k in keys if flags.get(k, False)) / max(1,len(keys))

    scores = {
        "Short-cover reversion up": weights["price_trend"]*float(price_block["reversion"])
                                    + weights["options_flow"]*float(flow_block["reversion"])
                                    + weights["volatility"]*float(vol_block["reversion"]),
        "Bear migration": weights["price_trend"]*float(price_block["bear"])
                                    + weights["options_flow"]*float(flow_block["bear"])
                                    + weights["volatility"]*float(vol_block["bear"]),
        "Bull migration / gamma carry": weights["price_trend"]*float(price_block["bull"])
                                    + weights["options_flow"]*float(flow_block["bull"])
                                    + weights["volatility"]*float(vol_block["bull"]),
        "Pin & decay day (IV crush)": weights["price_trend"]*float(price_block["pin"])
                                    + weights["options_flow"]*float(flow_block["pin"])
                                    + weights["volatility"]*float(vol_block["pin"]),
        "Squeeze continuation (one-way)": weights["price_trend"]*float(price_block["squeeze"])
                                    + weights["options_flow"]*float(flow_block["squeeze"])
                                    + weights["volatility"]*float(vol_block["squeeze"]),
        "Event knee-jerk then revert": weights["price_trend"]*float(price_block["event"])
                                    + weights["options_flow"]*float(flow_block["event"])
                                    + weights["volatility"]*float(vol_block["event"]),
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
            scores[name] = min(scores[name], gate_cap)

    probs = softmax(list(scores.values()))
    return ({k: round(v,3) for k,v in zip(scores.keys(), probs)}, block_ok)

# ---------- intraday (non-expiry) scoring ----------
def score_intraday(
    symbol: str,
    D: float,
    VND: float,
    adx5: float,
    dpcr_z: float,
    iv_z: float,
    iv_pct_hint: float,
    above_vwap: bool,
    below_vwap: bool,
    breakout_up: bool,
    breakdown: bool,
    trending_up_mtf: bool,
    trending_down_mtf: bool,
    short_covering: bool,
    oi_flags: Dict[str,bool],
    pin_distance_points: int,
    techs: Dict[str,float],
    weights: Optional[Dict[str,float]] = None,
    gate_cap: float = 0.49,
    mph_norm_thr: float = MPH_NORM_THR,
) -> Tuple[Dict[str,float], Dict[str,Dict[str,bool]]]:
    """
    Produce probabilities for the same SCENARIOS set, but using multi-timeframe trend/flow
    appropriate for non-expiry days. Keeps block gating logic identical for consistency.
    """
    if weights is None:
        weights = WEIGHTS
    vol_pressure = techs.get("vol_pressure", 0.0)
    far_bull = bool(techs.get("far_bull", False))
    far_bear = bool(techs.get("far_bear", False))
    term_iv_slope = float(techs.get("term_iv_slope", 0.0))

    # Price/Trend block (intraday)
    price_block = {
        "reversion": short_covering and above_vwap,
        "bear": (trending_down_mtf or (below_vwap and adx5 >= 18) or breakdown or (vol_pressure < -0.5)),
        "bull": (trending_up_mtf or (above_vwap and adx5 >= 18) or breakout_up or (vol_pressure > 0.5)),
        "pin": (VND < PIN_NORM and adx5 < ADX_LOW and abs(D) <= pin_distance_points),
        "squeeze": (breakout_up or breakdown),
        "event": (iv_z >= 2.0),
    }

    # Options Flow block (aggregate stance)
    flow_block = {
        "reversion": short_covering,
        "bear": (oi_flags.get("ce_write_above", False) and oi_flags.get("pe_unwind_below", False)) or oi_flags.get("pe_oi_down", False) or far_bear,
        "bull": (oi_flags.get("pe_write_above", False) and oi_flags.get("ce_unwind_below", False)) or oi_flags.get("ce_oi_down", False) or far_bull,
        "pin": oi_flags.get("two_sided_adjacent", False),
        "squeeze": oi_flags.get("same_side_add", False) and not oi_flags.get("unwind_present", False),
        "event": abs(dpcr_z) < 0.5,
    }

    # Volatility block
    vol_block = {
        "reversion": (iv_pct_hint < 60),
        "bear": (dpcr_z <= -1.0),
        "bull": (dpcr_z >= +1.0),
        "pin": (iv_pct_hint < 30 and iv_z <= 0.0),
        "squeeze": (iv_z >= +0.5),
        "event": (iv_z >= +2.0) or abs(term_iv_slope) > 0.03,
    }

    # Scores with same weights
    scores = {
        "Short-cover reversion up": weights["price_trend"]*float(price_block["reversion"]) + weights["options_flow"]*float(flow_block["reversion"]) + weights["volatility"]*float(vol_block["reversion"]),
        "Bear migration": weights["price_trend"]*float(price_block["bear"]) + weights["options_flow"]*float(flow_block["bear"]) + weights["volatility"]*float(vol_block["bear"]),
        "Bull migration / gamma carry": weights["price_trend"]*float(price_block["bull"]) + weights["options_flow"]*float(flow_block["bull"]) + weights["volatility"]*float(vol_block["bull"]),
        "Pin & decay day (IV crush)": weights["price_trend"]*float(price_block["pin"]) + weights["options_flow"]*float(flow_block["pin"]) + weights["volatility"]*float(vol_block["pin"]),
        "Squeeze continuation (one-way)": weights["price_trend"]*float(price_block["squeeze"]) + weights["options_flow"]*float(flow_block["squeeze"]) + weights["volatility"]*float(vol_block["squeeze"]),
        "Event knee-jerk then revert": weights["price_trend"]*float(price_block["event"]) + weights["options_flow"]*float(flow_block["event"]) + weights["volatility"]*float(vol_block["event"]),
    }

    # Gating: require at least one flag in each block else cap 0.49
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
            scores[name] = min(scores[name], gate_cap)

    probs = softmax(list(scores.values()))
    return ({k: round(v,3) for k,v in zip(scores.keys(), probs)}, block_ok)

# ---------- AI/GenAI-inspired ensemble ----------
def ai_predict_probs(
    symbol: str,
    above_vwap: bool,
    below_vwap: bool,
    macd_up: bool,
    macd_dn: bool,
    bull_flow: bool,
    bear_flow: bool,
    breakout_up: bool,
    breakdown: bool,
    adx5: float,
    dpcr_z: float,
    iv_z: float,
    mph_norm: float,
    VND: float,
) -> Dict[str, float]:
    """
    Lightweight ensemble that maps salient signals into scenario probabilities.
    Acts as a learned heuristic prior; blended with rule-based probs.
    """
    # Scores in [0,1]
    def nz(x):
        return 0.0 if x!=x else x
    adx_s = max(0.0, (nz(adx5) - 18.0) / 22.0)
    pcr_s_up = max(0.0, nz(dpcr_z)) / 2.0
    pcr_s_dn = max(0.0, -nz(dpcr_z)) / 2.0
    iv_s = min(1.0, abs(nz(iv_z)) / 2.0)
    drift_up = max(0.0, nz(mph_norm))
    drift_dn = max(0.0, -nz(mph_norm))
    range_s = max(0.0, (PIN_NORM - nz(VND))) / max(PIN_NORM, 1e-9)

    bull = (0.25*float(above_vwap) + 0.2*float(macd_up) + 0.25*float(bull_flow)
            + 0.15*pcr_s_up + 0.15*drift_up + 0.2*adx_s)
    bear = (0.25*float(below_vwap) + 0.2*float(macd_dn) + 0.25*float(bear_flow)
            + 0.15*pcr_s_dn + 0.15*drift_dn + 0.2*adx_s)
    squeeze = 0.6*float(breakout_up or breakdown) + 0.4*adx_s
    pin = 0.7*range_s + 0.3*(1.0 - adx_s)
    event = 1.0 if abs(nz(iv_z)) >= 2.0 and abs(nz(dpcr_z)) < 0.5 else 0.0

    raw = {
        "Short-cover reversion up": max(0.0, bull),
        "Bear migration": max(0.0, bear),
        "Bull migration / gamma carry": max(0.0, bull + 0.2*drift_up),
        "Pin & decay day (IV crush)": max(0.0, pin),
        "Squeeze continuation (one-way)": max(0.0, squeeze),
        "Event knee-jerk then revert": max(0.0, event),
    }
    vals = list(raw.values())
    if sum(vals) <= 1e-9:
        # fallback to uniform
        out = {k: 1.0/len(raw) for k in raw}
    else:
        s = sum(vals)
        out = {k: v/s for k, v in raw.items()}
    return out

def blend_probs(rule_probs: Dict[str,float], ai_probs: Dict[str,float], adx5: float, dpcr_z: float, iv_z: float, mph_norm: float) -> Dict[str,float]:
    """
    Blend AI prior with rule-based probs. Alpha rises with trend/clarity.
    """
    def clamp(x,a,b):
        return max(a, min(b, x))
    strength = (
        max(0.0, (adx5 - 18.0)/22.0) + min(1.0, abs(dpcr_z)/2.0) + min(1.0, abs(iv_z)/2.0) + min(1.0, abs(mph_norm))
    ) / 4.0
    alpha = clamp(0.2 + 0.5*strength, 0.2, 0.6)
    keys = rule_probs.keys()
    out = {k: clamp((1-alpha)*rule_probs.get(k,0.0) + alpha*ai_probs.get(k,0.0), 0.0, 1.0) for k in keys}
    # renormalize
    s = sum(out.values()) or 1.0
    return {k: round(v/s,3) for k,v in out.items()}

# ---------- decision evaluation ----------
def evaluate_decision(trend: TrendResult, threshold: float = DECISION_CONFIDENCE_THRESHOLD) -> Tuple[DecisionResult, bool]:
    """Convert TrendConsensus output into a DecisionResult and act flag."""
    dr = DecisionResult(
        trend_score=float(trend.score),
        trend_direction=str(trend.direction),
        confidence=float(trend.confidence),
        last_change_ts=trend.last_change_ts.isoformat() if trend.last_change_ts else None,
    )
    should_act = dr.trend_direction != "NEUTRAL" and dr.confidence >= threshold
    return dr, should_act

# ---------- main cycle ----------
def run_once(provider: provider_mod.MarketDataProvider, symbol: str, poll_secs: int, use_telegram: bool, slack_webhook: str, mode: str = "auto") -> Optional[Snapshot]:
    now = dt.datetime.now(IST)
    # ensure position variable exists across code paths
    pos = None
    # Use provider-selected earliest expiry to avoid calendar mismatches (weekly/monthly + symbol-specific rules)
    try:
        expiry = provider.get_current_expiry_date(symbol)
    except Exception:
        # Fallback to rules-based weekly date
        expiry = opt.nearest_weekly_expiry(now, symbol)

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
    rsi5 = float(tech.rsi(spot_5m["close"], 14).iloc[-1])
    macd_line, macd_sig, _ = tech.macd(spot_5m["close"]); macd_last, macd_sig_last = float(macd_line.iloc[-1]), float(macd_sig.iloc[-1])
    adx5 = float(tech.adx(spot_5m, 14).iloc[-1])
    vwap_spot = tech.session_vwap(spot_1m); vwap_fut = tech.session_vwap(fut_1m)
    vp = tech.volume_profile(spot_1m); poc = vp["POC"]
    sma20 = float(tech.sma(spot_5m["close"], 20).iloc[-1])
    ema20 = float(tech.ema(spot_5m["close"], 20).iloc[-1])
    bb_u, bb_m, bb_l = tech.bollinger(spot_5m["close"], 20, 2)
    bb_u = float(bb_u.iloc[-1]); bb_m = float(bb_m.iloc[-1]); bb_l = float(bb_l.iloc[-1])
    tenkan, kijun, span_a, span_b, _ = tech.ichimoku(spot_5m)
    tenkan_last = float(tenkan.iloc[-1]); kijun_last = float(kijun.iloc[-1])
    span_a_last = float(span_a.iloc[-1]); span_b_last = float(span_b.iloc[-1])
    pivot, r1_p, s1_p, r2_p, s2_p = tech.pivot_points(spot_5m)
    # CPR from previous daily
    daily = spot_1m.resample('1D').agg({'high':'max','low':'min','close':'last'})
    if daily.shape[0] >= 2:
        prev = daily.iloc[-2]
        cpr_pp, cpr_bc, cpr_tc = tech.cpr_from_daily(prev['high'], prev['low'], prev['close'])
    else:
        cpr_pp = cpr_bc = cpr_tc = float('nan')
    # Donchian and Keltner
    don_hi20, don_lo20 = tech.donchian(spot_5m['close'], 20)
    don_hi55, don_lo55 = tech.donchian(spot_5m['close'], 55)
    kel_u, kel_m, kel_l = tech.keltner(spot_5m, 20, 14, 1.5)
    don_hi20_l = float(don_hi20.iloc[-1]) if len(don_hi20) else float('nan')
    don_lo20_l = float(don_lo20.iloc[-1]) if len(don_lo20) else float('nan')
    kel_u_l = float(kel_u.iloc[-1]) if len(kel_u) else float('nan')
    kel_l_l = float(kel_l.iloc[-1]) if len(kel_l) else float('nan')
    # Multi-timeframe aggregates
    def resamp(df, mins):
        return df.resample(f"{mins}min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    spot_10m = resamp(spot_1m, 10)
    spot_15m = resamp(spot_1m, 15)
    ema1_20 = float(ema(spot_1m["close"], 20).iloc[-1]) if len(spot_1m) >= 20 else float('nan')
    ema5_20 = float(ema(spot_5m["close"], 20).iloc[-1]) if len(spot_5m) >= 20 else float('nan')
    ema15_20 = float(ema(spot_15m["close"], 20).iloc[-1]) if len(spot_15m) >= 20 else float('nan')
    rsi5_15 = float(rsi(spot_15m["close"], 14).iloc[-1]) if len(spot_15m) >= 20 else float('nan')
    adx15_series = adx(spot_15m, 14) if len(spot_15m) >= 30 else pd.Series([], dtype=float)
    adx15 = float(adx15_series.iloc[-1]) if len(adx15_series) else float('nan')
    adx5_series = adx(spot_5m, 14)
    adx5_up = bool(adx5_series.iloc[-1] > adx5_series.iloc[-3]) if len(adx5_series) >= 3 else False
    adx5_delta = float(adx5_series.iloc[-1] - adx5_series.iloc[-3]) if len(adx5_series) >= 3 else 0.0
    price_now = float(spot_1m["close"].iloc[-1])
    trending_up_mtf = (price_now > ema1_20 == ema1_20) and (float(spot_5m["close"].iloc[-1]) > ema5_20 == ema5_20) and (float(spot_15m["close"].iloc[-1]) > ema15_20 == ema15_20) and (adx5 >= 18 or (adx15==adx15 and adx15 >= 18))
    trending_down_mtf = (price_now < ema1_20 == ema1_20) and (float(spot_5m["close"].iloc[-1]) < ema5_20 == ema5_20) and (float(spot_15m["close"].iloc[-1]) < ema15_20 == ema15_20) and (adx5 >= 18 or (adx15==adx15 and adx15 >= 18))
    # Breakout sensitivity tightened: close > BB upper AND > R1, ADX rising by at least threshold; symmetric for breakdown
    settings = load_settings()
    adx_rise_min = float(settings.get("ADX_RISE_MIN", ADX_RISE_MIN))
    close_5m = float(spot_5m["close"].iloc[-1])
    breakout_up = (close_5m > bb_u) and (close_5m > (r1_p if r1_p==r1_p else bb_u)) and (adx5_delta >= adx_rise_min) and (price_now > vwap_spot)
    breakdown  = (close_5m < bb_l) and (close_5m < (s1_p if s1_p==s1_p else bb_l)) and (adx5_delta >= adx_rise_min) and (price_now < vwap_spot)

    # Microstructure features (best-effort): read aggregates if available else raw
    micro_bull = micro_bear = orb_up = orb_down = orb_thrust = False
    micro_imb = micro_cvd_slope = micro_qi = micro_stab = 0.0
    micro_spread_pct = 0.0
    thrust_up = thrust_down = False
    try:
        date_tag_stream = now.strftime('%Y%m%d')
        stream_1m_file = OUT_DIR / 'stream' / '1m' / f'{date_tag_stream}.csv'
        if stream_1m_file.exists():
            dfm = pd.read_csv(stream_1m_file, parse_dates=['ts_min'])
            dfm = dfm[dfm['symbol'].str.upper() == symbol.upper()].tail(5)
            if not dfm.empty:
                micro_spread_pct = float(dfm['spread'].mean() / price_now) if 'spread' in dfm.columns else 0.0
                micro_imb = float(dfm['imb'].mean()) if 'imb' in dfm.columns else 0.0
                micro_qi = float(dfm['qi'].mean()) if 'qi' in dfm.columns else 0.0
                micro_stab = float(dfm['quote_stab'].mean()) if 'quote_stab' in dfm.columns else 0.0
                if 'cvd' in dfm.columns:
                    vals = dfm['cvd'].values
                    micro_cvd_slope = float(vals[-1] - vals[0]) if len(vals) >= 2 else 0.0
                thrust_up = bool(dfm['thrust_up'].iloc[-1]) if 'thrust_up' in dfm.columns else False
                thrust_down = bool(dfm['thrust_down'].iloc[-1]) if 'thrust_down' in dfm.columns else False
                orb_up = bool(dfm['orb_up'].iloc[-1]) if 'orb_up' in dfm.columns else False
                orb_down = bool(dfm['orb_down'].iloc[-1]) if 'orb_down' in dfm.columns else False
        else:
            stream_file = OUT_DIR / 'stream' / f'{date_tag_stream}.csv'
            if stream_file.exists():
                dfm = pd.read_csv(stream_file)
                dfm = dfm[dfm['symbol'].str.upper() == symbol.upper()].tail(120)
                if not dfm.empty:
                    micro_spread_pct = float(dfm['spread'].tail(60).mean() / price_now) if 'spread' in dfm.columns else 0.0
                    micro_imb = float(dfm['imb'].tail(60).mean()) if 'imb' in dfm.columns else 0.0
                    micro_qi = float(dfm['qi'].tail(60).mean()) if 'qi' in dfm.columns else 0.0
                    micro_stab = float(dfm['quote_stab'].tail(60).mean()) if 'quote_stab' in dfm.columns else 0.0
                    cvd_series = dfm['cvd'].tail(60) if 'cvd' in dfm.columns else pd.Series([0])
                    micro_cvd_slope = float(cvd_series.iloc[-1] - cvd_series.iloc[0]) if len(cvd_series) >= 2 else 0.0
        # ORB (first 15 mins 9:15-9:30)
        try:
            day_str = now.astimezone(IST).strftime('%Y-%m-%d')
            day_start = pd.Timestamp(f"{day_str} 09:15:00", tz=IST)
            day_15 = day_start + pd.Timedelta(minutes=15)
            orb = spot_1m[(spot_1m.index >= day_start) & (spot_1m.index < day_15)]
            if len(orb) >= 5:
                orb_hi = float(orb['high'].max()); orb_lo = float(orb['low'].min())
                orb_range = orb_hi - orb_lo
                orb_up = orb_up or (price_now > orb_hi)
                orb_down = orb_down or (price_now < orb_lo)
                orb_hist_file = OUT_DIR / "orb_history.json"
                orb_hist = {}
                if orb_hist_file.exists():
                    try:
                        orb_hist = json.loads(orb_hist_file.read_text())
                    except Exception:
                        orb_hist = {}
                arr = orb_hist.get(symbol, [])
                arr.append({"date": day_str, "range": orb_range})
                arr = arr[-40:]
                orb_hist[symbol] = arr
                orb_hist_file.write_text(json.dumps(orb_hist))
                med = statistics.median([r["range"] for r in arr[-20:]]) if arr else float('nan')
                orb_thrust = (med==med) and (orb_range > 1.25*med) and adx5_up
            else:
                orb_thrust = False
        except Exception:
            orb_thrust = False
        micro_bull = (micro_imb > 0.10 and micro_cvd_slope > 0) or orb_up or thrust_up
        micro_bear = (micro_imb < -0.10 and micro_cvd_slope < 0) or orb_down or thrust_down
    except Exception:
        pass

    # FII/DII clients bias (optional) → integrate as micro bias
    inst_bull = inst_bear = False
    try:
        from src.features.clients import load_clients_features
        cli_path = ROOT / 'data' / 'clients.csv'
        if cli_path.exists():
            feats = load_clients_features(str(cli_path), now.date())
            fii_z = feats.get('fii_net_d1_z', 0.0)
            dii_z = feats.get('dii_net_d1_z', 0.0)
            inst_bull = (fii_z > 0.5) or (dii_z > 0.5)
            inst_bear = (fii_z < -0.5) or (dii_z < -0.5)
    except Exception:
        pass
    if inst_bull: micro_bull = True
    if inst_bear: micro_bear = True

    # Snapshot index & option chains (near + far for term structure)
    idx_snap = provider.get_indices_snapshot([symbol]); spot_now = float(idx_snap[symbol])
    exps = provider.get_upcoming_expiries(symbol, n=2)
    chains_map = provider.get_option_chains(symbol, exps[:2]) if exps else {}
    chain = chains_map.get(expiry) or provider.get_option_chain(symbol, expiry)
    far_chain = chains_map.get(exps[1]) if len(exps) > 1 else None
    far_exp = exps[1] if len(exps) > 1 else ""

    # Expiry sanity
    if chain["expiry"] != expiry:
        logger.warning(f"{symbol}: chain expiry mismatch ({chain['expiry']} != {expiry}) → NO-TRADE this tick.")
    # ATM with tie-high rule
    atm_k = opt.atm_strike_with_tie_high(spot_now, chain["strikes"]) if chain["strikes"] else None

    # Core metrics
    pcr = opt.pcr_from_chain(chain)
    mp  = opt.max_pain(chain)
    D   = float(spot_now - mp)
    session_hi, session_lo = float(spot_1m["high"].max()), float(spot_1m["low"].min())
    atr = float(tech.atr(spot_1m, 14)) if len(spot_1m) else 0.0
    ATR_D = max(1.0, atr)
    VND = abs(D) / ATR_D
    step = 100 if "BANK" in symbol.upper() else 50
    SSD  = abs(D) / step
    PD   = 100.0 * abs(D) / max(1.0, spot_now)

    volume_pressure = (spot_now - poc) / max(1.0, ATR_D)
    avg_oi = float(np.mean([chain["calls"][k]["oi"] + chain["puts"][k]["oi"] for k in chain["strikes"]])) if chain["strikes"] else 0.0
    mad_k, mph_thr = dynamic_thresholds(symbol, ATR_D, avg_oi)

    fibs = tech.fibonacci_levels(session_hi, session_lo)
    bb_pos = (spot_now - bb_m) / (bb_u - bb_m) if (bb_u > bb_m) else 0.0
    ich_trend = 1 if spot_now > max(span_a_last, span_b_last) else (-1 if spot_now < min(span_a_last, span_b_last) else 0)
    above_sma = spot_now > sma20
    above_ema = spot_now > ema20
    above_pivot = spot_now > pivot if pivot == pivot else False
    bullish_count = sum([
        rsi5 > 60,
        macd_last > macd_sig_last,
        above_sma,
        above_ema,
        ich_trend > 0,
        bb_pos > 0,
        above_pivot,
        spot_now > fibs['382']
    ])
    bearish_count = sum([
        rsi5 < 40,
        macd_last < macd_sig_last,
        not above_sma,
        not above_ema,
        ich_trend < 0,
        bb_pos < 0,
        (spot_now < pivot if pivot == pivot else False),
        spot_now < fibs['618']
    ])
    fut_move_up = fut_5m['close'].iloc[-1] > fut_5m['close'].iloc[-2] if len(fut_5m) >= 2 else False

    # Basis & VWAP relation
    basis = float(fut_1m["close"].iloc[-1] - spot_1m["close"].iloc[-1])
    # basis_series depends on persisted state; initialize after loading state

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
    mins_to_exp = opt.minutes_to_expiry(expiry)
    atm_iv = opt.atm_iv_from_chain(chain, spot_now, mins_to_exp, RISK_FREE)  # fraction (e.g., 0.12)
    zg, gex_map = opt.gamma_exposure(chain, spot_now, mins_to_exp, atm_iv, RISK_FREE)
    far_pcr = far_atm_iv = term_iv_slope = float('nan')
    far_bull = far_bear = False
    if far_chain:
        try:
            far_pcr = opt.pcr_from_chain(far_chain)
            minutes_far = opt.minutes_to_expiry(far_exp) if far_exp else 0.0
            far_atm_iv = opt.atm_iv_from_chain(far_chain, spot_now, minutes_far, RISK_FREE)
            if atm_iv==atm_iv and far_atm_iv==far_atm_iv:
                term_iv_slope = atm_iv - far_atm_iv
            if far_pcr==far_pcr:
                far_bull = (far_pcr < pcr - 0.1)
                far_bear = (far_pcr > pcr + 0.1)
        except Exception:
            pass
    # Rolling z-scores (20)
    state_file = OUT_DIR / f"state_{symbol}.json"
    state = {}
    if state_file.exists():
        try: state = json.loads(state_file.read_text())
        except: state = {}
    dpcr_series = state.get("dpcr_series", [])

    # Basis & VWAP relation (persist rolling series)
    basis_series = state.get("basis_series", [])
    basis_series.append(basis)
    basis_series = basis_series[-20:]
    state["basis_series"] = basis_series
    basis_slope = basis - basis_series[-2] if len(basis_series) >= 2 else 0.0

    prev_iv = state.get("atm_iv", float('nan'))
    div = (atm_iv - prev_iv) if (atm_iv==atm_iv and prev_iv==prev_iv) else float('nan')
    div_series = state.get("div_series", [])
    if div==div: div_series.append(div)
    div_series = div_series[-20:]
    if len(div_series)>=5 and statistics.pstdev(div_series)>0:
        iv_z = (div - statistics.mean(div_series))/ (statistics.pstdev(div_series)+1e-9) if div==div else 0.0
    else:
        iv_z = 0.0

    # IV percentile based on last 60 EOD observations
    iv_hist_file = OUT_DIR / "iv_eod_history.json"
    iv_hist = {}
    if iv_hist_file.exists():
        try:
            iv_hist = json.loads(iv_hist_file.read_text())
        except Exception:
            iv_hist = {}
    arr = iv_hist.get(symbol, [])
    if atm_iv==atm_iv:
        arr.append(atm_iv)
        iv_hist[symbol] = arr[-60:]
        iv_hist_file.write_text(json.dumps(iv_hist))
    vals = [x for x in arr[-60:] if x==x and x>0]
    iv_pct_hint = round(100.0 * (sum(1 for x in vals if x<=atm_iv)/len(vals)),1) if (vals and atm_iv==atm_iv) else float('nan')
    iv_obs = len(vals)

    # OI regime flags via MAD of ΔOI (liquidity: bid & ask must be >0)
    prev_chain = state.get("chain") or {"calls":{}, "puts":{}}
    # Dynamic banding and filters for OI analysis
    step = 100 if "BANK" in symbol.upper() else 50
    expiry_today = (dt.date.fromisoformat(expiry) == now.date())
    dm_above, dm_below, dm_far, dm_pin = compute_dynamic_bands(symbol, expiry_today, ATR_D, adx5, VND, D, step)
    # Persist dynamic choices for visibility
    _cfg = load_settings()
    _cfg.update({
        "BAND_MAX_STRIKES_ABOVE": dm_above,
        "BAND_MAX_STRIKES_BELOW": dm_below,
        "FAR_OTM_FILTER_POINTS": dm_far,
        "PIN_DISTANCE_POINTS": dm_pin,
    })
    save_settings(_cfg)
    def mad(arr):
        if not arr:
            return 0.0
        m = statistics.median(arr)
        return statistics.median([abs(x - m) for x in arr])
    d_oi_ce=[]; d_oi_pe=[]
    ce_write_above=False; pe_unwind_below=False; pe_write_above=False; ce_unwind_below=False
    same_side_add=False; unwind_present=False; two_sided_adjacent=False
    ce_shift = 0.0; pe_shift = 0.0
    w_ce = 0.0; w_pe = 0.0
    dpcr = 0.0
    if prev_chain and chain["strikes"]:
        atm = atm_k
        band_limit = step * 8
        considered = [k for k in chain["strikes"] if abs(k - atm) <= band_limit]
        above = set([k for k in considered if (k > atm and k <= atm + dm_above*step)])
        below = set([k for k in considered if (k < atm and k >= atm - dm_below*step)])
        ce_deltas={}; pe_deltas={}
        now_time = now.time()
        if dt.time(9,20) <= now_time <= dt.time(15,5):
            min_qty = 100 if "BANK" in symbol.upper() else 250
            for k in considered:
                c_now, p_now = chain["calls"][k], chain["puts"][k]
                c_prev = prev_chain.get("calls",{}).get(str(k), prev_chain.get("calls",{}).get(k, {"oi":0}))
                p_prev = prev_chain.get("puts",{}).get(str(k),  prev_chain.get("puts",{}).get(k, {"oi":0}))
                spread_c = (c_now.get("ask",0)-c_now.get("bid",0))/max(1.0,k)
                spread_p = (p_now.get("ask",0)-p_now.get("bid",0))/max(1.0,k)
                liq_c = (
                    c_now.get("bid",0)>0 and c_now.get("ask",0)>0 and
                    c_now.get("bid_qty",0)>=min_qty and c_now.get("ask_qty",0)>=min_qty and
                    spread_c <= 0.01
                )
                liq_p = (
                    p_now.get("bid",0)>0 and p_now.get("ask",0)>0 and
                    p_now.get("bid_qty",0)>=min_qty and p_now.get("ask_qty",0)>=min_qty and
                    spread_p <= 0.01
                )
                if liq_c:
                    ce_deltas[k] = c_now["oi"] - int(c_prev.get("oi",0))
                    d_oi_ce.append(ce_deltas[k])
                    mid_c = (c_now.get("bid",0) + c_now.get("ask",0)) / 2.0 if (c_now.get("bid",0)>0 and c_now.get("ask",0)>0) else c_now.get("ltp",0)
                    vol_c = min(c_now.get("bid_qty",0), c_now.get("ask_qty",0))
                    turn_c = mid_c * vol_c
                    if turn_c >= TURNOVER_MIN:
                        w_ce += ce_deltas[k] * turn_c
                if liq_p:
                    pe_deltas[k] = p_now["oi"] - int(p_prev.get("oi",0))
                    d_oi_pe.append(pe_deltas[k])
                    mid_p = (p_now.get("bid",0) + p_now.get("ask",0)) / 2.0 if (p_now.get("bid",0)>0 and p_now.get("ask",0)>0) else p_now.get("ltp",0)
                    vol_p = min(p_now.get("bid_qty",0), p_now.get("ask_qty",0))
                    turn_p = mid_p * vol_p
                    if turn_p >= TURNOVER_MIN:
                        w_pe += pe_deltas[k] * turn_p
        ce_m = mad(d_oi_ce); pe_m = mad(d_oi_pe)
        total_ce_oi = sum(v["oi"] for v in chain["calls"].values())
        total_pe_oi = sum(v["oi"] for v in chain["puts"].values())
        prev_total_ce = sum(v.get("oi",0) for v in prev_chain.get("calls",{}).values())
        prev_total_pe = sum(v.get("oi",0) for v in prev_chain.get("puts",{}).values())
        ce_shift = float(total_ce_oi - prev_total_ce)
        pe_shift = float(total_pe_oi - prev_total_pe)
        # flags
        ce_write_above = any( (k in above) and (ce_deltas.get(k,0) > mad_k*max(1,ce_m)) for k in above )
        pe_unwind_below = any( (k in below) and (pe_deltas.get(k,0) < -mad_k*max(1,pe_m)) for k in below )
        pe_write_above = any( (k in above) and (pe_deltas.get(k,0) > mad_k*max(1,pe_m)) for k in above )
        ce_unwind_below = any( (k in below) and (ce_deltas.get(k,0) < -mad_k*max(1,ce_m)) for k in below )
        same_side_add = ( (spot_now>vwap_spot and any(ce_deltas.get(k,0)>mad_k*max(1,ce_m) for k in above)) or
                          (spot_now<vwap_spot and any(pe_deltas.get(k,0)>mad_k*max(1,pe_m) for k in below)) )
        unwind_present = any( x < -mad_k*max(1,ce_m) for x in ce_deltas.values() ) or any( x < -mad_k*max(1,pe_m) for x in pe_deltas.values() )
        # adjacent two-sided
        if atm in chain["strikes"]:
            step = 100 if "BANK" in symbol.upper() else 50
            s1, s2 = atm-step, atm+step
            two_sided_adjacent = ( (ce_deltas.get(s2,0)>mad_k*max(1,ce_m)) and (pe_deltas.get(s1,0)>mad_k*max(1,pe_m)) )
        total_turn = abs(w_pe) + abs(w_ce)
        dpcr = (w_pe - w_ce) / total_turn if total_turn > 0 else 0.0
    if dpcr==dpcr:
        dpcr_series.append(dpcr)
    dpcr_series = dpcr_series[-20:]
    if len(dpcr_series) >= 5 and statistics.pstdev(dpcr_series) > 0:
        dpcr_z = (dpcr - statistics.mean(dpcr_series)) / (statistics.pstdev(dpcr_series) + 1e-9)
    else:
        dpcr_z = 0.0

    # consecutive confirmations
    conf = state.get("confirmations", {})
    def bump(flag, val):
        c = conf.get(flag,0)
        conf[flag] = c+1 if val else 0
    bump("ce_write_above", ce_write_above)
    bump("pe_unwind_below", pe_unwind_below)
    bump("pe_write_above", pe_write_above)
    bump("ce_unwind_below", ce_unwind_below)

    short_covering = (ce_shift < 0 and spot_now > vwap_spot)
    pe_short_cover = (pe_shift < 0 and spot_now < vwap_spot)

    oi_flags = {
        "ce_write_above": ce_write_above,
        "pe_unwind_below": pe_unwind_below,
        "pe_write_above": pe_write_above,
        "ce_unwind_below": ce_unwind_below,
        "same_side_add": same_side_add,
        "unwind_present": unwind_present,
        "two_sided_adjacent": two_sided_adjacent,
        "ce_oi_down": ce_shift < 0,
        "pe_oi_down": pe_shift < 0,
        "short_covering": short_covering or pe_short_cover,
    }

    techs = {
        "bullish": float(bullish_count),
        "bearish": float(bearish_count),
        "bb_pos": bb_pos,
        "fut_move_up": float(fut_move_up),
        "trending_up_mtf": bool(trending_up_mtf),
        "trending_down_mtf": bool(trending_down_mtf),
        "breakout_up": bool(breakout_up),
        "breakdown": bool(breakdown),
        "cpr_tc": float(cpr_tc),
        "cpr_bc": float(cpr_bc),
        "don_hi20": float(don_hi20_l),
        "don_lo20": float(don_lo20_l),
        "kel_u": float(kel_u_l),
        "kel_l": float(kel_l_l),
        "micro_bull": bool(micro_bull),
        "micro_bear": bool(micro_bear),
        "orb_up": bool(orb_up),
        "orb_down": bool(orb_down),
        "micro_imb": float(micro_imb),
        "micro_qi": float(micro_qi),
        "micro_stab": float(micro_stab),
        "micro_cvd_slope": float(micro_cvd_slope),
        "inst_bull": bool(inst_bull),
        "inst_bear": bool(inst_bear),
        "thrust_up": bool(thrust_up),
        "thrust_down": bool(thrust_down),
        "far_bull": bool(far_bull),
        "far_bear": bool(far_bear),
        "term_iv_slope": float(term_iv_slope),
        "vol_pressure": float(volume_pressure),
    }

    inst_bias = (1.0 if inst_bull and not inst_bear else (-1.0 if inst_bear and not inst_bull else 0.0))
    dyn_weights, gate_cap, tf_weights = dynamic_weight_gate(symbol, ATR_D, now, inst_bias)

    tc = TREND_ENGINES.setdefault(symbol, TrendConsensus())
    tc.weights = tf_weights
    trend = tc.evaluate(spot_1m)
    decision, should_act = evaluate_decision(trend)

    # Scoring: expiry vs non-expiry (mode override)
    if mode == "expiry":
        expiry_today = True
    elif mode == "intraday":
        expiry_today = False
    else:
        expiry_today = (dt.date.fromisoformat(expiry) == now.date())
    if expiry_today:
        probs, blocks_ok = score_normalized(
            symbol, D, ATR_D, VND, SSD, PD, pcr, dpcr, dpcr_z,
            mph_pts_hr, mph_norm, atm_iv, div, iv_z, iv_pct_hint,
            spot_now, vwap_spot, adx5, oi_flags, conf, techs,
            dm_pin, dyn_weights, gate_cap, mph_thr
        )
    else:
        probs, blocks_ok = score_intraday(
            symbol,
            D,
            VND,
            adx5,
            dpcr_z,
            iv_z,
            iv_pct_hint,
            above_vwap=(spot_now > vwap_spot),
            below_vwap=(spot_now < vwap_spot),
            breakout_up=bool(breakout_up or techs.get('micro_bull', False) or techs.get('orb_up', False)),
            breakdown=bool(breakdown or techs.get('micro_bear', False) or techs.get('orb_down', False)),
            trending_up_mtf=bool(trending_up_mtf),
            trending_down_mtf=bool(trending_down_mtf),
            short_covering=bool(oi_flags.get("short_covering", False)),
            oi_flags=oi_flags,
            pin_distance_points=dm_pin,
            techs=techs,
            weights=dyn_weights,
            gate_cap=gate_cap,
            mph_norm_thr=mph_thr,
        )
    # AI ensemble blending
    avw = (spot_now == spot_now) and (vwap_spot == vwap_spot) and (spot_now > vwap_spot)
    bvw = (spot_now == spot_now) and (vwap_spot == vwap_spot) and (spot_now < vwap_spot)
    macd_up = macd_last > macd_sig_last
    macd_dn = macd_last < macd_sig_last
    bull_flow = oi_flags.get("pe_write_above", False) and oi_flags.get("ce_unwind_below", False)
    bear_flow = oi_flags.get("ce_write_above", False) and oi_flags.get("pe_unwind_below", False)
    ai_p = ai_predict_probs(symbol, avw, bvw, macd_up, macd_dn, bull_flow, bear_flow, bool(breakout_up), bool(breakdown), adx5, dpcr_z, iv_z, mph_norm, VND)
    probs = blend_probs(probs, ai_p, adx5, dpcr_z, iv_z, mph_norm)
    if orb_thrust:
        probs[SCENARIOS[4]] = probs.get(SCENARIOS[4],0) + 0.05
        probs[SCENARIOS[3]] = max(0.0, probs.get(SCENARIOS[3],0) - 0.05)
    if zg==zg and abs(spot_now - zg) <= step*2 and iv_z >= 0:
        probs[SCENARIOS[4]] = probs.get(SCENARIOS[4],0) + 0.05  # Squeeze
    if basis_slope > 0 and iv_z >= 0:
        probs[SCENARIOS[2]] = probs.get(SCENARIOS[2],0) + 0.05
    if basis_slope < 0 and iv_z >= 0:
        probs[SCENARIOS[1]] = probs.get(SCENARIOS[1],0) + 0.05
    # Persist block-gating effect after AI blend: penalize scenarios lacking
    # at least one signal from each block to avoid unrealistic flips.
    try:
        pen = {k: (v*0.5 if not blocks_ok.get(k, True) else v) for k,v in probs.items()}
        s = sum(pen.values()) or 1.0
        probs = {k: round(v/s,3) for k,v in pen.items()}
    except Exception:
        pass
    # Scenario flip gating
    last_probs = state.get("last_probs", {})
    last_top = state.get("last_top", "")
    top = max(probs, key=probs.get)
    if last_top and top != last_top:
        gain = probs[top] - last_probs.get(last_top, 0.0)
        expected_direction = {
            "Bear migration": "BEAR",
            "Bull migration / gamma carry": "BULL",
            "Short-cover reversion up": "BULL",
            "Pin & decay day (IV crush)": None,
            "Squeeze continuation (one-way)": None,
            "Event knee-jerk then revert": None,
        }
        dir_req = expected_direction.get(top)
        trend_ok = True
        if dir_req == "BULL":
            trend_ok = trend.direction == "BULL" and trend.confidence >= tc.threshold
        elif dir_req == "BEAR":
            trend_ok = trend.direction == "BEAR" and trend.confidence >= tc.threshold
        if gain < SCENARIO_FLIP_DELTA or not trend_ok:
            top = last_top

    # Trade plan (mechanical)
    def trade_plan(top: str, atr_val: float, atr_mult: float) -> Dict:
        if top == "Bear migration":
            plan = {"action":"BUY_PE","instrument":f"PE {atm_k-200}",
                    "entry":"VWAP fail + 5m close below VWAP",
                    "invalidate":"VWAP reclaim + ΔPCR z ≥ +0.5"}
        elif top == "Bull migration / gamma carry":
            plan = {"action":"BUY_CE","instrument":f"CE {atm_k+200}",
                    "entry":"VWAP reclaim + MACD(5m) cross up",
                    "invalidate":"VWAP loss + ΔPCR z ≤ -0.5"}
        elif top == "Short-cover reversion up":
            plan = {"action":"BUY_CE","instrument":f"CE {atm_k+100}",
                    "entry":"VWAP reclaim + RSI(5m) > 50",
                    "invalidate":"VWAP loss + CE write above spot"}
        elif top == "Squeeze continuation (one-way)":
            side = "CE" if spot_now>vwap_spot else "PE"; k = atm_k+100 if side=="CE" else atm_k-100
            plan = {"action":f"BUY_{side}","instrument":f"{side} {k}",
                    "entry":"Pullback to 9-EMA(5m) with trend intact",
                    "invalidate":"ADX flattens + opposite OI flip"}
        elif top == "Pin & decay day (IV crush)":
            plan = {"action":"NO-TRADE","why":"Pin & decay: prefer credits / tiny scalps"}
        elif top == "Event knee-jerk then revert":
            side="PE" if spot_now>mp else "CE"; k = atm_k-100 if side=="PE" else atm_k+100
            plan = {"action":f"BUY_{side}","instrument":f"{side} {k}",
                    "entry":"After 15–30m OI flip + IV cools (div<0)",
                    "invalidate":"IV stays elevated + same-side OI persists"}
        else:
            plan = {"action":"NO-TRADE","why":"signals conflicted"}

        if plan.get("action") in ("BUY_CE", "BUY_PE") and atr_val > 0:
            direction = 1 if plan["action"] == "BUY_CE" else -1
            sl_price = spot_now - direction * atr_mult * atr_val
            tgt1 = spot_now + direction * atr_mult * atr_val
            tgt2 = spot_now + direction * atr_mult * 2 * atr_val
            plan["sl"] = f"{sl_price:.0f}"
            plan["targets"] = [f"{tgt1:.0f}", f"{tgt2:.0f}"]
        return plan

    atr_mult = 1.5 if inst_bias > 0 else 1.0
    tp = trade_plan(top, atr, atr_mult)

    # Microstructure guardrail: require strong queue imbalance and quote stability
    if tp.get("action") in ("BUY_CE", "BUY_PE"):
        direction = 1 if tp["action"] == "BUY_CE" else -1
        qi_ok = (micro_qi * direction) > 0.60
        stab_ok = micro_stab >= 0.3  # ≈3s of stable best bid/ask
        if not (qi_ok and stab_ok):
            logger.info(f"Abort entry: qi={micro_qi:.2f}, stab={micro_stab:.2f}")
            tp = {"action": "NO-TRADE", "why": "microstructure filter (qi/stab)"}

    if not should_act:
        tp = {"action": "NO-TRADE", "why": "confidence below threshold"}

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
        basis_slope=float(round(basis_slope,3)),
        ssd=float(round(SSD,2)), pdist_pct=float(round(PD,2)),
        sma20=float(round(sma20,2)) if sma20==sma20 else float('nan'),
        ema20=float(round(ema20,2)) if ema20==ema20 else float('nan'),
        bb_pos=float(round(bb_pos,2)),
        ichimoku_trend=float(ich_trend),
        pivot=float(round(pivot,2)) if pivot==pivot else float('nan'),
        fib382=float(round(fibs['382'],2)),
        fib618=float(round(fibs['618'],2)),
        bullish_signals=int(bullish_count),
        bearish_signals=int(bearish_count),
        ce_oi_shift=float(round(ce_shift,1)),
        pe_oi_shift=float(round(pe_shift,1)),
        zero_gamma=float(zg if zg==zg else float('nan')),
        trend_direction=trend.direction,
        trend_score=float(round(trend.score,2)),
        trend_confidence=float(round(trend.confidence,2)),
        scen_probs=probs, scen_top=top, trade=tp, eq_line=eq_line
    )

    update_metrics(snap, micro_spread_pct, micro_stab)

    ts_tag = now.strftime("%Y%m%d_%H%M%S")
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    (SNAP_DIR / f"{symbol}_{ts_tag}.json").write_text(json.dumps(asdict(snap), indent=2))
    (SNAP_DIR / f"decision_{symbol}_{ts_tag}.json").write_text(
        json.dumps({"decision": asdict(decision), "should_act": should_act}, indent=2)
    )

    csv_file = OUT_DIR / f"{symbol}_rollup.csv"
    header = not csv_file.exists()
    with open(csv_file, "a", newline="") as f:
        cols = ["ts","spot","vwap_spot","pcr","maxpain","dist_to_mp","atr_d","vnd","mph_pts_per_hr","mph_norm","atm","atm_iv","dpcr_z","iv_z","scenario"]
        row  = [snap.ts, snap.spot, snap.vwap_spot, snap.pcr, snap.maxpain, snap.dist_to_mp, snap.atr_d, snap.vnd, snap.mph_pts_per_hr, snap.mph_norm, snap.atm, snap.atm_iv, snap.dpcr_z, snap.iv_z, snap.scen_top]
        if header: f.write(",".join(cols)+"\n")
        f.write(",".join(map(str,row))+"\n")

    # Save state
    state = {
        "pcr": pcr,
        "atm_iv": atm_iv,
        "dpcr_series": dpcr_series,
        "div_series": div_series,
        "chain": {"calls": chain["calls"], "puts": chain["puts"]},
        "last_probs": probs,
        "last_top": top,
        "confirmations": conf,
        "position": pos,
        "basis_series": basis_series,
        "decision": asdict(decision),
    }
    state_file.write_text(json.dumps(to_native(state)))
    drift_file.write_text(json.dumps(to_native(mp_hist)))

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
    # ----- Override: formatted output with action, checklist, and verdict -----
    def pretty_scn(name: str) -> str:
        mapping = {
            "Short-cover reversion up": "Short-Cover Reversion Up",
            "Bear migration": "Bear Migration",
            "Bull migration / gamma carry": "Bull Migration",
            "Pin & decay day (IV crush)": "Pin and Decay (IV crush)",
            "Squeeze continuation (one-way)": "Squeeze Continuation (One-way)",
            "Event knee-jerk then revert": "Event Knee-jerk then Revert",
        }
        return mapping.get(name, name)

    iv_crush = (div == div and div <= 0) and (iv_pct_hint == iv_pct_hint and iv_pct_hint < 33)
    step_sz = 100 if "BANK" in symbol.upper() else 50
    def suggest_spread(action: str):
        if action == "BUY_CE":
            return ("BUY", (int(atm_k + step_sz), int(atm_k + 2*step_sz)))
        if action == "BUY_PE":
            return ("BUY", (int(atm_k - step_sz), int(atm_k - 2*step_sz)))
        return ("NO-TRADE", None)

    side, spread = suggest_spread(tp.get("action", "NO-TRADE"))
    _sorted = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    alt_name = pretty_scn(_sorted[1][0]) if len(_sorted) > 1 else ""
    alt_name = alt_name.replace(" (IV crush)", "") if alt_name else ""
    alt_pct  = int(_sorted[1][1]*100) if len(_sorted) > 1 else 0
    top_disp = pretty_scn(top)
    if iv_crush and "(IV crush)" not in top_disp:
        top_disp = f"{top_disp} (IV crush)"

    above_vwap = (spot_now == spot_now) and (vwap_spot == vwap_spot) and (spot_now > vwap_spot)
    below_vwap = (spot_now == spot_now) and (vwap_spot == vwap_spot) and (spot_now < vwap_spot)
    macd_up = macd_last > macd_sig_last
    macd_dn = macd_last < macd_sig_last
    trend_strong = adx5 >= 18
    bull_flow = oi_flags.get("pe_write_above", False) and oi_flags.get("ce_unwind_below", False)
    bear_flow = oi_flags.get("ce_write_above", False) and oi_flags.get("pe_unwind_below", False)
    bull_pcr = dpcr_z >= 0.5
    bear_pcr = dpcr_z <= -0.5
    bull_drift = (mph_norm == mph_norm) and (mph_norm >= mph_thr)
    bear_drift = (mph_norm == mph_norm) and (mph_norm <= -mph_thr)

    conds = []
    if side == "BUY":
        if tp.get("action") == "BUY_CE":
            conds = [
                ("Price above VWAP", above_vwap),
                ("ADX >= 18 (trend strength)", trend_strong),
                ("OI flow supports up (PE write, CE unwind)", bull_flow),
                ("PCR momentum up (Δz >= 0.5)", bull_pcr),
                ("MACD(5m) cross up", macd_up),
                ("MaxPain drift up (norm >= thr)", bull_drift),
            ]
        elif tp.get("action") == "BUY_PE":
            conds = [
                ("Price below VWAP", below_vwap),
                ("ADX >= 18 (trend strength)", trend_strong),
                ("OI flow supports down (CE write, PE unwind)", bear_flow),
                ("PCR momentum down (Δz <= -0.5)", bear_pcr),
                ("MACD(5m) cross down", macd_dn),
                ("MaxPain drift down (norm <= -thr)", bear_drift),
            ]
    else:
        conds = [
            ("VND < pin norm (range-bound)", VND < PIN_NORM),
            ("ADX < low (chop)", adx5 < ADX_LOW),
            ("Two-sided OI near ATM", oi_flags.get("two_sided_adjacent", False)),
            ("IV crushing (div<=0, pctile<33)", iv_crush),
        ]

    total = len(conds)
    threshold = max(3, (total * 3)//5) if total else 0
    met = sum(1 for _, ok in conds if ok)

    def tick(ok: bool) -> str:
        # Use unicode escapes to avoid source encoding issues
        return (Fore.GREEN + "\u2714" + Style.RESET_ALL) if ok else (Fore.RED + "\u2718" + Style.RESET_ALL)

    cond_lines = []
    for idx, (name, ok) in enumerate(conds, start=1):
        cond_lines.append(f"{idx}. {name} - {tick(ok)}")

    spot_print = int(spot_now) if not math.isnan(spot_now) else "NA"
    vwap_fut_print = int(vwap_fut) if not math.isnan(vwap_fut) else "NA"
    header = f"{now.strftime('%H:%M')} IST | {symbol} {spot_print} | VWAP(fut) {vwap_fut_print}"
    l1 = f"D={int(D)} | ATR_D={int(ATR_D)} | VND={snap.vnd} | SSD={snap.ssd} | PD={snap.pdist_pct}%"
    l2 = f"PCR {snap.pcr} (Δz={snap.dpcr_z}) | MaxPain {mp} | Drift {snap.mph_pts_per_hr}/hr (norm {snap.mph_norm})"
    l3 = f"ATM {atm_k} IV {snap.atm_iv}% (ΔIV_z={snap.iv_z}) | Basis {snap.basis}"
    l4 = f"Scenario: {top_disp} {int(probs[top]*100)}%" + (f" (alt: {alt_name} {alt_pct}%)" if alt_name else "")

    if side == "BUY" and spread:
        k1, k2 = spread
        cepe = 'CE' if tp.get('action')=='BUY_CE' else 'PE'
        action_line = Style.BRIGHT + Fore.CYAN + f"Action: TRADE | BUY {k1} {cepe} / {k2} {cepe}" + Style.RESET_ALL
    else:
        reason = tp.get('why', 'No favorable setup')
        action_line = Style.BRIGHT + Fore.YELLOW + f"Action: NO-TRADE | {reason}" + Style.RESET_ALL

    entry_gate = f"Enter when atleast {threshold} of below {total} are satisfied:"
    verdict_ok = (side == "BUY") and (met >= threshold)
    verdict = (Style.BRIGHT + Fore.GREEN + "Final Verdict: Enter Now" + Style.RESET_ALL) if verdict_ok else (Style.BRIGHT + Fore.YELLOW + "Final Verdict: Hold" + Style.RESET_ALL)

    from src.output.format import format_output_line
    snap_dict = {
        "vnd": snap.vnd,
        "ssd": snap.ssd,
        "pdist_pct": snap.pdist_pct,
        "pcr": snap.pcr,
        "dpcr_z": snap.dpcr_z,
        "mph_pts_per_hr": snap.mph_pts_per_hr,
        "mph_norm": snap.mph_norm,
        "atm_iv": snap.atm_iv,
        "iv_z": snap.iv_z,
        "basis": snap.basis,
    }
    line = format_output_line(
        now, symbol, spot_now, vwap_fut, D, ATR_D, snap_dict, mp, atm_k,
        probs, top, tp, oi_flags, vwap_spot, adx5, div, iv_pct_hint,
        macd_last, macd_sig_last, VND, PIN_NORM, mph_thr
    )
    # Track assumed positions and exit signals between ticks
    try:
        pos = state.get('position') if isinstance(state, dict) else None
    except Exception:
        pos = None
    action = (tp.get('action') or '').upper()
    # Entry: assume user enters if verdict says Enter Now and action is BUY_*
    if action.startswith('BUY_') and 'Final Verdict: Enter Now' in line:
        if not pos:
            pos = {
                'action': action,
                'instrument': tp.get('instrument',''),
                'opened_at': now.isoformat()
            }
    # Exit: if pos open, fire exit when any exit cond satisfies
    exit_now = False
    if pos and isinstance(pos, dict):
        macd_dn = macd_last < macd_sig_last
        macd_up = macd_last > macd_sig_last
        trend_weak = adx5 < 16
        bear_flow = oi_flags.get('ce_write_above', False) and oi_flags.get('pe_unwind_below', False)
        bull_flow = oi_flags.get('pe_write_above', False) and oi_flags.get('ce_unwind_below', False)
        bear_pcr = dpcr_z <= -0.5
        bull_pcr = dpcr_z >= 0.5
        bear_drift = (mph_norm == mph_norm) and (mph_norm <= -mph_thr)
        bull_drift = (mph_norm == mph_norm) and (mph_norm >= mph_thr)
        if pos.get('action') == 'BUY_CE':
            exit_now = (spot_now < vwap_spot) or trend_weak or bear_flow or bear_pcr or macd_dn or bear_drift
        elif pos.get('action') == 'BUY_PE':
            exit_now = (spot_now > vwap_spot) or trend_weak or bull_flow or bull_pcr or macd_up or bull_drift
        if exit_now:
            line = line + "\n" + (Style.BRIGHT + Fore.RED + "EXIT NOW: Exit open position (" + pos.get('action','') + ")" + Style.RESET_ALL)
            pos = None
    # Optional: risk and strategy hints appended to output
    try:
        from src.models.prob import scp_probability, erp_probability
        from src.features.oi import compute_voi, pin_density
        from src.features.greeks import gex_from_chain, zero_gamma_level
        from src.strategy.select import select_strategy
        from src.risk.sizing import vol_target_size, capped_kelly
        # IV term structure (near/next expiries)
        term_near = term_next = float('nan')
        try:
            exps = provider.get_upcoming_expiries(symbol, n=2)
            if exps:
                chains_map = provider.get_option_chains(symbol, exps)
                ivs = []
                for e in exps:
                    ch = chains_map.get(e)
                    if ch and ch.get('strikes'):
                        iv_e = opt.atm_iv_from_chain(ch, spot_now, opt.minutes_to_expiry(e), RISK_FREE)
                        if iv_e==iv_e and iv_e>0:
                            ivs.append((e, iv_e))
                if ivs:
                    term_near = ivs[0][1]
                    if len(ivs) > 1:
                        term_next = ivs[1][1]
        except Exception:
            pass
        dt_minutes = max(1.0, poll_secs/60.0)
        prev_chain_state = state.get("chain") if isinstance(state, dict) else None
        voi = compute_voi(prev_chain_state or {"calls":{}, "puts":{}}, {"calls": chain["calls"], "puts": chain["puts"]}, dt_minutes, atm_k or 0, 100 if "BANK" in symbol.upper() else 50)
        pin_d = pin_density(chain, atm_k or 0, 100 if "BANK" in symbol.upper() else 50, n=3)
        gex = gex_from_chain(chain, spot_now, mins_to_exp, RISK_FREE, atm_iv or 0.2)
        zg = zero_gamma_level(chain, spot_now, mins_to_exp, RISK_FREE, atm_iv or 0.2, 100 if "BANK" in symbol.upper() else 50)
        prev_gex = (state or {}).get("gex", float('nan')) if isinstance(state, dict) else float('nan')
        d_gex_toward_zero = (abs(prev_gex) - abs(gex)) if (prev_gex==prev_gex and gex==gex) else 0.0
        macd_hist = (macd_line - macd_sig)
        macd_hist_slope = float(macd_hist.iloc[-1] - macd_hist.iloc[-2]) if len(macd_hist) >= 2 else 0.0
        rsi_series_tmp = rsi(spot_5m["close"], 14)
        rsi_slope = float(rsi_series_tmp.iloc[-1] - rsi_series_tmp.iloc[-2]) if len(rsi_series_tmp) >= 2 else 0.0
        scp = scp_probability({
            "dbasis": (basis - ((state or {}).get("basis", basis))) if basis==basis else 0.0,
            "voi_ce_atm": voi.get("voi_ce_atm", 0.0),
            "voi_pe_atm": voi.get("voi_pe_atm", 0.0),
            "rsi_slope": rsi_slope/10.0,
            "macd_hist_slope": macd_hist_slope,
            "price_above_avwap": spot_now > vwap_spot,
            "d_gex_toward_zero": d_gex_toward_zero/(abs(prev_gex)+1e-6) if prev_gex==prev_gex else 0.0,
            "adx": adx5,
        })
        dist_to_pin = abs((atm_k or 0) - (zg if zg==zg else atm_k or 0))
        erp = erp_probability({
            "pin_density": 0 if pin_d!=pin_d else pin_d,
            "dist_to_pin": dist_to_pin / (100 if "BANK" in symbol.upper() else 50),
            "ivp_low": iv_pct_hint==iv_pct_hint and iv_pct_hint < 33,
            "gex_small": abs(gex) < 1e6,
            "oi_conc_at_atm": 0.0,
        })
        strategy_hint = select_strategy(scp, erp, iv_pct_hint if iv_pct_hint==iv_pct_hint else 50.0, mins_to_exp, (spot_now - (zg if zg==zg else spot_now)))
        realized_vol = (ATR_D / max(1.0, spot_now))
        target_vol = 0.02
        size_vol = vol_target_size(target_vol, max(1e-6, realized_vol))
        kelly = capped_kelly(max(scp, 1.0 - erp), cap=0.1)
        # Update IV rolling history and compute IVP/IVR
        from src.features.ivterm import update_iv_history, ivp_ivr
        term_note = ""
        if term_near==term_near:
            term_note = f" | Term near {term_near*100:.2f}%"
            try:
                update_iv_history(symbol, 'near', term_near, now)
            except Exception:
                pass
        if term_next==term_next:
            term_note += f" / next {term_next*100:.2f}%"
            try:
                update_iv_history(symbol, 'next', term_next, now)
            except Exception:
                pass
        ivpx = ivp_ivr(symbol, 'near')
        ivp_val = ivpx.get('ivp', float('nan'))
        ivr_val = ivpx.get('ivr', float('nan'))
        if ivp_val==ivp_val:
            term_note += f" | IVP {ivp_val:.1f}%"
        if ivr_val==ivr_val:
            term_note += f" IVR {ivr_val:.1f}%"
        line = line + "\n" + f"SCP {int(scp*100)}% | ERP {int(erp*100)}% | Strategy: {strategy_hint} | Size(vol) {size_vol:.2f} | Kelly {kelly:.2f}{term_note}"
        # refresh state file with added metrics
        try:
            state_extra = {
                "pcr": pcr, "atm_iv": atm_iv, "dpcr_series": dpcr_series, "div_series": div_series,
                "chain": {"calls": chain["calls"], "puts": chain["puts"]},
                "last_probs": probs, "last_top": top, "confirmations": conf,
                "basis": basis, "gex": gex, "zero_gamma": zg
            }
            state_file.write_text(json.dumps(state_extra))
        except Exception:
            pass
    except Exception:
        pass
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

def replay_snapshots(pattern: str, speed: float = 1.0) -> None:
    files = sorted(glob.glob(pattern))
    if not files:
        logger.error(f"No snapshot files found for pattern: {pattern}")
        return
    logger.info(f"Replaying {len(files)} snapshots at {speed}x speed")
    prev_scen = None
    prev_ts = None
    open_trade = None
    transitions = {}
    results = []
    for fp in files:
        data = json.loads(pathlib.Path(fp).read_text())
        ts = dt.datetime.fromisoformat(data.get("ts"))
        spot = float(data.get("spot", float("nan")))
        scen = data.get("scen_top")
        trade = data.get("trade") or {}

        if open_trade:
            move = open_trade["direction"] * (spot - open_trade["entry"])
            open_trade["mfe"] = max(open_trade["mfe"], move)
            open_trade["mae"] = min(open_trade["mae"], move)
            tgt = open_trade.get("target")
            if tgt is not None and not open_trade["hit"]:
                if open_trade["direction"] * (spot - tgt) >= 0:
                    open_trade["hit"] = True

        if prev_scen is not None and scen != prev_scen:
            logger.info(f"Scenario flip: {prev_scen} -> {scen} @ {ts.strftime('%H:%M:%S')}")
            transitions[(prev_scen, scen)] = transitions.get((prev_scen, scen), 0) + 1
            if open_trade:
                results.append(open_trade)
                open_trade = None

        action = trade.get("action")
        if action and action != "NO-TRADE" and open_trade is None:
            open_trade = {
                "entry": spot,
                "direction": 1 if "BUY_CE" in action else -1,
                "instrument": trade.get("instrument", ""),
                "target": float(trade.get("targets", [None])[0]) if trade.get("targets") else None,
                "mfe": 0.0,
                "mae": 0.0,
                "hit": False,
            }
            logger.info(f"Trade entry: {action} {open_trade['instrument']} @ {spot}")
        elif action == "NO-TRADE" and open_trade:
            results.append(open_trade)
            open_trade = None

        prev_scen = scen
        if speed > 0 and prev_ts is not None:
            delta = (ts - prev_ts).total_seconds()
            if delta > 0:
                time.sleep(delta / speed)
        prev_ts = ts

    if open_trade:
        results.append(open_trade)

    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    hit_rate = hits / total if total else 0.0
    avg_mfe = sum(r["mfe"] for r in results) / total if total else 0.0
    avg_mae = sum(r["mae"] for r in results) / total if total else 0.0
    confusion = {}
    for (a, b), cnt in transitions.items():
        confusion.setdefault(a, {})[b] = cnt
    report = {
        "trades": total,
        "hit_rate": hit_rate,
        "avg_mfe": avg_mfe,
        "avg_mae": avg_mae,
        "confusion": confusion,
    }
    report_file = OUT_DIR / "replay_report.json"
    report_file.write_text(json.dumps(to_native(report), indent=2))
    logger.info(
        f"Day-end report: trades={total} hit-rate={hit_rate:.1%} avg MFE={avg_mfe:.2f} avg MAE={avg_mae:.2f}"
    )
    logger.info(f"Scenario confusion: {confusion}")
    logger.info(f"Report saved to {report_file}")

def main():
    global WEIGHTS
    WEIGHTS = load_weights()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="Comma-separated, e.g., BANKNIFTY,NIFTY")
    ap.add_argument("--provider", default=DEFAULT_PROVIDER, help="KITE (default)")
    ap.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECS)
    ap.add_argument("--run-once", action="store_true")
    ap.add_argument("--use-telegram", action="store_true")
    ap.add_argument("--slack-webhook", default=os.getenv("SLACK_WEBHOOK",""))
    ap.add_argument("--mode", choices=["auto","intraday","expiry"], default="auto")
    ap.add_argument("--replay", help="Glob pattern for snapshot JSON files")
    ap.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier (1.0=real-time)")
    args = ap.parse_args()

    if args.replay:
        replay_snapshots(args.replay, args.speed)
        return

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.provider.upper() != "KITE":
        logger.error("Only KITE provider is wired in this build. Use --provider KITE.")
        sys.exit(2)

    provider = provider_mod.KiteProvider()
    logger.info(
        f"Engine started | provider=KITE | symbols={symbols} | poll={args.poll_seconds}s | mode={args.mode}"
    )
    if args.run_once:
        for i, sym in enumerate(symbols):
            run_once(provider, sym, args.poll_seconds, args.use_telegram, args.slack_webhook, args.mode)
            # blank line between different indexes when running together
            if i < len(symbols) - 1:
                logger.info("")
        return
    while True:
        start = time.time()
        if market_open_now():
            for i, sym in enumerate(symbols):
                try:
                    run_once(
                        provider,
                        sym,
                        args.poll_seconds,
                        args.use_telegram,
                        args.slack_webhook,
                        args.mode,
                    )
                except Exception as e:
                    logger.exception(f"Run error for {sym}: {e}")
                # blank line between different indexes
                if i < len(symbols) - 1:
                    logger.info("")
            # one line break between cycles in continuous mode
            logger.info("")
            slp = max(10, args.poll_seconds - int(time.time() - start))
            time.sleep(slp)
        else:
            logger.info("Market closed (IST). Stopping engine.")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
