from __future__ import annotations
import math
import datetime as dt
from typing import Dict

import numpy as np
import pandas as pd

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, length: int = 14) -> float:
    if df.empty:
        return float('nan')
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=length).mean()
    return float(atr.iloc[-1]) if len(atr) else float('nan')

def wilder_atr(df: pd.DataFrame, length: int = 14) -> float:
    if df.empty:
        return float('nan')
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return float(atr.iloc[-1]) if len(atr) else float('nan')

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

# --- Additional features ---
def cpr_from_daily(prev_high: float, prev_low: float, prev_close: float):
    pp = (prev_high + prev_low + prev_close)/3.0
    bc = (prev_high + prev_low)/2.0
    tc = 2*pp - bc
    return float(pp), float(bc), float(tc)

def donchian(series: pd.Series, window: int = 20):
    high = series.rolling(window).max()
    low = series.rolling(window).min()
    return high, low

def keltner(df: pd.DataFrame, ema_len: int = 20, atr_len: int = 14, mult: float = 1.5):
    mid = ema(df['close'], ema_len)
    tr = (pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1))
    atr = tr.ewm(alpha=1/atr_len, adjust=False).mean()
    upper = mid + mult*atr
    lower = mid - mult*atr
    return upper, mid, lower

