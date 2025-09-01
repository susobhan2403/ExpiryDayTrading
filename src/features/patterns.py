import numpy as np
import pandas as pd
from typing import Dict
from .technicals import rsi, macd, ichimoku, adx


def _pivot(series: pd.Series, left: int = 2, right: int = 2, func=np.greater) -> pd.Series:
    """Return boolean series marking pivot points."""
    pivots = pd.Series(False, index=series.index)
    for i in range(left, len(series) - right):
        window = series[(i - left):(i + right + 1)]
        center = window.iloc[left]
        if func(center, window.drop(window.index[left]).max() if func is np.greater else window.drop(window.index[left]).min()):
            pivots.iat[i] = True
    return pivots


def pivot_highs(series: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    return _pivot(series, left, right, func=np.greater)


def pivot_lows(series: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    return _pivot(series, left, right, func=np.less)


def detect_double_top(series: pd.Series, left: int = 2, right: int = 2, tol: float = 0.01) -> pd.Series:
    highs = pivot_highs(series, left, right)
    idx = series.index
    flags = pd.Series(False, index=idx)
    prev_high = None
    prev_idx = None
    for i in range(len(series)):
        if highs.iat[i]:
            if prev_high is not None and abs(series.iat[i] - prev_high) / prev_high <= tol:
                flags.iat[i] = True
            prev_high = series.iat[i]
            prev_idx = i
    return flags


def detect_double_bottom(series: pd.Series, left: int = 2, right: int = 2, tol: float = 0.01) -> pd.Series:
    lows = pivot_lows(series, left, right)
    idx = series.index
    flags = pd.Series(False, index=idx)
    prev_low = None
    for i in range(len(series)):
        if lows.iat[i]:
            if prev_low is not None and abs(series.iat[i] - prev_low) / prev_low <= tol:
                flags.iat[i] = True
            prev_low = series.iat[i]
    return flags


def detect_head_shoulders(series: pd.Series, left: int = 2, right: int = 2, tol: float = 0.03) -> pd.Series:
    highs = pivot_highs(series, left, right)
    idx = series.index
    flags = pd.Series(False, index=idx)
    piv_idxs = np.where(highs)[0]
    for i in range(len(piv_idxs) - 2):
        l, h, r = piv_idxs[i:i+3]
        lh, hh, rh = series.iat[l], series.iat[h], series.iat[r]
        if hh > lh and hh > rh and abs(lh - rh) / max(lh, rh) <= tol:
            flags.iat[r] = True
    return flags


def detect_cup_handle(series: pd.Series, window: int = 20, tol: float = 0.05) -> pd.Series:
    flags = pd.Series(False, index=series.index)
    for i in range(window, len(series)):
        w = series[i - window:i]
        left, bottom, right = w.iloc[0], w.min(), w.iloc[-1]
        if left > bottom and right > bottom and abs(left - right) / max(left, right) <= tol:
            flags.iat[i-1] = True
    return flags


def detect_triangle(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    flags = pd.Series(False, index=high.index)
    for i in range(window, len(high)):
        hh = high[i - window:i]
        ll = low[i - window:i]
        slope_high = (hh.iloc[-1] - hh.iloc[0]) / window
        slope_low = (ll.iloc[-1] - ll.iloc[0]) / window
        if slope_high < 0 and slope_low > 0:
            flags.iat[i-1] = True
    return flags


def candlestick_flags(df: pd.DataFrame) -> pd.DataFrame:
    o = df['open']; h = df['high']; l = df['low']; c = df['close']
    body = (c - o).abs()
    range_ = h - l
    eps = 1e-9
    doji = body / (range_ + eps) < 0.1
    lower = (o.where(o < c, c) - l)
    upper = (h - o.where(o > c, c))
    hammer = (lower > 2 * body) & (upper < body)
    engulf_bull = (c > o) & (o.shift(1) > c.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1))
    engulf_bear = (c < o) & (o.shift(1) < c.shift(1)) & (o >= c.shift(1)) & (c <= o.shift(1))
    return pd.DataFrame({
        'doji': doji,
        'hammer': hammer,
        'engulfing_bull': engulf_bull,
        'engulfing_bear': engulf_bear
    })


def false_breakout_filter(df: pd.DataFrame, breakout: pd.Series, volume_col: str = 'volume', adx_len: int = 14, atr_len: int = 14, n: int = 3, adx_th: float = 20) -> pd.Series:
    vol = df[volume_col]
    vol_avg = vol.rolling(20).mean()
    price = df['close']
    tr = (pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1))
    atr = tr.ewm(alpha=1/atr_len, adjust=False).mean()
    adx_val = adx(df, length=adx_len)
    valid = breakout & (vol > vol_avg) & (adx_val > adx_th)
    reenter = price.where(~breakout).rolling(n).max().shift(-n) > price
    return valid & ~reenter


def mtf_consensus(df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_d: pd.DataFrame) -> Dict[str, bool]:
    res = {}
    for name, df in {'15m': df_15m, '1h': df_1h, 'daily': df_d}.items():
        r = rsi(df['close']).iloc[-1]
        m_line, m_sig, _ = macd(df['close'])
        tenkan, kijun, span_a, span_b, _ = ichimoku(df)
        res[name] = {
            'rsi': r,
            'macd_cross': m_line.iloc[-1] > m_sig.iloc[-1],
            'price_above_cloud': df['close'].iloc[-1] > max(span_a.iloc[-1], span_b.iloc[-1])
        }
    bullish = all(v['macd_cross'] and v['price_above_cloud'] and v['rsi'] > 50 for v in res.values())
    bearish = all((not v['macd_cross']) and (not v['price_above_cloud']) and v['rsi'] < 50 for v in res.values())
    return {'bullish': bullish, 'bearish': bearish}
