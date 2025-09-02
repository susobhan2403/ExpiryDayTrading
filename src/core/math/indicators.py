from __future__ import annotations

"""Deterministic technical indicators with diagnostics.

Each function returns ``(value_or_series_or_none, diagnostics)`` where
``diagnostics`` is a dictionary containing useful metadata.  When input
values are insufficient the value is ``None`` and diagnostics include a
``{"reason": ...}`` entry.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _check_series(series: pd.Series, min_len: int) -> Optional[Dict[str, str]]:
    """Return diagnostics dict if series is invalid."""
    if series is None or len(series) < min_len:
        return {"reason": "insufficient_data"}
    if series.isna().any():
        return {"reason": "nan_in_input"}
    return None


def _check_df(df: pd.DataFrame, cols: Tuple[str, ...], min_len: int) -> Optional[Dict[str, str]]:
    if df is None or len(df) < min_len:
        return {"reason": "insufficient_data"}
    missing = [c for c in cols if c not in df]
    if missing:
        return {"reason": f"missing_columns:{','.join(missing)}"}
    if df[list(cols)].isna().any().any():
        return {"reason": "nan_in_input"}
    return None


def sma(series: pd.Series, window: int) -> Tuple[Optional[pd.Series], Dict]:
    diag = _check_series(series, window)
    if diag:
        return None, diag
    out = series.rolling(window).mean().dropna()
    if out.empty:
        return None, {"reason": "insufficient_data"}
    return out, {}


def ema(series: pd.Series, span: int) -> Tuple[Optional[pd.Series], Dict]:
    diag = _check_series(series, span)
    if diag:
        return None, diag
    out = series.ewm(span=span, adjust=False).mean().dropna()
    if out.empty:
        return None, {"reason": "insufficient_data"}
    return out, {}


def rsi(series: pd.Series, length: int = 14) -> Tuple[Optional[pd.Series], Dict]:
    diag = _check_series(series, length + 1)
    if diag:
        return None, diag
    diff = series.diff().fillna(0.0)
    up = diff.clip(lower=0.0)
    down = -diff.clip(upper=0.0)
    alpha = 1 / float(length)
    gain = up.ewm(alpha=alpha, adjust=False).mean()
    loss = down.ewm(alpha=alpha, adjust=False).mean()
    rs = gain / loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.replace([np.inf, -np.inf], 100.0).fillna(50.0)
    rsi = rsi.iloc[length:]
    if rsi.empty:
        return None, {"reason": "insufficient_data"}
    return rsi, {}


def atr(df: pd.DataFrame, length: int = 14) -> Tuple[Optional[pd.Series], Dict]:
    diag = _check_df(df, ("high", "low", "close"), length)
    if diag:
        return None, diag
    prev_close = df["close"].shift(1).fillna(df["close"])
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1/float(length), adjust=False).mean().dropna()
    if atr_series.empty:
        return None, {"reason": "insufficient_data"}
    return atr_series, {}


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[Tuple[pd.Series, pd.Series, pd.Series]], Dict]:
    need = max(fast, slow, signal)
    diag = _check_series(series, need)
    if diag:
        return None, diag
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    signal_series = line.ewm(span=signal, adjust=False).mean()
    hist = line - signal_series
    df = pd.DataFrame({
        "line": line,
        "signal": signal_series,
        "hist": hist,
    }).dropna()
    if df.empty:
        return None, {"reason": "insufficient_data"}
    return (df["line"], df["signal"], df["hist"]), {}


def bollinger(series: pd.Series, length: int = 20, numsd: float = 2.0) -> Tuple[Optional[Tuple[pd.Series, pd.Series, pd.Series]], Dict]:
    diag = _check_series(series, length)
    if diag:
        return None, diag
    mid = series.rolling(length).mean()
    std = series.rolling(length).std()
    upper = mid + numsd * std
    lower = mid - numsd * std
    df = pd.DataFrame({"upper": upper, "mid": mid, "lower": lower}).dropna()
    if df.empty:
        return None, {"reason": "insufficient_data"}
    return (df["upper"], df["mid"], df["lower"]), {}


def ichimoku(df: pd.DataFrame) -> Tuple[Optional[Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]], Dict]:
    diag = _check_df(df, ("high", "low", "close"), 52)
    if diag:
        return None, diag
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = close.shift(-26)
    df_out = pd.DataFrame({
        "tenkan": tenkan,
        "kijun": kijun,
        "span_a": span_a,
        "span_b": span_b,
        "chikou": chikou,
    }).dropna()
    if df_out.empty:
        return None, {"reason": "insufficient_data"}
    return (
        df_out["tenkan"],
        df_out["kijun"],
        df_out["span_a"],
        df_out["span_b"],
        df_out["chikou"],
    ), {}

