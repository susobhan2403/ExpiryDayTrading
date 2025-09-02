import pandas as pd
import numpy as np

from src.core.math import indicators

def test_rsi_insufficient():
    s = pd.Series(np.arange(10, dtype=float))
    r, d = indicators.rsi(s, length=14)
    assert r is None
    assert d["reason"] == "insufficient_data"


def test_rsi_increasing():
    s = pd.Series(np.arange(1, 40, dtype=float))
    r, d = indicators.rsi(s, length=14)
    assert d == {}
    assert r is not None
    assert np.isclose(r.iloc[-1], 100.0)


def test_atr():
    df_short = pd.DataFrame({
        "high": [1,2,3],
        "low": [0,1,2],
        "close": [0.5,1.5,2.5]
    })
    atr, d = indicators.atr(df_short, length=5)
    assert atr is None
    assert d["reason"] == "insufficient_data"

    df = pd.DataFrame({
        "high": np.arange(1, 40),
        "low": np.arange(0, 39),
        "close": np.arange(0.5, 39.5)
    })
    atr, d = indicators.atr(df, length=14)
    assert d == {}
    assert atr is not None
    assert len(atr) > 0


def test_macd():
    s = pd.Series(np.arange(20, dtype=float))
    val, d = indicators.macd(s)
    assert val is None
    assert d["reason"] == "insufficient_data"

    s = pd.Series(np.arange(100, dtype=float))
    val, d = indicators.macd(s)
    assert d == {}
    line, sig, hist = val
    assert len(line) == len(sig) == len(hist) > 0


def test_bollinger():
    s = pd.Series(np.arange(10, dtype=float))
    val, d = indicators.bollinger(s)
    assert val is None
    assert d["reason"] == "insufficient_data"

    s = pd.Series(np.arange(50, dtype=float))
    val, d = indicators.bollinger(s)
    assert d == {}
    upper, mid, lower = val
    assert len(upper) == len(mid) == len(lower) > 0


def test_ichimoku():
    df = pd.DataFrame({"high": np.arange(10), "low": np.arange(10), "close": np.arange(10)})
    val, d = indicators.ichimoku(df)
    assert val is None
    assert d["reason"] == "insufficient_data"

    n = 120
    df = pd.DataFrame({
        "high": np.linspace(1, 2, n),
        "low": np.linspace(0, 1, n),
        "close": np.linspace(0.5, 1.5, n)
    })
    val, d = indicators.ichimoku(df)
    assert d == {}
    tenkan, kijun, span_a, span_b, chikou = val
    assert len(tenkan) == len(kijun) == len(span_a) == len(span_b) == len(chikou) > 0
