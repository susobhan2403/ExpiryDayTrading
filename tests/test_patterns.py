import pandas as pd
from src.features.patterns import candlestick_flags, detect_double_top


def test_candlestick_patterns():
    df = pd.DataFrame({
        'open': [100, 101, 99],
        'high': [101, 102, 103],
        'low': [99, 98, 98],
        'close': [100, 99, 102]
    })
    flags = candlestick_flags(df)
    assert flags['doji'].iloc[0]
    assert flags['engulfing_bull'].iloc[2]


def test_double_top():
    prices = pd.Series([100, 110, 105, 110, 100])
    flags = detect_double_top(prices, left=1, right=1, tol=0.05)
    assert flags.any()
