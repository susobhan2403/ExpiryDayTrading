from __future__ import annotations
import pandas as pd
import numpy as np

def triple_barrier_labels(df: pd.DataFrame, close_col: str = 'close', atr_col: str | None = None,
                          k_atr: float = 1.0, time_barrier: int = 15) -> pd.Series:
    """
    Lopez de Prado triple-barrier labels: 1 (upper), -1 (lower), 0 (time barrier), NaN for insufficient horizon.
    If atr_col provided, barriers = k_atr * ATR; else use k_atr * rolling stddev as proxy.
    time_barrier in bars (e.g., 15 bars ahead resembles ~15 minutes if 1m bars).
    """
    c = df[close_col].astype(float).values
    if atr_col and atr_col in df.columns:
        atr = df[atr_col].astype(float).values
    else:
        atr = pd.Series(c).diff().rolling(20).std().fillna(method='bfill').values
    n = len(df)
    labels = np.full(n, np.nan)
    for i in range(n):
        up = c[i] + k_atr * atr[i]
        dn = c[i] - k_atr * atr[i]
        j = min(n-1, i + time_barrier)
        # scan path
        path = c[i:j+1]
        if (path >= up).any():
            labels[i] = 1
        elif (path <= dn).any():
            labels[i] = -1
        else:
            labels[i] = 0
    return pd.Series(labels, index=df.index, name='label')

