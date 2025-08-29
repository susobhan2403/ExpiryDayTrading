from __future__ import annotations
import datetime as dt
from typing import Dict
import pandas as pd

def load_clients_features(csv_path: str, as_of: dt.date | None = None) -> Dict[str, float]:
    """
    Ingest client-wise positions CSV and compute day-over-day deltas, z-scores, and lags (t-1..t-5).
    Expected columns (flexible):
      date, fii_index_fut_net (or fii_fut_net), dii_index_fut_net (or dii_fut_net)
    Returns dict for the most recent available date (<= as_of if provided).
    """
    df = pd.read_csv(csv_path)
    # normalize date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date
    else:
        raise ValueError('CSV must include a date column')
    if as_of:
        df = df[df['date'] <= as_of]
    df = df.sort_values('date')
    # choose net columns
    def pick(col_opts):
        for c in col_opts:
            if c in df.columns:
                return c
        return None
    fii_col = pick(['fii_index_fut_net','fii_fut_net','fii_fut_index_net'])
    dii_col = pick(['dii_index_fut_net','dii_fut_net','dii_fut_index_net'])
    out: Dict[str,float] = {}
    for col, tag in ((fii_col, 'fii'), (dii_col, 'dii')):
        if not col: continue
        s = df[col].astype(float)
        d = s.diff().fillna(0.0)
        out[f'{tag}_net'] = float(s.iloc[-1])
        out[f'{tag}_net_d1'] = float(d.iloc[-1])
        if len(d) >= 20 and d.std() > 0:
            z = (d - d.mean())/(d.std()+1e-9)
            out[f'{tag}_net_d1_z'] = float(z.iloc[-1])
        # lags
        for L in range(1,6):
            if len(d) > L:
                out[f'{tag}_net_d1_lag{L}'] = float(d.iloc[-L])
    return out

