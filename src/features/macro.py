import pandas as pd

def macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Integrate macro variables like rates, CPI, crude, VIX."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        out[col] = df[col].pct_change()
        out[f'{col}_roll'] = df[col].rolling(5).mean()
    return out
