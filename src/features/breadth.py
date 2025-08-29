import pandas as pd

def breadth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market breadth and options sentiment features.

    df should contain columns like 'advances','declines','pcr','iv'.
    """
    out = pd.DataFrame(index=df.index)
    adv = df['advances']; dec = df['declines']
    out['ad_ratio'] = adv / (dec.replace(0, pd.NA))
    out['pcr'] = df.get('pcr', pd.Series(index=df.index))
    out['iv_percentile'] = df.get('iv', pd.Series(index=df.index)).rolling(252).rank(pct=True)
    return out
