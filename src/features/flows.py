import pandas as pd

def flow_features(df: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """Compute FII/DII flow based features.

    Parameters
    ----------
    df : DataFrame with columns ['fii','dii','fut','usdinr'] indexed by date.
    price : Series of index prices.
    """
    out = pd.DataFrame(index=df.index)
    out['net_flow'] = df['fii'] - df['dii']
    out['net_roll_5'] = out['net_flow'].rolling(5).mean()
    out['usd_change'] = df['usdinr'].pct_change()
    out['price_div'] = out['net_flow'].pct_change().subtract(price.pct_change())
    return out
