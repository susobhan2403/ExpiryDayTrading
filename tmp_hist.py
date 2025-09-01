import sys
sys.path.append('.')
import pandas as pd
from src.provider.kite import KiteProvider
p = KiteProvider()
df_spot = p.get_spot_ohlcv('NIFTY', '1m', 420)
print('spot rows', len(df_spot))
if not df_spot.empty:
    print(df_spot.head(3).to_string())
    print(df_spot.tail(3).to_string())
fut = p._nearest_future_row('NIFTY')
print('fut token', int(fut['instrument_token']), fut['tradingsymbol'], fut['expiry'])
df_fut = p.get_futures_ohlcv('NIFTY', '1m', 'NA', 420)
print('fut rows', len(df_fut))
if not df_fut.empty:
    print(df_fut.head(3).to_string())
    print(df_fut.tail(3).to_string())
