import pandas as pd, os
p='out/instruments_cache.pkl'
if os.path.exists(p):
    df=pd.read_pickle(p)
    print('NFO cache rows:', len(df))
    for sym in ['NIFTY','BANKNIFTY','MIDCPNIFTY','SENSEX']:
        sdf=df[(df['name']==sym) & (df['instrument_type']=='FUT')]
        print(sym, 'FUT rows:', len(sdf))
        if not sdf.empty:
            r=sdf.sort_values('expiry').iloc[0]
            print(sym, 'nearest FUT token', int(r['instrument_token']), r['tradingsymbol'], r['expiry'])
else:
    print('No NFO cache file')
