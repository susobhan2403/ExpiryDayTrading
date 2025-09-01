from __future__ import annotations
import argparse, pathlib, datetime as dt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
IN_DIR = ROOT / 'out' / 'stream'
OUT_DIR = IN_DIR / '1m'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def aggregate_day(date_tag: str, symbols: list[str]):
    f = IN_DIR / f'{date_tag}.csv'
    if not f.exists():
        raise SystemExit(f'No stream file: {f}')
    df = pd.read_csv(f, parse_dates=['ts'])
    if symbols:
        df = df[df['symbol'].str.upper().isin([s.upper() for s in symbols])]
    if df.empty:
        return
    df['ts_min'] = pd.to_datetime(df['ts']).dt.floor('1min')
    aggs = df.groupby(['symbol','ts_min']).agg(
        ltp=('ltp','last'),
        spread=('spread','median'),
        imb=('imb','mean'),
        qi=('qi','mean'),
        cvd=('cvd','last'),
        quote_stab=('quote_stab','mean')
    ).reset_index()
    # compute cvd delta within minute (approx by diff)
    aggs = aggs.sort_values(['symbol','ts_min'])
    aggs['cvd_delta'] = aggs.groupby('symbol')['cvd'].diff().fillna(0.0)
    # ORB thrust flags (first 15m from 09:15 IST)
    def thrust_flags(g):
        out = g.copy()
        # local time naive: we rely on IST CSV timestamps already
        day = out['ts_min'].dt.date.iloc[0]
        start = pd.Timestamp(f"{day} 09:15:00")
        first_15 = out[(out['ts_min']>=start)&(out['ts_min']<start+pd.Timedelta(minutes=15))]
        orb_hi = first_15['ltp'].max() if len(first_15) else float('nan')
        orb_lo = first_15['ltp'].min() if len(first_15) else float('nan')
        out['orb_up'] = (out['ltp']>orb_hi) if orb_hi==orb_hi else False
        out['orb_down'] = (out['ltp']<orb_lo) if orb_lo==orb_lo else False
        # thrust if imbalance and cvd_delta align strongly
        out['thrust_up'] = (out['imb']>0.1) & (out['cvd_delta']>0)
        out['thrust_down'] = (out['imb']<-0.1) & (out['cvd_delta']<0)
        return out
    try:
        aggs = aggs.groupby('symbol', group_keys=False).apply(thrust_flags, include_groups=False)
    except TypeError:
        aggs = aggs.groupby('symbol', group_keys=False).apply(thrust_flags)
    out_f = OUT_DIR / f'{date_tag}.csv'
    header = not out_f.exists()
    aggs.to_csv(out_f, mode='a', index=False, header=header)
    print('Wrote aggregates:', out_f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=dt.datetime.now().strftime('%Y%m%d'))
    ap.add_argument('--symbols', default='')
    args = ap.parse_args()
    syms = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    aggregate_day(args.date, syms)

if __name__ == '__main__':
    main()
