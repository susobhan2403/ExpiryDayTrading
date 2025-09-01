from __future__ import annotations
import argparse, pathlib
import pandas as pd
from src.offline.labels import triple_barrier_labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--bar-mins', type=int, default=1)
    ap.add_argument('--k-atr', type=float, default=1.0)
    ap.add_argument('--time-barrier', type=int, default=15)
    args = ap.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    f = root / 'out' / f'{args.symbol.upper()}_rollup.csv'
    if not f.exists():
        raise SystemExit(f'No rollup at {f}')
    df = pd.read_csv(f)
    if 'close' not in df.columns and 'spot' in df.columns:
        df = df.rename(columns={'spot':'close'})
    lbl = triple_barrier_labels(df, close_col='close', atr_col=None, k_atr=args.k_atr, time_barrier=args.time_barrier)
    out = df.copy()
    out['label'] = lbl
    out_f = root / 'out' / f'{args.symbol.upper()}_dataset.csv'
    out.to_csv(out_f, index=False)
    print('Wrote', out_f)

if __name__ == '__main__':
    main()

