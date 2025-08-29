from __future__ import annotations
import argparse, pathlib, json
import pandas as pd

def load_rollup(symbol: str):
    root = pathlib.Path(__file__).resolve().parents[2]
    f = root / 'out' / f'{symbol}_rollup.csv'
    if not f.exists():
        raise SystemExit(f'No rollup found at {f}')
    df = pd.read_csv(f)
    return df

def simple_calibration_report(df: pd.DataFrame, prob_col: str, label_col: str):
    if prob_col not in df.columns or label_col not in df.columns:
        print('Columns missing for calibration (need prob_col and label_col).')
        return
    try:
        import numpy as np
        # Brier score
        p = df[prob_col].astype(float).values
        y = df[label_col].astype(int).values
        brier = float(((p - y)**2).mean())
        # naive reliability by bins
        bins = np.linspace(0,1,6)
        df['bin'] = pd.cut(p, bins, include_lowest=True)
        calib = df.groupby('bin').agg(pred=('prob', 'mean'), obs=(label_col,'mean'), n=('prob','size'))
        print('Brier score:', round(brier,4))
        print('Calibration by bins:\n', calib)
    except Exception as e:
        print('Calibration error:', e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--prob-col', default='prob')
    ap.add_argument('--label-col', default='label')
    args = ap.parse_args()
    df = load_rollup(args.symbol.upper())
    print('Loaded', args.symbol, df.shape)
    simple_calibration_report(df, args.prob_col, args.label_col)

if __name__ == '__main__':
    main()

