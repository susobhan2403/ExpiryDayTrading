from __future__ import annotations
import argparse, json, pathlib, sys
from typing import List

import numpy as np
import pandas as pd
import joblib


def main():
    ap = argparse.ArgumentParser(description="Predict with trained LightGBM model")
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', default='')
    args = ap.parse_args()

    model_dir = pathlib.Path(args.model_dir)
    if not (model_dir / 'meta.json').exists():
        raise SystemExit(f"meta.json not found in {model_dir}")
    meta = json.loads((model_dir / 'meta.json').read_text())
    features: List[str] = meta.get('features') or []
    task = meta.get('task','clf')
    model = joblib.load(model_dir / 'model.pkl')
    calibrator = None
    if (model_dir / 'calib.pkl').exists():
        try:
            calibrator = joblib.load(model_dir / 'calib.pkl')
        except Exception:
            calibrator = None

    df = pd.read_csv(args.input)
    if not features:
        # fallback: numeric columns removal of obvious non-features
        non = {'ts','symbol','label'}
        features = [c for c in df.columns if c not in non and pd.api.types.is_numeric_dtype(df[c])]
    X = df[features].copy()

    if task == 'clf':
        if calibrator is not None:
            prob = calibrator.predict_proba(X)[:,1]
        else:
            prob = model.predict_proba(X)[:,1]
        pred = (prob >= 0.5).astype(int)
        out = df.copy()
        out['prob'] = prob
        out['pred'] = pred
    else:
        yhat = model.predict(X)
        out = df.copy()
        out['pred'] = yhat
        # conformal interval if residual quantile provided
        if (model_dir / 'resid_quantile.json').exists():
            q = json.loads((model_dir / 'resid_quantile.json').read_text()).get('q', 0.0)
            out['pred_lo'] = out['pred'] - q
            out['pred_hi'] = out['pred'] + q

    if args.output:
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        print('Wrote', args.output)
    else:
        print(out.head(10).to_string(index=False))

if __name__ == '__main__':
    main()

