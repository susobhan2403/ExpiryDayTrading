from __future__ import annotations
import argparse, json, pathlib, sys
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception as e:
    print("lightgbm is required: pip install lightgbm", file=sys.stderr)
    raise

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import joblib


def auto_feature_cols(df: pd.DataFrame, target: str, drop_cols: List[str]) -> List[str]:
    cols = [c for c in df.columns if c not in drop_cols and c != target]
    # keep numeric only
    feats = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return feats


def make_binary_target(y: pd.Series, mode: str = "up_down") -> pd.Series:
    # up_down: map 1->1, -1->0, drop 0
    if mode == "up_down":
        mask = y.isin([1, -1])
        yb = y[mask].map({1: 1, -1: 0})
        return yb
    # up_vs_nonup: 1->1 else 0
    if mode == "up_vs_nonup":
        return y.map(lambda v: 1 if v == 1 else 0)
    return y


def main():
    ap = argparse.ArgumentParser(description="Train LightGBM model with optional calibration")
    ap.add_argument("--symbol", default="", help="Symbol tag (for default paths)")
    ap.add_argument("--input", default="", help="Input CSV. Default: out/{symbol}_dataset.csv")
    ap.add_argument("--target", default="label", help="Target column name")
    ap.add_argument("--task", choices=["clf","reg"], default="clf")
    ap.add_argument("--binary-mode", choices=["up_down","up_vs_nonup"], default="up_down")
    ap.add_argument("--drop-cols", default="ts,symbol", help="Comma-separated columns to drop if present")
    ap.add_argument("--valid-size", type=float, default=0.2)
    ap.add_argument("--calibrate", choices=["none","platt","isotonic"], default="platt")
    ap.add_argument("--model-dir", default="", help="Output models directory. Default: models/{symbol}_lgbm")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parents[2]
    in_path = pathlib.Path(args.input) if args.input else (root / 'out' / f"{args.symbol.upper()}_dataset.csv")
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)
    if args.target not in df.columns:
        raise SystemExit(f"Target column '{args.target}' not in {in_path}")

    drop_cols = [c.strip() for c in args.drop_cols.split(',') if c.strip()]
    features = auto_feature_cols(df, args.target, drop_cols)
    if not features:
        raise SystemExit("No numeric features found. Provide a dataset with numeric columns.")

    y = df[args.target]
    X = df[features]

    if args.task == 'clf':
        # make binary classification target
        yb = make_binary_target(y, args.binary_mode)
        X = X.loc[yb.index]
        y = yb
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=args.valid_size, random_state=42, stratify=y)
        model = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(Xtr, ytr,
                  eval_set=[(Xva, yva)],
                  eval_metric='logloss',
                  verbose=False)

        calibrator = None
        yva_prob = model.predict_proba(Xva)[:,1]
        metrics = {
            'logloss': float(log_loss(yva, yva_prob)),
            'auc': float(roc_auc_score(yva, yva_prob)),
            'brier': float(brier_score_loss(yva, yva_prob)),
        }
        if args.calibrate in ('platt','isotonic'):
            # Wrap using sklearn CalibratedClassifierCV
            method = 'sigmoid' if args.calibrate=='platt' else 'isotonic'
            base = lgb.LGBMClassifier()
            base.classes_ = np.array([0,1])
            base._Booster = model.booster_
            # sklearn requires predict_proba/decision_function; LightGBM has predict_proba
            base.n_features_in_ = Xtr.shape[1]
            calibrator = CalibratedClassifierCV(base_estimator=base, method=method, cv='prefit')
            calibrator.fit(Xva, yva)
            yva_prob = calibrator.predict_proba(Xva)[:,1]
            metrics.update({
                'calibrated_logloss': float(log_loss(yva, yva_prob)),
                'calibrated_auc': float(roc_auc_score(yva, yva_prob)),
                'calibrated_brier': float(brier_score_loss(yva, yva_prob)),
            })

        out_dir = pathlib.Path(args.model_dir) if args.model_dir else (root / 'models' / (args.symbol.upper() + '_lgbm'))
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_dir / 'model.pkl')
        if calibrator is not None:
            joblib.dump(calibrator, out_dir / 'calib.pkl')
        meta = {
            'task': args.task,
            'features': features,
            'target': args.target,
            'binary_mode': args.binary_mode,
            'calibrate': args.calibrate,
            'metrics': metrics,
        }
        (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
        print('Saved model to', out_dir)
        print('Metrics:', json.dumps(metrics, indent=2))

    else:  # regression
        y = y.astype(float)
        mask = y==y
        X = X.loc[mask]
        y = y.loc[mask]
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=args.valid_size, random_state=42)
        model = lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=127,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(Xtr, ytr,
                  eval_set=[(Xva, yva)],
                  eval_metric='l2',
                  verbose=False)
        yhat = model.predict(Xva)
        rmse = float(np.sqrt(np.mean((yva - yhat)**2)))
        out_dir = pathlib.Path(args.model_dir) if args.model_dir else (root / 'models' / (args.symbol.upper() + '_lgbm_reg'))
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_dir / 'model.pkl')
        meta = {
            'task': args.task,
            'features': features,
            'target': args.target,
            'metrics': {'rmse': rmse},
        }
        (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
        print('Saved model to', out_dir)
        print('RMSE:', rmse)


if __name__ == '__main__':
    main()

