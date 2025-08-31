from __future__ import annotations
import datetime as dt
import itertools
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.config import load_settings, save_settings

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "out"
CATEGORIES = ["close_vwap", "range", "iv_change"]
BASELINE = np.array([0.35, 0.45, 0.20])


def _daily_signals(df: pd.DataFrame) -> List[np.ndarray]:
    """Return normalized signals for each day from the dataset."""
    if df.empty:
        return []
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    df["date"] = df["ts"].dt.date
    # take last row per day
    last = df.sort_values("ts").groupby("date").tail(1)
    # restrict to last 30 days
    cutoff = dt.date.today() - dt.timedelta(days=30)
    last = last[last["date"] >= cutoff]
    sigs: List[np.ndarray] = []
    for _, row in last.iterrows():
        close = float(row.get("close", np.nan))
        vwap = float(row.get("vwap_spot", np.nan))
        vnd = float(row.get("vnd", np.nan))
        iv_z = float(row.get("iv_z", np.nan))
        if any(np.isnan(x) for x in (close, vwap, vnd, iv_z)):
            continue
        price_sig = max(0.0, 1.0 - abs(close - vwap) / max(close, 1e-9))
        range_sig = max(0.0, 1.0 - vnd / 0.4)
        iv_sig = min(1.0, abs(iv_z) / 2.0)
        sigs.append(np.array([price_sig, range_sig, iv_sig]))
    return sigs


def classify(signals: np.ndarray, weights: np.ndarray | None = None) -> str:
    scores = signals if weights is None else signals * weights
    idx = int(np.argmax(scores))
    return CATEGORIES[idx]


def confusion(signals: List[np.ndarray], weights: np.ndarray) -> pd.DataFrame:
    pred = [classify(s, weights) for s in signals]
    real = [classify(s, None) for s in signals]
    return pd.crosstab(pd.Series(real, name="real"), pd.Series(pred, name="pred"), dropna=False)


def grid_search(signals: List[np.ndarray]) -> np.ndarray:
    grid = np.linspace(-0.10, 0.10, 5)
    best_w = BASELINE
    best_err = 1.0
    for dx, dy, dz in itertools.product(grid, repeat=3):
        w = BASELINE + np.array([dx, dy, dz])
        if np.any(w <= 0):
            continue
        w = w / w.sum()
        err = np.mean([classify(s, w) != classify(s, None) for s in signals])
        if err < best_err:
            best_err = err
            best_w = w
    return best_w


def main() -> None:
    signals: List[np.ndarray] = []
    for path in DATA_DIR.glob("*_dataset.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        signals.extend(_daily_signals(df))
    if not signals:
        print("No data available for reweighting")
        return
    cm = confusion(signals, BASELINE)
    print("Baseline confusion matrix:\n", cm)
    best = grid_search(signals)
    print("Best weights:", {k: round(v, 3) for k, v in zip(CATEGORIES, best)})
    cfg = load_settings() or {}
    cfg["WEIGHTS"] = {
        "price_trend": round(float(best[0]), 3),
        "options_flow": round(float(best[1]), 3),
        "volatility": round(float(best[2]), 3),
    }
    save_settings(cfg)


if __name__ == "__main__":
    main()
