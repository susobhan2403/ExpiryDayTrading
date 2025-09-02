from __future__ import annotations
import itertools
import pandas as pd
from typing import Callable, Dict, Iterable


def sweep_thresholds(
    data: pd.DataFrame,
    regimes: Iterable[str],
    metric_fn: Callable[[pd.DataFrame], Dict[str, float]],
    thresholds: Dict[str, Iterable[float]],
) -> pd.DataFrame:
    """Grid-search thresholds per regime and report KPIs."""
    rows = []
    for regime in regimes:
        regime_df = data[data["regime"] == regime]
        for combo in itertools.product(*thresholds.values()):
            cfg = dict(zip(thresholds.keys(), combo))
            kpi = metric_fn(regime_df, **cfg)
            rows.append({"regime": regime, **cfg, **kpi})
    return pd.DataFrame(rows)


__all__ = ["sweep_thresholds"]
