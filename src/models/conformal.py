from __future__ import annotations
import numpy as np

def naive_conformal_interval(residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Return quantile q such that |residual| <= q with prob 1-alpha. Use on validation residuals."""
    r = np.abs(residuals)
    return float(np.quantile(r, 1 - alpha))

