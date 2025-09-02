"""Compatibility wrappers for option metric utilities.

This module historically exposed a set of helpers under ``src.features``.
The canonical implementations now live in :mod:`src.metrics.core`.  To keep
existing imports working we re-export thin wrappers that delegate to the new
functions while preserving the previous function signatures as much as
possible.

All wrappers return ``None`` instead of ``NaN`` for unavailable results and
otherwise forward the diagnostics from :mod:`src.metrics.core`.
"""

from __future__ import annotations

import math
import datetime as dt
from typing import Mapping, Sequence, Tuple, Dict, Literal

from src.metrics.core import (
    infer_strike_step,
    choose_expiry,
    compute_forward as _compute_forward,
    pick_atm_strike as _pick_atm_strike,
    implied_vol as _implied_vol,
    compute_atm_iv as _compute_atm_iv,
    compute_iv_stats as _compute_iv_stats,
    compute_pcr as _compute_pcr,
)


# ---------------------------------------------------------------------------
# wrappers matching historical function names


def detect_strike_step(strikes: Sequence[float]) -> int:
    """Alias of :func:`src.metrics.core.infer_strike_step`."""

    return infer_strike_step(strikes)


def pick_expiry(now_ist: dt.datetime, expiries: Sequence[dt.datetime], min_tau_h: float) -> dt.datetime:
    """Alias of :func:`src.metrics.core.choose_expiry`."""

    return choose_expiry(now_ist, expiries, min_tau_h)


def compute_forward(spot: float, fut_mid: float | None, r: float, q: float, tau_y: float) -> float:
    """Alias of :func:`src.metrics.core.compute_forward`."""

    return _compute_forward(spot, fut_mid, r, q, tau_y)


def pick_atm_strike(F: float, strikes: Sequence[float], step: int,
                     ce_mid: Mapping[float, float], pe_mid: Mapping[float, float]) -> Tuple[float, Dict]:
    """Alias of :func:`src.metrics.core.pick_atm_strike`."""

    return _pick_atm_strike(F, strikes, step, ce_mid, pe_mid)


def implied_vol_bs(price: float, S: float, K: float, tau_y: float, r: float, q: float,
                   opt_type: Literal["C", "P"]) -> Tuple[float | None, Dict]:
    """Backward compatible implied volatility solver.

    The original function accepted spot ``S`` and dividend yield ``q``.  The
    new core solver expects the forward price ``F`` directly, so we derive it
    using :func:`compute_forward` and delegate to
    :func:`src.metrics.core.implied_vol`.
    """

    F = compute_forward(S, None, r, q, tau_y)
    iv, diag = _implied_vol(price, F, K, tau_y, r, opt_type)
    diag["F"] = F
    return iv, diag


def compute_atm_iv(ce_mid: float | None, pe_mid: float | None, S: float, K_atm: float,
                   tau_y: float, r: float, q: float, F: float | None = None) -> Tuple[float | None, Dict]:
    """Backward compatible wrapper for ATM IV.

    Parameters mirror the historical implementation.  ``F`` is optional and
    will be computed from ``S``/``r``/``q`` when absent.  ``None`` is returned
    when neither leg yields a valid IV.
    """

    F = F if F is not None else compute_forward(S, None, r, q, tau_y)
    return _compute_atm_iv(ce_mid, pe_mid, F, K_atm, tau_y, r)


def compute_iv_percentile(series: Sequence[float], current: float | None) -> Tuple[float | None, float | None]:
    """Return percentile and rank of ``current`` within ``series``.

    This wraps :func:`src.metrics.core.compute_iv_stats` and extracts the two
    headline numbers, returning ``(percentile, iv_rank)``.
    """

    stats = _compute_iv_stats(series, current)
    return stats.get("percentile"), stats.get("iv_rank")


def compute_pcr(oi_put: Mapping[float, int], oi_call: Mapping[float, int], strikes: Sequence[float],
                K_atm: float, step: int, m: int = 6) -> Dict:
    """Alias of :func:`src.metrics.core.compute_pcr`."""

    return _compute_pcr(oi_put, oi_call, strikes, K_atm, step, m)


__all__ = [
    "detect_strike_step",
    "pick_expiry",
    "compute_forward",
    "pick_atm_strike",
    "implied_vol_bs",
    "compute_atm_iv",
    "compute_iv_percentile",
    "compute_pcr",
]

