"""Signal gating utilities.

This module provides rolling z-score calculators for PCR and IV deltas and
implements a debounce mechanism that requires multiple consecutive spikes
before muting signals.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple, Dict
import statistics

from ..config import load_settings


@dataclass
class RollingZGate:
    """Track a series and return z-scores with spike debouncing.

    Parameters
    ----------
    window: int
        Number of observations to retain for z-score calculation.
    threshold: float
        Absolute z-score required to consider the observation a spike.
    confirm: int
        Number of consecutive spikes required before the gate mutes signals.
    """

    window: int = 20
    threshold: float = 2.0
    confirm: int = 2
    history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    _spikes: int = 0

    def update(self, value: float) -> Tuple[float, bool]:
        """Update series with ``value`` and return (z, muted).

        ``muted`` becomes ``True`` only after ``confirm`` consecutive
        observations exceed ``threshold`` in absolute z-score terms.
        """
        self.history.append(float(value))
        if len(self.history) >= 5 and statistics.pstdev(self.history) > 0:
            m = statistics.mean(self.history)
            s = statistics.pstdev(self.history) + 1e-9
            z = (value - m) / s
        else:
            z = 0.0
        if abs(z) >= self.threshold:
            self._spikes += 1
        else:
            self._spikes = 0
        muted = self._spikes >= self.confirm
        return z, muted


def load_gate_settings() -> Dict[str, float]:
    """Return gate thresholds from settings.json with fallbacks."""
    cfg = load_settings() or {}
    thr_cfg = cfg.get("GATE_THRESHOLDS", {}) if isinstance(cfg, dict) else {}
    try:
        pcr = float(thr_cfg.get("PCR_Z", 2.0))
    except Exception:
        pcr = 2.0
    try:
        iv = float(thr_cfg.get("IV_Z", 2.0))
    except Exception:
        iv = 2.0
    return {"pcr": pcr, "iv": iv}


__all__ = ["RollingZGate", "load_gate_settings"]
