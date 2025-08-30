from __future__ import annotations
import json
import pathlib
from typing import Dict

ROOT = pathlib.Path(__file__).resolve().parents[2]
FLAG_PATH = ROOT / "data" / "headlines.json"

IV_Z_TRIGGER = 2.0
INR_MOVE_TRIGGER = 0.3  # percentage move in USDINR

def _load_flags() -> Dict[str, bool]:
    """Best-effort read of headline flags from optional data/headlines.json."""
    try:
        if FLAG_PATH.exists():
            return json.loads(FLAG_PATH.read_text())
    except Exception:
        pass
    return {}

def macro_triggered(iv_z: float = 0.0, inr_move: float = 0.0) -> bool:
    """Return True if macro triggers like INR move or IV spike occur."""
    return abs(inr_move) >= INR_MOVE_TRIGGER or abs(iv_z) >= IV_Z_TRIGGER

def headline_active(iv_z: float = 0.0, inr_move: float = 0.0) -> bool:
    """Check if manual headline flags or macro triggers are active.

    Flags are read from data/headlines.json with optional keys like
    "RBI_window" or "tariff_news". Macro triggers include INR moves and
    implied volatility z-score spikes.
    """
    flags = _load_flags()
    manual = bool(flags.get("RBI_window")) or bool(flags.get("tariff_news"))
    return manual or macro_triggered(iv_z=iv_z, inr_move=inr_move)
