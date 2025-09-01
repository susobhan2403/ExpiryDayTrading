from __future__ import annotations
import json
import pathlib
from typing import Dict, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]

STEP_MAP = {
    "BANKNIFTY": 100,
    "SENSEX": 100,
    "NIFTY": 50,
    "MIDCPNIFTY": 25,
}

def load_settings() -> Dict:
    path = ROOT / "settings.json"
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}

def save_settings(cfg: Dict) -> None:
    path = ROOT / "settings.json"
    try:
        path.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass

def compute_dynamic_bands(symbol: str, expiry_today: bool, ATR_D: float, adx5: float, VND: float, D: float, step: int | None = None) -> Tuple[int,int,int,int]:
    sym = symbol.upper()
    step = step if step is not None else STEP_MAP.get(sym, 50)
    is_bank = "BANK" in sym
    base_above = 3
    base_below = 2
    base_far = 1200 if is_bank else 600
    base_pin = 200 if is_bank else 100
    # presets from settings.json (optional)
    try:
        cfg = load_settings()
        presets = (cfg.get("PRESETS") or {}).get(symbol.upper()) or {}
        base_above = int(
            presets.get(
                "BAND_MAX_STRIKES_ABOVE",
                cfg.get("BAND_MAX_STRIKES_ABOVE", base_above),
            )
        )
        base_below = int(
            presets.get(
                "BAND_MAX_STRIKES_BELOW",
                cfg.get("BAND_MAX_STRIKES_BELOW", base_below),
            )
        )
        base_far = int(
            presets.get(
                "FAR_OTM_FILTER_POINTS",
                cfg.get("FAR_OTM_FILTER_POINTS", base_far),
            )
        )
        base_pin = int(
            presets.get(
                "PIN_DISTANCE_POINTS",
                cfg.get("PIN_DISTANCE_POINTS", base_pin),
            )
        )
    except Exception:
        pass

    PIN_NORM = 0.4
    ADX_LOW = 15

    range_bound = (VND < PIN_NORM and adx5 < ADX_LOW and abs(D) <= base_pin)
    trending = (adx5 >= 25) or (VND >= 1.5) or (ATR_D >= (300 if is_bank else 150))

    above = base_above
    below = base_below
    far_pts = base_far
    pin_pts = base_pin

    if expiry_today:
        if trending:
            above += 1; below += 1
            far_pts += (400 if is_bank else 200)
            pin_pts = int(pin_pts * 1.25)
        else:
            above = max(2, above); below = max(2, below)
            # Previously we reduced the far OTM filter on non-trending expiry
            # days which yielded overly tight bands (e.g. 550pt for
            # MIDCPNIFTY).  Keeping the baseline distance matches external
            # references and test expectations.
            far_pts = base_far
            pin_pts = max(int(pin_pts * 0.7), step)
    else:
        if trending:
            above += 1; below += 1
            far_pts += (400 if is_bank else 200)
            pin_pts = int(pin_pts * 1.5)
        if range_bound:
            above = min(above, 2)
            below = min(below, 2)
            far_pts = max(base_far - (400 if is_bank else 200), base_far//2)
            pin_pts = max(int(pin_pts * 0.7), step)

    above = int(max(1, min(6, above)))
    below = int(max(1, min(6, below)))
    far_pts = int(max(6*step, min(30*step, far_pts)))
    pin_pts = int(max(1*step, min(6*step, pin_pts)))
    return above, below, far_pts, pin_pts

# backward-compat helpers
_load_settings = load_settings
_save_settings = save_settings
