from __future__ import annotations
import json
import datetime as dt
import pathlib
from typing import Dict, List

ROOT = pathlib.Path(__file__).resolve().parents[2]

def _load_events() -> List[Dict]:
    path = ROOT / "data" / "events.json"
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return []

def event_features(now_ist: dt.datetime, symbol: str) -> Dict[str, float]:
    """
    Returns simple one-hot for upcoming event within 7 days and T-days to nearest event for the symbol group.
    Expects optional data/events.json with entries: {"date":"YYYY-MM-DD","tags":["RBI","MSCI",...],"symbols":["NIFTY","BANKNIFTY",...]}
    """
    evs = _load_events()
    today = now_ist.date()
    tmin = 999
    flag = 0.0
    for e in evs:
        try:
            d = dt.date.fromisoformat(e.get("date",""))
        except Exception:
            continue
        syms = [s.upper() for s in (e.get("symbols") or [])]
        if symbol.upper() in syms or not syms:
            dt_days = (d - today).days
            if dt_days >= 0:
                tmin = min(tmin, dt_days)
                if dt_days <= 7:
                    flag = 1.0
    out = {
        "event_within_7d": flag,
        "days_to_event": float(tmin if tmin != 999 else 999.0)
    }
    return out

