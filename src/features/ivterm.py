from __future__ import annotations
import json, pathlib, datetime as dt
from typing import Dict, List

ROOT = pathlib.Path(__file__).resolve().parents[2]

def _store_path(symbol: str, term: str) -> pathlib.Path:
    d = ROOT / 'out' / 'iv_term'
    d.mkdir(parents=True, exist_ok=True)
    return d / f'{symbol.upper()}_{term}.json'

def update_iv_history(symbol: str, term: str, iv: float, now: dt.datetime) -> None:
    """Append daily IV value (once per day) for rolling IVP/IVR calculations."""
    p = _store_path(symbol, term)
    arr: List[Dict] = []
    if p.exists():
        try:
            arr = json.loads(p.read_text())
        except Exception:
            arr = []
    # keep only last 252 entries
    day = now.date().isoformat()
    if iv==iv and iv>0:
        if arr and arr[-1].get('date') == day:
            arr[-1]['iv'] = iv
        else:
            arr.append({'date': day, 'iv': iv})
        arr = arr[-252:]
        p.write_text(json.dumps(arr))

def ivp_ivr(symbol: str, term: str) -> Dict[str, float]:
    """Compute IVP (percentile) and IVR (range percentile) from stored history."""
    p = _store_path(symbol, term)
    if not p.exists():
        return {'ivp': float('nan'), 'ivr': float('nan')}
    try:
        arr = json.loads(p.read_text())
        vals = [float(x['iv']) for x in arr if x.get('iv',0)>0]
        if len(vals) < 30:
            return {'ivp': float('nan'), 'ivr': float('nan')}
        cur = vals[-1]
        rank = sum(1 for v in vals if v <= cur)/len(vals)
        ivp = 100.0*rank
        lo, hi = min(vals), max(vals)
        ivr = 100.0*((cur - lo)/max(1e-9, (hi - lo)))
        return {'ivp': ivp, 'ivr': ivr}
    except Exception:
        return {'ivp': float('nan'), 'ivr': float('nan')}

