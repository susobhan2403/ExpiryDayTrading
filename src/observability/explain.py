from __future__ import annotations
import json
from typing import Any, Dict, Sequence
import datetime as dt


def emit_explain(
    index: str,
    expiry: dt.datetime,
    tau: float,
    step: int,
    F: float,
    K_atm: float,
    ivs: Dict[str, Any],
    pcr: Dict[str, Any],
    signals: Sequence[str],
    gates: Dict[str, Any],
    decision: str,
    dq_flags: Sequence[str],
) -> str:
    """Return a structured JSON blob explaining a single run.

    Parameters capture the core metrics and decisions for transparency and
    debugging.  ``expiry`` is expected to be timezone aware.
    """

    payload = {
        "index": index,
        "expiry": expiry,
        "tau": tau,
        "step": step,
        "forward": F,
        "K_atm": K_atm,
        "ivs": ivs,
        "pcr": pcr,
        "signals": list(signals),
        "gates": gates,
        "decision": decision,
        "dq_flags": list(dq_flags),
    }
    return json.dumps(payload, default=str)


__all__ = ["emit_explain"]
