from __future__ import annotations
import json
from typing import Any, Dict


def emit_explain(
    signal_name: str,
    inputs: Dict[str, Any],
    gates: Dict[str, bool],
    decision: str,
    meta: Dict[str, Any],
) -> str:
    """Return structured explain JSON for logging."""
    payload = {
        "signal": signal_name,
        "inputs": inputs,
        "gates": gates,
        "decision": decision,
        **meta,
    }
    return json.dumps(payload, default=str)


__all__ = ["emit_explain"]
