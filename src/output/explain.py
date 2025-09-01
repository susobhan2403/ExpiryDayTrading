"""Utility for writing lightweight explanation snapshots.

The engine produces rich :class:`Snapshot` objects containing numerous
metrics.  Downstream dashboards only need a small subset of these fields to
show the current state.  :func:`write_explain` extracts a stable schema from a
snapshot dictionary and writes it out as JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

# Keys that make up the explanation schema.  The values are short field
# descriptions purely for documentation.  The order of keys is preserved when
# writing the JSON output for readability.
EXPLAIN_SCHEMA: Dict[str, str] = {
    "ts": "timestamp of the snapshot in ISO format",
    "symbol": "instrument symbol",
    "expiry": "selected weekly expiry (ISO date)",
    "step": "strike step size",
    "atm": "at-the-money strike",
    "pcr": "put-call ratio",
    "dpcr_z": "delta PCR z-score",
    "vnd": "volatility normalised distance",
    "mph_norm": "max pain drift normalised",
    "iv_z": "implied volatility z-score",
}


def write_explain(snapshot: Dict[str, Any], path: Path) -> None:
    """Write an explanation snapshot to ``path`` in JSON format.

    Parameters
    ----------
    snapshot:
        Mapping containing snapshot metrics.  Only the keys defined in
        :data:`EXPLAIN_SCHEMA` are persisted which keeps the written file
        stable even if the caller passes additional data.
    path:
        Destination path for the JSON output.
    """

    data = {key: snapshot.get(key) for key in EXPLAIN_SCHEMA}
    path.write_text(json.dumps(data, indent=2))


__all__ = ["write_explain", "EXPLAIN_SCHEMA"]

