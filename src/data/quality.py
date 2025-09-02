from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Tick:
    ts: dt.datetime
    bid: float
    ask: float
    ltp: float
    oi: int
    volume: int
    quote_age: float          # seconds since exchange timestamp
    meta: Dict[str, str] = field(default_factory=dict)


class TickValidator:
    """Validate and quarantine bad ticks.

    Rolls a window of recent rejects so that recurring offenders are muted
    upstream.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.quarantine: Dict[str, dt.datetime] = {}

    def _is_quarantined(self, symbol: str, now: dt.datetime) -> bool:
        until = self.quarantine.get(symbol)
        return bool(until and now < until)

    def _quarantine(self, symbol: str, now: dt.datetime) -> None:
        win = dt.timedelta(seconds=self.cfg["data"]["quarantine_window"])
        self.quarantine[symbol] = now + win

    def validate(self, symbol: str, tick: Tick) -> bool:
        now = dt.datetime.now(tz=dt.timezone.utc)
        if self._is_quarantined(symbol, now):
            return False
        # sanity checks
        if tick.quote_age > self.cfg["data"]["stale_quote_sec"]:
            self._quarantine(symbol, now)
            return False
        if tick.bid <= 0 or tick.ask <= 0 or tick.bid >= tick.ask:
            self._quarantine(symbol, now)
            return False
        if tick.oi < self.cfg["data"]["min_oi"] or tick.volume <= 0:
            return False
        return True
