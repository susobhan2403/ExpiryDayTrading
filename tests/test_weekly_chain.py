import datetime as dt
import pandas as pd
import pytest

import engine as eng
import src.provider.kite as kite


def _make_df(expiries):
    return pd.DataFrame(
        {
            "name": ["NIFTY"] * len(expiries),
            "segment": ["NFO-OPT"] * len(expiries),
            "expiry": expiries,
            "strike": [100] * len(expiries),
        }
    )


def test_engine_skips_monthly(monkeypatch):
    expiries = [
        dt.date(2024, 1, 25),  # monthly
        dt.date(2024, 2, 8),
        dt.date(2024, 2, 15),
        dt.date(2024, 2, 22),
        dt.date(2024, 2, 29),
    ]
    e = eng.KiteProvider.__new__(eng.KiteProvider)
    e._instruments = lambda exch: _make_df(expiries)

    class Fixed(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime(2024, 1, 20, 10, 0, tzinfo=tz)

    monkeypatch.setattr(eng.dt, "datetime", Fixed)
    _, expiry = e._nearest_weekly_chain_df("NIFTY")
    assert expiry == "2024-02-08"


def test_provider_skips_monthly(monkeypatch):
    expiries = [
        dt.date(2024, 1, 18),
        dt.date(2024, 1, 25),  # monthly
        dt.date(2024, 2, 8),
        dt.date(2024, 2, 15),
        dt.date(2024, 2, 22),
    ]
    p = kite.KiteProvider.__new__(kite.KiteProvider)
    p._instruments = lambda exch: _make_df(expiries)

    class Fixed(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime(2024, 1, 10, 10, 0, tzinfo=tz)

    monkeypatch.setattr(kite.dt, "datetime", Fixed)
    _, expiry = p._nearest_weekly_chain_df("NIFTY")
    assert expiry == "2024-01-18"

