import datetime as dt
import pandas as pd
import engine

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


class DummyProvider:
    def get_spot_ohlcv(self, *args, **kwargs):
        return pd.DataFrame()


def make_df(rows=21, value=100):
    idx = pd.date_range(dt.datetime(2024, 1, 1, 9, 15, tzinfo=IST), periods=rows, freq="1min")
    return pd.DataFrame(
        {
            "open": [value] * rows,
            "high": [value] * rows,
            "low": [value] * rows,
            "close": [value] * rows,
            "volume": [100] * rows,
        },
        index=idx,
    )


def test_iv_range_spike_promotes_to_act():
    chain = {"strikes": [], "calls": {}, "puts": {}}
    vp = {"POC": 100.0, "VAL": 95.0, "VAH": 105.0}
    state = {}

    spot1 = make_df()
    spot1.iloc[-1] = [80, 120, 80, 80, 100]
    spot5 = make_df()
    now = dt.datetime(2024, 1, 1, 9, 30, tzinfo=IST)

    common_args = [
        DummyProvider(),
        "TEST",
        now,
        spot1,
        spot5,
        100.0,
        1.0,
        2.5,
        50.0,
        1.0,
        25.0,
        50.0,
        0.0,
        0.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        0.005,
        0.0,
        4.0,
        200.0,
        80.0,
        10.0,
        0.0,
        0.0,
        0.0,
        chain,
        10.0,
        100.0,
        vp,
    ]

    alerts1 = engine.detect_spike_traps(
        *common_args, oi_flags={"pe_write_above": True}, state=state
    )
    assert any(a.action == "IGNORE" and a.message == "IV range spike" for a in alerts1)
    assert state.get("iv_range_event", {}).get("active")

    now2 = now + dt.timedelta(minutes=1)
    idx2 = spot1.index[-1] + pd.Timedelta(minutes=1)
    row2 = pd.DataFrame(
        {"open": [90], "high": [95], "low": [85], "close": [90], "volume": [100]},
        index=[idx2],
    )
    spot1_2 = pd.concat([spot1, row2])

    alerts2 = engine.detect_spike_traps(
        DummyProvider(),
        "TEST",
        now2,
        spot1_2,
        spot5,
        100.0,
        1.0,
        -0.5,
        50.0,
        1.0,
        25.0,
        50.0,
        0.0,
        0.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        0.005,
        0.0,
        4.0,
        200.0,
        80.0,
        10.0,
        0.0,
        0.0,
        0.0,
        chain,
        10.0,
        100.0,
        vp,
        oi_flags={"ce_write_above": True},
        state=state,
    )
    assert not any(a.action == "ACT" for a in alerts2)
    assert state.get("iv_range_event", {}).get("oi_flip_count") == 1

    now3 = now2 + dt.timedelta(minutes=1)
    idx3 = spot1_2.index[-1] + pd.Timedelta(minutes=1)
    row3 = pd.DataFrame(
        {"open": [102], "high": [103], "low": [99], "close": [102], "volume": [100]},
        index=[idx3],
    )
    spot1_3 = pd.concat([spot1_2, row3])

    alerts3 = engine.detect_spike_traps(
        DummyProvider(),
        "TEST",
        now3,
        spot1_3,
        spot5,
        100.0,
        1.0,
        -0.5,
        50.0,
        1.0,
        25.0,
        50.0,
        0.0,
        0.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        0.005,
        0.0,
        4.0,
        200.0,
        80.0,
        10.0,
        0.0,
        0.0,
        0.0,
        chain,
        10.0,
        100.0,
        vp,
        oi_flags={"ce_write_above": True},
        state=state,
    )
    assert any(a.action == "ACT" and a.message == "IV range spike reversal" for a in alerts3)
