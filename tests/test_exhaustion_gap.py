import datetime as dt
import pandas as pd
import engine

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


class DummyProvider:
    def get_spot_ohlcv(self, *args, **kwargs):
        return pd.DataFrame()


def make_gap_df():
    idx = pd.date_range(dt.datetime(2024, 1, 1, 9, 15, tzinfo=IST), periods=30, freq="1min")
    data = {
        "open": [115] + [110] * 29,
        "high": [120] * 6 + [110] * 24,
        "low": [110] * 6 + [90] * 24,
        "close": [112] * 6 + [90] * 24,
        "volume": [100] * 30,
    }
    return pd.DataFrame(data, index=idx)


def make_df(rows=30, value=100):
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


def test_exhaustion_gap_act_and_ignore():
    spot1 = make_gap_df()
    spot5 = make_df()
    now = dt.datetime(2024, 1, 1, 9, 30, tzinfo=IST)
    chain = {"strikes": [], "calls": {}, "puts": {}}
    common_args = [
        DummyProvider(),
        "TEST",
        now,
        spot1,
        spot5,
        100.0,  # vwap_spot
        1.0,  # pcr
        2.0,  # iv_z
        50.0,  # iv_pct
        1.0,  # VND
        25.0,  # adx5
        50.0,  # rsi5
        0.0,  # macd_last
        0.0,  # macd_sig_last
        100.0,  # span_a_last
        100.0,  # span_b_last
        100.0,  # bb_u
        100.0,  # bb_l
        100.0,  # bb_m
        0.005,  # micro_spread_pct
        -0.2,  # micro_cvd_slope
        4.0,  # micro_stab
        200.0,  # don_hi20
        80.0,  # don_lo20
        10.0,  # ATR_D
        0.0,  # D
        0.0,  # basis
        0.0,  # basis_slope
        chain,
        10.0,  # atm_iv
        100.0,  # prev_close
        {},  # vp
    ]

    alerts = engine.detect_spike_traps(
        *common_args, oi_flags={"ce_unwind_below": True}
    )
    assert any(
        a.action == "ACT" and a.message == "Exhaustion gap failure" for a in alerts
    )

    alerts2 = engine.detect_spike_traps(*common_args, oi_flags={})
    assert any(
        a.action == "IGNORE" and a.message == "Exhaustion gap" for a in alerts2
    )

    common_args_bad = common_args.copy()
    common_args_bad[19] = 0.012  # micro_spread_pct too wide
    alerts3 = engine.detect_spike_traps(
        *common_args_bad, oi_flags={"ce_unwind_below": True}
    )
    assert any(
        a.action == "IGNORE" and a.message == "Exhaustion gap" for a in alerts3
    )

