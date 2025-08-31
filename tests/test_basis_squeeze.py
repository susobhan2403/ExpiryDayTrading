import datetime as dt
import pandas as pd
import engine

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

class DummyProvider:
    def get_spot_ohlcv(self, *args, **kwargs):
        return pd.DataFrame()

def make_df(rows=30, value=100):
    idx = pd.date_range(dt.datetime(2024,1,1,12,0,tzinfo=IST) - pd.Timedelta(minutes=rows-1), periods=rows, freq="1min")
    return pd.DataFrame({
        "open": [value]*rows,
        "high": [value]*rows,
        "low": [value]*rows,
        "close": [value]*rows,
        "volume": [100]*rows,
    }, index=idx)

def test_basis_squeeze_act_and_ignore():
    spot1 = make_df()
    spot5 = make_df()
    now = dt.datetime(2024,1,1,12,0,tzinfo=IST)
    chain = {"strikes": [], "calls": {}, "puts": {}}
    common_args = [
        DummyProvider(),
        "TEST",
        now,
        spot1,
        spot5,
        100.0,  # vwap_spot
        1.0,    # pcr
        1.0,    # iv_z
        50.0,   # iv_pct
        1.0,    # VND
        25.0,   # adx5
        50.0,   # rsi5
        0.0,    # macd_last
        0.0,    # macd_sig_last
        100.0,  # span_a_last
        100.0,  # span_b_last
        100.0,  # bb_u
        100.0,  # bb_l
        100.0,  # bb_m
        0.005,  # micro_spread_pct
        0.0,    # micro_cvd_slope
        5.0,    # micro_stab
        100.0,  # don_hi20
        90.0,   # don_lo20
        10.0,   # ATR_D
        0.0,    # D
        0.0,    # basis
        0.6,    # basis_slope
        chain,  # chain
        10.0,   # atm_iv
        100.0,  # prev_close
        {},     # vp
    ]
    alerts = engine.detect_spike_traps(*common_args, oi_flags={"ce_write_above": True})
    assert any(a.action == "ACT" and a.message == "Basis squeeze trend" for a in alerts)
    alerts2 = engine.detect_spike_traps(*common_args, oi_flags={})
    assert any(a.action == "IGNORE" and a.message == "Basis squeeze" for a in alerts2)


def test_apply_alert_bias():
    probs = {sc: 0.1 for sc in engine.SCENARIOS}
    alerts = [engine.AlertEvent("ACT", "Basis squeeze trend")]
    adjusted = engine.apply_alert_bias(probs, alerts)
    assert adjusted[engine.SCENARIOS[4]] > probs[engine.SCENARIOS[4]]
