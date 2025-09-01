import engine

def test_stop_moved_and_no_add_after_ignore():
    state = {}
    trade = {"direction": 1, "stop": 100.0}
    alerts = [engine.AlertEvent("IGNORE", "test")]
    exit_frac = engine.manage_trade_on_ignore(state, trade, alerts, vwap=110.0, last_swing=108.0, snap_idx=1)
    assert exit_frac == 0.0
    assert trade["stop"] == 110.0
    assert state["add_allowed"] is False
    assert state["last_ignore_snap"] == 1

def test_second_ignore_triggers_scale_out():
    state = {}
    trade = {"direction": 1, "stop": 100.0}
    alerts = [engine.AlertEvent("IGNORE", "first")]
    engine.manage_trade_on_ignore(state, trade, alerts, vwap=110.0, last_swing=108.0, snap_idx=1)
    # intermediate snapshot without ignore
    engine.manage_trade_on_ignore(state, trade, [], vwap=111.0, last_swing=109.0, snap_idx=2)
    # second ignore within two snapshots
    alerts2 = [engine.AlertEvent("IGNORE", "second")]
    exit_frac = engine.manage_trade_on_ignore(state, trade, alerts2, vwap=112.0, last_swing=111.0, snap_idx=3)
    assert exit_frac == 0.25
    assert trade["stop"] == 112.0
    assert trade["scale_out"] == 0.25
    assert state["add_allowed"] is False
    assert state["last_ignore_snap"] == 3

def test_short_direction_stop_moves_down():
    state = {}
    trade = {"direction": -1, "stop": 120.0}
    alerts = [engine.AlertEvent("IGNORE", "test")]
    engine.manage_trade_on_ignore(state, trade, alerts, vwap=115.0, last_swing=117.0, snap_idx=5)
    assert trade["stop"] == 115.0
    assert state["add_allowed"] is False
