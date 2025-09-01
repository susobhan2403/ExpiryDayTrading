import json
import engine


def test_replay(tmp_path):
    snap1 = {
        "ts": "2025-08-29T09:30:00+05:30",
        "spot": 100.0,
        "scen_top": "S1",
        "trade": {"action": "BUY_CE", "instrument": "CE 100", "targets": ["110"]},
    }
    snap2 = {
        "ts": "2025-08-29T09:31:00+05:30",
        "spot": 112.0,
        "scen_top": "S2",
        "trade": {"action": "NO-TRADE"},
    }
    (tmp_path / "a.json").write_text(json.dumps(snap1))
    (tmp_path / "b.json").write_text(json.dumps(snap2))
    engine.replay_snapshots(str(tmp_path / "*.json"), speed=0)
    report = json.loads((engine.OUT_DIR / "replay_report.json").read_text())
    assert report["trades"] == 1
    assert report["hit_rate"] == 1.0
    assert report["avg_mfe"] == 12.0
    assert report["avg_mae"] == 0.0
    assert report["confusion"] == {"S1": {"S2": 1}}
