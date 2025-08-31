import json
import pytest
import src.config as config
from src.config import compute_dynamic_bands, load_settings


def test_dynamic_bands_basic():
    above, below, far_pts, pin_pts = compute_dynamic_bands(
        symbol="NIFTY", expiry_today=False, ATR_D=120.0, adx5=20.0, VND=0.5, D=30.0
    )
    assert 1 <= above <= 6
    assert 1 <= below <= 6
    assert 300 <= far_pts <= 1500
    assert 50 <= pin_pts <= 300


def test_dynamic_bands_sensex():
    above, below, far_pts, pin_pts = compute_dynamic_bands(
        symbol="SENSEX", expiry_today=True, ATR_D=100.0, adx5=10.0, VND=0.1, D=50.0
    )
    assert (far_pts, pin_pts) == (600, 105)


def test_dynamic_bands_midcpnifty():
    above, below, far_pts, pin_pts = compute_dynamic_bands(
        symbol="MIDCPNIFTY", expiry_today=True, ATR_D=100.0, adx5=10.0, VND=0.1, D=50.0
    )
    assert (far_pts, pin_pts) == (600, 105)


def test_settings_without_snapshot_minutes():
    cfg = load_settings()
    assert "SNAPSHOT_MINUTES" not in cfg


def test_dynamic_bands_respect_global_defaults(tmp_path, monkeypatch):
    cfg = {
        "BAND_MAX_STRIKES_ABOVE": 5,
        "BAND_MAX_STRIKES_BELOW": 4,
        "FAR_OTM_FILTER_POINTS": 700,
        "PIN_DISTANCE_POINTS": 120,
        "PRESETS": {},
    }
    (tmp_path / "settings.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(config, "ROOT", tmp_path)
    above, below, far_pts, pin_pts = compute_dynamic_bands(
        symbol="NIFTY",
        expiry_today=False,
        ATR_D=0.0,
        adx5=15.0,
        VND=0.4,
        D=50.0,
    )
    assert (above, below, far_pts, pin_pts) == (5, 4, 700, 120)


def test_dynamic_bands_presets_override_globals(tmp_path, monkeypatch):
    cfg = {
        "BAND_MAX_STRIKES_ABOVE": 5,
        "BAND_MAX_STRIKES_BELOW": 4,
        "FAR_OTM_FILTER_POINTS": 700,
        "PIN_DISTANCE_POINTS": 120,
        "PRESETS": {
            "NIFTY": {
                "BAND_MAX_STRIKES_ABOVE": 2,
                "BAND_MAX_STRIKES_BELOW": 1,
                "FAR_OTM_FILTER_POINTS": 300,
                "PIN_DISTANCE_POINTS": 60,
            }
        },
    }
    (tmp_path / "settings.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(config, "ROOT", tmp_path)
    above, below, far_pts, pin_pts = compute_dynamic_bands(
        symbol="NIFTY",
        expiry_today=False,
        ATR_D=0.0,
        adx5=15.0,
        VND=0.4,
        D=50.0,
    )
    assert (above, below, far_pts, pin_pts) == (2, 1, 300, 60)

