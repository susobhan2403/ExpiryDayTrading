import pytest
from src.config import compute_dynamic_bands

def test_dynamic_bands_basic():
    # NIFTY, non-expiry, modest trend
    above, below, far_pts, pin_pts = compute_dynamic_bands(
        symbol="NIFTY", expiry_today=False, ATR_D=120.0, adx5=20.0, VND=0.5, D=30.0, step=50
    )
    assert 1 <= above <= 6
    assert 1 <= below <= 6
    assert 300 <= far_pts <= 1500
    assert 50 <= pin_pts <= 300

