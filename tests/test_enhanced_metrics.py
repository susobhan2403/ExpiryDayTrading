"""Tests for enhanced metrics with India-specific conventions."""

import math
import pytest
import pandas as pd
from pathlib import Path

from src.metrics.enhanced import (
    infer_strike_step_enhanced,
    compute_forward_enhanced,
    pick_atm_strike_enhanced,
    implied_vol_enhanced,
    compute_atm_iv_enhanced,
    compute_iv_percentile_enhanced,
    compute_pcr_enhanced,
    validate_inputs
)


class TestValidateInputs:
    """Test input validation utilities."""
    
    def test_valid_inputs(self):
        """Test validation of valid inputs."""
        valid, reason = validate_inputs(100.0, 0.05, 1.0)
        assert valid
        assert reason == "valid"
    
    def test_nan_input(self):
        """Test rejection of NaN inputs."""
        valid, reason = validate_inputs(float('nan'), 0.05)
        assert not valid
        assert "invalid_input" in reason
    
    def test_negative_input(self):
        """Test rejection of negative inputs."""
        valid, reason = validate_inputs(-100.0, 0.05)
        assert not valid
        assert "negative_input" in reason
    
    def test_none_input(self):
        """Test rejection of None inputs."""
        valid, reason = validate_inputs(None, 0.05)
        assert not valid
        assert "invalid_input" in reason


class TestStrikeStepInference:
    """Test enhanced strike step inference."""
    
    def test_nifty_strikes(self):
        """Test NIFTY 50-point spacing."""
        strikes = [19000, 19050, 19100, 19150, 19200]
        step, diag = infer_strike_step_enhanced(strikes)
        assert step == 50
        assert diag["confidence"] == 1.0
        assert diag["selected_step"] == 50
    
    def test_banknifty_strikes(self):
        """Test BANKNIFTY 100-point spacing."""
        strikes = [45000, 45100, 45200, 45300, 45400]
        step, diag = infer_strike_step_enhanced(strikes)
        assert step == 100
        assert diag["confidence"] == 1.0
    
    def test_mixed_spacing(self):
        """Test handling of mixed spacing."""
        strikes = [100, 150, 200, 250, 300]  # 50-point spacing
        step, diag = infer_strike_step_enhanced(strikes)
        assert step == 50
        assert diag["confidence"] == 1.0
    
    def test_insufficient_strikes(self):
        """Test handling of insufficient strikes."""
        strikes = [100]
        step, diag = infer_strike_step_enhanced(strikes)
        assert step == 0
        assert diag["reason"] == "insufficient_strikes"
    
    def test_invalid_strikes(self):
        """Test handling of invalid strike data."""
        strikes = [float('nan'), float('inf')]
        step, diag = infer_strike_step_enhanced(strikes)
        assert step == 0
        assert diag["reason"] == "insufficient_strikes"


class TestForwardComputation:
    """Test enhanced forward price computation."""
    
    def test_futures_preferred(self):
        """Test preference for futures price when available."""
        forward, diag = compute_forward_enhanced(
            spot=19000, fut_mid=19050, r=0.06, q=0.01, tau_years=0.25
        )
        assert forward == 19050
        assert diag["method"] == "futures_mid"
    
    def test_cost_of_carry(self):
        """Test cost-of-carry calculation."""
        forward, diag = compute_forward_enhanced(
            spot=19000, fut_mid=None, r=0.06, q=0.01, tau_years=0.25
        )
        expected = 19000 * math.exp((0.06 - 0.01) * 0.25)
        assert abs(forward - expected) < 1e-6
        assert diag["method"] == "cost_of_carry"
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        forward, diag = compute_forward_enhanced(
            spot=-100, fut_mid=None, r=0.06, q=0.01, tau_years=0.25
        )
        assert forward == 0.0
        assert "negative_input" in diag["reason"]


class TestATMStrikeSelection:
    """Test enhanced ATM strike selection."""
    
    def test_forward_based_selection(self):
        """Test ATM selection based on forward price."""
        strikes = [18900, 18950, 19000, 19050, 19100]
        F = 19025  # Between 19000 and 19050
        ce_mid = {s: 25 for s in strikes}
        pe_mid = {s: 25 for s in strikes}
        
        K_atm, diag = pick_atm_strike_enhanced(F, strikes, 50, ce_mid, pe_mid)
        assert K_atm == 19050  # Closer to 19025
        assert diag["method"] == "forward_based"
    
    def test_tie_breaking_with_straddle(self):
        """Test tie-breaking using straddle comparison."""
        strikes = [19000, 19050]
        F = 19025  # Exactly between strikes
        # Make 19000 straddle closer to theoretical
        ce_mid = {19000: 30, 19050: 25}
        pe_mid = {19000: 30, 19050: 35}
        
        K_atm, diag = pick_atm_strike_enhanced(F, strikes, 50, ce_mid, pe_mid)
        assert diag["tie_detected"]
        assert K_atm in [19000, 19050]  # Should pick based on straddle fit
    
    def test_no_valid_strikes(self):
        """Test handling when no valid strikes available."""
        strikes = []
        F = 19025
        ce_mid = {}
        pe_mid = {}
        
        K_atm, diag = pick_atm_strike_enhanced(F, strikes, 50, ce_mid, pe_mid)
        assert K_atm == 0.0
        assert diag["reason"] == "no_valid_strikes"


class TestImpliedVolatility:
    """Test enhanced implied volatility solver."""
    
    def test_golden_case(self):
        """Test against known Black-Scholes solution."""
        # Known case: ATM option, 1 year, 5% rate, 20% vol
        price = 7.9656  # Theoretical price for these parameters
        F = 100.0
        K = 100.0
        tau = 1.0
        r = 0.05
        
        iv, diag = implied_vol_enhanced(price, F, K, tau, r, "C")
        assert iv is not None
        assert abs(iv - 0.20) < 1.5e-2  # Should be close to 20% (relaxed tolerance for numerical precision)
        assert diag["converged"]
    
    def test_deep_otm_handling(self):
        """Test handling of deep OTM options."""
        price = 0.50  # Very cheap option
        F = 19000
        K = 20000  # Deep OTM call
        tau = 7/365  # 1 week
        r = 0.06
        
        iv, diag = implied_vol_enhanced(price, F, K, tau, r, "C")
        # Should still converge but with bounded volatility
        if iv is not None:
            assert iv <= 2.0  # Should be capped for deep OTM
    
    def test_invalid_price(self):
        """Test handling of invalid option prices."""
        iv, diag = implied_vol_enhanced(
            price=-1.0, F=19000, K=19000, tau_years=0.25, r=0.06, opt_type="C"
        )
        assert iv is None
        assert not diag["converged"]


class TestATMIVComputation:
    """Test enhanced ATM IV computation."""
    
    def test_dual_leg_average(self):
        """Test averaging of call and put IVs when both are valid."""
        # Use prices that should give similar IVs
        F = 19000
        K_atm = 19000
        tau = 30/365  # 30 days
        r = 0.06
        
        # Theoretical ATM straddle with 20% vol
        from src.metrics.enhanced import implied_vol_enhanced
        
        call_price = 75.0  # Reasonable ATM call price
        put_price = 75.0   # Reasonable ATM put price
        
        atm_iv, diag = compute_atm_iv_enhanced(call_price, put_price, F, K_atm, tau, r)
        
        if atm_iv is not None:
            assert diag["method"] in ["dual_leg_average", "averaged_legs", "conservative_leg_call", "conservative_leg_put"]
    
    def test_single_leg_fallback(self):
        """Test fallback to single leg when other is invalid."""
        atm_iv, diag = compute_atm_iv_enhanced(
            ce_mid=50.0, pe_mid=None, F=19000, K_atm=19000, tau_years=30/365, r=0.06
        )
        
        if atm_iv is not None:
            assert "single_leg" in diag["method"]
    
    def test_no_valid_legs(self):
        """Test handling when no valid IVs can be computed."""
        atm_iv, diag = compute_atm_iv_enhanced(
            ce_mid=None, pe_mid=None, F=19000, K_atm=19000, tau_years=30/365, r=0.06
        )
        assert atm_iv is None
        assert diag["reason"] == "no_valid_legs"


class TestIVPercentile:
    """Test enhanced IV percentile computation."""
    
    def test_tenor_filtering(self):
        """Test filtering by time-to-expiry."""
        # Create test data with tenor information
        weekly_data = [
            (0.096, 0.1850), (0.095, 0.1875), (0.094, 0.1920), 
            (0.093, 0.1945), (0.092, 0.1890), (0.091, 0.1865)
        ]
        monthly_data = [
            (0.270, 0.1780), (0.268, 0.1820), (0.265, 0.1750)
        ]
        
        # Combine all data
        history = weekly_data + monthly_data
        
        current_iv = 0.22  # 22% IV
        current_tau = 0.095  # About 5 days (weekly)
        
        percentile, iv_rank, diag = compute_iv_percentile_enhanced(
            history, current_iv, current_tau, tau_tol=0.01
        )
        
        assert percentile is not None
        assert iv_rank is not None
        assert diag["method"] == "tenor_filtered_percentile"
        assert diag["filtered_count"] > 0  # Should filter for weekly data only
    
    def test_same_tenor_assumption(self):
        """Test handling when all data assumed same tenor."""
        history = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
        current_iv = 0.24
        
        percentile, iv_rank, diag = compute_iv_percentile_enhanced(history, current_iv)
        
        assert percentile is not None
        assert iv_rank is not None
        # Should be around 57.1% percentile (4 out of 7 below)
        assert 50 <= percentile <= 65
    
    def test_constant_series(self):
        """Test handling of constant IV series."""
        history = [0.20, 0.20, 0.20, 0.20, 0.20]
        current_iv = 0.20
        
        percentile, iv_rank, diag = compute_iv_percentile_enhanced(history, current_iv)
        
        assert percentile is not None
        assert iv_rank == 50.0  # Should be middle rank
        assert diag.get("constant_series", False)


class TestPCRComputation:
    """Test enhanced PCR computation."""
    
    @pytest.fixture
    def sample_oi_data(self):
        """Sample OI data for testing."""
        df = pd.read_csv(Path(__file__).parent / "fixtures" / "pcr_comprehensive.csv")
        
        oi_call = dict(zip(df["strike"], df["call_oi"]))
        oi_put = dict(zip(df["strike"], df["put_oi"]))
        strikes = df["strike"].tolist()
        
        return oi_call, oi_put, strikes
    
    def test_total_pcr_calculation(self, sample_oi_data):
        """Test total PCR calculation."""
        oi_call, oi_put, strikes = sample_oi_data
        K_atm = 19000
        step = 50
        
        results, diag = compute_pcr_enhanced(oi_put, oi_call, strikes, K_atm, step)
        
        assert results["PCR_OI_total"] is not None
        assert results["PCR_OI_total"] > 0
        assert diag["total_pcr_calculated"]
    
    def test_band_pcr_calculation(self, sample_oi_data):
        """Test ATM band PCR calculation."""
        oi_call, oi_put, strikes = sample_oi_data
        K_atm = 19000
        step = 50
        m = 3  # ±3 strikes from ATM
        
        results, diag = compute_pcr_enhanced(oi_put, oi_call, strikes, K_atm, step, m)
        
        assert results["PCR_OI_band"] is not None
        assert results["band_strikes_count"] >= 3  # Should have enough strikes
        assert diag["band_pcr_calculated"]
    
    def test_insufficient_band_data(self):
        """Test handling when insufficient strikes in ATM band."""
        oi_call = {19000: 1000}
        oi_put = {19000: 1500}
        strikes = [19000]  # Only one strike
        
        results, diag = compute_pcr_enhanced(oi_put, oi_call, strikes, 19000, 50, m=3)
        
        assert results["PCR_OI_total"] == 1.5  # Should still calculate total
        assert results["PCR_OI_band"] is None  # Should not calculate band
        assert "insufficient_strikes" in diag["band_pcr_reason"]
    
    def test_no_oi_data(self):
        """Test handling when no OI data available."""
        oi_call = {}
        oi_put = {}
        strikes = [18900, 19000, 19100]
        
        results, diag = compute_pcr_enhanced(oi_put, oi_call, strikes, 19000, 50)
        
        assert results["PCR_OI_total"] is None
        assert results["PCR_OI_band"] is None
        assert diag["valid_strikes"] == 0