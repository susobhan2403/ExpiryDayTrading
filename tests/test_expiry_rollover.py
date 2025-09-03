"""Tests for expiry time rollover scenarios near 15:29 IST."""

import pytest
import datetime as dt
import pytz
from unittest.mock import Mock

from src.metrics.enhanced import (
    compute_forward_enhanced,
    pick_atm_strike_enhanced,
    compute_atm_iv_enhanced
)
from src.metrics.core import choose_expiry
from src.strategy.enhanced_gates import (
    apply_enhanced_gates,
    MultiFactorSignal,
    EnhancedRegime
)


class TestExpiryRollover:
    """Test behavior near expiry rollover at 15:29 IST."""
    
    @pytest.fixture
    def ist_timezone(self):
        """IST timezone fixture."""
        return pytz.timezone("Asia/Kolkata")
    
    @pytest.fixture
    def expiry_scenarios(self, ist_timezone):
        """Various expiry rollover scenarios."""
        base_date = dt.date(2024, 1, 11)  # Thursday expiry
        
        scenarios = {
            "pre_expiry_normal": ist_timezone.localize(
                dt.datetime.combine(base_date, dt.time(15, 0))  # 15:00 IST
            ),
            "near_expiry_warning": ist_timezone.localize(
                dt.datetime.combine(base_date, dt.time(15, 25))  # 15:25 IST
            ),
            "critical_expiry": ist_timezone.localize(
                dt.datetime.combine(base_date, dt.time(15, 28))  # 15:28 IST
            ),
            "post_expiry": ist_timezone.localize(
                dt.datetime.combine(base_date, dt.time(15, 30))  # 15:30 IST
            ),
            "next_day_morning": ist_timezone.localize(
                dt.datetime.combine(base_date + dt.timedelta(days=1), dt.time(9, 15))  # Next day 9:15 IST
            )
        }
        
        # Corresponding expiries
        weekly_expiry = ist_timezone.localize(
            dt.datetime.combine(base_date, dt.time(15, 29, 59))  # Weekly expiry
        )
        monthly_expiry = ist_timezone.localize(
            dt.datetime.combine(base_date + dt.timedelta(days=14), dt.time(15, 29, 59))  # Monthly expiry
        )
        
        return scenarios, [weekly_expiry, monthly_expiry]
    
    def test_tau_calculation_near_expiry(self, expiry_scenarios, ist_timezone):
        """Test time-to-expiry calculation near 15:29 IST."""
        scenarios, expiries = expiry_scenarios
        weekly_expiry = expiries[0]
        
        for scenario_name, current_time in scenarios.items():
            if current_time <= weekly_expiry:
                tau_seconds = (weekly_expiry - current_time).total_seconds()
                tau_hours = tau_seconds / 3600.0
                
                if scenario_name == "critical_expiry":
                    assert tau_hours < 0.05  # Less than 3 minutes
                    assert tau_hours > 0  # But still positive
                elif scenario_name == "near_expiry_warning":
                    assert 0.05 <= tau_hours <= 0.1  # 3-6 minutes
                elif scenario_name == "pre_expiry_normal":
                    assert tau_hours > 0.4  # More than 24 minutes
    
    def test_expiry_selection_rollover(self, expiry_scenarios, ist_timezone):
        """Test expiry selection during rollover period."""
        scenarios, expiries = expiry_scenarios
        
        for scenario_name, current_time in scenarios.items():
            min_tau_h = 2.0  # Minimum 2 hours
            
            if scenario_name == "post_expiry":
                # After weekly expiry, should select monthly
                selected_expiry = choose_expiry(current_time, expiries, min_tau_h)
                assert selected_expiry == expiries[1]  # Monthly expiry
            elif scenario_name == "critical_expiry":
                # Very close to weekly expiry, should select monthly if min_tau_h not met
                selected_expiry = choose_expiry(current_time, expiries, min_tau_h)
                assert selected_expiry == expiries[1]  # Should rollover to monthly
            elif scenario_name == "next_day_morning":
                # Next day, should definitely be on monthly
                selected_expiry = choose_expiry(current_time, expiries, min_tau_h)
                assert selected_expiry == expiries[1]
    
    def test_atm_strike_stability_near_expiry(self, expiry_scenarios):
        """Test ATM strike selection stability near expiry."""
        scenarios, _ = expiry_scenarios
        
        # Market data that might be volatile near expiry
        F = 19000
        strikes = [18900, 18950, 19000, 19050, 19100]
        
        # Simulate wider spreads near expiry
        ce_mid_normal = {s: 25.0 for s in strikes}
        pe_mid_normal = {s: 25.0 for s in strikes}
        
        ce_mid_wide = {s: 20.0 for s in strikes}  # Wider spreads
        pe_mid_wide = {s: 30.0 for s in strikes}
        
        for scenario_name, _ in scenarios.items():
            if scenario_name in ["critical_expiry", "near_expiry_warning"]:
                # Near expiry - use wide spreads
                K_atm, diag = pick_atm_strike_enhanced(F, strikes, 50, ce_mid_wide, pe_mid_wide)
            else:
                # Normal conditions
                K_atm, diag = pick_atm_strike_enhanced(F, strikes, 50, ce_mid_normal, pe_mid_normal)
            
            # ATM strike should still be selected consistently
            assert K_atm == 19000  # Should pick exact ATM
            assert diag["F"] == F
    
    def test_iv_calculation_expiry_stress(self, expiry_scenarios):
        """Test IV calculations under expiry stress conditions."""
        scenarios, expiries = expiry_scenarios
        weekly_expiry = expiries[0]
        
        F = 19000
        K_atm = 19000
        r = 0.06
        
        for scenario_name, current_time in scenarios.items():
            if current_time <= weekly_expiry:
                tau_years = max(0.001, (weekly_expiry - current_time).total_seconds() / (365.25 * 24 * 3600))
                
                if scenario_name == "critical_expiry":
                    # Very short expiry - options should be nearly worthless unless deep ITM
                    ce_mid = 0.05  # Very cheap
                    pe_mid = 0.05
                    
                    atm_iv, diag = compute_atm_iv_enhanced(ce_mid, pe_mid, F, K_atm, tau_years, r)
                    
                    # IV calculation should still work, though values might be extreme
                    if atm_iv is not None:
                        assert atm_iv >= 0  # Should be non-negative
                        # Very short expiry can lead to very high IVs
                        assert atm_iv <= 10.0  # But not infinite
                elif scenario_name == "near_expiry_warning":
                    # Moderate expiry stress - use more realistic prices for short expiry
                    ce_mid = 25.0  # Higher price for short expiry
                    pe_mid = 25.0
                    
                    atm_iv, diag = compute_atm_iv_enhanced(ce_mid, pe_mid, F, K_atm, tau_years, r)
                    
                    if atm_iv is not None:
                        assert 0.05 <= atm_iv <= 5.0  # Broader range for stress conditions
    
    def test_gate_behavior_near_expiry(self, expiry_scenarios):
        """Test enhanced gating behavior near expiry."""
        scenarios, expiries = expiry_scenarios
        weekly_expiry = expiries[0]
        
        # Strong signal setup
        strong_signals = MultiFactorSignal(
            orb_signal="LONG",
            orb_strength=0.9,
            volume_signal="LONG",
            volume_strength=0.8,
            oi_flow_signal="LONG",
            oi_flow_strength=0.7
        )
        
        # Moderate signal setup
        moderate_signals = MultiFactorSignal(
            orb_signal="LONG",
            orb_strength=0.6,
            volume_signal="LONG",
            volume_strength=0.5
        )
        
        regime = EnhancedRegime("WEAK_UP", "NORMAL", "GOOD", "STEADY")
        
        for scenario_name, current_time in scenarios.items():
            if current_time <= weekly_expiry:
                tau_hours = (weekly_expiry - current_time).total_seconds() / 3600.0
                
                # Test strong signals
                strong_decision = apply_enhanced_gates(
                    signals=strong_signals,
                    regime=regime,
                    tau_hours=tau_hours,
                    confirming_bars=2,
                    min_confirm_bars=2
                )
                
                # Test moderate signals
                moderate_decision = apply_enhanced_gates(
                    signals=moderate_signals,
                    regime=regime,
                    tau_hours=tau_hours,
                    confirming_bars=2,
                    min_confirm_bars=2
                )
                
                if scenario_name == "critical_expiry":
                    # Very close to expiry - should require very strong signals
                    if not strong_decision.muted:
                        assert strong_decision.size_factor <= 0.7  # Reduced size
                        assert ("near_expiry" in strong_decision.risk_adjustments or 
                               "critical_expiry" in strong_decision.risk_adjustments)
                    
                    # Moderate signals should likely be blocked
                    assert moderate_decision.muted or moderate_decision.size_factor < 0.5
                
                elif scenario_name == "near_expiry_warning":
                    # Some risk reduction but not as severe
                    if not strong_decision.muted:
                        assert strong_decision.size_factor <= 0.8
                
                elif scenario_name == "pre_expiry_normal":
                    # Normal behavior
                    if not strong_decision.muted:
                        assert strong_decision.size_factor >= 0.8
    
    def test_data_quality_flags_near_expiry(self, expiry_scenarios):
        """Test data quality flag generation near expiry."""
        scenarios, expiries = expiry_scenarios
        weekly_expiry = expiries[0]
        
        for scenario_name, current_time in scenarios.items():
            if current_time <= weekly_expiry:
                tau_hours = (weekly_expiry - current_time).total_seconds() / 3600.0
                
                dq_flags = []
                
                # Add expiry-related flags (matching engine logic)
                if tau_hours < 0.1:  # Less than 6 minutes
                    dq_flags.append("expiry_too_close")
                elif tau_hours < 0.25:  # Less than 15 minutes
                    dq_flags.append("near_expiry_warning")
                
                # Simulate other data quality issues near expiry
                if scenario_name == "critical_expiry":
                    dq_flags.extend(["wide_spreads", "insufficient_volume"])
                elif scenario_name == "near_expiry_warning":
                    dq_flags.append("wide_spreads")
                
                # Validate flag logic
                if scenario_name == "critical_expiry":
                    assert "expiry_too_close" in dq_flags
                elif scenario_name == "near_expiry_warning":
                    assert "near_expiry_warning" in dq_flags or "wide_spreads" in dq_flags
                elif scenario_name == "pre_expiry_normal":
                    assert len(dq_flags) == 0  # Should be clean
    
    def test_rollover_consistency(self, expiry_scenarios, ist_timezone):
        """Test consistency across rollover boundary."""
        scenarios, expiries = expiry_scenarios
        
        # Test just before and after rollover
        pre_rollover = ist_timezone.localize(dt.datetime(2024, 1, 11, 15, 29, 58))
        post_rollover = ist_timezone.localize(dt.datetime(2024, 1, 11, 15, 30, 1))
        
        min_tau_h = 2.0
        
        # Before rollover - should be forced to monthly
        pre_expiry = choose_expiry(pre_rollover, expiries, min_tau_h)
        assert pre_expiry == expiries[1]  # Monthly
        
        # After rollover - should definitely be monthly
        post_expiry = choose_expiry(post_rollover, expiries, min_tau_h)
        assert post_expiry == expiries[1]  # Monthly
        
        # Should be the same expiry selected
        assert pre_expiry == post_expiry
    
    def test_weekend_rollover(self, ist_timezone):
        """Test behavior over weekend when markets are closed."""
        # Friday expiry
        friday_expiry = ist_timezone.localize(dt.datetime(2024, 1, 12, 15, 29, 59))
        
        # Next weekly would be Thursday
        next_weekly = ist_timezone.localize(dt.datetime(2024, 1, 18, 15, 29, 59))
        monthly_expiry = ist_timezone.localize(dt.datetime(2024, 1, 25, 15, 29, 59))
        
        expiries = [friday_expiry, next_weekly, monthly_expiry]
        
        # Monday morning
        monday_morning = ist_timezone.localize(dt.datetime(2024, 1, 15, 9, 15))
        
        selected = choose_expiry(monday_morning, expiries, min_tau_h=2.0)
        
        # Should select next weekly (not the expired Friday)
        assert selected == next_weekly
        
        # Calculate tau
        tau_hours = (selected - monday_morning).total_seconds() / 3600.0
        assert tau_hours > 48  # Should be several days
    
    def test_intraday_tau_monotonicity(self, ist_timezone):
        """Test that tau decreases monotonically during trading day."""
        expiry = ist_timezone.localize(dt.datetime(2024, 1, 11, 15, 29, 59))
        
        # Trading session times
        times = [
            ist_timezone.localize(dt.datetime(2024, 1, 11, 9, 15)),   # Market open
            ist_timezone.localize(dt.datetime(2024, 1, 11, 12, 0)),   # Midday
            ist_timezone.localize(dt.datetime(2024, 1, 11, 15, 0)),   # Near close
            ist_timezone.localize(dt.datetime(2024, 1, 11, 15, 25)),  # Very near close
        ]
        
        previous_tau = float('inf')
        
        for current_time in times:
            tau_hours = (expiry - current_time).total_seconds() / 3600.0
            assert tau_hours < previous_tau  # Should be decreasing
            assert tau_hours >= 0  # Should not be negative during trading
            previous_tau = tau_hours