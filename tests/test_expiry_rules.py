import datetime as dt
from src.features.options import nearest_weekly_expiry

def wd(iso):
    d = dt.date.fromisoformat(iso)
    return d.weekday()  # Mon=0 .. Sun=6

def aware(y,m,d,h=10,mi=0):
    # naive ok; function uses .date()
    return dt.datetime(y,m,d,h,mi)

def test_nifty_switch_to_tuesday_after_sep2025():
    # After effective date, NIFTY should be Tuesday (1)
    now = aware(2025,9,2,10,0)  # Tue
    exp = nearest_weekly_expiry(now, 'NIFTY')
    assert wd(exp) == 1

def test_nifty_before_sep2025_is_thursday():
    now = aware(2025,8,28,10,0)  # Thu
    exp = nearest_weekly_expiry(now, 'NIFTY')
    assert wd(exp) == 3

def test_sensex_switch_to_thursday_after_sep2025():
    now = aware(2025,9,4,10,0)  # Thu
    exp = nearest_weekly_expiry(now, 'SENSEX')
    assert wd(exp) == 3

def test_sensex_before_sep2025_is_tuesday():
    now = aware(2025,8,27,10,0)  # Wed -> nearest Tue
    exp = nearest_weekly_expiry(now, 'SENSEX')
    assert wd(exp) == 1


def test_banknifty_monthly_expiry():
    # BANKNIFTY should use monthly expiry (last Tuesday of month)
    now = aware(2026,1,2,10,0)  # Jan 2, 2026 (Thursday)
    exp = nearest_weekly_expiry(now, 'BANKNIFTY')
    assert wd(exp) == 1  # Should be Tuesday
    exp_date = dt.date.fromisoformat(exp)
    # Should be last Tuesday of January 2026
    assert exp_date.month == 1 and exp_date.year == 2026
    # Verify it's the last Tuesday of the month
    next_tuesday = exp_date + dt.timedelta(days=7)
    assert next_tuesday.month == 2  # Next Tuesday should be in next month


def test_midcpnifty_monthly_expiry():
    # MIDCPNIFTY should use monthly expiry (last Tuesday of month)
    now = aware(2026,1,2,10,0)  # Jan 2, 2026 (Thursday)
    exp = nearest_weekly_expiry(now, 'MIDCPNIFTY')
    assert wd(exp) == 1  # Should be Tuesday
    exp_date = dt.date.fromisoformat(exp)
    # Should be last Tuesday of January 2026
    assert exp_date.month == 1 and exp_date.year == 2026
    # Verify it's the last Tuesday of the month
    next_tuesday = exp_date + dt.timedelta(days=7)
    assert next_tuesday.month == 2  # Next Tuesday should be in next month

