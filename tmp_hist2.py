import sys, datetime as dt, pytz
sys.path.append('.')
from src.provider.kite import KiteProvider, IST
p = KiteProvider()
fut = p._nearest_future_row('NIFTY')
tok = int(fut['instrument_token'])
print('token', tok)
from kiteconnect import KiteConnect
# Use underlying kite client
k = p.kite
now = dt.datetime.now()
# pick a window inside last Friday 10:00-15:29 IST
# find last weekday <=4 before now
for days_back in range(1, 8):
    day = (now - dt.timedelta(days=days_back)).date()
    if day.weekday() <= 4:
        last_weekday = day
        break
start = dt.datetime.combine(last_weekday, dt.time(10, 0))
end = dt.datetime.combine(last_weekday, dt.time(15, 29))
print('window naive', start, end)
try:
    data = k.historical_data(tok, start, end, 'minute', continuous=False, oi=False)
    print('len naive', len(data))
except Exception as e:
    print('error naive', e)
# try timezone aware IST
start_ist = IST.localize(start)
end_ist = IST.localize(end)
print('window IST', start_ist, end_ist)
try:
    data2 = k.historical_data(tok, start_ist, end_ist, 'minute', continuous=False, oi=False)
    print('len IST', len(data2))
except Exception as e:
    print('error IST', e)
