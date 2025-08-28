from __future__ import annotations
import os, json
import datetime as dt
from typing import Dict, List, Tuple, Optional

import pandas as pd
import pytz
import logging

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger("engine")

class MarketDataProvider:
    def get_spot_ohlcv(self, symbol: str, interval: str, lookback_minutes: int) -> pd.DataFrame: ...
    def get_futures_ohlcv(self, symbol: str, interval: str, expiry: str, lookback_minutes: int) -> pd.DataFrame: ...
    def get_option_chain(self, symbol: str, expiry: str) -> Dict: ...
    def get_indices_snapshot(self, symbols: List[str]) -> Dict[str, float]: ...

class KiteProvider(MarketDataProvider):
    """Zerodha Kite provider using existing auth: .kite_session.json + KITE_API_KEY.
    If env KITE_API_KEY is absent, tries to read settings.json and set it.
    """
    def __init__(self):
        try:
            from kiteconnect import KiteConnect
        except Exception:
            logger.error("kiteconnect not installed. pip install kiteconnect")
            raise
        if not os.path.exists(".kite_session.json"):
            raise SystemExit("Run get_access_token.py first (creates .kite_session.json).")
        sess = json.loads(open(".kite_session.json","r").read())
        api_key = os.getenv("KITE_API_KEY")
        if not api_key:
            # Try settings.json
            try:
                cfg = json.loads(open("settings.json","r").read())
                api_key = cfg.get("KITE_API_KEY", "")
                if api_key:
                    os.environ["KITE_API_KEY"] = api_key
                    logger.info("KITE_API_KEY loaded from settings.json and set in environment.")
            except Exception:
                pass
        if not api_key:
            raise SystemExit("Set KITE_API_KEY env var or add to settings.json.")
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(sess["access_token"])
        self._nfo = None
        self._nse = None

    def _instruments(self, exch: str) -> pd.DataFrame:
        if exch=="NFO":
            if getattr(self, "_nfo", None) is None:
                self._nfo = pd.DataFrame(self.kite.instruments("NFO"))
            return self._nfo
        if exch=="BFO":
            if getattr(self, "_bfo", None) is None:
                self._bfo = pd.DataFrame(self.kite.instruments("BFO"))
            return self._bfo
        if exch=="NSE":
            if getattr(self, "_nse", None) is None:
                self._nse = pd.DataFrame(self.kite.instruments("NSE"))
            return self._nse
        if exch=="BSE":
            if getattr(self, "_bse", None) is None:
                self._bse = pd.DataFrame(self.kite.instruments("BSE"))
            return self._bse
        # default to NSE
        if getattr(self, "_nse", None) is None:
            self._nse = pd.DataFrame(self.kite.instruments("NSE"))
        return self._nse

    @staticmethod
    def _index_ltp_key(symbol: str) -> str:
        m = symbol.upper()
        if m=="BANKNIFTY": return "NSE:NIFTY BANK"
        if m=="NIFTY": return "NSE:NIFTY 50"
        if m=="FINNIFTY": return "NSE:NIFTY FIN SERVICE"
        if m=="MIDCPNIFTY": return "NSE:NIFTY MID SELECT"
        if m=="SENSEX": return "BSE:SENSEX"
        return "NSE:NIFTY 50"

    def _resolve_index_token(self, symbol: str) -> Optional[int]:
        df = self._instruments("NSE")
        name_map = {"NIFTY":"NIFTY 50", "BANKNIFTY":"NIFTY BANK",
                    "FINNIFTY":"NIFTY FIN SERVICE", "MIDCPNIFTY":"NIFTY MID SELECT"}
        ts = name_map.get(symbol.upper(), "NIFTY 50")
        m = df[df["tradingsymbol"].str.upper()==ts.upper()]
        if not m.empty:
            return int(m.iloc[0]["instrument_token"])
        return None

    def _nearest_weekly_chain_df(self, symbol: str) -> Tuple[pd.DataFrame, str]:
        sym = symbol.upper()
        if sym=="SENSEX":
            ex = "BFO"; seg = "BFO-OPT"
        else:
            ex = "NFO"; seg = "NFO-OPT"
        inst = self._instruments(ex).copy()
        df = inst[(inst["name"] == sym) & (inst["segment"] == seg)].copy()
        if df.empty:
            raise RuntimeError(f"No NFO options for {symbol}")
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
        now_ist = dt.datetime.now(IST)
        today = now_ist.date()
        if now_ist.time() > dt.time(15, 30):
            exps = sorted([e for e in df["expiry"].unique() if e > today])
        else:
            exps = sorted([e for e in df["expiry"].unique() if e >= today])
        if not exps:
            raise RuntimeError("No future expiries")
        expiry = exps[0]
        chain = df[df["expiry"] == expiry].copy()
        chain["strike"] = chain["strike"].astype(int)
        return chain, expiry.isoformat()

    def _nearest_future_row(self, symbol: str) -> pd.Series:
        sym = symbol.upper()
        ex = "BFO" if sym=="SENSEX" else "NFO"
        inst = self._instruments(ex).copy()
        futs = inst[(inst["name"] == sym) & (inst["instrument_type"] == "FUT")].copy()
        futs["expiry"] = pd.to_datetime(futs["expiry"]).dt.date
        now_ist = dt.datetime.now(IST)
        today = now_ist.date()
        if now_ist.time() > dt.time(15, 30):
            futs = futs[futs["expiry"] > today]
        else:
            futs = futs[futs["expiry"] >= today]
        futs = futs.sort_values("expiry")
        if futs.empty:
            raise RuntimeError(f"No futures for {symbol}")
        return futs.iloc[0]

    def _hist(self, token: int, interval: str, lookback_minutes: int, is_future: bool = False) -> pd.DataFrame:
        now = dt.datetime.now().replace(tzinfo=None)
        start = now - dt.timedelta(minutes=max(lookback_minutes + 15, 240))
        data = []
        try:
            data = self.kite.historical_data(
                token, start, now, "minute",
                continuous=False,
                oi=False
            )
        except Exception as e:
            logger.warning(f"[hist] token={token} error: {e}")
            data = []
        if not data:
            try:
                alt_start = now - dt.timedelta(days=2)
                data = self.kite.historical_data(
                    token, alt_start, now, "minute",
                    continuous=False,
                    oi=False
                )
            except Exception as e:
                logger.warning(f"[hist-retry] token={token} error: {e}")
                data = []
        if not data:
            return pd.DataFrame(columns=["open","high","low","close","volume"])
        df = pd.DataFrame(data)
        dates = pd.to_datetime(df["date"])
        try:
            if dates.dt.tz is None:
                dates = dates.dt.tz_localize(dt.timezone.utc).dt.tz_convert(IST)
            else:
                dates = dates.dt.tz_convert(IST)
        except Exception:
            dates = pd.to_datetime(df["date"]).tz_localize(dt.timezone.utc).tz_convert(IST)
        df = df.set_index(dates)[["open","high","low","close","volume"]]
        df.index.name = "date"
        if interval == "5m":
            df = df.resample("5min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(how="any")
        return df.tail(lookback_minutes)

    def get_spot_ohlcv(self, symbol: str, interval: str, lookback_minutes: int) -> pd.DataFrame:
        tok = self._resolve_index_token(symbol)
        df = pd.DataFrame()
        if tok:
            df = self._hist(tok, interval, lookback_minutes, is_future=False)
        if df.empty:
            fut = self._nearest_future_row(symbol)
            df = self._hist(int(fut["instrument_token"]), interval, lookback_minutes, is_future=True)
            if df.empty:
                logger.warning(f"{symbol}: spot proxy via futures also empty (fut token {int(fut['instrument_token'])}).")
        return df

    def get_futures_ohlcv(self, symbol: str, interval: str, expiry: str, lookback_minutes: int) -> pd.DataFrame:
        fut = self._nearest_future_row(symbol)
        return self._hist(int(fut["instrument_token"]), interval, lookback_minutes, is_future=True)

    def get_option_chain(self, symbol: str, expiry: str) -> Dict:
        chain_df, exp_sel = self._nearest_weekly_chain_df(symbol)
        if exp_sel != expiry:
            raise RuntimeError(f"Expiry mismatch chain={exp_sel} selected={expiry}")
        # Resolve exchange prefix dynamically (NFO vs BFO)
        seg = str(chain_df["segment"].iloc[0]) if not chain_df.empty else "NFO-OPT"
        prefix = "BFO" if "BFO" in seg else "NFO"
        syms = [f"{prefix}:{ts}" for ts in chain_df["tradingsymbol"].tolist()]
        chain = {"symbol": symbol, "expiry": expiry, "strikes": sorted(chain_df["strike"].unique()),
                 "calls": {}, "puts": {}}
        for i in range(0, len(syms), 450):
            q = self.kite.quote(syms[i:i+450])
            for v in q.values():
                tok = v["instrument_token"]
                row = chain_df[chain_df["instrument_token"]==tok]
                if row.empty: continue
                strike = int(row.iloc[0]["strike"])
                typ = row.iloc[0]["instrument_type"]
                bid=ask=ltp=0.0
                dep = v.get("depth")
                if dep and dep.get("buy") and dep.get("sell"):
                    try:
                        bid = float(dep["buy"][0]["price"]); ask = float(dep["sell"][0]["price"])
                    except: pass
                ltp = float(v.get("last_price") or 0.0)
                oi  = int(v.get("oi") or v.get("open_interest") or 0)
                node = {"oi": oi, "ltp": ltp, "bid": bid, "ask": ask}
                if typ=="CE": chain["calls"][strike]=node
                else: chain["puts"][strike]=node
        for k in chain["strikes"]:
            chain["calls"].setdefault(k, {"oi":0,"ltp":0.0,"bid":0.0,"ask":0.0})
            chain["puts"].setdefault(k,  {"oi":0,"ltp":0.0,"bid":0.0,"ask":0.0})
        return chain

    def get_indices_snapshot(self, symbols: List[str]) -> Dict[str, float]:
        keys = []
        map_key = {}
        for s in symbols:
            k = self._index_ltp_key(s)
            keys.append(k); map_key[k]=s
        out = {}
        try:
            q = self.kite.ltp(keys)
            for k,v in q.items():
                out[map_key[k]] = float(v["last_price"])
        except Exception as e:
            logger.warning(f"ltp error: {e}")
        for s in symbols:
            if s not in out:
                try:
                    fut = self._nearest_future_row(s)
                    seg = "BFO" if s.upper()=="SENSEX" else "NFO"
                    q = self.kite.quote([f"{seg}:{fut['tradingsymbol']}"])
                    out[s] = float(list(q.values())[0]["last_price"])
                except: out[s] = float('nan')
        return out

    # Public helper: earliest upcoming expiry used by option chain selection
    def get_current_expiry_date(self, symbol: str) -> str:
        _, exp_sel = self._nearest_weekly_chain_df(symbol)
        return exp_sel
