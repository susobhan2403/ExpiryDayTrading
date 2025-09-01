from __future__ import annotations
import os, json, asyncio
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
        self._index_cache: Dict[str, float] = {}

    def _instruments(self, exch: str) -> pd.DataFrame:
        if exch=="NFO":
            if getattr(self, "_nfo", None) is None:
                cache_file = os.path.join("out", "instruments_cache.pkl")
                if os.path.exists(cache_file):
                    try:
                        self._nfo = pd.read_pickle(cache_file)
                    except Exception:
                        self._nfo = None
                if getattr(self, "_nfo", None) is None:
                    self._nfo = pd.DataFrame(self.kite.instruments("NFO"))
                    try:
                        os.makedirs("out", exist_ok=True)
                        self._nfo.to_pickle(cache_file)
                    except Exception as e:
                        logger.warning(f"Failed to write instruments cache: {e}")
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
        sym = symbol.upper()
        if sym == "SENSEX":
            df = self._instruments("BSE")
            m = df[df["tradingsymbol"].str.upper()=="SENSEX"]
            if not m.empty:
                return int(m.iloc[0]["instrument_token"])
            return None
        df = self._instruments("NSE")
        name_map = {"NIFTY":"NIFTY 50", "BANKNIFTY":"NIFTY BANK",
                    "FINNIFTY":"NIFTY FIN SERVICE", "MIDCPNIFTY":"NIFTY MID SELECT"}
        ts = name_map.get(sym, "NIFTY 50")
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
            raise RuntimeError(f"No options for {symbol}")
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
        today = dt.datetime.now(IST).date()
        weekly = sorted(e for e in df["expiry"].unique() if e >= today)[:5]
        if dt.datetime.now(IST).time() > dt.time(15, 30):
            weekly = [e for e in weekly if e > today]
        if not weekly:
            raise RuntimeError("No future expiries")
        expiry = weekly[0]
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
        """Fetch minute candles with off-hours resilience.

        - Anchors the end time to the last trading session close when market is
          closed (weekends or pre-open), so queries return data instead of empty.
        - Falls back to a wider 7-day window if the immediate lookback returns empty.
        """
        now_ist = dt.datetime.now(IST)
        # Determine an appropriate 'end' anchored to the latest trading session
        end_ist = now_ist
        if now_ist.weekday() >= 5:  # Sat/Sun -> last Friday 15:29:59 IST
            days_back = (now_ist.weekday() - 4)  # 1 for Sat, 2 for Sun
            last_weekday = (now_ist - dt.timedelta(days=days_back)).date()
            end_ist = IST.localize(dt.datetime.combine(last_weekday, dt.time(15, 29, 59)))
        elif now_ist.time() < dt.time(9, 15):  # before market open -> previous weekday close
            # move to previous business day (skip weekends)
            prev = now_ist - dt.timedelta(days=1)
            while prev.weekday() >= 5:
                prev -= dt.timedelta(days=1)
            end_ist = IST.localize(dt.datetime.combine(prev.date(), dt.time(15, 29, 59)))

        # Use at least 4 hours of history to build resamples/indicators
        lb = max(lookback_minutes + 15, 240)
        start_ist = end_ist - dt.timedelta(minutes=lb)

        # Primary query
        data = []
        try:
            data = self.kite.historical_data(
                token, start_ist, end_ist, "minute",
                continuous=False,
                oi=False
            )
        except Exception as e:
            logger.warning(f"[hist] token={token} error: {e}")
            data = []

        # Fallback: widen to last 7 days ending at end_ist
        if not data:
            try:
                alt_start = end_ist - dt.timedelta(days=7)
                data = self.kite.historical_data(
                    token, alt_start, end_ist, "minute",
                    continuous=False,
                    oi=False
                )
            except Exception as e:
                logger.warning(f"[hist-retry-wide] token={token} error: {e}")
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

        batches = [syms[i:i+450] for i in range(0, len(syms), 450)]
        q: Dict[str, Dict] = {}

        async def gather_quotes() -> None:
            loop = asyncio.get_running_loop()
            for j in range(0, len(batches), 3):
                group = batches[j:j+3]
                tasks = [loop.run_in_executor(None, self.kite.quote, b) for b in group]
                res = await asyncio.gather(*tasks, return_exceptions=True)
                for r in res:
                    if isinstance(r, dict):
                        q.update(r)
                if j + 3 < len(batches):
                    await asyncio.sleep(1)

        if batches:
            asyncio.run(gather_quotes())

        for v in q.values():
            tok = v["instrument_token"]
            row = chain_df[chain_df["instrument_token"]==tok]
            if row.empty: continue
            strike = int(row.iloc[0]["strike"])
            typ = row.iloc[0]["instrument_type"]
            bid=ask=ltp=bid_q=ask_q=0.0
            dep = v.get("depth")
            if dep and dep.get("buy") and dep.get("sell"):
                try:
                    bid = float(dep["buy"][0]["price"]); ask = float(dep["sell"][0]["price"])
                    bid_q = float(dep["buy"][0].get("quantity",0)); ask_q = float(dep["sell"][0].get("quantity",0))
                except Exception:
                    pass
            ltp = float(v.get("last_price") or 0.0)
            ltt = v.get("last_trade_time")
            try:
                ltt = pd.to_datetime(ltt).tz_localize(dt.timezone.utc).tz_convert(IST)
            except Exception:
                ltt = None
            oi  = int(v.get("oi") or v.get("open_interest") or 0)
            node = {"oi": oi, "ltp": ltp, "bid": bid, "ask": ask,
                    "bid_qty": bid_q, "ask_qty": ask_q,
                    "ltp_ts": ltt.isoformat() if ltt else None}
            if typ=="CE": chain["calls"][strike]=node
            else: chain["puts"][strike]=node
        for k in chain["strikes"]:
            chain["calls"].setdefault(k, {"oi":0,"ltp":0.0,"bid":0.0,"ask":0.0,"bid_qty":0.0,"ask_qty":0.0,"ltp_ts":None})
            chain["puts"].setdefault(k,  {"oi":0,"ltp":0.0,"bid":0.0,"ask":0.0,"bid_qty":0.0,"ask_qty":0.0,"ltp_ts":None})
        return chain

    def get_indices_snapshot(self, symbols: List[str]) -> Dict[str, float]:
        """Return the latest spot price for each index.

        The previous implementation fell back to near-month futures prices when
        index quotes were unavailable.  This introduced large basis-driven
        errors, resulting in incorrect ATM strikes.  Instead of using futures,
        we now attempt a secondary quote call and ultimately return ``nan`` when
        spot data cannot be fetched."""
        keys = []
        map_key = {}
        for s in symbols:
            k = self._index_ltp_key(s)
            keys.append(k)
            map_key[k] = s

        out: Dict[str, float] = {}
        try:
            q = self.kite.quote(keys)
            for k, v in q.items():
                price = float(v.get("last_price") or float("nan"))
                sym = map_key[k]
                if price == price:
                    self._index_cache[sym] = price
                out[sym] = price
        except Exception as e:
            logger.warning(f"quote error: {e}")

        # Fallback to last cached quote if current fetch fails
        for s in symbols:
            if s not in out or out[s] != out[s]:
                out[s] = self._index_cache.get(s, float("nan"))
        return out

    def get_option_chains(self, symbol: str, expiries: List[str]) -> Dict[str, Dict]:
        """Throttle-friendly multi-expiry option chains.
        Returns mapping expiry->chain dict (same schema as get_option_chain).
        """
        chains: Dict[str, Dict] = {}
        if not expiries:
            return chains
        sym = symbol.upper()
        # choose exchange segment by symbol
        ex = "BFO" if sym=="SENSEX" else "NFO"
        seg = f"{ex}-OPT"
        inst = self._instruments(ex).copy()
        df = inst[(inst["name"] == sym) & (inst["segment"] == seg)].copy()
        if df.empty:
            return chains
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
        exp_dates = set()
        for e in expiries:
            try:
                exp_dates.add(pd.to_datetime(e).date())
            except Exception:
                pass
        by_exp = {e: df[df["expiry"]==e].copy() for e in exp_dates}
        # Build a combined symbol list and quote in batches
        syms_all: List[str] = []
        exp_for_ts: Dict[str, dt.date] = {}
        for e, dfe in by_exp.items():
            dfe["strike"] = dfe["strike"].astype(int)
            for ts in dfe["tradingsymbol"].tolist():
                syms_all.append(f"{ex}:{ts}")
                exp_for_ts[ts] = e
        # quote in chunks
        q_all: Dict[str, Dict] = {}
        for i in range(0, len(syms_all), 400):
            try:
                q = self.kite.quote(syms_all[i:i+400])
                q_all.update(q)
            except Exception:
                continue
        # Build chains per expiry
        for e, dfe in by_exp.items():
            chain = {"symbol": symbol, "expiry": e.isoformat(), "strikes": sorted(dfe["strike"].unique()),
                     "calls": {}, "puts": {}}
            for sym_key, v in q_all.items():
                ts = sym_key.split(":",1)[1] if ":" in sym_key else sym_key
                # map ts -> row
                row = dfe[dfe["tradingsymbol"] == ts]
                if row.empty: continue
                strike = int(row.iloc[0]["strike"])
                typ = row.iloc[0]["instrument_type"]
                bid=ask=ltp=bid_q=ask_q=0.0
                dep = v.get("depth")
                if dep and dep.get("buy") and dep.get("sell"):
                    try:
                        bid = float(dep["buy"][0]["price"]); ask = float(dep["sell"][0]["price"])
                        bid_q = float(dep["buy"][0].get("quantity",0)); ask_q = float(dep["sell"][0].get("quantity",0))
                    except Exception:
                        pass
                ltp = float(v.get("last_price") or 0.0)
                ltt = v.get("last_trade_time")
                try:
                    ltt = pd.to_datetime(ltt).tz_localize(dt.timezone.utc).tz_convert(IST)
                except Exception:
                    ltt = None
                oi  = int(v.get("oi") or v.get("open_interest") or 0)
                node = {"oi": oi, "ltp": ltp, "bid": bid, "ask": ask,
                        "bid_qty": bid_q, "ask_qty": ask_q,
                        "ltp_ts": ltt.isoformat() if ltt else None}
                if typ=="CE": chain["calls"][strike]=node
                else: chain["puts"][strike]=node
            # ensure all strikes have nodes
            for k in chain["strikes"]:
                chain["calls"].setdefault(k, {"oi":0,"ltp":0.0,"bid":0.0,"ask":0.0,"bid_qty":0.0,"ask_qty":0.0,"ltp_ts":None})
                chain["puts"].setdefault(k,  {"oi":0,"ltp":0.0,"bid":0.0,"ask":0.0,"bid_qty":0.0,"ask_qty":0.0,"ltp_ts":None})
            chains[e.isoformat()] = chain
        return chains

    # Public helper: earliest upcoming expiry used by option chain selection
    def get_current_expiry_date(self, symbol: str) -> str:
        _, exp_sel = self._nearest_weekly_chain_df(symbol)
        return exp_sel

    # Public helper: first N upcoming expiries (weekly+monthly), string ISO dates
    def get_upcoming_expiries(self, symbol: str, n: int = 3) -> list[str]:
        sym = symbol.upper()
        if sym=="SENSEX":
            ex = "BFO"; seg = "BFO-OPT"
        else:
            ex = "NFO"; seg = "NFO-OPT"
        inst = self._instruments(ex).copy()
        if inst.empty:
            return []
        df = inst[(inst["name"] == sym) & (inst["segment"] == seg)].copy()
        if df.empty:
            return []
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
        today = dt.datetime.now(IST).date()
        exps = sorted([e for e in df["expiry"].unique() if e >= today])
        return [e.isoformat() for e in exps[:max(1,n)]]
