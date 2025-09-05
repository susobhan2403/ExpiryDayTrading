from __future__ import annotations
import os, json, asyncio
import datetime as dt
from typing import Dict, List, Tuple, Optional

import pandas as pd
import pytz
import logging

from .option_chain_builder import OptionChainBuilder, convert_to_legacy_format

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger("engine")

class MarketDataProvider:
    def get_spot_ohlcv(self, symbol: str, interval: str, lookback_minutes: int) -> pd.DataFrame: ...
    def get_futures_ohlcv(self, symbol: str, interval: str, expiry: str, lookback_minutes: int) -> pd.DataFrame: ...
    def get_option_chain(self, symbol: str, expiry: str) -> Dict: ...
    def get_option_chains(self, symbol: str, expiries: List[str]) -> Dict[str, Dict]: ...
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
        
        # Initialize option chain builder (lazy loaded)
        self._option_chain_builder: Optional[OptionChainBuilder] = None

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
        now_ist = dt.datetime.now(IST)
        today = now_ist.date()
        weekly = sorted(e for e in df["expiry"].unique() if e >= today)[:5]
        if now_ist.time() > dt.time(15, 30):
            weekly = [e for e in weekly if e > today]
        if not weekly:
            raise RuntimeError("No future expiries")

        # Remove monthly expiries: any expiry followed by a gap >7 days
        monthly = {
            weekly[i]
            for i in range(len(weekly) - 1)
            if (weekly[i + 1] - weekly[i]).days > 7
        }
        weekly = [e for e in weekly if e not in monthly]
        if not weekly:
            raise RuntimeError("No future weekly expiries")

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

    # Old get_option_chain method removed - replaced with OptionChainBuilder
    # This method has been deprecated as part of the option chain architecture redesign

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

    # Old get_option_chains method removed - replaced with OptionChainBuilder
    # This method has been deprecated as part of the option chain architecture redesign

    def _get_option_chain_builder(self) -> OptionChainBuilder:
        """Get or create option chain builder instance."""
        if self._option_chain_builder is None:
            # Load NFO instruments for option chain building
            nfo_instruments = self._instruments("NFO")
            # Also load BFO for SENSEX if needed
            bfo_instruments = self._instruments("BFO") 
            
            # Combine instruments DataFrames
            combined_instruments = pd.concat([nfo_instruments, bfo_instruments], ignore_index=True)
            
            self._option_chain_builder = OptionChainBuilder(self.kite, combined_instruments)
            logger.info("Initialized OptionChainBuilder with instruments data")
        
        return self._option_chain_builder

    def get_option_chain(self, symbol: str, expiry: str) -> Dict:
        """Build option chain using new OptionChainBuilder architecture.
        
        This method provides backward compatibility with the legacy format
        while using the improved option chain building architecture.
        """
        try:
            # Get current spot price for strike range calculation
            quotes = self.get_indices_snapshot([symbol])
            if not quotes or symbol not in quotes:
                logger.warning(f"Could not get spot price for {symbol}")
                return {"symbol": symbol, "expiry": expiry, "strikes": [], "calls": {}, "puts": {}}
            
            spot_price = quotes[symbol]
            if spot_price <= 0 or spot_price != spot_price:  # Check for invalid/NaN
                logger.warning(f"Invalid spot price {spot_price} for {symbol}")
                return {"symbol": symbol, "expiry": expiry, "strikes": [], "calls": {}, "puts": {}}
            
            # Build option chain using new architecture
            builder = self._get_option_chain_builder()
            option_chain = builder.build_chain(symbol, expiry, spot_price)
            
            if option_chain is None:
                logger.warning(f"Failed to build option chain for {symbol} {expiry}")
                return {"symbol": symbol, "expiry": expiry, "strikes": [], "calls": {}, "puts": {}}
            
            # Convert to legacy format for backward compatibility
            legacy_chain = convert_to_legacy_format(option_chain)
            
            logger.info(f"Built option chain for {symbol} {expiry}: {len(legacy_chain['strikes'])} strikes, "
                       f"quality: {option_chain.data_quality_score:.1%}")
            
            return legacy_chain
            
        except Exception as e:
            logger.error(f"Error building option chain for {symbol} {expiry}: {e}")
            return {"symbol": symbol, "expiry": expiry, "strikes": [], "calls": {}, "puts": {}}

    def get_option_chains(self, symbol: str, expiries: List[str]) -> Dict[str, Dict]:
        """Build multiple option chains using new OptionChainBuilder architecture.
        
        Returns mapping expiry->chain dict (same schema as get_option_chain).
        """
        chains = {}
        
        if not expiries:
            return chains
        
        try:
            # Get current spot price
            quotes = self.get_indices_snapshot([symbol])
            if not quotes or symbol not in quotes:
                logger.warning(f"Could not get spot price for {symbol}")
                return chains
            
            spot_price = quotes[symbol]
            if spot_price <= 0 or spot_price != spot_price:
                logger.warning(f"Invalid spot price {spot_price} for {symbol}")
                return chains
            
            # Build chains for each expiry
            builder = self._get_option_chain_builder()
            
            for expiry in expiries:
                try:
                    option_chain = builder.build_chain(symbol, expiry, spot_price)
                    if option_chain is not None:
                        legacy_chain = convert_to_legacy_format(option_chain)
                        chains[expiry] = legacy_chain
                    else:
                        logger.warning(f"Failed to build chain for {symbol} {expiry}")
                        
                except Exception as e:
                    logger.error(f"Error building chain for {symbol} {expiry}: {e}")
                    continue
            
            logger.info(f"Built {len(chains)} option chains for {symbol} from {len(expiries)} requested expiries")
            return chains
            
        except Exception as e:
            logger.error(f"Error building option chains for {symbol}: {e}")
            return chains

    # Public helper: earliest upcoming expiry used by option chain selection  
    def get_current_expiry_date(self, symbol: str) -> str:
        """Get current expiry date using OptionChainBuilder for consistency."""
        try:
            builder = self._get_option_chain_builder()
            expiry = builder.get_nearest_expiry(symbol)
            return expiry if expiry else ""
        except Exception as e:
            logger.error(f"Error getting current expiry for {symbol}: {e}")
            return ""

    # Public helper: first N upcoming expiries (weekly+monthly), string ISO dates
    def get_upcoming_expiries(self, symbol: str, n: int = 3) -> list[str]:
        """Get upcoming expiries using OptionChainBuilder for consistency."""
        try:
            builder = self._get_option_chain_builder()
            expiries = builder.get_available_expiries(symbol)
            return expiries[:max(1, n)]
        except Exception as e:
            logger.error(f"Error getting upcoming expiries for {symbol}: {e}")
            return []
