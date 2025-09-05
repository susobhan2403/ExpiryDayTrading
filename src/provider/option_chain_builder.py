"""
Modern Option Chain Builder for Kite Connect

This module provides a clean, efficient implementation for building option chains
from Kite Connect data following their recommended patterns:
1. Filter instruments list for index/expiry/strikes
2. Batch quote requests in chunks ≤500 tokens  
3. Assemble CE/PE rows by strike
4. Compute indicators client-side

Architecture:
- InstrumentFilter: Filters and validates instruments data
- QuoteBatcher: Handles efficient batch quote requests with rate limiting
- OptionDataAssembler: Assembles final option chain structure
- OptionChainBuilder: Main orchestrator class
"""

from __future__ import annotations
import datetime as dt
import logging
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger("option_chain_builder")


@dataclass
class OptionData:
    """Structured option data for a single strike/type."""
    strike: int
    option_type: str  # "CE" or "PE"
    bid: float = 0.0
    ask: float = 0.0
    ltp: float = 0.0
    bid_qty: int = 0
    ask_qty: int = 0
    oi: int = 0
    volume: int = 0
    last_trade_time: Optional[dt.datetime] = None
    instrument_token: int = 0
    trading_symbol: str = ""


@dataclass
class OptionChain:
    """Complete option chain for a symbol/expiry."""
    symbol: str
    expiry: str
    spot_price: float
    strikes: List[int]
    options: Dict[Tuple[int, str], OptionData]  # (strike, type) -> OptionData
    build_timestamp: dt.datetime
    data_quality_score: float = 1.0  # 0.0 to 1.0


class InstrumentFilter:
    """Filters instruments list for option chain building."""
    
    def __init__(self, instruments_df: pd.DataFrame):
        """Initialize with instruments DataFrame."""
        self.instruments_df = instruments_df.copy()
        if not self.instruments_df.empty:
            self.instruments_df["expiry"] = pd.to_datetime(self.instruments_df["expiry"]).dt.date
            self.instruments_df["strike"] = self.instruments_df["strike"].astype(int)
    
    def get_option_instruments(
        self, 
        symbol: str, 
        expiry_date: dt.date,
        strike_range: Optional[Tuple[int, int]] = None
    ) -> pd.DataFrame:
        """Get option instruments for symbol/expiry, optionally filtered by strike range."""
        symbol = symbol.upper()
        
        # Determine exchange and segment
        exchange = "BFO" if symbol == "SENSEX" else "NFO"
        segment = f"{exchange}-OPT"
        
        # Memory optimization: use vectorized operations
        mask = (
            (self.instruments_df["name"] == symbol) &
            (self.instruments_df["segment"] == segment) &
            (self.instruments_df["expiry"] == expiry_date)
        )
        
        # Apply filter in one operation
        filtered_df = self.instruments_df.loc[mask].copy()
        
        if filtered_df.empty:
            logger.warning(f"No option instruments found for {symbol} expiry {expiry_date}")
            return filtered_df
        
        # Apply strike range filter if provided
        if strike_range:
            min_strike, max_strike = strike_range
            strike_mask = (
                (filtered_df["strike"] >= min_strike) &
                (filtered_df["strike"] <= max_strike)
            )
            filtered_df = filtered_df.loc[strike_mask]
        
        logger.info(f"Found {len(filtered_df)} option instruments for {symbol} {expiry_date}")
        return filtered_df
    
    def get_available_expiries(self, symbol: str, min_expiry: Optional[dt.date] = None) -> List[dt.date]:
        """Get list of available expiry dates for symbol."""
        symbol = symbol.upper()
        exchange = "BFO" if symbol == "SENSEX" else "NFO"
        segment = f"{exchange}-OPT"
        
        mask = (
            (self.instruments_df["name"] == symbol) &
            (self.instruments_df["segment"] == segment)
        )
        
        if min_expiry:
            mask &= (self.instruments_df["expiry"] >= min_expiry)
        
        expiries = sorted(self.instruments_df[mask]["expiry"].unique())
        return expiries
    
    def infer_strike_step(self, instruments_df: pd.DataFrame) -> int:
        """Infer strike step from instruments data."""
        if instruments_df.empty:
            return 50  # Default
        
        strikes = sorted(instruments_df["strike"].unique())
        if len(strikes) < 2:
            return 50
        
        # Find most common step size
        steps = [strikes[i+1] - strikes[i] for i in range(len(strikes)-1)]
        step_counts = {}
        for step in steps:
            step_counts[step] = step_counts.get(step, 0) + 1
        
        most_common_step = max(step_counts.keys(), key=lambda k: step_counts[k])
        return int(most_common_step)


class QuoteBatcher:
    """Handles efficient batch quote requests with rate limiting."""
    
    def __init__(self, kite_client, max_batch_size: int = 500, delay_between_batches: float = 0.2):
        """Initialize quote batcher.
        
        Args:
            kite_client: KiteConnect client instance
            max_batch_size: Maximum symbols per batch (Kite limit is 500)
            delay_between_batches: Delay in seconds between batches
        """
        self.kite = kite_client
        self.max_batch_size = max_batch_size
        self.delay_between_batches = delay_between_batches
    
    def get_quotes_batch(self, trading_symbols: List[str]) -> Dict[str, Dict]:
        """Get quotes for list of trading symbols with batching and rate limiting."""
        if not trading_symbols:
            return {}
        
        all_quotes = {}
        batches = [
            trading_symbols[i:i + self.max_batch_size] 
            for i in range(0, len(trading_symbols), self.max_batch_size)
        ]
        
        logger.info(f"Fetching quotes in {len(batches)} batches of max {self.max_batch_size} symbols")
        
        for i, batch in enumerate(batches):
            try:
                start_time = time.time()
                quotes = self.kite.quote(batch)
                fetch_time = time.time() - start_time
                
                if quotes:
                    all_quotes.update(quotes)
                
                logger.debug(f"Batch {i+1}/{len(batches)}: {len(batch)} symbols, {len(quotes)} quotes, {fetch_time:.2f}s")
                
                # Rate limiting: delay between batches except for the last one
                if i < len(batches) - 1:
                    time.sleep(self.delay_between_batches)
                    
            except Exception as e:
                logger.error(f"Error fetching batch {i+1}: {e}")
                # Continue with next batch rather than failing completely
                continue
        
        logger.info(f"Successfully fetched {len(all_quotes)} quotes from {len(trading_symbols)} requested symbols")
        return all_quotes


class OptionDataAssembler:
    """Assembles option chain data from instruments and quotes."""
    
    @staticmethod
    def parse_quote_data(quote: Dict) -> Tuple[float, float, float, int, int, int, int, Optional[dt.datetime]]:
        """Parse quote data into standardized format."""
        # Price data
        ltp = float(quote.get("last_price", 0.0))
        bid = ask = 0.0
        bid_qty = ask_qty = 0
        
        depth = quote.get("depth", {})
        if depth and depth.get("buy") and depth.get("sell"):
            try:
                bid = float(depth["buy"][0]["price"])
                ask = float(depth["sell"][0]["price"])
                bid_qty = int(depth["buy"][0].get("quantity", 0))
                ask_qty = int(depth["sell"][0].get("quantity", 0))
            except (IndexError, KeyError, ValueError):
                pass
        
        # Volume and OI data
        oi = int(quote.get("oi", 0) or quote.get("open_interest", 0))
        volume = int(quote.get("volume", 0))
        
        # Last trade time
        ltt = quote.get("last_trade_time")
        last_trade_time = None
        if ltt:
            try:
                last_trade_time = pd.to_datetime(ltt).tz_localize(dt.timezone.utc).tz_convert(IST)
            except Exception:
                pass
        
        return bid, ask, ltp, bid_qty, ask_qty, oi, volume, last_trade_time
    
    @staticmethod
    def build_option_chain(
        symbol: str,
        expiry: str,
        spot_price: float,
        instruments_df: pd.DataFrame,
        quotes: Dict[str, Dict]
    ) -> OptionChain:
        """Build complete option chain from instruments and quotes data."""
        options = {}
        strikes = set()
        valid_quotes = 0
        total_instruments = len(instruments_df)
        
        # Determine exchange prefix
        exchange = "BFO" if symbol.upper() == "SENSEX" else "NFO"
        
        for _, row in instruments_df.iterrows():
            strike = int(row["strike"])
            option_type = row["instrument_type"]  # "CE" or "PE"
            trading_symbol = row["tradingsymbol"]
            instrument_token = int(row["instrument_token"])
            
            # Find quote data for this instrument
            quote_key = f"{exchange}:{trading_symbol}"
            quote = quotes.get(quote_key, {})
            
            if quote:
                bid, ask, ltp, bid_qty, ask_qty, oi, volume, ltt = OptionDataAssembler.parse_quote_data(quote)
                valid_quotes += 1
            else:
                # No quote data available
                bid = ask = ltp = 0.0
                bid_qty = ask_qty = oi = volume = 0
                ltt = None
            
            option_data = OptionData(
                strike=strike,
                option_type=option_type,
                bid=bid,
                ask=ask,
                ltp=ltp,
                bid_qty=bid_qty,
                ask_qty=ask_qty,
                oi=oi,
                volume=volume,
                last_trade_time=ltt,
                instrument_token=instrument_token,
                trading_symbol=trading_symbol
            )
            
            options[(strike, option_type)] = option_data
            strikes.add(strike)
        
        # Calculate data quality score
        data_quality_score = valid_quotes / total_instruments if total_instruments > 0 else 0.0
        
        option_chain = OptionChain(
            symbol=symbol,
            expiry=expiry,
            spot_price=spot_price,
            strikes=sorted(strikes),
            options=options,
            build_timestamp=dt.datetime.now(IST),
            data_quality_score=data_quality_score
        )
        
        logger.info(f"Built option chain for {symbol} {expiry}: {len(strikes)} strikes, "
                   f"{total_instruments} instruments, {valid_quotes} quotes "
                   f"(quality: {data_quality_score:.1%})")
        
        return option_chain


class OptionChainBuilder:
    """Main orchestrator for building option chains from Kite Connect data."""
    
    def __init__(self, kite_client, instruments_df: pd.DataFrame):
        """Initialize option chain builder.
        
        Args:
            kite_client: KiteConnect client instance
            instruments_df: Complete instruments DataFrame from Kite
        """
        self.kite = kite_client
        self.instrument_filter = InstrumentFilter(instruments_df)
        self.quote_batcher = QuoteBatcher(kite_client)
        self.build_count = 0
        
        # Performance optimization: cache frequently used data
        self._expiry_cache = {}
        self._last_cache_time = 0
        self._cache_ttl = 300  # 5 minutes cache TTL
    
    def build_chain(
        self, 
        symbol: str, 
        expiry: str,
        spot_price: float,
        strike_range_pct: float = 0.20,  # ±20% around spot by default
        max_strikes: int = 100  # Performance limit on strikes
    ) -> Optional[OptionChain]:
        """Build option chain for symbol/expiry with performance optimizations.
        
        Args:
            symbol: Index symbol (e.g., "NIFTY", "BANKNIFTY")
            expiry: Target expiry in YYYY-MM-DD format
            spot_price: Current spot price for strike range calculation
            strike_range_pct: Percentage range around spot to include (e.g., 0.20 = ±20%)
            max_strikes: Maximum number of strikes to process (performance limit)
        
        Returns:
            OptionChain instance or None if build failed
        """
        self.build_count += 1
        start_time = time.time()
        
        try:
            # Parse expiry date
            expiry_date = pd.to_datetime(expiry).date()
            
            # Calculate strike range around spot price
            range_buffer = spot_price * strike_range_pct
            min_strike = int(spot_price - range_buffer)
            max_strike = int(spot_price + range_buffer)
            
            # Get option instruments for this symbol/expiry
            instruments_df = self.instrument_filter.get_option_instruments(
                symbol, expiry_date, (min_strike, max_strike)
            )
            
            if instruments_df.empty:
                logger.warning(f"No instruments found for {symbol} {expiry}")
                return None
            
            # Performance optimization: limit strikes if too many
            if len(instruments_df) > max_strikes * 2:  # *2 for CE and PE
                logger.info(f"Limiting strikes for {symbol} from {len(instruments_df)} to {max_strikes * 2} instruments")
                # Keep strikes closest to spot
                instruments_df['distance_from_spot'] = abs(instruments_df['strike'] - spot_price)
                instruments_df = instruments_df.nsmallest(max_strikes * 2, 'distance_from_spot').drop('distance_from_spot', axis=1)
            
            # Build trading symbols list for quote requests
            exchange = "BFO" if symbol.upper() == "SENSEX" else "NFO"
            trading_symbols = [
                f"{exchange}:{ts}" for ts in instruments_df["tradingsymbol"].tolist()
            ]
            
            # Fetch quotes in batches
            quotes = self.quote_batcher.get_quotes_batch(trading_symbols)
            
            # Assemble final option chain
            option_chain = OptionDataAssembler.build_option_chain(
                symbol, expiry, spot_price, instruments_df, quotes
            )
            
            build_time = time.time() - start_time
            logger.info(f"Built option chain #{self.build_count} for {symbol} {expiry} in {build_time:.2f}s")
            
            return option_chain
            
        except Exception as e:
            logger.error(f"Failed to build option chain for {symbol} {expiry}: {e}")
            return None
    
    def get_available_expiries(self, symbol: str, min_days_ahead: int = 0) -> List[str]:
        """Get available expiry dates for symbol with caching for performance.
        
        Args:
            symbol: Index symbol
            min_days_ahead: Minimum days from today
            
        Returns:
            List of expiry dates in YYYY-MM-DD format
        """
        # Performance optimization: use cache for expiry data
        current_time = time.time()
        cache_key = f"{symbol}_{min_days_ahead}"
        
        if (cache_key in self._expiry_cache and 
            current_time - self._last_cache_time < self._cache_ttl):
            return self._expiry_cache[cache_key]
        
        min_expiry = None
        if min_days_ahead > 0:
            min_expiry = (dt.datetime.now(IST).date() + dt.timedelta(days=min_days_ahead))
        
        expiry_dates = self.instrument_filter.get_available_expiries(symbol, min_expiry)
        expiry_list = [d.isoformat() for d in expiry_dates]
        
        # Cache the result
        self._expiry_cache[cache_key] = expiry_list
        self._last_cache_time = current_time
        
        return expiry_list
    
    def get_nearest_expiry(self, symbol: str) -> Optional[str]:
        """Get the nearest available expiry for symbol."""
        expiries = self.get_available_expiries(symbol)
        return expiries[0] if expiries else None


# Legacy compatibility functions for gradual migration
def convert_to_legacy_format(option_chain: OptionChain) -> Dict:
    """Convert new OptionChain to legacy format for backward compatibility."""
    legacy_chain = {
        "symbol": option_chain.symbol,
        "expiry": option_chain.expiry,
        "strikes": option_chain.strikes,
        "calls": {},
        "puts": {}
    }
    
    for (strike, option_type), option_data in option_chain.options.items():
        legacy_data = {
            "oi": option_data.oi,
            "ltp": option_data.ltp,
            "bid": option_data.bid,
            "ask": option_data.ask,
            "bid_qty": option_data.bid_qty,
            "ask_qty": option_data.ask_qty,
            "ltp_ts": option_data.last_trade_time.isoformat() if option_data.last_trade_time else None
        }
        
        if option_type == "CE":
            legacy_chain["calls"][strike] = legacy_data
        else:
            legacy_chain["puts"][strike] = legacy_data
    
    # Ensure all strikes have entries for both calls and puts
    for strike in option_chain.strikes:
        if strike not in legacy_chain["calls"]:
            legacy_chain["calls"][strike] = {
                "oi": 0, "ltp": 0.0, "bid": 0.0, "ask": 0.0,
                "bid_qty": 0, "ask_qty": 0, "ltp_ts": None
            }
        if strike not in legacy_chain["puts"]:
            legacy_chain["puts"][strike] = {
                "oi": 0, "ltp": 0.0, "bid": 0.0, "ask": 0.0,
                "bid_qty": 0, "ask_qty": 0, "ltp_ts": None
            }
    
    return legacy_chain