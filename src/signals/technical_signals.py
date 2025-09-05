"""
Technical Signal Integration for Trading Engine

This module integrates technical indicators into the trading decision framework,
providing comprehensive signal analysis for Indian Stock Market trading.

Key Features:
- Real-time technical indicator calculation using Kite Connect data
- Multi-timeframe analysis (1min, 5min, 15min, 1hour, 1day)
- Signal aggregation and consensus building
- Risk-adjusted signal confidence scoring
- Integration with existing options analytics
"""

from __future__ import annotations
import datetime as dt
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pytz

from ..calculations.technical_indicators import (
    TechnicalIndicatorEngine,
    calculate_technical_indicators_with_validation
)

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class TechnicalSignalConfig:
    """Configuration for technical signal generation."""
    timeframes: List[str] = None  # ['1minute', '5minute', '15minute', '1hour', '1day']
    lookback_periods: Dict[str, int] = None  # Lookback periods for each timeframe
    min_data_points: int = 100  # Minimum data points required
    signal_weights: Dict[str, float] = None  # Weights for different indicators
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['5minute', '15minute', '1hour']
        
        if self.lookback_periods is None:
            self.lookback_periods = {
                '1minute': 300,    # 5 hours of 1-min data
                '5minute': 288,    # 24 hours of 5-min data  
                '15minute': 96,    # 24 hours of 15-min data
                '1hour': 72,       # 3 days of hourly data
                '1day': 50         # 50 days of daily data
            }
        
        if self.signal_weights is None:
            self.signal_weights = {
                'rsi': 0.15,
                'bollinger': 0.15,
                'adx': 0.20,
                'macd': 0.20,
                'ichimoku': 0.10,
                'vwap': 0.10,
                'ma_crossover': 0.10
            }


@dataclass
class TimeframeSignal:
    """Signal for a specific timeframe."""
    timeframe: str
    signal: str  # BUY, SELL, NEUTRAL
    confidence: float  # 0.0 to 1.0
    strength: str  # WEAK, MODERATE, STRONG
    indicators: Dict[str, Any]
    timestamp: dt.datetime


@dataclass
class ComprehensiveTechnicalSignal:
    """Comprehensive technical signal across multiple timeframes."""
    symbol: str
    spot_price: float
    overall_signal: str
    overall_confidence: float
    overall_strength: str
    timeframe_signals: Dict[str, TimeframeSignal]
    consensus_score: float  # Agreement across timeframes
    risk_score: float  # Risk assessment based on volatility indicators
    timestamp: dt.datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert datetime objects to ISO strings
        result['timestamp'] = self.timestamp.isoformat()
        for tf_signal in result['timeframe_signals'].values():
            tf_signal['timestamp'] = tf_signal['timestamp'].isoformat()
        return result


class TechnicalSignalGenerator:
    """
    Generates technical signals from real-time market data.
    
    This class fetches OHLCV data from Kite Connect, calculates technical indicators
    across multiple timeframes, and provides comprehensive trading signals.
    """
    
    def __init__(self, market_data_provider, config: TechnicalSignalConfig = None):
        """
        Initialize the technical signal generator.
        
        Parameters
        ----------
        market_data_provider : MarketDataProvider
            Kite Connect data provider instance
        config : TechnicalSignalConfig, optional
            Configuration for signal generation
        """
        self.provider = market_data_provider
        self.config = config or TechnicalSignalConfig()
        self.indicator_engine = TechnicalIndicatorEngine()
        
        # Cache for recent data to avoid repeated API calls
        self._data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._cache_timestamps: Dict[str, Dict[str, dt.datetime]] = {}
        self._cache_duration = dt.timedelta(minutes=1)  # Cache for 1 minute
    
    def _get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get cached data if available and fresh."""
        if symbol not in self._data_cache:
            return None
        
        if timeframe not in self._data_cache[symbol]:
            return None
        
        cache_time = self._cache_timestamps.get(symbol, {}).get(timeframe)
        if not cache_time:
            return None
        
        now = dt.datetime.now(IST)
        if now - cache_time > self._cache_duration:
            return None
        
        return self._data_cache[symbol][timeframe]
    
    def _cache_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Cache data with timestamp."""
        if symbol not in self._data_cache:
            self._data_cache[symbol] = {}
            self._cache_timestamps[symbol] = {}
        
        self._data_cache[symbol][timeframe] = data.copy()
        self._cache_timestamps[symbol][timeframe] = dt.datetime.now(IST)
    
    def _fetch_ohlcv_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol and timeframe.
        
        Returns None if data fetching fails (no synthetic fallbacks).
        """
        # Check cache first
        cached_data = self._get_cached_data(symbol, timeframe)
        if cached_data is not None:
            return cached_data
        
        try:
            lookback_minutes = self.config.lookback_periods.get(timeframe, 300)
            
            # Map timeframe to Kite Connect intervals
            interval_map = {
                '1minute': 'minute',
                '5minute': '5minute', 
                '15minute': '15minute',
                '1hour': '60minute',
                '1day': 'day'
            }
            
            interval = interval_map.get(timeframe, '5minute')
            
            # Fetch data using provider
            data = self.provider.get_spot_ohlcv(symbol, interval, lookback_minutes)
            
            if data.empty:
                raise ValueError(f"No data returned for {symbol} {timeframe}")
            
            # Validate minimum data points
            if len(data) < self.config.min_data_points:
                raise ValueError(f"Insufficient data: {len(data)} < {self.config.min_data_points}")
            
            # Cache the data
            self._cache_data(symbol, timeframe, data)
            
            return data
            
        except Exception as e:
            # No synthetic fallbacks - return None to indicate failure
            print(f"Failed to fetch {symbol} {timeframe} data: {e}")
            return None
    
    def _calculate_timeframe_signal(self, symbol: str, timeframe: str, spot_price: float) -> Optional[TimeframeSignal]:
        """
        Calculate technical signal for a specific timeframe.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        timeframe : str
            Timeframe (e.g., '5minute', '1hour')
        spot_price : float
            Current spot price
            
        Returns
        -------
        Optional[TimeframeSignal]
            Technical signal or None if calculation fails
        """
        try:
            # Fetch OHLCV data
            ohlcv_data = self._fetch_ohlcv_data(symbol, timeframe)
            if ohlcv_data is None:
                return None
            
            # Calculate technical indicators
            indicators, status = calculate_technical_indicators_with_validation(ohlcv_data)
            if indicators is None:
                print(f"Indicator calculation failed for {symbol} {timeframe}: {status}")
                return None
            
            # Generate comprehensive signal
            signal_result = self.indicator_engine.get_comprehensive_signal(indicators, spot_price)
            
            return TimeframeSignal(
                timeframe=timeframe,
                signal=signal_result['signal'],
                confidence=signal_result['confidence'],
                strength=signal_result['strength'],
                indicators=indicators,
                timestamp=dt.datetime.now(IST)
            )
            
        except Exception as e:
            print(f"Error calculating signal for {symbol} {timeframe}: {e}")
            return None
    
    def _calculate_risk_score(self, timeframe_signals: Dict[str, TimeframeSignal]) -> float:
        """
        Calculate risk score based on volatility indicators.
        
        Higher score indicates higher risk/volatility.
        """
        risk_factors = []
        
        for tf_signal in timeframe_signals.values():
            if tf_signal.indicators:
                # Check Bollinger Band width (volatility measure)
                if 'bollinger' in tf_signal.indicators:
                    bb_data = tf_signal.indicators['bollinger']
                    if 'width' in bb_data and not bb_data['width'].empty:
                        latest_width = bb_data['width'].iloc[-1]
                        if not pd.isna(latest_width):
                            # Normalize width to 0-1 scale (assume max reasonable width is 10%)
                            normalized_width = min(latest_width / 10.0, 1.0)
                            risk_factors.append(normalized_width)
                
                # Check ADX for trend strength (lower ADX = higher risk)
                if 'adx' in tf_signal.indicators:
                    adx_data = tf_signal.indicators['adx']
                    if 'adx' in adx_data and not adx_data['adx'].empty:
                        latest_adx = adx_data['adx'].iloc[-1]
                        if not pd.isna(latest_adx):
                            # Higher ADX = lower risk (strong trend)
                            risk_from_adx = max(0, (50 - latest_adx) / 50)
                            risk_factors.append(risk_from_adx)
        
        if not risk_factors:
            return 0.5  # Default moderate risk
        
        return sum(risk_factors) / len(risk_factors)
    
    def _calculate_consensus_score(self, timeframe_signals: Dict[str, TimeframeSignal]) -> float:
        """
        Calculate consensus score across timeframes.
        
        Returns value between 0.0 (no agreement) and 1.0 (full agreement).
        """
        if not timeframe_signals:
            return 0.0
        
        signals = [tf_signal.signal for tf_signal in timeframe_signals.values()]
        
        # Count signal types
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        neutral_count = signals.count('NEUTRAL')
        total_count = len(signals)
        
        # Calculate agreement
        max_agreement = max(buy_count, sell_count, neutral_count)
        consensus_score = max_agreement / total_count if total_count > 0 else 0.0
        
        return consensus_score
    
    def _aggregate_signals(self, timeframe_signals: Dict[str, TimeframeSignal]) -> Tuple[str, float, str]:
        """
        Aggregate signals across timeframes with weighted voting.
        
        Returns
        -------
        Tuple[str, float, str]
            (overall_signal, overall_confidence, overall_strength)
        """
        if not timeframe_signals:
            return "NEUTRAL", 0.0, "WEAK"
        
        # Timeframe weights (longer timeframes have higher weight)
        tf_weights = {
            '1minute': 0.1,
            '5minute': 0.15,
            '15minute': 0.25,
            '1hour': 0.3,
            '1day': 0.2
        }
        
        weighted_buy_score = 0.0
        weighted_sell_score = 0.0
        weighted_neutral_score = 0.0
        total_weight = 0.0
        total_confidence = 0.0
        
        for tf_signal in timeframe_signals.values():
            weight = tf_weights.get(tf_signal.timeframe, 0.1)
            total_weight += weight
            total_confidence += tf_signal.confidence * weight
            
            if tf_signal.signal == "BUY":
                weighted_buy_score += weight * tf_signal.confidence
            elif tf_signal.signal == "SELL":
                weighted_sell_score += weight * tf_signal.confidence
            else:
                weighted_neutral_score += weight * tf_signal.confidence
        
        # Normalize scores
        if total_weight > 0:
            weighted_buy_score /= total_weight
            weighted_sell_score /= total_weight
            weighted_neutral_score /= total_weight
            avg_confidence = total_confidence / total_weight
        else:
            avg_confidence = 0.0
        
        # Determine overall signal
        max_score = max(weighted_buy_score, weighted_sell_score, weighted_neutral_score)
        
        if max_score == weighted_buy_score:
            overall_signal = "BUY"
            overall_confidence = weighted_buy_score
        elif max_score == weighted_sell_score:
            overall_signal = "SELL" 
            overall_confidence = weighted_sell_score
        else:
            overall_signal = "NEUTRAL"
            overall_confidence = weighted_neutral_score
        
        # Determine strength
        if overall_confidence >= 0.7:
            strength = "STRONG"
        elif overall_confidence >= 0.4:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        return overall_signal, overall_confidence, strength
    
    def generate_comprehensive_signal(self, symbol: str) -> Optional[ComprehensiveTechnicalSignal]:
        """
        Generate comprehensive technical signal for a symbol.
        
        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., 'NIFTY', 'BANKNIFTY')
            
        Returns
        -------
        Optional[ComprehensiveTechnicalSignal]
            Comprehensive signal or None if generation fails
        """
        try:
            # Get current spot price
            indices_data = self.provider.get_indices_snapshot([symbol])
            if symbol not in indices_data:
                print(f"Failed to get spot price for {symbol}")
                return None
            
            spot_price = indices_data[symbol]
            
            # Calculate signals for each timeframe
            timeframe_signals = {}
            
            for timeframe in self.config.timeframes:
                tf_signal = self._calculate_timeframe_signal(symbol, timeframe, spot_price)
                if tf_signal is not None:
                    timeframe_signals[timeframe] = tf_signal
            
            if not timeframe_signals:
                print(f"No valid timeframe signals generated for {symbol}")
                return None
            
            # Aggregate signals
            overall_signal, overall_confidence, overall_strength = self._aggregate_signals(timeframe_signals)
            
            # Calculate additional metrics
            consensus_score = self._calculate_consensus_score(timeframe_signals)
            risk_score = self._calculate_risk_score(timeframe_signals)
            
            return ComprehensiveTechnicalSignal(
                symbol=symbol,
                spot_price=spot_price,
                overall_signal=overall_signal,
                overall_confidence=overall_confidence,
                overall_strength=overall_strength,
                timeframe_signals=timeframe_signals,
                consensus_score=consensus_score,
                risk_score=risk_score,
                timestamp=dt.datetime.now(IST)
            )
            
        except Exception as e:
            print(f"Error generating comprehensive signal for {symbol}: {e}")
            return None
    
    def generate_signals_for_symbols(self, symbols: List[str]) -> Dict[str, ComprehensiveTechnicalSignal]:
        """
        Generate technical signals for multiple symbols.
        
        Parameters
        ----------
        symbols : List[str]
            List of trading symbols
            
        Returns
        -------
        Dict[str, ComprehensiveTechnicalSignal]
            Technical signals by symbol
        """
        results = {}
        
        for symbol in symbols:
            signal = self.generate_comprehensive_signal(symbol)
            if signal is not None:
                results[symbol] = signal
            else:
                print(f"Failed to generate signal for {symbol}")
        
        return results
    
    def get_signal_summary(self, signal: ComprehensiveTechnicalSignal) -> Dict[str, Any]:
        """
        Get a summary of the technical signal for reporting.
        
        Parameters
        ----------
        signal : ComprehensiveTechnicalSignal
            Technical signal to summarize
            
        Returns
        -------
        Dict[str, Any]
            Signal summary
        """
        return {
            'symbol': signal.symbol,
            'spot_price': signal.spot_price,
            'signal': signal.overall_signal,
            'confidence': round(signal.overall_confidence, 3),
            'strength': signal.overall_strength,
            'consensus': round(signal.consensus_score, 3),
            'risk': round(signal.risk_score, 3),
            'timeframes_analyzed': len(signal.timeframe_signals),
            'timestamp': signal.timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')
        }


class TechnicalSignalValidator:
    """
    Validates technical signals against market conditions and data quality.
    """
    
    @staticmethod
    def validate_signal_quality(signal: ComprehensiveTechnicalSignal) -> Tuple[bool, str]:
        """
        Validate the quality of a technical signal.
        
        Returns
        -------
        Tuple[bool, str]
            (is_valid, validation_message)
        """
        if not signal:
            return False, "Empty signal"
        
        if signal.spot_price <= 0:
            return False, "Invalid spot price"
        
        if len(signal.timeframe_signals) == 0:
            return False, "No timeframe signals available"
        
        if signal.overall_confidence < 0 or signal.overall_confidence > 1:
            return False, "Invalid confidence score"
        
        if signal.consensus_score < 0 or signal.consensus_score > 1:
            return False, "Invalid consensus score"
        
        if signal.risk_score < 0 or signal.risk_score > 1:
            return False, "Invalid risk score"
        
        # Check for recent timestamp (within last 30 minutes)
        now = dt.datetime.now(IST)
        if now - signal.timestamp > dt.timedelta(minutes=30):
            return False, "Signal timestamp too old"
        
        return True, "Valid signal"
    
    @staticmethod
    def check_data_freshness(signal: ComprehensiveTechnicalSignal, max_age_minutes: int = 5) -> bool:
        """
        Check if signal data is fresh enough for trading decisions.
        
        Parameters
        ----------
        signal : ComprehensiveTechnicalSignal
            Signal to check
        max_age_minutes : int, default 5
            Maximum acceptable age in minutes
            
        Returns
        -------
        bool
            True if data is fresh enough
        """
        now = dt.datetime.now(IST)
        age = now - signal.timestamp
        return age.total_seconds() / 60 <= max_age_minutes
    
    @staticmethod
    def validate_trading_hours(timestamp: dt.datetime) -> bool:
        """
        Check if signal was generated during trading hours.
        
        Indian market hours: 9:15 AM - 3:30 PM IST (Mon-Fri)
        """
        if timestamp.weekday() >= 5:  # Weekend
            return False
        
        market_open = dt.time(9, 15)
        market_close = dt.time(15, 30)
        signal_time = timestamp.time()
        
        return market_open <= signal_time <= market_close


def create_technical_signal_config(
    aggressive: bool = False,
    conservative: bool = False
) -> TechnicalSignalConfig:
    """
    Create predefined technical signal configurations.
    
    Parameters
    ----------
    aggressive : bool, default False
        Use aggressive trading parameters (shorter timeframes, lower thresholds)
    conservative : bool, default False
        Use conservative trading parameters (longer timeframes, higher thresholds)
        
    Returns
    -------
    TechnicalSignalConfig
        Configured technical signal parameters
    """
    if aggressive:
        return TechnicalSignalConfig(
            timeframes=['1minute', '5minute', '15minute'],
            lookback_periods={
                '1minute': 240,  # 4 hours
                '5minute': 144,  # 12 hours
                '15minute': 48   # 12 hours
            },
            min_data_points=50,
            signal_weights={
                'rsi': 0.20,
                'bollinger': 0.20,
                'adx': 0.15,
                'macd': 0.25,
                'ichimoku': 0.05,
                'vwap': 0.15,
                'ma_crossover': 0.00
            }
        )
    elif conservative:
        return TechnicalSignalConfig(
            timeframes=['1hour', '1day'],
            lookback_periods={
                '1hour': 168,  # 7 days
                '1day': 100    # 100 days
            },
            min_data_points=150,
            signal_weights={
                'rsi': 0.10,
                'bollinger': 0.10,
                'adx': 0.25,
                'macd': 0.15,
                'ichimoku': 0.20,
                'vwap': 0.05,
                'ma_crossover': 0.15
            }
        )
    else:
        # Default balanced configuration
        return TechnicalSignalConfig()