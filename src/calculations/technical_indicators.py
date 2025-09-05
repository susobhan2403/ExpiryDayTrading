"""
Technical Indicators for Indian Stock Market - Pure Mathematical Implementation

This module implements industry-standard technical indicators using pure mathematical
formulas per Indian Stock Market standards. No parameter tuning or benchmark adjustments.

Indicators Implemented:
- RSI (Relative Strength Index)
- Bollinger Bands
- ADX/DMS (Average Directional Index/Directional Movement System)
- MACD (Moving Average Convergence Divergence)
- Ichimoku Cloud
- Fibonacci Retracement
- VWAP (Volume Weighted Average Price)
- Moving Averages (SMA, EMA, WMA)

All calculations use real-time data from Kite Connect with zero synthetic fallbacks.
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class TechnicalIndicatorResult:
    """Container for technical indicator calculation results."""
    indicator: str
    value: Union[float, Dict[str, float], List[float]]
    timestamp: pd.Timestamp
    signal: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None


class RSICalculator:
    """
    Relative Strength Index (RSI) - Momentum Oscillator
    
    RSI measures the speed and change of price movements.
    
    Mathematical Formula (Wilder's smoothing):
    RS = Average Gain / Average Loss
    RSI = 100 - (100 / (1 + RS))
    
    Where:
    - Average Gain = (Previous Average Gain * (n-1) + Current Gain) / n
    - Average Loss = (Previous Average Loss * (n-1) + Current Loss) / n
    - Standard period: 14
    
    Interpretation (Indian Market Standards):
    - RSI > 70: Overbought (potential sell signal)
    - RSI < 30: Oversold (potential buy signal)
    - RSI 50: Neutral momentum
    """
    
    @staticmethod
    def calculate(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI using Wilder's smoothing method.
        
        Parameters
        ----------
        prices : pd.Series
            Price series (typically closing prices)
        period : int, default 14
            RSI calculation period
            
        Returns
        -------
        pd.Series
            RSI values (0-100 scale)
        """
        if len(prices) < period + 1:
            raise ValueError(f"Insufficient data: need {period + 1} points, got {len(prices)}")
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate initial averages (SMA for first period)
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Apply Wilder's smoothing for subsequent periods
        for i in range(period, len(prices)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gains.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + losses.iloc[i]) / period
        
        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def get_signal(rsi_value: float) -> Tuple[str, float]:
        """
        Get trading signal based on RSI value.
        
        Returns
        -------
        Tuple[str, float]
            (signal, confidence)
        """
        if rsi_value >= 70:
            return "SELL", min((rsi_value - 70) / 30, 1.0)
        elif rsi_value <= 30:
            return "BUY", min((30 - rsi_value) / 30, 1.0)
        else:
            return "NEUTRAL", abs(rsi_value - 50) / 50


class BollingerBandsCalculator:
    """
    Bollinger Bands - Volatility-based Trading Bands
    
    Mathematical Formula:
    Middle Band = Simple Moving Average (SMA)
    Upper Band = SMA + (Standard Deviation * multiplier)
    Lower Band = SMA - (Standard Deviation * multiplier)
    
    Standard Parameters (Indian Market):
    - Period: 20
    - Multiplier: 2.0
    - Calculation: Close prices
    
    Interpretation:
    - Price touching upper band: Potential resistance (overbought)
    - Price touching lower band: Potential support (oversold)
    - Band width: Volatility measure
    """
    
    @staticmethod
    def calculate(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Parameters
        ----------
        prices : pd.Series
            Price series (typically closing prices)
        period : int, default 20
            Moving average period
        std_dev : float, default 2.0
            Standard deviation multiplier
            
        Returns
        -------
        Dict[str, pd.Series]
            Dictionary with 'upper', 'middle', 'lower' bands and 'width'
        """
        if len(prices) < period:
            raise ValueError(f"Insufficient data: need {period} points, got {len(prices)}")
        
        # Calculate middle band (SMA)
        middle = prices.rolling(window=period).mean()
        
        # Calculate standard deviation
        std = prices.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        # Calculate band width (volatility measure)
        width = (upper - lower) / middle * 100
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width
        }
    
    @staticmethod
    def get_signal(price: float, bands: Dict[str, float]) -> Tuple[str, float]:
        """
        Get trading signal based on price position relative to bands.
        
        Parameters
        ----------
        price : float
            Current price
        bands : Dict[str, float]
            Current band values {'upper', 'middle', 'lower'}
            
        Returns
        -------
        Tuple[str, float]
            (signal, confidence)
        """
        upper, middle, lower = bands['upper'], bands['middle'], bands['lower']
        
        if price >= upper:
            # Price at or above upper band - potential sell
            confidence = min((price - upper) / (upper - middle), 1.0)
            return "SELL", confidence
        elif price <= lower:
            # Price at or below lower band - potential buy
            confidence = min((lower - price) / (middle - lower), 1.0)
            return "BUY", confidence
        else:
            # Price within bands - neutral
            distance_from_middle = abs(price - middle) / (upper - middle)
            return "NEUTRAL", 1.0 - distance_from_middle


class ADXCalculator:
    """
    Average Directional Index (ADX) with Directional Movement System (DMS)
    
    ADX measures trend strength (0-100 scale), while +DI and -DI indicate direction.
    
    Mathematical Formula:
    1. True Range (TR) = max(H-L, |H-C₁|, |L-C₁|)
    2. +DM = max(H-H₁, 0) if H-H₁ > L₁-L, else 0
    3. -DM = max(L₁-L, 0) if L₁-L > H-H₁, else 0
    4. +DI₁₄ = (+DM₁₄ / TR₁₄) * 100
    5. -DI₁₄ = (-DM₁₄ / TR₁₄) * 100
    6. DX = |(+DI₁₄ - -DI₁₄)| / (+DI₁₄ + -DI₁₄) * 100
    7. ADX₁₄ = Wilder's smoothing of DX
    
    Interpretation (Indian Market):
    - ADX > 25: Strong trend
    - ADX < 20: Weak/sideways trend
    - +DI > -DI: Uptrend
    - -DI > +DI: Downtrend
    """
    
    @staticmethod
    def calculate(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """
        Calculate ADX and Directional Movement indicators.
        
        Parameters
        ----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        close : pd.Series
            Close prices
        period : int, default 14
            Calculation period
            
        Returns
        -------
        Dict[str, pd.Series]
            Dictionary with 'adx', 'plus_di', 'minus_di' values
        """
        if len(high) < period + 1:
            raise ValueError(f"Insufficient data: need {period + 1} points, got {len(high)}")
        
        # Calculate True Range (TR)
        high_low = high - low
        high_close_prev = (high - close.shift(1)).abs()
        low_close_prev = (low - close.shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)
        
        high_diff = high - high.shift(1)
        low_diff = low.shift(1) - low
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        # Calculate smoothed TR, +DM, -DM using Wilder's smoothing
        tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
        
        # Calculate Directional Indicators
        plus_di = (plus_dm_smooth / tr_smooth) * 100
        minus_di = (minus_dm_smooth / tr_smooth) * 100
        
        # Calculate DX
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        
        # Calculate ADX using Wilder's smoothing
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    @staticmethod
    def get_signal(adx_value: float, plus_di: float, minus_di: float) -> Tuple[str, float]:
        """
        Get trading signal based on ADX and directional indicators.
        
        Returns
        -------
        Tuple[str, float]
            (signal, confidence)
        """
        # Trend strength based on ADX
        if adx_value < 20:
            return "SIDEWAYS", adx_value / 20
        
        # Trend direction based on DI
        trend_strength = min(adx_value / 100, 1.0)
        
        if plus_di > minus_di:
            return "BUY", trend_strength
        elif minus_di > plus_di:
            return "SELL", trend_strength
        else:
            return "NEUTRAL", trend_strength


class MACDCalculator:
    """
    Moving Average Convergence Divergence (MACD)
    
    Mathematical Formula:
    MACD Line = EMA₁₂ - EMA₂₆
    Signal Line = EMA₉(MACD Line)
    Histogram = MACD Line - Signal Line
    
    Standard Parameters (Indian Market):
    - Fast EMA: 12 periods
    - Slow EMA: 26 periods
    - Signal EMA: 9 periods
    
    Interpretation:
    - MACD > Signal: Bullish momentum
    - MACD < Signal: Bearish momentum
    - Histogram: Momentum strength
    - Zero line crossover: Trend change
    """
    
    @staticmethod
    def calculate(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD indicator.
        
        Parameters
        ----------
        prices : pd.Series
            Price series (typically closing prices)
        fast : int, default 12
            Fast EMA period
        slow : int, default 26
            Slow EMA period
        signal : int, default 9
            Signal line EMA period
            
        Returns
        -------
        Dict[str, pd.Series]
            Dictionary with 'macd', 'signal', 'histogram' values
        """
        if len(prices) < slow:
            raise ValueError(f"Insufficient data: need {slow} points, got {len(prices)}")
        
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate Signal line
        signal_line = macd_line.ewm(span=signal).mean()
        
        # Calculate Histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def get_signal(macd_value: float, signal_value: float, histogram: float) -> Tuple[str, float]:
        """
        Get trading signal based on MACD values.
        
        Returns
        -------
        Tuple[str, float]
            (signal, confidence)
        """
        # Signal based on MACD line vs Signal line
        if macd_value > signal_value and histogram > 0:
            confidence = min(abs(histogram) / abs(macd_value), 1.0) if macd_value != 0 else 0.5
            return "BUY", confidence
        elif macd_value < signal_value and histogram < 0:
            confidence = min(abs(histogram) / abs(macd_value), 1.0) if macd_value != 0 else 0.5
            return "SELL", confidence
        else:
            return "NEUTRAL", 0.1


class IchimokuCalculator:
    """
    Ichimoku Cloud (Ichimoku Kinko Hyo) - Comprehensive Trend Analysis
    
    Mathematical Formulas:
    Tenkan-sen (Conversion Line) = (Highest High + Lowest Low) / 2 over 9 periods
    Kijun-sen (Base Line) = (Highest High + Lowest Low) / 2 over 26 periods
    Senkou Span A = (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
    Senkou Span B = (Highest High + Lowest Low) / 2 over 52 periods, shifted 26 periods ahead
    Chikou Span = Close price shifted 26 periods behind
    
    Standard Parameters:
    - Tenkan: 9 periods
    - Kijun: 26 periods
    - Senkou B: 52 periods
    - Displacement: 26 periods
    """
    
    @staticmethod
    def calculate(high: pd.Series, low: pd.Series, close: pd.Series,
                 tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> Dict[str, pd.Series]:
        """
        Calculate Ichimoku Cloud components.
        
        Parameters
        ----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        close : pd.Series
            Close prices
        tenkan : int, default 9
            Tenkan-sen period
        kijun : int, default 26
            Kijun-sen period
        senkou_b : int, default 52
            Senkou Span B period
            
        Returns
        -------
        Dict[str, pd.Series]
            Dictionary with all Ichimoku components
        """
        if len(high) < senkou_b:
            raise ValueError(f"Insufficient data: need {senkou_b} points, got {len(high)}")
        
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(window=senkou_b).max() + 
                         low.rolling(window=senkou_b).min()) / 2).shift(kijun)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-kijun)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def get_signal(price: float, tenkan: float, kijun: float, 
                  span_a: float, span_b: float) -> Tuple[str, float]:
        """
        Get trading signal based on Ichimoku components.
        
        Returns
        -------
        Tuple[str, float]
            (signal, confidence)
        """
        signals = []
        
        # Price vs Cloud
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if price > cloud_top:
            signals.append(("BUY", 0.4))
        elif price < cloud_bottom:
            signals.append(("SELL", 0.4))
        
        # Tenkan vs Kijun
        if tenkan > kijun:
            signals.append(("BUY", 0.3))
        elif tenkan < kijun:
            signals.append(("SELL", 0.3))
        
        # Price vs Kijun
        if price > kijun:
            signals.append(("BUY", 0.3))
        elif price < kijun:
            signals.append(("SELL", 0.3))
        
        # Aggregate signals
        buy_confidence = sum(conf for sig, conf in signals if sig == "BUY")
        sell_confidence = sum(conf for sig, conf in signals if sig == "SELL")
        
        if buy_confidence > sell_confidence:
            return "BUY", min(buy_confidence, 1.0)
        elif sell_confidence > buy_confidence:
            return "SELL", min(sell_confidence, 1.0)
        else:
            return "NEUTRAL", 0.1


class FibonacciCalculator:
    """
    Fibonacci Retracement Levels
    
    Mathematical Formula:
    Fibonacci Levels = High - (High - Low) * Fibonacci_Ratio
    
    Standard Fibonacci Ratios:
    - 0.0% (High)
    - 23.6% (0.236)
    - 38.2% (0.382)
    - 50.0% (0.500)
    - 61.8% (0.618)
    - 78.6% (0.786)
    - 100.0% (Low)
    
    Interpretation:
    - Support/Resistance levels for trend continuation/reversal
    - 38.2% and 61.8% are key levels in Indian markets
    """
    
    @staticmethod
    def calculate(high: float, low: float) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels.
        
        Parameters
        ----------
        high : float
            Period high
        low : float
            Period low
            
        Returns
        -------
        Dict[str, float]
            Dictionary with Fibonacci levels
        """
        if high <= low:
            raise ValueError(f"High ({high}) must be greater than Low ({low})")
        
        range_val = high - low
        
        levels = {
            'level_0': high,                           # 0.0%
            'level_236': high - range_val * 0.236,     # 23.6%
            'level_382': high - range_val * 0.382,     # 38.2%
            'level_500': high - range_val * 0.500,     # 50.0%
            'level_618': high - range_val * 0.618,     # 61.8%
            'level_786': high - range_val * 0.786,     # 78.6%
            'level_100': low                           # 100.0%
        }
        
        return levels
    
    @staticmethod
    def get_nearest_level(price: float, levels: Dict[str, float]) -> Tuple[str, float, float]:
        """
        Find the nearest Fibonacci level to current price.
        
        Returns
        -------
        Tuple[str, float, float]
            (level_name, level_value, distance_percentage)
        """
        min_distance = float('inf')
        nearest_level = None
        nearest_value = None
        
        for level_name, level_value in levels.items():
            distance = abs(price - level_value)
            if distance < min_distance:
                min_distance = distance
                nearest_level = level_name
                nearest_value = level_value
        
        distance_pct = (min_distance / price) * 100 if price != 0 else 0
        
        return nearest_level, nearest_value, distance_pct


class VWAPCalculator:
    """
    Volume Weighted Average Price (VWAP)
    
    Mathematical Formula:
    VWAP = Σ(Price × Volume) / Σ(Volume)
    
    Where Price is typically (H + L + C) / 3 (typical price)
    
    Interpretation (Indian Market):
    - Price > VWAP: Bullish (above average weighted price)
    - Price < VWAP: Bearish (below average weighted price)
    - VWAP acts as dynamic support/resistance
    - Institutional traders often use VWAP for execution
    """
    
    @staticmethod
    def calculate(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate VWAP.
        
        Parameters
        ----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        close : pd.Series
            Close prices
        volume : pd.Series
            Volume data
            
        Returns
        -------
        pd.Series
            VWAP values
        """
        if len(high) == 0:
            raise ValueError("Empty price data")
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate VWAP
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap
    
    @staticmethod
    def get_signal(price: float, vwap_value: float) -> Tuple[str, float]:
        """
        Get trading signal based on price vs VWAP.
        
        Returns
        -------
        Tuple[str, float]
            (signal, confidence)
        """
        deviation = abs(price - vwap_value) / vwap_value * 100
        confidence = min(deviation / 2.0, 1.0)  # Max confidence at 2% deviation
        
        if price > vwap_value:
            return "BUY", confidence
        elif price < vwap_value:
            return "SELL", confidence
        else:
            return "NEUTRAL", 0.0


class MovingAverageCalculator:
    """
    Moving Averages - Multiple Types
    
    Types Implemented:
    1. Simple Moving Average (SMA): Arithmetic mean
    2. Exponential Moving Average (EMA): Exponentially weighted
    3. Weighted Moving Average (WMA): Linearly weighted
    
    Mathematical Formulas:
    SMA = Σ(Price) / n
    EMA = (Price × α) + (Previous EMA × (1 - α)), where α = 2/(n+1)
    WMA = Σ(Price × Weight) / Σ(Weight), where Weight = period position
    """
    
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            raise ValueError(f"Insufficient data: need {period} points, got {len(prices)}")
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            raise ValueError(f"Insufficient data: need {period} points, got {len(prices)}")
        return prices.ewm(span=period).mean()
    
    @staticmethod
    def wma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Weighted Moving Average."""
        if len(prices) < period:
            raise ValueError(f"Insufficient data: need {period} points, got {len(prices)}")
        
        weights = np.arange(1, period + 1)
        
        def weighted_mean(x):
            return np.dot(x, weights) / weights.sum()
        
        return prices.rolling(window=period).apply(weighted_mean, raw=True)
    
    @staticmethod
    def get_crossover_signal(price: float, ma_fast: float, ma_slow: float) -> Tuple[str, float]:
        """
        Get signal based on moving average crossover.
        
        Parameters
        ----------
        price : float
            Current price
        ma_fast : float
            Fast moving average
        ma_slow : float
            Slow moving average
            
        Returns
        -------
        Tuple[str, float]
            (signal, confidence)
        """
        # Price vs MA signals
        price_signal = 0.5 if price > ma_fast else -0.5
        
        # MA crossover signal
        if ma_fast > ma_slow:
            crossover_signal = 0.5
        elif ma_fast < ma_slow:
            crossover_signal = -0.5
        else:
            crossover_signal = 0.0
        
        total_signal = price_signal + crossover_signal
        confidence = abs(total_signal)
        
        if total_signal > 0.5:
            return "BUY", confidence
        elif total_signal < -0.5:
            return "SELL", confidence
        else:
            return "NEUTRAL", confidence


class TechnicalIndicatorEngine:
    """
    Comprehensive Technical Indicator Engine
    
    Integrates all technical indicators for unified analysis.
    Provides comprehensive signals based on multiple indicators.
    """
    
    def __init__(self):
        self.rsi = RSICalculator()
        self.bollinger = BollingerBandsCalculator()
        self.adx = ADXCalculator()
        self.macd = MACDCalculator()
        self.ichimoku = IchimokuCalculator()
        self.fibonacci = FibonacciCalculator()
        self.vwap = VWAPCalculator()
        self.ma = MovingAverageCalculator()
    
    def calculate_all_indicators(self, ohlcv_data: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate all technical indicators from OHLCV data.
        
        Parameters
        ----------
        ohlcv_data : pd.DataFrame
            OHLCV data with columns: open, high, low, close, volume
            
        Returns
        -------
        Dict[str, any]
            All calculated indicators
        """
        if ohlcv_data.empty:
            raise ValueError("Empty OHLCV data provided")
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in ohlcv_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        high = ohlcv_data['high']
        low = ohlcv_data['low']
        close = ohlcv_data['close']
        volume = ohlcv_data['volume']
        
        results = {}
        
        try:
            # RSI
            results['rsi'] = self.rsi.calculate(close)
            
            # Bollinger Bands
            results['bollinger'] = self.bollinger.calculate(close)
            
            # ADX/DMS
            results['adx'] = self.adx.calculate(high, low, close)
            
            # MACD
            results['macd'] = self.macd.calculate(close)
            
            # Ichimoku Cloud
            results['ichimoku'] = self.ichimoku.calculate(high, low, close)
            
            # VWAP
            results['vwap'] = self.vwap.calculate(high, low, close, volume)
            
            # Moving Averages
            results['ma'] = {
                'sma_20': self.ma.sma(close, 20),
                'sma_50': self.ma.sma(close, 50),
                'ema_12': self.ma.ema(close, 12),
                'ema_26': self.ma.ema(close, 26)
            }
            
            # Fibonacci (using recent high/low)
            if len(high) >= 20:
                recent_high = high.tail(20).max()
                recent_low = low.tail(20).min()
                results['fibonacci'] = self.fibonacci.calculate(recent_high, recent_low)
            
        except Exception as e:
            raise ValueError(f"Technical indicator calculation failed: {str(e)}")
        
        return results
    
    def get_comprehensive_signal(self, indicators: Dict[str, any], current_price: float) -> Dict[str, any]:
        """
        Generate comprehensive trading signal from all indicators.
        
        Parameters
        ----------
        indicators : Dict[str, any]
            Calculated indicators
        current_price : float
            Current market price
            
        Returns
        -------
        Dict[str, any]
            Comprehensive signal analysis
        """
        signals = []
        confidences = []
        
        try:
            # RSI Signal
            if 'rsi' in indicators and not indicators['rsi'].empty:
                latest_rsi = indicators['rsi'].iloc[-1]
                if not pd.isna(latest_rsi):
                    signal, confidence = self.rsi.get_signal(latest_rsi)
                    signals.append(signal)
                    confidences.append(confidence)
            
            # Bollinger Bands Signal
            if 'bollinger' in indicators:
                bands = indicators['bollinger']
                if all(not bands[key].empty for key in ['upper', 'middle', 'lower']):
                    latest_bands = {
                        'upper': bands['upper'].iloc[-1],
                        'middle': bands['middle'].iloc[-1],
                        'lower': bands['lower'].iloc[-1]
                    }
                    if not any(pd.isna(val) for val in latest_bands.values()):
                        signal, confidence = self.bollinger.get_signal(current_price, latest_bands)
                        signals.append(signal)
                        confidences.append(confidence)
            
            # ADX Signal
            if 'adx' in indicators:
                adx_data = indicators['adx']
                if all(not adx_data[key].empty for key in ['adx', 'plus_di', 'minus_di']):
                    latest_adx = adx_data['adx'].iloc[-1]
                    latest_plus_di = adx_data['plus_di'].iloc[-1]
                    latest_minus_di = adx_data['minus_di'].iloc[-1]
                    if not any(pd.isna(val) for val in [latest_adx, latest_plus_di, latest_minus_di]):
                        signal, confidence = self.adx.get_signal(latest_adx, latest_plus_di, latest_minus_di)
                        signals.append(signal)
                        confidences.append(confidence)
            
            # MACD Signal
            if 'macd' in indicators:
                macd_data = indicators['macd']
                if all(not macd_data[key].empty for key in ['macd', 'signal', 'histogram']):
                    latest_macd = macd_data['macd'].iloc[-1]
                    latest_signal = macd_data['signal'].iloc[-1]
                    latest_histogram = macd_data['histogram'].iloc[-1]
                    if not any(pd.isna(val) for val in [latest_macd, latest_signal, latest_histogram]):
                        signal, confidence = self.macd.get_signal(latest_macd, latest_signal, latest_histogram)
                        signals.append(signal)
                        confidences.append(confidence)
            
            # VWAP Signal
            if 'vwap' in indicators and not indicators['vwap'].empty:
                latest_vwap = indicators['vwap'].iloc[-1]
                if not pd.isna(latest_vwap):
                    signal, confidence = self.vwap.get_signal(current_price, latest_vwap)
                    signals.append(signal)
                    confidences.append(confidence)
            
            # Aggregate signals
            buy_votes = sum(1 for s in signals if s == "BUY")
            sell_votes = sum(1 for s in signals if s == "SELL")
            neutral_votes = sum(1 for s in signals if s == "NEUTRAL")
            
            total_votes = len(signals)
            if total_votes == 0:
                return {
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'strength': 'WEAK',
                    'consensus': 0.0,
                    'individual_signals': {}
                }
            
            # Calculate consensus
            if buy_votes > sell_votes and buy_votes > neutral_votes:
                consensus_signal = "BUY"
                consensus_strength = buy_votes / total_votes
            elif sell_votes > buy_votes and sell_votes > neutral_votes:
                consensus_signal = "SELL"
                consensus_strength = sell_votes / total_votes
            else:
                consensus_signal = "NEUTRAL"
                consensus_strength = neutral_votes / total_votes
            
            # Calculate weighted confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            final_confidence = consensus_strength * avg_confidence
            
            # Determine strength
            if final_confidence >= 0.7:
                strength = "STRONG"
            elif final_confidence >= 0.4:
                strength = "MODERATE"
            else:
                strength = "WEAK"
            
            return {
                'signal': consensus_signal,
                'confidence': final_confidence,
                'strength': strength,
                'consensus': consensus_strength,
                'individual_signals': {
                    'buy_votes': buy_votes,
                    'sell_votes': sell_votes,
                    'neutral_votes': neutral_votes,
                    'total_votes': total_votes
                }
            }
            
        except Exception as e:
            return {
                'signal': 'ERROR',
                'confidence': 0.0,
                'strength': 'NONE',
                'consensus': 0.0,
                'individual_signals': {},
                'error': str(e)
            }


# Validation functions for input data
def validate_ohlcv_data(data: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate OHLCV data for technical indicator calculations.
    
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    if data.empty:
        return False, "Empty OHLCV data"
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing columns: {missing_columns}"
    
    # Check for negative values
    for col in ['open', 'high', 'low', 'close']:
        if (data[col] <= 0).any():
            return False, f"Non-positive values found in {col}"
    
    if (data['volume'] < 0).any():
        return False, "Negative volume values found"
    
    # Check OHLC relationships
    if (data['high'] < data['low']).any():
        return False, "High < Low found in data"
    
    if (data['high'] < data['close']).any() or (data['high'] < data['open']).any():
        return False, "High is not the highest price"
    
    if (data['low'] > data['close']).any() or (data['low'] > data['open']).any():
        return False, "Low is not the lowest price"
    
    return True, "Valid"


def calculate_technical_indicators_with_validation(
    ohlcv_data: pd.DataFrame
) -> Tuple[Optional[Dict[str, any]], str]:
    """
    Calculate technical indicators with comprehensive validation.
    
    Parameters
    ----------
    ohlcv_data : pd.DataFrame
        OHLCV data
        
    Returns
    -------
    Tuple[Optional[Dict[str, any]], str]
        (indicators or None, status_message)
    """
    # Validate input data
    is_valid, error_msg = validate_ohlcv_data(ohlcv_data)
    if not is_valid:
        return None, f"Data validation failed: {error_msg}"
    
    try:
        engine = TechnicalIndicatorEngine()
        indicators = engine.calculate_all_indicators(ohlcv_data)
        return indicators, "Success"
    except Exception as e:
        return None, f"Technical indicator calculation failed: {str(e)}"