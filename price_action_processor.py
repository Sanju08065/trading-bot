import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CandlePattern(Enum):
    DOJI = "doji"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    PIN_BAR_BULLISH = "pin_bar_bullish"
    PIN_BAR_BEARISH = "pin_bar_bearish"
    INSIDE_BAR = "inside_bar"
    OUTSIDE_BAR = "outside_bar"

class MarketStructure(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    CONSOLIDATION = "consolidation"
    BREAK_OF_STRUCTURE_BULLISH = "bos_bullish"
    BREAK_OF_STRUCTURE_BEARISH = "bos_bearish"

@dataclass
class CandleFeatures:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    body_size: float
    upper_wick: float
    lower_wick: float
    body_ratio: float  # Body size relative to total range
    is_bullish: bool
    patterns: List[CandlePattern]
    structure: MarketStructure
    liquidity_touched: bool
    order_block_formed: bool
    order_block_retested: bool
    break_of_structure: bool

class PriceActionProcessor:
    def __init__(self, window_size: int = 20):
        """
        Initialize the price action processor.
        
        Args:
            window_size: Number of candles to consider for pattern detection and structure analysis
        """
        self.window_size = window_size
        self.min_body_ratio = 0.1  # Minimum body size ratio to be considered a valid candle
        self.min_wick_ratio = 0.3  # Minimum wick size ratio for pin bars
        
    def _calculate_candle_features(self, candle: Dict) -> CandleFeatures:
        """Calculate basic features for a single candle."""
        open_price = candle['open']
        high = candle['high']
        low = candle['low']
        close = candle['close']
        
        # Basic candle properties
        body_size = abs(close - open_price)
        total_range = high - low
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        body_ratio = body_size / total_range if total_range > 0 else 0
        is_bullish = close > open_price
        
        # Initialize patterns list
        patterns = []
        
        # Detect basic patterns
        if body_ratio < 0.1:  # Doji
            patterns.append(CandlePattern.DOJI)
            
        # More pattern detection will be added in sequence processing
        
        return CandleFeatures(
            timestamp=candle['timestamp'],
            open=open_price,
            high=high,
            low=low,
            close=close,
            body_size=body_size,
            upper_wick=upper_wick,
            lower_wick=lower_wick,
            body_ratio=body_ratio,
            is_bullish=is_bullish,
            patterns=patterns,
            structure=MarketStructure.CONSOLIDATION,  # Will be updated in sequence processing
            liquidity_touched=False,  # Will be updated in sequence processing
            order_block_formed=False,  # Will be updated in sequence processing
            order_block_retested=False,  # Will be updated in sequence processing
            break_of_structure=False  # Will be updated in sequence processing
        )
    
    def _detect_sequence_patterns(self, candles: List[CandleFeatures]) -> List[CandleFeatures]:
        """Detect patterns that require multiple candles."""
        if len(candles) < 2:
            return candles
            
        for i in range(1, len(candles)):
            current = candles[i]
            previous = candles[i-1]
            
            # Detect engulfing patterns
            if (current.body_size > previous.body_size * 1.5 and  # Current body is significantly larger
                current.is_bullish != previous.is_bullish):  # Opposite colors
                if current.is_bullish:
                    current.patterns.append(CandlePattern.ENGULFING_BULLISH)
                else:
                    current.patterns.append(CandlePattern.ENGULFING_BEARISH)
            
            # Detect pin bars
            if (current.body_ratio < 0.3 and  # Small body
                (current.upper_wick > current.body_size * 2 or  # Long upper wick
                 current.lower_wick > current.body_size * 2)):  # Long lower wick
                if current.upper_wick > current.lower_wick:
                    current.patterns.append(CandlePattern.PIN_BAR_BEARISH)
                else:
                    current.patterns.append(CandlePattern.PIN_BAR_BULLISH)
            
            # Detect inside/outside bars
            if (current.high <= previous.high and current.low >= previous.low):
                current.patterns.append(CandlePattern.INSIDE_BAR)
            elif (current.high > previous.high and current.low < previous.low):
                current.patterns.append(CandlePattern.OUTSIDE_BAR)
        
        return candles
    
    def _analyze_market_structure(self, candles: List[CandleFeatures]) -> List[CandleFeatures]:
        """Analyze market structure and detect breaks of structure."""
        if len(candles) < self.window_size:
            return candles
            
        # Calculate swing highs and lows
        for i in range(2, len(candles)-2):
            # Check for swing high
            if (candles[i].high > candles[i-1].high and 
                candles[i].high > candles[i-2].high and
                candles[i].high > candles[i+1].high and
                candles[i].high > candles[i+2].high):
                candles[i].structure = MarketStructure.BREAK_OF_STRUCTURE_BULLISH
                
            # Check for swing low
            if (candles[i].low < candles[i-1].low and 
                candles[i].low < candles[i-2].low and
                candles[i].low < candles[i+1].low and
                candles[i].low < candles[i+2].low):
                candles[i].structure = MarketStructure.BREAK_OF_STRUCTURE_BEARISH
        
        return candles
    
    def _detect_liquidity_and_orderblocks(self, candles: List[CandleFeatures]) -> List[CandleFeatures]:
        """Detect liquidity sweeps and order blocks."""
        if len(candles) < 3:
            return candles
            
        for i in range(2, len(candles)):
            # Detect liquidity sweeps (failed breakouts)
            if (candles[i-2].high < candles[i-1].high and  # Higher high
                candles[i-1].high > candles[i].high and  # Failed to continue
                candles[i].close < candles[i-1].open):  # Bearish close
                candles[i].liquidity_touched = True
                
            # Detect order blocks
            if (candles[i-1].is_bullish and  # Bullish candle
                candles[i].low < candles[i-1].low and  # Retested the low
                candles[i].close > candles[i-1].close):  # Closed above
                candles[i].order_block_retested = True
                candles[i-1].order_block_formed = True
        
        return candles
    
    def process_candles(self, raw_candles: List[Dict]) -> List[CandleFeatures]:
        """
        Process a sequence of raw candles into enhanced features.
        
        Args:
            raw_candles: List of dictionaries containing OHLC data
            
        Returns:
            List of CandleFeatures objects with enhanced analysis
        """
        # Convert raw candles to CandleFeatures
        processed_candles = [self._calculate_candle_features(c) for c in raw_candles]
        
        # Apply sequence-based analysis
        processed_candles = self._detect_sequence_patterns(processed_candles)
        processed_candles = self._analyze_market_structure(processed_candles)
        processed_candles = self._detect_liquidity_and_orderblocks(processed_candles)
        
        return processed_candles
    
    def prepare_model_input(self, processed_candles: List[CandleFeatures]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare the processed candles for model input.
        
        Returns:
            Tuple of (features, labels) where:
            - features: numpy array of shape (n_samples, window_size, n_features)
            - labels: numpy array of shape (n_samples, 3) for [UP, DOWN, NEUTRAL] probabilities
        """
        # Convert to DataFrame for easier processing
        df = pd.DataFrame([{
            'timestamp': c.timestamp,
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'body_size': c.body_size,
            'upper_wick': c.upper_wick,
            'lower_wick': c.lower_wick,
            'body_ratio': c.body_ratio,
            'is_bullish': int(c.is_bullish),
            'has_doji': int(CandlePattern.DOJI in c.patterns),
            'has_engulfing_bullish': int(CandlePattern.ENGULFING_BULLISH in c.patterns),
            'has_engulfing_bearish': int(CandlePattern.ENGULFING_BEARISH in c.patterns),
            'has_pin_bar_bullish': int(CandlePattern.PIN_BAR_BULLISH in c.patterns),
            'has_pin_bar_bearish': int(CandlePattern.PIN_BAR_BEARISH in c.patterns),
            'has_inside_bar': int(CandlePattern.INSIDE_BAR in c.patterns),
            'has_outside_bar': int(CandlePattern.OUTSIDE_BAR in c.patterns),
            'is_uptrend': int(c.structure == MarketStructure.UPTREND),
            'is_downtrend': int(c.structure == MarketStructure.DOWNTREND),
            'is_bos_bullish': int(c.structure == MarketStructure.BREAK_OF_STRUCTURE_BULLISH),
            'is_bos_bearish': int(c.structure == MarketStructure.BREAK_OF_STRUCTURE_BEARISH),
            'liquidity_touched': int(c.liquidity_touched),
            'order_block_formed': int(c.order_block_formed),
            'order_block_retested': int(c.order_block_retested),
            'break_of_structure': int(c.break_of_structure)
        } for c in processed_candles])
        
        # Create sliding windows
        n_samples = len(df) - self.window_size
        n_features = len(df.columns) - 1  # Exclude timestamp
        
        X = np.zeros((n_samples, self.window_size, n_features))
        y = np.zeros((n_samples, 3))  # [UP, DOWN, NEUTRAL]
        
        for i in range(n_samples):
            # Get window of features
            window = df.iloc[i:i+self.window_size]
            X[i] = window.drop('timestamp', axis=1).values
            
            # Calculate label (next candle's direction)
            if i + self.window_size < len(df):
                next_close = df.iloc[i+self.window_size]['close']
                current_close = df.iloc[i+self.window_size-1]['close']
                price_change = next_close - current_close
                
                # Define thresholds for direction
                threshold = 0.0001  # 0.01% change threshold
                if price_change > threshold:
                    y[i] = [1, 0, 0]  # UP
                elif price_change < -threshold:
                    y[i] = [0, 1, 0]  # DOWN
                else:
                    y[i] = [0, 0, 1]  # NEUTRAL
        
        return X, y 