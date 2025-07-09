# regime_detector_core.py
"""
Core regime detection functions for cryptocurrency trading
Detects 9 market regimes based on trend and volatility

Regimes:
- Bull Market: High/Normal/Low Volatility
- Range Market: High/Normal/Low Volatility  
- Bear Market: High/Normal/Low Volatility
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sqlite3

class RegimeDetector:
    def __init__(self, 
                 trend_window_days: int = 60,
                 volatility_window_days: int = 30,
                 high_vol_threshold: float = 0.035,
                 low_vol_threshold: float = 0.015,
                 bull_trend_threshold: float = 0.20,
                 bear_trend_threshold: float = -0.20):
        """
        Initialize regime detector with configurable parameters
        
        Args:
            trend_window_days: Days to look back for trend calculation
            volatility_window_days: Days to look back for volatility calculation
            high_vol_threshold: Daily volatility threshold for high volatility regime
            low_vol_threshold: Daily volatility threshold for low volatility regime
            bull_trend_threshold: Price change threshold for bull market
            bear_trend_threshold: Price change threshold for bear market
        """
        self.trend_window_days = trend_window_days
        self.volatility_window_days = volatility_window_days
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.bull_trend_threshold = bull_trend_threshold
        self.bear_trend_threshold = bear_trend_threshold
        
        # Convert to minutes for 1-minute data
        self.trend_window_minutes = trend_window_days * 24 * 60
        self.vol_window_minutes = volatility_window_days * 24 * 60
        
        # Define all possible regimes
        self.all_regimes = [
            'bull_high_vol', 'bull_normal_vol', 'bull_low_vol',
            'range_high_vol', 'range_normal_vol', 'range_low_vol',
            'bear_high_vol', 'bear_normal_vol', 'bear_low_vol'
        ]
    
    def detect_current_regime(self, df: pd.DataFrame, price_col: str = 'close') -> str:
        """
        Detect the current market regime based on the latest data
        
        Args:
            df: DataFrame with OHLCV data (sorted by timestamp)
            price_col: Name of the price column to use
            
        Returns:
            String representing the current market regime
        """
        if len(df) < max(self.trend_window_minutes, self.vol_window_minutes):
            return 'range_normal_vol'  # Default for insufficient data
        
        # Get recent data for calculations
        recent_df = df.tail(max(self.trend_window_minutes, self.vol_window_minutes)).copy()
        
        # Calculate trend (price change over trend window)
        if len(recent_df) >= self.trend_window_minutes:
            trend_start_price = recent_df[price_col].iloc[-self.trend_window_minutes]
            current_price = recent_df[price_col].iloc[-1]
            trend = (current_price - trend_start_price) / trend_start_price
        else:
            trend = 0.0
        
        # Calculate volatility (rolling standard deviation)
        if len(recent_df) >= self.vol_window_minutes:
            vol_window_df = recent_df.tail(self.vol_window_minutes)
            returns = vol_window_df[price_col].pct_change().dropna()
            volatility = returns.std() * np.sqrt(1440)  # Annualized daily volatility
        else:
            volatility = 0.02  # Default volatility
        
        # Determine trend type
        if trend >= self.bull_trend_threshold:
            trend_type = 'bull'
        elif trend <= self.bear_trend_threshold:
            trend_type = 'bear'
        else:
            trend_type = 'range'
        
        # Determine volatility type
        if volatility >= self.high_vol_threshold:
            vol_type = 'high_vol'
        elif volatility <= self.low_vol_threshold:
            vol_type = 'low_vol'
        else:
            vol_type = 'normal_vol'
        
        return f"{trend_type}_{vol_type}"
    
    def detect_regime_history(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        Detect market regime for the entire history in the dataframe
        
        Args:
            df: DataFrame with OHLCV data (sorted by timestamp)
            price_col: Name of the price column to use
            
        Returns:
            DataFrame with additional 'market_regime' column
        """
        df = df.copy()
        df.reset_index(drop=True, inplace=True)
        
        # Initialize regime column
        df['market_regime'] = 'range_normal_vol'
        
        # Calculate rolling trend (price change over trend window)
        df['price_change_trend'] = df[price_col].pct_change(periods=self.trend_window_minutes)
        
        # Calculate rolling volatility
        df['returns'] = df[price_col].pct_change()
        df['volatility_rolling'] = df['returns'].rolling(
            window=self.vol_window_minutes, min_periods=1
        ).std() * np.sqrt(1440)  # Daily volatility
        
        # Apply regime detection to each row
        for i in range(len(df)):
            trend = df.loc[i, 'price_change_trend']
            volatility = df.loc[i, 'volatility_rolling']
            
            # Skip if data is insufficient
            if pd.isna(trend) or pd.isna(volatility):
                continue
            
            # Determine trend type
            if trend >= self.bull_trend_threshold:
                trend_type = 'bull'
            elif trend <= self.bear_trend_threshold:
                trend_type = 'bear'
            else:
                trend_type = 'range'
            
            # Determine volatility type
            if volatility >= self.high_vol_threshold:
                vol_type = 'high_vol'
            elif volatility <= self.low_vol_threshold:
                vol_type = 'low_vol'
            else:
                vol_type = 'normal_vol'
            
            # Set regime
            df.loc[i, 'market_regime'] = f"{trend_type}_{vol_type}"
        
        # Clean up temporary columns
        df.drop(['price_change_trend', 'returns', 'volatility_rolling'], axis=1, inplace=True)
        
        return df
    
    def get_regime_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about regime distribution in the data
        
        Args:
            df: DataFrame with 'market_regime' column
            
        Returns:
            Dictionary with regime statistics
        """
        if 'market_regime' not in df.columns:
            return {}
        
        regime_counts = df['market_regime'].value_counts()
        total_count = len(df)
        
        stats = {
            'total_periods': total_count,
            'regime_counts': regime_counts.to_dict(),
            'regime_percentages': (regime_counts / total_count * 100).to_dict(),
            'most_common_regime': regime_counts.index[0] if len(regime_counts) > 0 else None,
            'regime_transitions': self._count_regime_transitions(df)
        }
        
        return stats
    
    def _count_regime_transitions(self, df: pd.DataFrame) -> int:
        """Count the number of regime transitions in the data"""
        if 'market_regime' not in df.columns or len(df) < 2:
            return 0
        
        transitions = 0
        for i in range(1, len(df)):
            if df.iloc[i]['market_regime'] != df.iloc[i-1]['market_regime']:
                transitions += 1
        
        return transitions
    
    def analyze_regime_performance(self, df: pd.DataFrame, return_col: str = 'net_return_after_costs') -> Dict:
        """
        Analyze trading performance by regime
        
        Args:
            df: DataFrame with 'market_regime' and return columns
            return_col: Name of the return column to analyze
            
        Returns:
            Dictionary with performance statistics by regime
        """
        if 'market_regime' not in df.columns or return_col not in df.columns:
            return {}
        
        performance = {}
        
        for regime in df['market_regime'].unique():
            regime_data = df[df['market_regime'] == regime]
            
            if len(regime_data) > 0:
                returns = regime_data[return_col].dropna()
                
                performance[regime] = {
                    'count': len(regime_data),
                    'mean_return': returns.mean() if len(returns) > 0 else 0,
                    'median_return': returns.median() if len(returns) > 0 else 0,
                    'std_return': returns.std() if len(returns) > 0 else 0,
                    'win_rate': (returns > 0).mean() if len(returns) > 0 else 0,
                    'max_return': returns.max() if len(returns) > 0 else 0,
                    'min_return': returns.min() if len(returns) > 0 else 0
                }
        
        return performance

def load_and_analyze_regime_data(db_file: str, pair: str, table_suffix: str = "_1m_institutional") -> Dict:
    """
    Load data for a pair and analyze regime characteristics
    
    Args:
        db_file: Path to the database file
        pair: Trading pair (e.g., 'XBTUSDT')
        table_suffix: Suffix for the table name
        
    Returns:
        Dictionary with regime analysis results
    """
    # Initialize regime detector
    detector = RegimeDetector()
    
    # Connect to database
    conn = sqlite3.connect(db_file)
    
    try:
        # Load data
        table_name = f"features_{pair}{table_suffix}"
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=["timestamp"])
        df.sort_values("timestamp", inplace=True)
        
        # Detect regimes
        df_with_regimes = detector.detect_regime_history(df)
        
        # Get statistics
        regime_stats = detector.get_regime_statistics(df_with_regimes)
        
        # Analyze performance if return data exists
        performance_stats = {}
        if 'net_return_after_costs' in df.columns:
            performance_stats = detector.analyze_regime_performance(df_with_regimes)
        
        return {
            'pair': pair,
            'total_periods': len(df_with_regimes),
            'regime_statistics': regime_stats,
            'performance_by_regime': performance_stats,
            'data_range': {
                'start': df_with_regimes['timestamp'].min(),
                'end': df_with_regimes['timestamp'].max()
            }
        }
        
    except Exception as e:
        return {'error': str(e), 'pair': pair}
    
    finally:
        conn.close()

if __name__ == "__main__":
    # FIXED: Correct database path and all 9 pairs
    DB_FILE = "../data/kraken_v2.db"  # Fixed path from src/ directory
    PAIRS = [
        "XBTUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT",
        "LTCUSDT", "XRPUSDT", "LINKUSDT", "AVAXUSDT"
    ]  # All 9 pairs
    
    print("🔍 REGIME ANALYSIS FOR ALL 9 PAIRS")
    print("=" * 50)
    
    for pair in PAIRS:
        print(f"\n📊 Analyzing {pair}...")
        result = load_and_analyze_regime_data(DB_FILE, pair)
        
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
            continue
        
        stats = result['regime_statistics']
        print(f"  📈 Total periods: {stats['total_periods']:,}")
        print(f"  🔄 Regime transitions: {stats['regime_transitions']:,}")
        print(f"  📊 Most common regime: {stats['most_common_regime']}")
        
        print(f"  📋 Regime distribution:")
        for regime, percentage in sorted(stats['regime_percentages'].items()):
            print(f"    {regime}: {percentage:.1f}%")