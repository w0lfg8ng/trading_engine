# enhanced_features_v2.py
"""
Enhanced Feature Engineering for Kraken Trading Model V2
Builds comprehensive technical features for institutional crypto trading
"""

import pandas as pd
import sqlite3
import ta  # Technical Analysis library
import numpy as np
import os
from typing import Dict, List
import logging

class EnhancedFeatureBuilder:
    def __init__(self, db_path: str = "../../data/kraken_v2.db"):
        self.db_path = db_path
        self.conn = None
        
        # Trading pairs
        self.pairs = [
            "ADAUSDT", "AVAXUSDT", "DOTUSDT", "ETHUSDT", 
            "LINKUSDT", "LTCUSDT", "SOLUSDT", "XBTUSDT", "XRPUSDT"
        ]
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('../../logs/feature_engineering.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def connect_database(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)
        self.logger.info(f"Connected to {self.db_path}")
    
    def calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate existing V1 features"""
        
        # RSI
        df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # EMAs
        df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(close=df['close'], window=26).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(close=df['close'], window=200).ema_indicator()
        
        # EMA Trend
        df['ema_trend'] = (df['ema_50'] > df['ema_200']).astype(int)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # VWAP
        df['cum_vp'] = (df['close'] * df['volume']).cumsum()
        df['cum_vol'] = df['volume'].cumsum()
        df['vwap'] = df['cum_vp'] / df['cum_vol']
        
        # Price momentum
        df['pct_change_5'] = df['close'].pct_change(periods=5)
        df['pct_change_15'] = df['close'].pct_change(periods=15)
        df['pct_change_60'] = df['close'].pct_change(periods=60)  # 1 hour
        
        # Volume features
        df['volume_change'] = df['volume'].diff()
        df['vol_avg_50'] = df['volume'].rolling(window=50, min_periods=1).mean()
        df['volume_spike'] = (df['volume'] > 2 * df['vol_avg_50']).astype(int)
        
        return df
    
    def calculate_advanced_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced momentum indicators"""
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], 
                                               close=df['close'], window=14)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], 
                                                         close=df['close'], lbp=14).williams_r()
        
        # Rate of Change (Multiple periods)
        df['roc_60'] = ta.momentum.ROCIndicator(close=df['close'], window=60).roc()  # 1 hour
        df['roc_240'] = ta.momentum.ROCIndicator(close=df['close'], window=240).roc()  # 4 hours
        df['roc_1440'] = ta.momentum.ROCIndicator(close=df['close'], window=1440).roc()  # 1 day
        
        # Commodity Channel Index
        df['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], 
                                         close=df['close'], window=20).cci()
        
        return df
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        
        # Average True Range
        df['atr_14'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], 
                                                     close=df['close'], window=14).average_true_range()
        
        # Realized Volatility (rolling standard deviation of returns)
        returns = df['close'].pct_change()
        df['realized_vol_24h'] = returns.rolling(window=1440).std() * np.sqrt(1440)  # 24h
        df['realized_vol_7d'] = returns.rolling(window=10080).std() * np.sqrt(1440)  # 7d
        
        # ATR-based volatility breakout
        df['atr_avg'] = df['atr_14'].rolling(window=20).mean()
        df['vol_breakout'] = (df['atr_14'] > 1.5 * df['atr_avg']).astype(int)
        
        # Volatility percentile (where current ATR sits historically)
        df['atr_percentile'] = df['atr_14'].rolling(window=1000).rank(pct=True)
        
        return df
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        
        # On-Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], 
                                                      volume=df['volume']).on_balance_volume()
        
        # Money Flow Index
        df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], 
                                          close=df['close'], volume=df['volume'], 
                                          window=14).money_flow_index()
        
        # Accumulation/Distribution Line
        df['ad_line'] = ta.volume.AccDistIndexIndicator(high=df['high'], low=df['low'], 
                                                       close=df['close'], 
                                                       volume=df['volume']).acc_dist_index()
        
        # Volume Rate of Change
        df['volume_roc'] = ta.momentum.ROCIndicator(close=df['volume'], window=14).roc()
        
        # Volume moving averages
        df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma_20']
        
        return df
    
    def calculate_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market structure features"""
        
        # Higher highs, higher lows detection
        df['high_20'] = df['high'].rolling(window=20).max()
        df['low_20'] = df['low'].rolling(window=20).min()
        
        df['near_high'] = (df['close'] > 0.95 * df['high_20']).astype(int)
        df['near_low'] = (df['close'] < 1.05 * df['low_20']).astype(int)
        
        # Price position in range
        df['range_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
        
        # Trend strength (based on EMA alignment)
        df['ema_alignment'] = ((df['ema_12'] > df['ema_26']) & 
                              (df['ema_26'] > df['ema_50']) & 
                              (df['ema_50'] > df['ema_200'])).astype(int)
        
        return df
    
    def calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features"""
        
        # Convert timestamp to datetime for time features
        df['datetime_obj'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Hour of day (0-23)
        df['hour'] = df['datetime_obj'].dt.hour
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['datetime_obj'].dt.dayofweek
        
        # Session indicators (rough approximations)
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['european_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
        df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        
        # Weekend effect (crypto still trades but lower volume)
        df['weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def calculate_multi_timeframe_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Calculate features from higher timeframes"""
        
        if timeframe == '1m':
            # Add 5-minute features to 1-minute data
            
            # Resample to 5-minute for higher timeframe indicators
            df_5m = df.set_index('datetime_obj').resample('5T').agg({
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Calculate 5m RSI
            df_5m['rsi_5m'] = ta.momentum.RSIIndicator(close=df_5m['close'], window=14).rsi()
            
            # Calculate 5m EMA trend
            df_5m['ema_50_5m'] = ta.trend.EMAIndicator(close=df_5m['close'], window=50).ema_indicator()
            df_5m['ema_200_5m'] = ta.trend.EMAIndicator(close=df_5m['close'], window=200).ema_indicator()
            df_5m['ema_trend_5m'] = (df_5m['ema_50_5m'] > df_5m['ema_200_5m']).astype(int)
            
            # Merge back to 1-minute data
            df = df.set_index('datetime_obj')
            df = df.merge(df_5m[['rsi_5m', 'ema_trend_5m']], 
                         left_index=True, right_index=True, how='left')
            df = df.fillna(method='ffill')  # Forward fill higher timeframe data
            df = df.reset_index()
        
        return df
    
    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up intermediate calculations and handle NaNs"""
        
        # Drop intermediate calculation columns
        cols_to_drop = ['cum_vp', 'cum_vol', 'vol_avg_50', 'atr_avg', 
                       'high_20', 'low_20', 'vol_ma_20', 'datetime_obj']
        
        for col in cols_to_drop:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        
        # Drop rows with NaNs (from indicator calculation periods)
        initial_rows = len(df)
        df.dropna(inplace=True)
        final_rows = len(df)
        
        self.logger.info(f"Dropped {initial_rows - final_rows} rows with NaNs")
        
        return df
    
    def build_features_for_pair(self, pair: str, timeframe: str):
        """Build all features for a specific pair and timeframe"""
        
        table_name = f"ohlcv_{timeframe}"
        feature_table_name = f"features_{pair.lower()}_{timeframe}"
        
        self.logger.info(f"Building enhanced features for {pair} {timeframe}...")
        
        try:
            # Load OHLCV data
            df = pd.read_sql(f"""
                SELECT * FROM {table_name} 
                WHERE pair = '{pair}' 
                ORDER BY timestamp ASC
            """, self.conn, parse_dates=['datetime'])
            
            if df.empty:
                self.logger.warning(f"No data found for {pair} {timeframe}")
                return False
            
            # Calculate all feature groups
            df = self.calculate_basic_features(df)
            df = self.calculate_advanced_momentum(df)
            df = self.calculate_volatility_features(df)
            df = self.calculate_volume_features(df)
            df = self.calculate_market_structure(df)
            df = self.calculate_time_features(df)
            df = self.calculate_multi_timeframe_features(df, timeframe)
            
            # Clean up
            df = self.clean_features(df)
            
            # Save to database
            df.to_sql(feature_table_name, self.conn, if_exists="replace", index=False)
            
            self.logger.info(f"✅ Features saved to {feature_table_name} ({len(df):,} rows)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process {pair} {timeframe}: {e}")
            return False
    
    def build_all_features(self):
        """Build features for all pairs and timeframes"""
        
        self.logger.info("Starting enhanced feature building...")
        
        results = {}
        
        for timeframe in ['1m', '5m']:
            self.logger.info(f"\n🔥 BUILDING {timeframe.upper()} FEATURES...")
            results[timeframe] = {}
            
            for pair in self.pairs:
                success = self.build_features_for_pair(pair, timeframe)
                results[timeframe][pair] = success
        
        # Summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ENHANCED FEATURE BUILDING SUMMARY")
        self.logger.info("=" * 60)
        
        for timeframe in ['1m', '5m']:
            successful = sum(1 for success in results[timeframe].values() if success)
            self.logger.info(f"{timeframe.upper()}: {successful}/9 pairs completed")
            
            for pair, success in results[timeframe].items():
                status = "✅" if success else "❌"
                self.logger.info(f"  {status} {pair}")
        
        return results
    
    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")

def main():
    """Main execution function"""
    
    print("=" * 60)
    print("ENHANCED FEATURE BUILDER V2")
    print("=" * 60)
    
    builder = EnhancedFeatureBuilder()
    
    try:
        builder.connect_database()
        results = builder.build_all_features()
        
        print("\n✅ Enhanced feature building completed!")
        
    except Exception as e:
        print(f"\n❌ Feature building failed: {e}")
        
    finally:
        builder.close_connection()

if __name__ == "__main__":
    main()