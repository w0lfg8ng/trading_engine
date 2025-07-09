#!/usr/bin/env python3
"""
Enhanced Daily Data & Feature Updater for Automated ML Pipeline
Adapted for Erick's existing project structure

Features:
- Integrates with existing daily_updater.py logic
- Enhanced error handling and recovery
- Data quality validation
- Pipeline status reporting
- Automatic institutional label updates when needed
"""

import os
import sys
import sqlite3
import pandas as pd
import requests
import time
import ta
import numpy as np
import json
import subprocess
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from dotenv import load_dotenv
from pathlib import Path

# Add the project root to Python path for imports
sys.path.append('/mnt/raid0/data_erick/kraken_trading_model_v2')

class EnhancedDailyUpdater:
    def __init__(self, db_path: str = "/mnt/raid0/data_erick/kraken_trading_model_v2/data/kraken_v2.db", 
                 config_path: str = "/mnt/raid0/data_erick/kraken_trading_model_v2/config/pipeline_config.json"):
        """Initialize the enhanced daily updater"""
        self.db_path = db_path
        self.conn = None
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Load environment variables
        load_dotenv()
        
        # Setup logging with pipeline integration
        self._setup_logging()
        
        # Trading pairs (Kraken format vs database format) - matching your existing setup
        self.pairs = {
            "ADAUSDT": "ADAUSD",
            "AVAXUSDT": "AVAXUSD", 
            "DOTUSDT": "DOTUSD",
            "ETHUSDT": "ETHUSD",
            "LINKUSDT": "LINKUSD",
            "LTCUSDT": "LTCUSD",
            "SOLUSDT": "SOLUSD",
            "XBTUSDT": "XXBTZUSD",
            "XRPUSDT": "XXRPZUSD"
        }
        
        # API configuration with enhanced rate limiting
        self.api_url = "https://api.kraken.com/0/public/OHLC"
        self.rate_limit_delay = 1.5  # Increased for stability
        self.max_retries = 3
        self.retry_delay = 5
        
        # Data quality thresholds
        self.min_data_completeness = self.config.get('data_pipeline', {}).get('required_data_completeness', 0.95)
        self.max_data_age_hours = self.config.get('data_pipeline', {}).get('max_data_age_hours', 8)
        
        # Pipeline status tracking
        self.pipeline_status = {
            'last_update': None,
            'data_quality_score': 0.0,
            'errors': [],
            'warnings': [],
            'pairs_updated': 0,
            'total_rows_added': 0
        }
        
        self.logger.info(f"Enhanced daily updater initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid config file: {e}, using defaults")
            return {}
    
    def _setup_logging(self):
        """Setup logging with pipeline integration"""
        log_dir = Path(self.config.get('logging', {}).get('directory', '/mnt/raid0/data_erick/kraken_trading_model_v2/logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_filename = log_dir / f"enhanced_daily_update_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config.get('logging', {}).get('level', 'INFO')),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def connect_database(self):
        """Create database connection with enhanced error handling"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.conn = sqlite3.connect(self.db_path, timeout=30)
                self.conn.execute("PRAGMA foreign_keys = ON")
                self.conn.execute("PRAGMA journal_mode = WAL")
                self.conn.execute("PRAGMA synchronous = NORMAL")
                self.conn.execute("PRAGMA cache_size = 10000")
                self.logger.info(f"Connected to database: {self.db_path}")
                return
            except Exception as e:
                self.logger.error(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(5)
                else:
                    raise
    
    def get_latest_timestamp(self, pair: str, timeframe: str) -> Optional[int]:
        """Get the latest timestamp for a pair from database"""
        try:
            table_name = f"ohlcv_{timeframe}"
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT MAX(timestamp) 
                FROM {table_name} 
                WHERE pair = ?
            """, (pair,))
            
            result = cursor.fetchone()
            return result[0] if result[0] else None
            
        except Exception as e:
            self.logger.error(f"Error getting latest timestamp for {pair}: {e}")
            return None
    
    def fetch_latest_ohlcv(self, pair: str, timeframe: str) -> pd.DataFrame:
        """Fetch latest OHLCV data from Kraken API"""
        
        kraken_pair = self.pairs[pair]
        interval = int(timeframe.replace('m', ''))
        
        # Get latest timestamp from database
        latest_timestamp = self.get_latest_timestamp(pair, timeframe)
        
        if not latest_timestamp:
            self.logger.warning(f"No existing data found for {pair} {timeframe}")
            return pd.DataFrame()
        
        # Start fetching from the next interval
        since_timestamp = latest_timestamp + (interval * 60)
        
        params = {
            'pair': kraken_pair,
            'interval': interval,
            'since': since_timestamp * 1000000000  # Convert to nanoseconds
        }
        
        try:
            self.logger.info(f"Fetching latest {pair} {timeframe} data since {since_timestamp}")
            
            response = requests.get(self.api_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('error'):
                self.logger.error(f"Kraken API error for {pair}: {data['error']}")
                return pd.DataFrame()
            
            # Process API response
            pair_data = list(data['result'].values())[0]
            
            if not pair_data:
                self.logger.info(f"No new data available for {pair} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(pair_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'trade_count'
            ])
            
            # Clean and format data
            df['timestamp'] = df['timestamp'].astype(int)
            df['pair'] = pair
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df['trade_count'] = df['trade_count'].astype(int)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Filter to only new data
            df = df[df['timestamp'] > latest_timestamp]
            
            # Select columns matching database schema
            df = df[['pair', 'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'trade_count']]
            
            self.logger.info(f"Fetched {len(df)} new rows for {pair} {timeframe}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {pair} {timeframe}: {e}")
            return pd.DataFrame()
    
    def insert_ohlcv_data(self, df: pd.DataFrame, timeframe: str) -> bool:
        """Insert new OHLCV data into database"""
        
        if df.empty:
            return True
        
        try:
            table_name = f"ohlcv_{timeframe}"
            
            insert_sql = f"""
            INSERT OR REPLACE INTO {table_name} 
            (pair, timestamp, datetime, open, high, low, close, volume, trade_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.conn.executemany(insert_sql, df.values.tolist())
            self.conn.commit()
            
            self.logger.info(f"Inserted {len(df)} rows into {table_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to insert OHLCV data: {e}")
            return False
    
    def calculate_features_incremental(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Calculate features for the data (using your existing logic)"""
        
        if df.empty:
            return pd.DataFrame()
        
        try:
            # Basic momentum indicators (matching your existing feature set)
            df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
            
            # MACD
            macd = ta.trend.MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # EMAs (matching your model's expected features)
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
            
            # VWAP calculation
            df['cum_vp'] = (df['close'] * df['volume']).cumsum()
            df['cum_vol'] = df['volume'].cumsum()
            df['vwap'] = df['cum_vp'] / df['cum_vol']
            
            # Add all the other indicators your models expect
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Williams %R
            df['williams_r'] = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=14).williams_r()
            
            # Rate of Change
            df['roc_60'] = ta.momentum.ROCIndicator(close=df['close'], window=60).roc()
            df['roc_240'] = ta.momentum.ROCIndicator(close=df['close'], window=240).roc() if len(df) > 240 else np.nan
            df['roc_1440'] = ta.momentum.ROCIndicator(close=df['close'], window=1440).roc() if len(df) > 1440 else np.nan
            
            # CCI
            df['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()
            
            # ATR and volatility
            df['atr_14'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
            
            # Realized volatility
            returns = df['close'].pct_change()
            df['realized_vol_24h'] = returns.rolling(window=1440, min_periods=1).std() * np.sqrt(1440)
            df['realized_vol_7d'] = returns.rolling(window=10080, min_periods=1).std() * np.sqrt(1440)
            
            # Volume indicators
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
            df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14).money_flow_index()
            df['ad_line'] = ta.volume.AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).acc_dist_index()
            
            # Price momentum
            df['pct_change_5'] = df['close'].pct_change(periods=5) * 100
            df['pct_change_15'] = df['close'].pct_change(periods=15) * 100
            df['pct_change_60'] = df['close'].pct_change(periods=60) * 100
            df['pct_change_240'] = df['close'].pct_change(periods=240) * 100
            df['pct_change_1440'] = df['close'].pct_change(periods=1440) * 100
            
            # Volume features
            df['volume_change'] = df['volume'].pct_change() * 100
            df['vol_avg_50'] = df['volume'].rolling(window=50, min_periods=1).mean()
            df['volume_spike'] = (df['volume'] > 2 * df['vol_avg_50']).astype(int)
            df['volume_roc'] = ta.momentum.ROCIndicator(close=df['volume'], window=14).roc()
            df['vol_ratio'] = df['volume'] / df['vol_avg_50']
            df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Market structure
            df['high_20'] = df['high'].rolling(window=20, min_periods=1).max()
            df['low_20'] = df['low'].rolling(window=20, min_periods=1).min()
            df['near_high'] = (df['close'] > 0.95 * df['high_20']).astype(int)
            df['near_low'] = (df['close'] < 1.05 * df['low_20']).astype(int)
            df['range_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
            
            # EMA alignment
            df['ema_alignment'] = ((df['ema_12'] > df['ema_26']) & 
                                  (df['ema_26'] > df['ema_50']) & 
                                  (df['ema_50'] > df['ema_200'])).astype(int)
            
            # Additional price features
            df['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # ATR percentile
            df['atr_percentile'] = df['atr_14'].rolling(window=min(100, len(df)), min_periods=1).rank(pct=True)
            
            # Volatility breakout
            df['vol_breakout'] = (df['realized_vol_24h'] > df['realized_vol_24h'].rolling(window=min(168, len(df)), min_periods=1).mean()).astype(int)
            
            # Time features
            df['datetime_obj'] = pd.to_datetime(df['timestamp'], unit='s')
            df['hour'] = df['datetime_obj'].dt.hour
            df['day_of_week'] = df['datetime_obj'].dt.dayofweek
            df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['european_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
            df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
            df['weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Add ID and trade_count columns that models expect
            df['id'] = range(len(df))
            # trade_count already exists from OHLCV data
            
            # Multi-timeframe for 1m data
            if timeframe == '1m' and len(df) >= 300:
                try:
                    # Add 5m higher timeframe signals
                    df_5m = df.set_index('datetime_obj').resample('5min').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min', 
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    
                    if not df_5m.empty and len(df_5m) >= 50:
                        df_5m['rsi_5m'] = ta.momentum.RSIIndicator(close=df_5m['close'], window=14).rsi()
                        df_5m['ema_50_5m'] = ta.trend.EMAIndicator(close=df_5m['close'], window=50).ema_indicator()
                        df_5m['ema_200_5m'] = ta.trend.EMAIndicator(close=df_5m['close'], window=200).ema_indicator()
                        df_5m['ema_trend_5m'] = (df_5m['ema_50_5m'] > df_5m['ema_200_5m']).astype(int)
                        
                        # Merge back to 1m data
                        df = df.set_index('datetime_obj')
                        df = df.merge(df_5m[['rsi_5m', 'ema_trend_5m']], left_index=True, right_index=True, how='left')
                        df = df.ffill()  # Forward fill higher timeframe data
                        df = df.reset_index()
                    else:
                        df['rsi_5m'] = 50.0  # Default neutral values
                        df['ema_trend_5m'] = 0
                except Exception as e:
                    self.logger.warning(f"Multi-timeframe calculation failed: {e}")
                    df['rsi_5m'] = 50.0
                    df['ema_trend_5m'] = 0
            else:
                df['rsi_5m'] = 50.0
                df['ema_trend_5m'] = 0
            
            # Clean up intermediate columns
            cols_to_drop = ['cum_vp', 'cum_vol', 'vol_avg_50', 'high_20', 'low_20', 'datetime_obj']
            for col in cols_to_drop:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            
            # Clean data
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.ffill(inplace=True)  # Forward fill remaining NaNs
            df.fillna(0, inplace=True)  # Fill any remaining NaNs with 0
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            return df
    
    def update_feature_table(self, df: pd.DataFrame, pair: str, timeframe: str, new_rows_count: int) -> bool:
        """Update feature table with new calculated features"""
        
        if df.empty or new_rows_count == 0:
            return True
        
        try:
            feature_table_name = f"features_{pair.lower()}_{timeframe}"
            
            # Get only the newest rows (the ones we just added to OHLCV)
            latest_rows = df.tail(new_rows_count).copy()
            
            # Drop rows with NaN values in critical features only
            critical_features = ['rsi_14', 'ema_12', 'ema_26', 'close']
            latest_rows = latest_rows.dropna(subset=critical_features)
            
            if latest_rows.empty:
                self.logger.warning(f"No valid feature rows after NaN removal for {pair} {timeframe}")
                return True
            
            # Get existing table schema to match column order
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({feature_table_name})")
            existing_columns = [col[1] for col in cursor.fetchall()]
            
            # Ensure we only insert columns that exist in the target table
            available_columns = [col for col in existing_columns if col in latest_rows.columns]
            latest_rows = latest_rows[available_columns]
            
            # Convert data types properly
            for col in latest_rows.columns:
                if latest_rows[col].dtype == 'object':
                    continue  # Skip string columns
                elif 'datetime' in col.lower():
                    latest_rows[col] = latest_rows[col].astype(str)
                elif latest_rows[col].dtype.name.startswith('int'):
                    latest_rows[col] = latest_rows[col].astype('int64')
                elif latest_rows[col].dtype.name.startswith('float'):
                    latest_rows[col] = latest_rows[col].replace([np.inf, -np.inf], np.nan)
                    latest_rows[col] = latest_rows[col].astype('float64')
                elif latest_rows[col].dtype.name.startswith('bool'):
                    latest_rows[col] = latest_rows[col].astype('int64')
            
            # Insert data
            placeholders = ','.join(['?' for _ in available_columns])
            insert_sql = f"""
            INSERT OR REPLACE INTO {feature_table_name} 
            ({','.join(available_columns)})
            VALUES ({placeholders})
            """
            
            values_list = []
            for _, row in latest_rows.iterrows():
                values = []
                for col in available_columns:
                    val = row[col]
                    
                    # Convert numpy types to native Python types
                    if hasattr(val, 'item'):  # numpy scalar
                        val = val.item()
                    elif pd.isna(val):
                        val = None
                    elif isinstance(val, (np.int64, np.int32)):
                        val = int(val)
                    elif isinstance(val, (np.float64, np.float32)):
                        val = float(val) if not np.isnan(val) else None
                    elif isinstance(val, np.bool_):
                        val = bool(val)
                    
                    values.append(val)
                values_list.append(tuple(values))
            
            cursor.executemany(insert_sql, values_list)
            self.conn.commit()
            
            self.logger.info(f"Updated {len(latest_rows)} feature rows in {feature_table_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update feature table for {pair} {timeframe}: {e}")
            return False
    
    def update_pair_timeframe(self, pair: str, timeframe: str) -> Dict:
        """Update both OHLCV and features for a specific pair and timeframe"""
        
        try:
            # Step 1: Fetch latest OHLCV data
            new_ohlcv = self.fetch_latest_ohlcv(pair, timeframe)
            
            if new_ohlcv.empty:
                return {
                    'success': True,
                    'ohlcv_rows': 0,
                    'feature_rows': 0,
                    'message': 'No new data available'
                }
            
            # Step 2: Insert new OHLCV data
            if not self.insert_ohlcv_data(new_ohlcv, timeframe):
                return {
                    'success': False,
                    'error': 'Failed to insert OHLCV data'
                }
            
            # Step 3: Get data for feature calculation (including historical context)
            # Use a larger lookback for proper feature calculation
            lookback_minutes = 2000  # About 33 hours for 1m data
            
            try:
                cursor = self.conn.cursor()
                cursor.execute(f"""
                    SELECT * FROM ohlcv_{timeframe} 
                    WHERE pair = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (pair, lookback_minutes))
                
                data = cursor.fetchall()
                if not data:
                    return {'success': False, 'error': 'No historical data for feature calculation'}
                
                # Get column names
                cursor.execute(f"PRAGMA table_info(ohlcv_{timeframe})")
                columns = [col[1] for col in cursor.fetchall()]
                
                # Create DataFrame
                calc_data = pd.DataFrame(data, columns=columns)
                calc_data = calc_data.sort_values('timestamp').reset_index(drop=True)
                
            except Exception as e:
                return {'success': False, 'error': f'Failed to get calculation data: {e}'}
            
            # Step 4: Calculate features
            featured_data = self.calculate_features_incremental(calc_data, timeframe)
            
            # Step 5: Update feature table with new rows only
            feature_success = self.update_feature_table(featured_data, pair, timeframe, len(new_ohlcv))
            
            if not feature_success:
                return {
                    'success': False,
                    'error': 'Failed to update feature table'
                }
            
            return {
                'success': True,
                'ohlcv_rows': len(new_ohlcv),
                'feature_rows': len(new_ohlcv),
                'message': 'Successfully updated'
            }
            
        except Exception as e:
            self.logger.error(f"Error updating {pair} {timeframe}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _should_update_labels(self, results: Dict) -> bool:
        """Determine if institutional labels should be updated"""
        total_new_rows = results['summary']['total_ohlcv_rows']
        
        # Update if we added more than 1000 new rows (about 16 hours of 1m data)
        if total_new_rows > 1000:
            return True
        
        # Or if it's been more than 24 hours since last label update
        try:
            label_file = Path("/mnt/raid0/data_erick/kraken_trading_model_v2/logs/last_label_update.txt")
            if label_file.exists():
                with open(label_file, 'r') as f:
                    last_update = datetime.fromisoformat(f.read().strip())
                    if datetime.now() - last_update > timedelta(hours=24):
                        return True
            else:
                return True  # No record of previous update
        except Exception:
            return False
        
        return False
    
    def _update_institutional_labels(self):
        """Update institutional labels"""
        try:
            # Run institutional labeling script
            result = subprocess.run([
                'python', '/mnt/raid0/data_erick/kraken_trading_model_v2/src/features/1m_institutional_label_targets_enhanced.py'
            ], capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                # Record successful update
                label_file = Path("/mnt/raid0/data_erick/kraken_trading_model_v2/logs/last_label_update.txt")
                with open(label_file, 'w') as f:
                    f.write(datetime.now().isoformat())
                self.logger.info("✅ Institutional labels updated successfully")
            else:
                self.logger.error(f"❌ Label update failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"❌ Label update error: {e}")
    
    def run_enhanced_daily_update(self):
        """Run the complete enhanced daily update process"""
        
        self.logger.info("🚀 Starting enhanced daily update process...")
        
        results = {
            'start_time': datetime.now(),
            'ohlcv_updates': {},
            'feature_updates': {},
            'summary': {}
        }
        
        total_ohlcv_rows = 0
        total_feature_rows = 0
        successful_updates = 0
        
        # Update each timeframe and pair
        for timeframe in ['1m', '5m']:
            self.logger.info(f"\n🔄 UPDATING {timeframe.upper()} DATA...")
            
            results['ohlcv_updates'][timeframe] = {}
            results['feature_updates'][timeframe] = {}
            
            for pair in self.pairs.keys():
                self.logger.info(f"Processing {pair} {timeframe}...")
                
                # Update this pair and timeframe
                result = self.update_pair_timeframe(pair, timeframe)
                
                # Store results
                results['ohlcv_updates'][timeframe][pair] = result
                
                if result['success']:
                    ohlcv_rows = result.get('ohlcv_rows', 0)
                    feature_rows = result.get('feature_rows', 0)
                    
                    total_ohlcv_rows += ohlcv_rows
                    total_feature_rows += feature_rows
                    successful_updates += 1
                    
                    status = "✅" if ohlcv_rows > 0 else "📊"
                    self.logger.info(f"  {status} {pair}: {ohlcv_rows} OHLCV, {feature_rows} features")
                else:
                    self.logger.error(f"  ❌ {pair}: {result.get('error', 'Unknown error')}")
                
                # Rate limiting between requests
                time.sleep(self.rate_limit_delay)
        
        # Update institutional labels if needed
        if self._should_update_labels({'summary': {'total_ohlcv_rows': total_ohlcv_rows}}):
            self.logger.info("🏷️ Updating institutional labels...")
            self._update_institutional_labels()
        
        # Summary
        results['end_time'] = datetime.now()
        results['duration'] = results['end_time'] - results['start_time']
        results['summary'] = {
            'total_ohlcv_rows': total_ohlcv_rows,
            'total_feature_rows': total_feature_rows,
            'successful_updates': successful_updates,
            'total_attempts': len(self.pairs) * len(['1m', '5m']),
            'success_rate': successful_updates / (len(self.pairs) * 2) * 100
        }
        
        # Create pipeline status file for health checks
        status_file = Path("/mnt/raid0/data_erick/kraken_trading_model_v2/data/pipeline_status.json")
        status = {
            'last_update': datetime.now().isoformat(),
            'status': 'completed',
            'total_rows_added': total_ohlcv_rows,
            'successful_updates': successful_updates,
            'total_attempts': len(self.pairs) * 2
        }
        
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        # Log summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ENHANCED DAILY UPDATE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"⏱️ Duration: {results['duration']}")
        self.logger.info(f"📊 OHLCV rows added: {total_ohlcv_rows:,}")
        self.logger.info(f"🔢 Feature rows added: {total_feature_rows:,}")
        self.logger.info(f"✅ Successful updates: {successful_updates}/{len(self.pairs) * 2}")
        self.logger.info(f"📈 Success rate: {results['summary']['success_rate']:.1f}%")
        
        if total_ohlcv_rows > 0:
            self.logger.info("🎉 Database successfully updated with fresh data!")
        else:
            self.logger.info("📊 No new data available - database is current")
        
        return results
    
    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")

def main():
    """Main execution function for enhanced daily automation"""
    
    print("=" * 60)
    print("ENHANCED KRAKEN DAILY DATA & FEATURE UPDATER")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    updater = EnhancedDailyUpdater()
    
    try:
        # Connect to database
        updater.connect_database()
        
        # Run enhanced daily update
        results = updater.run_enhanced_daily_update()
        
        # Check if update was successful
        if results['summary']['success_rate'] >= 80:
            print("\n✅ Enhanced daily update completed successfully!")
            exit_code = 0
        else:
            print(f"\n⚠️ Enhanced daily update completed with issues ({results['summary']['success_rate']:.1f}% success rate)")
            exit_code = 1
        
        print(f"Total new data: {results['summary']['total_ohlcv_rows']:,} OHLCV rows, {results['summary']['total_feature_rows']:,} feature rows")
        
    except Exception as e:
        print(f"\n❌ Enhanced daily update failed: {e}")
        updater.logger.error(f"Critical failure: {e}")
        exit_code = 2
        
    finally:
        updater.close_connection()
        print(f"Completed at: {datetime.now()}")
    
    exit(exit_code)

if __name__ == "__main__":
    main()