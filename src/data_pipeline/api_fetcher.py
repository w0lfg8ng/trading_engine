#!/usr/bin/env python3
"""
API Fetcher for Kraken Trading Model V2
Fetches missing OHLCV data from Kraken API to bridge the gap from CSV data.
"""

import os
import sqlite3
import pandas as pd
import requests
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../../logs/api_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KrakenAPIFetcher:
    def __init__(self, db_path: str = "../../data/kraken_v2.db"):
        """Initialize the API fetcher with database connection."""
        self.db_path = db_path
        self.conn = None
        
        # Load environment variables
        load_dotenv()
        
        # Trading pairs (Kraken format vs database format)
        self.pairs = {
            "ADAUSDT": "ADAUSD",    # Kraken uses different naming
            "AVAXUSDT": "AVAXUSD",
            "DOTUSDT": "DOTUSD", 
            "ETHUSDT": "ETHUSD",
            "LINKUSDT": "LINKUSD",
            "LTCUSDT": "LTCUSD",
            "SOLUSDT": "SOLUSD",
            "XBTUSDT": "XXBTZUSD",  # Bitcoin has special naming
            "XRPUSDT": "XXRPZUSD"   # XRP has special naming
        }
        
        # API configuration
        self.api_url = "https://api.kraken.com/0/public/OHLC"
        self.rate_limit_delay = 1.0  # 1 second between requests
        
        # Date range for catch-up (from end of CSV data to present)
        self.start_date = "2025-03-31"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"API Fetcher initialized for date range: {self.start_date} to {self.end_date}")
        
    def connect_database(self):
        """Create database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA journal_mode = WAL")
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def get_latest_timestamp(self, pair: str, timeframe: str) -> Optional[int]:
        """Get the latest timestamp for a pair from database."""
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
            logger.error(f"Error getting latest timestamp for {pair}: {e}")
            return None
    
    def timestamp_to_kraken_since(self, timestamp: int) -> int:
        """Convert Unix timestamp to Kraken 'since' parameter (nanoseconds)."""
        return timestamp * 1000000000
    
    def fetch_ohlcv_data(self, kraken_pair: str, interval: int, since: Optional[int] = None) -> Dict:
        """Fetch OHLCV data from Kraken API."""
        
        params = {
            'pair': kraken_pair,
            'interval': interval
        }
        
        if since:
            params['since'] = self.timestamp_to_kraken_since(since)
        
        try:
            logger.info(f"Fetching {kraken_pair} (interval={interval}min) since={since}")
            
            response = requests.get(self.api_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('error'):
                logger.error(f"Kraken API error: {data['error']}")
                return {'success': False, 'error': data['error']}
            
            # Kraken returns data in format: {pair_name: [[timestamp, open, high, low, close, vwap, volume, count], ...]}
            pair_data = list(data['result'].values())[0]  # Get the OHLCV array
            last_timestamp = data['result']['last']  # Get last timestamp for pagination
            
            return {
                'success': True,
                'data': pair_data,
                'last': last_timestamp,
                'count': len(pair_data)
            }
            
        except requests.RequestException as e:
            logger.error(f"Request failed for {kraken_pair}: {e}")
            return {'success': False, 'error': str(e)}
        except Exception as e:
            logger.error(f"Unexpected error fetching {kraken_pair}: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_api_data(self, api_data: List, pair: str, timeframe: str) -> pd.DataFrame:
        """Convert Kraken API data to DataFrame matching database schema."""
        
        if not api_data:
            return pd.DataFrame()
        
        # Convert API data to DataFrame
        # Kraken format: [timestamp, open, high, low, close, vwap, volume, count]
        df = pd.DataFrame(api_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'trade_count'
        ])
        
        # Convert timestamp from float to int
        df['timestamp'] = df['timestamp'].astype(int)
        
        # Convert prices to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Convert trade_count to int
        df['trade_count'] = df['trade_count'].astype(int)
        
        # Create datetime column
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Add pair column
        df['pair'] = pair
        
        # Select columns to match database schema (drop vwap)
        df = df[['pair', 'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'trade_count']]
        
        logger.info(f"Processed {len(df)} rows for {pair} {timeframe}")
        return df
    
    def insert_ohlcv_data(self, df: pd.DataFrame, timeframe: str) -> bool:
        """Insert OHLCV data into database."""
        
        if df.empty:
            logger.warning(f"No data to insert for {timeframe}")
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
            
            logger.info(f"Inserted {len(df)} rows into {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert data into {timeframe}: {e}")
            return False
    
    def fetch_pair_timeframe(self, pair: str, timeframe: str) -> Dict:
        """Fetch missing data for a specific pair and timeframe."""
        
        kraken_pair = self.pairs[pair]
        interval = int(timeframe.replace('m', ''))  # Convert '1m' to 1, '5m' to 5
        
        # Get latest timestamp from database
        latest_timestamp = self.get_latest_timestamp(pair, timeframe)
        
        if not latest_timestamp:
            logger.warning(f"No existing data found for {pair} {timeframe}")
            return {'success': False, 'error': 'No existing data'}
        
        # Start fetching from the next timestamp
        since_timestamp = latest_timestamp + (interval * 60)  # Add interval in seconds
        
        total_fetched = 0
        all_data = []
        
        while True:
            # Fetch data from API
            result = self.fetch_ohlcv_data(kraken_pair, interval, since_timestamp)
            
            if not result['success']:
                break
            
            api_data = result['data']
            
            if not api_data:
                logger.info(f"No more data available for {pair} {timeframe}")
                break
            
            # Process and store data
            df = self.process_api_data(api_data, pair, timeframe)
            
            if not df.empty:
                # Filter to only new data (after our latest timestamp)
                df = df[df['timestamp'] > latest_timestamp]
                
                if not df.empty:
                    if self.insert_ohlcv_data(df, timeframe):
                        total_fetched += len(df)
                        all_data.append(df)
                        
                        # Update since_timestamp for next iteration
                        since_timestamp = df['timestamp'].max() + (interval * 60)
                    else:
                        logger.error(f"Failed to insert data for {pair} {timeframe}")
                        break
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Check if we got less than expected (API might be caught up)
            if len(api_data) < 720:  # Kraken typically returns up to 720 records
                logger.info(f"Received {len(api_data)} records, likely caught up for {pair} {timeframe}")
                break
        
        return {
            'success': True,
            'rows_fetched': total_fetched,
            'start_timestamp': latest_timestamp,
            'end_timestamp': since_timestamp if total_fetched > 0 else latest_timestamp
        }
    
    def fetch_all_missing_data(self):
        """Fetch missing data for all pairs and timeframes."""
        
        logger.info("Starting API fetch for missing data")
        logger.info(f"Target date range: {self.start_date} to {self.end_date}")
        
        results = {}
        
        for timeframe in ['1m', '5m']:
            logger.info(f"\n🔥 FETCHING {timeframe.upper()} DATA...")
            results[timeframe] = {}
            
            for pair in self.pairs.keys():
                result = self.fetch_pair_timeframe(pair, timeframe)
                results[timeframe][pair] = result
                
                if result['success']:
                    logger.info(f"✅ {pair}: {result['rows_fetched']} new rows")
                else:
                    logger.error(f"❌ {pair}: {result.get('error', 'Unknown error')}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("API FETCH SUMMARY")
        logger.info("=" * 60)
        
        for timeframe in ['1m', '5m']:
            total_rows = sum(r['rows_fetched'] for r in results[timeframe].values() if r['success'])
            successful = sum(1 for r in results[timeframe].values() if r['success'])
            
            logger.info(f"{timeframe.upper()} Data: {successful}/9 pairs, {total_rows:,} total rows")
            
            for pair, result in results[timeframe].items():
                status = "✅" if result['success'] else "❌"
                rows = result['rows_fetched'] if result['success'] else 0
                logger.info(f"  {status} {pair}: {rows:,} rows")
        
        return results
    
    def verify_data_continuity(self):
        """Verify that data is continuous from CSV to API data."""
        
        logger.info("Verifying data continuity...")
        
        try:
            cursor = self.conn.cursor()
            
            for timeframe in ['1m', '5m']:
                table_name = f"ohlcv_{timeframe}"
                
                # Check new date ranges
                cursor.execute(f"""
                    SELECT pair, 
                           COUNT(*) as total_rows,
                           MIN(datetime) as start_date, 
                           MAX(datetime) as end_date
                    FROM {table_name} 
                    GROUP BY pair 
                    ORDER BY pair
                """)
                
                results = cursor.fetchall()
                
                logger.info(f"\n{timeframe.upper()} DATA VERIFICATION:")
                logger.info("-" * 50)
                
                for pair, count, start, end in results:
                    logger.info(f"{pair}: {count:,} rows ({start} to {end})")
        
        except Exception as e:
            logger.error(f"Verification failed: {e}")
    
    def close_connection(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

def main():
    """Main execution function."""
    
    print("=" * 60)
    print("KRAKEN API FETCHER V2")
    print("=" * 60)
    
    fetcher = KrakenAPIFetcher()
    
    try:
        # Connect to database
        fetcher.connect_database()
        
        # Fetch all missing data
        results = fetcher.fetch_all_missing_data()
        
        # Verify the results
        fetcher.verify_data_continuity()
        
        print("\n✅ API fetch completed successfully!")
        print(f"Database updated at: {fetcher.db_path}")
        
    except Exception as e:
        logger.error(f"API fetch failed: {e}")
        print(f"\n❌ API fetch failed: {e}")
        
    finally:
        fetcher.close_connection()

if __name__ == "__main__":
    main()