#!/usr/bin/env python3
"""
CSV Importer for Kraken Trading Model V2
Imports historical OHLCV data from CSV files into SQLite database.
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../../logs/csv_import.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KrakenCSVImporter:
    def __init__(self, db_path: str = "../../data/kraken_v2.db"):
        """Initialize the CSV importer with database connection."""
        self.db_path = db_path
        self.conn = None
        
        # Trading pairs to import
        self.pairs = [
            "ADAUSDT", "AVAXUSDT", "DOTUSDT", "ETHUSDT", 
            "LINKUSDT", "LTCUSDT", "SOLUSDT", "XBTUSDT", "XRPUSDT"
        ]
        
        # Date range for consistent data (from your analysis)
        self.start_date = "2022-10-11"
        self.end_date = "2025-03-31"
        
        # CSV data directory
        self.csv_base_dir = "/home/erick/Desktop/RAID0/data_erick/OHLCV Data"
        
    def connect_database(self):
        """Create database connection and setup tables."""
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA journal_mode = WAL")
            logger.info(f"Connected to database: {self.db_path}")
            
            self.create_tables()
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def create_tables(self):
        """Create OHLCV tables for different timeframes."""
        
        # 1-minute OHLCV table
        create_1m_table = """
        CREATE TABLE IF NOT EXISTS ohlcv_1m (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            datetime TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            trade_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(pair, timestamp)
        )
        """
        
        # 5-minute OHLCV table
        create_5m_table = """
        CREATE TABLE IF NOT EXISTS ohlcv_5m (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            datetime TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            trade_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(pair, timestamp)
        )
        """
        
        # Create indexes for better query performance
        create_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_1m_pair_timestamp ON ohlcv_1m(pair, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_1m_datetime ON ohlcv_1m(datetime)",
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_5m_pair_timestamp ON ohlcv_5m(pair, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_5m_datetime ON ohlcv_5m(datetime)"
        ]
        
        try:
            self.conn.execute(create_1m_table)
            self.conn.execute(create_5m_table)
            
            for index_sql in create_indexes:
                self.conn.execute(index_sql)
            
            self.conn.commit()
            logger.info("Database tables and indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def import_pair_csv(self, pair: str, timeframe: str = "1m") -> Dict:
        """Import CSV data for a specific trading pair."""
        
        # Determine file path and table name
        csv_file = f"{pair}_{timeframe.replace('m', '')}.csv"
        csv_path = os.path.join(self.csv_base_dir, pair, csv_file)
        table_name = f"ohlcv_{timeframe}"
        
        logger.info(f"Importing {pair} {timeframe} from {csv_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                logger.warning(f"CSV file not found: {csv_path}")
                return {"success": False, "error": "File not found", "rows": 0}
            
            # Read CSV file
            df = pd.read_csv(csv_path)
            logger.info(f"Read {len(df):,} rows from {csv_file}")
            
            # Handle column names (timestamps are typically first column)
            if len(df.columns) >= 7:
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count'] + list(df.columns[7:])
            elif len(df.columns) >= 6:
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + list(df.columns[6:])
            else:
                logger.error(f"Insufficient columns in {csv_file}: {df.columns}")
                return {"success": False, "error": "Insufficient columns", "rows": 0}
            
            # Convert Unix timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Filter to consistent date range
            start_dt = pd.to_datetime(self.start_date)
            end_dt = pd.to_datetime(self.end_date)
            
            original_count = len(df)
            df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)]
            filtered_count = len(df)
            
            logger.info(f"Filtered {pair} {timeframe}: {original_count:,} → {filtered_count:,} rows ({self.start_date} to {self.end_date})")
            
            if filtered_count == 0:
                logger.warning(f"No data in date range for {pair} {timeframe}")
                return {"success": False, "error": "No data in range", "rows": 0}
            
            # Add pair column
            df['pair'] = pair
            
            # Prepare data for database insert
            df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Ensure trade_count column exists with proper defaults
            if 'trade_count' not in df.columns:
                df['trade_count'] = 1
            else:
                df['trade_count'] = df['trade_count'].fillna(1)
            
            # Select columns in correct order
            columns = ['pair', 'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'trade_count']
            df_insert = df[columns]
            
            # Insert into database
            insert_sql = f"""
            INSERT OR REPLACE INTO {table_name} 
            (pair, timestamp, datetime, open, high, low, close, volume, trade_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.conn.executemany(insert_sql, df_insert.values.tolist())
            self.conn.commit()
            
            logger.info(f"Successfully imported {filtered_count:,} rows for {pair} {timeframe}")
            
            return {
                "success": True, 
                "rows": filtered_count,
                "start_date": df['datetime'].min(),
                "end_date": df['datetime'].max()
            }
            
        except Exception as e:
            logger.error(f"Failed to import {pair} {timeframe}: {e}")
            return {"success": False, "error": str(e), "rows": 0}
    
    def import_all_pairs(self, timeframe: str = "1m"):
        """Import all trading pairs for specified timeframe."""
        
        logger.info(f"Starting import of all {len(self.pairs)} pairs for {timeframe} timeframe")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        
        results = {}
        total_rows = 0
        successful_imports = 0
        
        for pair in self.pairs:
            result = self.import_pair_csv(pair, timeframe)
            results[pair] = result
            
            if result["success"]:
                total_rows += result["rows"]
                successful_imports += 1
            
            logger.info(f"Progress: {len(results)}/{len(self.pairs)} pairs processed")
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"IMPORT SUMMARY - {timeframe.upper()}")
        logger.info("=" * 60)
        logger.info(f"Successful imports: {successful_imports}/{len(self.pairs)}")
        logger.info(f"Total rows imported: {total_rows:,}")
        
        for pair, result in results.items():
            status = "✅" if result["success"] else "❌"
            rows = result["rows"] if result["success"] else 0
            logger.info(f"{status} {pair}: {rows:,} rows")
        
        return results
    
    def verify_import(self, timeframe: str = "1m"):
        """Verify the imported data for specified timeframe."""
        
        table_name = f"ohlcv_{timeframe}"
        logger.info(f"Verifying {timeframe} imported data...")
        
        try:
            # Check total rows per pair
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT pair, COUNT(*) as row_count, 
                       MIN(datetime) as start_date, 
                       MAX(datetime) as end_date
                FROM {table_name} 
                GROUP BY pair 
                ORDER BY pair
            """)
            
            results = cursor.fetchall()
            
            if results:
                logger.info(f"\n{timeframe.upper()} DATABASE VERIFICATION:")
                logger.info("-" * 50)
                for pair, count, start, end in results:
                    logger.info(f"{pair}: {count:,} rows ({start} to {end})")
                
                # Check for any data quality issues
                cursor.execute(f"""
                    SELECT pair, COUNT(*) as null_count
                    FROM {table_name} 
                    WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
                    GROUP BY pair
                """)
                
                null_results = cursor.fetchall()
                if null_results:
                    logger.warning(f"{timeframe} data quality issues found:")
                    for pair, null_count in null_results:
                        logger.warning(f"{pair}: {null_count} rows with NULL values")
                else:
                    logger.info(f"✅ No {timeframe} data quality issues found")
            else:
                logger.info(f"No {timeframe} data found in database")
            
        except Exception as e:
            logger.error(f"{timeframe} verification failed: {e}")
    
    def close_connection(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

def main():
    """Main execution function."""
    
    print("=" * 60)
    print("KRAKEN CSV IMPORTER V2 - DUAL TIMEFRAME")
    print("=" * 60)
    
    importer = KrakenCSVImporter()
    
    try:
        # Connect to database
        importer.connect_database()
        
        # Import 1-minute data first
        print("\n🔥 IMPORTING 1-MINUTE DATA...")
        results_1m = importer.import_all_pairs(timeframe="1m")
        
        # Import 5-minute data second
        print("\n🔥 IMPORTING 5-MINUTE DATA...")
        results_5m = importer.import_all_pairs(timeframe="5m")
        
        # Verify both imports
        print("\n🔍 VERIFYING IMPORTS...")
        importer.verify_import(timeframe="1m")
        importer.verify_import(timeframe="5m")
        
        print("\n✅ CSV import completed successfully!")
        print(f"Database created at: {importer.db_path}")
        print("📊 Both 1-minute and 5-minute data imported!")
        
        # Final summary
        total_1m = sum(r["rows"] for r in results_1m.values() if r["success"])
        total_5m = sum(r["rows"] for r in results_5m.values() if r["success"])
        print(f"📈 Total 1-minute rows: {total_1m:,}")
        print(f"📈 Total 5-minute rows: {total_5m:,}")
        
    except Exception as e:
        logger.error(f"Import failed: {e}")
        print(f"\n❌ Import failed: {e}")
        
    finally:
        importer.close_connection()

if __name__ == "__main__":
    main()