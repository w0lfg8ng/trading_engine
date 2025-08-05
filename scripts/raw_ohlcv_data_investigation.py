#!/usr/bin/env python3
"""
Investigate if raw OHLCV data exists from October 2022 for all pairs
"""

import sqlite3
import pandas as pd
from datetime import datetime

# Database path
DB_FILE = "../data/kraken_v2.db"

# Your trading pairs
PAIRS = ["XBTUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", 
         "LTCUSDT", "XRPUSDT", "LINKUSDT", "AVAXUSDT"]

def investigate_raw_data():
    """Check if raw OHLCV data exists from October 2022"""
    
    print("🔍 RAW OHLCV DATA INVESTIGATION")
    print("=" * 70)
    print("Checking if your original October 2022 aligned data still exists...")
    
    conn = sqlite3.connect(DB_FILE)
    
    # First, let's see what tables exist
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    all_tables = [row[0] for row in cursor.fetchall()]
    
    print(f"\n📋 Available Tables in Database:")
    
    # Group tables by type
    feature_tables = [t for t in all_tables if 'features_' in t and '_1m' in t]
    ohlc_tables = [t for t in all_tables if any(pair in t for pair in PAIRS) and 'ohlc' in t.lower()]
    raw_tables = [t for t in all_tables if any(pair in t for pair in PAIRS) and 'raw' in t.lower()]
    other_pair_tables = [t for t in all_tables if any(pair.lower() in t.lower() for pair in PAIRS) 
                        and t not in feature_tables and t not in ohlc_tables and t not in raw_tables]
    
    print(f"  Feature Tables: {len(feature_tables)}")
    for table in sorted(feature_tables)[:5]:  # Show first 5
        print(f"    {table}")
    if len(feature_tables) > 5:
        print(f"    ... and {len(feature_tables) - 5} more")
    
    if ohlc_tables:
        print(f"  OHLC Tables: {len(ohlc_tables)}")
        for table in sorted(ohlc_tables):
            print(f"    {table}")
    
    if raw_tables:
        print(f"  Raw Tables: {len(raw_tables)}")
        for table in sorted(raw_tables):
            print(f"    {table}")
            
    if other_pair_tables:
        print(f"  Other Pair Tables: {len(other_pair_tables)}")
        for table in sorted(other_pair_tables)[:10]:  # Show first 10
            print(f"    {table}")
        if len(other_pair_tables) > 10:
            print(f"    ... and {len(other_pair_tables) - 10} more")
    
    # Now check each pair for different table types
    print(f"\n📊 PER-PAIR DATA AVAILABILITY CHECK:")
    
    target_date = datetime(2022, 10, 1)
    
    for pair in PAIRS:
        print(f"\n💰 {pair}:")
        
        # Check different possible table names
        possible_tables = [
            f"{pair.lower()}_1m",
            f"{pair}_1m", 
            f"ohlc_{pair.lower()}_1m",
            f"ohlc_{pair}_1m",
            f"raw_{pair.lower()}",
            f"raw_{pair}",
            f"kraken_{pair.lower()}_1m",
            f"kraken_{pair}_1m"
        ]
        
        found_raw_data = False
        
        for table_name in possible_tables:
            if table_name in all_tables:
                try:
                    # Check this table's date range
                    cursor.execute(f"SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM {table_name}")
                    result = cursor.fetchone()
                    
                    if result and result[0] is not None:
                        min_ts, max_ts, count = result
                        
                        # Try to parse timestamp
                        try:
                            if isinstance(min_ts, str):
                                min_date = pd.to_datetime(min_ts)
                                max_date = pd.to_datetime(max_ts)
                            else:
                                min_date = pd.to_datetime(min_ts, unit='s')
                                max_date = pd.to_datetime(max_ts, unit='s')
                            
                            print(f"  📋 {table_name}:")
                            print(f"    📅 {min_date.strftime('%Y-%m-%d')} → {max_date.strftime('%Y-%m-%d')}")
                            print(f"    📏 {count:,} rows ({(max_date - min_date).days} days)")
                            
                            # Check if this goes back to October 2022
                            if min_date.date() <= target_date.date():
                                print(f"    ✅ COVERS TARGET PERIOD (Oct 2022)")
                                found_raw_data = True
                            else:
                                days_missing = (min_date.date() - target_date.date()).days
                                print(f"    ⚠️ STARTS {days_missing} days after target")
                                
                        except Exception as e:
                            print(f"    ❌ Timestamp parsing error: {e}")
                            
                except Exception as e:
                    print(f"    ❌ Query error for {table_name}: {e}")
        
        # Check the feature table we know exists
        feature_table = f"features_{pair}_1m"
        if feature_table in all_tables:
            try:
                cursor.execute(f"SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM {feature_table}")
                result = cursor.fetchone()
                
                if result and result[0] is not None:
                    min_ts, max_ts, count = result
                    min_date = pd.to_datetime(min_ts, unit='s')
                    max_date = pd.to_datetime(max_ts, unit='s')
                    
                    print(f"  📊 {feature_table} (current):")
                    print(f"    📅 {min_date.strftime('%Y-%m-%d')} → {max_date.strftime('%Y-%m-%d')}")
                    print(f"    📏 {count:,} rows")
                    
            except Exception as e:
                print(f"    ❌ Feature table error: {e}")
        
        if not found_raw_data:
            print(f"  ❌ NO RAW DATA FOUND going back to October 2022")
    
    conn.close()
    
    print(f"\n💡 ANALYSIS:")
    print(f"=" * 70)
    print(f"1. If raw OHLC tables exist with October 2022 data → Feature calculation issue")
    print(f"2. If no raw tables found with October 2022 data → Data collection issue") 
    print(f"3. If some pairs have older raw data → Inconsistent data processing")
    print(f"4. Check your original data collection scripts/process")

if __name__ == "__main__":
    investigate_raw_data()