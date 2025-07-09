#!/usr/bin/env python3
"""
Quick script to debug database structure and check table contents
"""

import sqlite3
import pandas as pd
from datetime import datetime

def check_database_structure():
    db_path = "/mnt/raid0/data_erick/kraken_trading_model_v2/data/kraken_v2.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("=== DATABASE STRUCTURE CHECK ===")
        
        # Check all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"\nFound {len(tables)} tables:")
        feature_tables = []
        ohlcv_tables = []
        
        for table in tables:
            table_name = table[0]
            print(f"  - {table_name}")
            
            if table_name.startswith('features_'):
                feature_tables.append(table_name)
            elif table_name.startswith('ohlcv_'):
                ohlcv_tables.append(table_name)
        
        print(f"\nFeature tables ({len(feature_tables)}):")
        for table in feature_tables:
            print(f"  - {table}")
            
            # Check latest timestamp and data count
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                
                cursor.execute(f"SELECT MAX(timestamp), MIN(timestamp) FROM {table}")
                timestamps = cursor.fetchone()
                
                print(f"    Rows: {count:,}")
                print(f"    Latest timestamp: {timestamps[0]} (type: {type(timestamps[0])})")
                
                if timestamps[0]:
                    # Try to convert to datetime
                    try:
                        if isinstance(timestamps[0], str):
                            # Try parsing as ISO format
                            dt = datetime.fromisoformat(timestamps[0].replace('Z', '+00:00'))
                            print(f"    Latest date: {dt}")
                        else:
                            # Assume Unix timestamp
                            dt = datetime.fromtimestamp(float(timestamps[0]))
                            print(f"    Latest date: {dt}")
                    except Exception as e:
                        print(f"    Could not parse timestamp: {e}")
                
            except Exception as e:
                print(f"    Error checking table: {e}")
            
            print()
        
        print(f"\nOHLCV tables ({len(ohlcv_tables)}):")
        for table in ohlcv_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  - {table}: {count:,} rows")
            except Exception as e:
                print(f"  - {table}: Error - {e}")
        
        # Check for institutional tables
        print("\nLooking for institutional tables:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%institutional%'")
        institutional_tables = cursor.fetchall()
        
        if institutional_tables:
            for table in institutional_tables:
                print(f"  - {table[0]}")
        else:
            print("  - No institutional tables found")
        
        conn.close()
        
    except Exception as e:
        print(f"Error checking database: {e}")

def check_models_directory():
    import os
    from pathlib import Path
    
    models_dir = Path("/mnt/raid0/data_erick/kraken_trading_model_v2/models/xgboost_regime_specific")
    
    print(f"\n=== MODELS DIRECTORY CHECK ===")
    print(f"Directory: {models_dir}")
    
    if not models_dir.exists():
        print("❌ Models directory does not exist!")
        return
    
    model_files = list(models_dir.glob("*.pkl"))
    print(f"\nFound {len(model_files)} model files:")
    
    # Group by pair and regime
    pairs = set()
    regimes = set()
    
    for model_file in model_files:
        print(f"  - {model_file.name}")
        
        # Extract pair and regime from filename
        name_parts = model_file.stem.split('_')
        if len(name_parts) >= 4:  # e.g., XBTUSDT_bull_high_vol_xgb_model
            pair = name_parts[0]
            regime = '_'.join(name_parts[1:-2])  # Join middle parts as regime
            pairs.add(pair)
            regimes.add(regime)
    
    print(f"\nUnique pairs found: {sorted(pairs)}")
    print(f"Unique regimes found: {sorted(regimes)}")

if __name__ == "__main__":
    check_database_structure()
    check_models_directory()