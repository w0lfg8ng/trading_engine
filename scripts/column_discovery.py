#!/usr/bin/env python3
"""
Discover what columns actually exist in your feature tables
"""

import sqlite3
import pandas as pd

# Database path
DB_FILE = "../data/kraken_v2.db"

def discover_columns():
    """Discover actual column structure"""
    
    print("🔍 COLUMN STRUCTURE DISCOVERY")
    print("=" * 60)
    
    conn = sqlite3.connect(DB_FILE)
    
    # Check XBTUSDT as example
    pair = "XBTUSDT"
    regular_table = f"features_{pair}_1m"
    institutional_table = f"features_{pair}_1m_institutional"
    
    try:
        # Get column info for regular table
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({regular_table})")
        regular_columns = cursor.fetchall()
        
        print(f"📊 {regular_table} columns:")
        for col in regular_columns:
            print(f"  {col[1]} ({col[2]})")  # name (type)
        
        print(f"\n📊 {institutional_table} columns:")
        cursor.execute(f"PRAGMA table_info({institutional_table})")
        inst_columns = cursor.fetchall()
        
        for col in inst_columns:
            print(f"  {col[1]} ({col[2]})")  # name (type)
        
        # Sample a few rows to see data
        print(f"\n📋 Sample data from {regular_table}:")
        sample_df = pd.read_sql(f"SELECT * FROM {regular_table} LIMIT 3", conn)
        for col in sample_df.columns[:10]:  # First 10 columns
            print(f"  {col}: {sample_df[col].iloc[0]}")
        
        if len(sample_df.columns) > 10:
            print(f"  ... and {len(sample_df.columns) - 10} more columns")
        
        # Check for technical indicators
        tech_indicators = []
        for col in sample_df.columns:
            if any(indicator in col.lower() for indicator in ['rsi', 'macd', 'sma', 'bb', 'bollinger', 'volume']):
                tech_indicators.append(col)
        
        print(f"\n🎯 Technical Indicator Columns Found:")
        for indicator in tech_indicators:
            print(f"  ✅ {indicator}")
        
        if not tech_indicators:
            print(f"  ❌ No obvious technical indicators found")
            print(f"  📋 Available columns that might be indicators:")
            for col in sample_df.columns:
                if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                    print(f"    {col}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    conn.close()
    
    print(f"\n💡 NEXT STEPS:")
    print(f"1. Use the actual column names found above")
    print(f"2. Update the data quality investigation script")
    print(f"3. Check why institutional labeling started at different dates")

if __name__ == "__main__":
    discover_columns()