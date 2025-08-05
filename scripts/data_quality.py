#!/usr/bin/env python3
"""
Proper investigation using actual column names
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

# Database path
DB_FILE = "../data/kraken_v2.db"

# Your trading pairs
PAIRS = ["XBTUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", 
         "LTCUSDT", "XRPUSDT", "LINKUSDT", "AVAXUSDT"]

def investigate_labeling_delays():
    """Investigate why institutional labeling started at different dates"""
    
    print("🔍 INSTITUTIONAL LABELING DELAY INVESTIGATION")
    print("=" * 70)
    
    conn = sqlite3.connect(DB_FILE)
    
    results = []
    
    for pair in PAIRS:
        try:
            print(f"\n📊 {pair} Analysis:")
            
            # Check regular table data availability
            regular_table = f"features_{pair}_1m"
            
            # Get first few hundred rows to check data quality
            query = f"""
            SELECT timestamp, datetime, rsi_14, macd, macd_signal, volume, 
                   bb_upper, bb_lower, close
            FROM {regular_table} 
            ORDER BY timestamp 
            LIMIT 500
            """
            
            df_regular = pd.read_sql(query, conn)
            
            if len(df_regular) == 0:
                print(f"  ❌ No data found")
                continue
            
            # Convert timestamps to datetime for analysis
            df_regular['dt'] = pd.to_datetime(df_regular['timestamp'], unit='s')
            
            print(f"  📅 Regular table starts: {df_regular['dt'].iloc[0]}")
            print(f"  📏 First 500 rows span: {df_regular['dt'].iloc[0]} → {df_regular['dt'].iloc[-1]}")
            
            # Check when all required indicators become available
            required_cols = ['rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower']
            
            # Find first row where all indicators are non-null
            all_available = df_regular[required_cols].notna().all(axis=1)
            
            if all_available.any():
                first_complete_idx = all_available.idxmax()
                first_complete_date = df_regular.loc[first_complete_idx, 'dt']
                warmup_hours = first_complete_idx  # Assuming 1-minute data
                
                print(f"  ✅ All indicators available from: {first_complete_date}")
                print(f"  🕒 Warmup period: {warmup_hours} minutes ({warmup_hours/60:.1f} hours)")
            else:
                print(f"  ❌ Some indicators still missing after 500 rows")
                
                # Check individual indicators
                for col in required_cols:
                    first_valid = df_regular[col].first_valid_index()
                    if first_valid is not None:
                        first_date = df_regular.loc[first_valid, 'dt']
                        print(f"    {col}: Available from row {first_valid} ({first_date})")
                    else:
                        print(f"    {col}: ❌ Not available in first 500 rows")
                
                first_complete_date = None
                warmup_hours = None
            
            # Check institutional table start
            inst_table = f"features_{pair}_1m_institutional"
            
            try:
                inst_query = f"""
                SELECT MIN(timestamp) as first_label, COUNT(*) as total_labels,
                       SUM(CASE WHEN institutional_label_4h != 'hold' THEN 1 ELSE 0 END) as viable_labels
                FROM {inst_table}
                """
                
                inst_df = pd.read_sql(inst_query, conn)
                first_label_date = pd.to_datetime(inst_df['first_label'].iloc[0])
                total_labels = inst_df['total_labels'].iloc[0]
                viable_labels = inst_df['viable_labels'].iloc[0]
                
                print(f"  🏷️ Institutional labels start: {first_label_date}")
                print(f"  🏷️ Total labels: {total_labels:,}")
                print(f"  🏷️ Viable trades: {viable_labels:,} ({viable_labels/total_labels*100:.1f}%)")
                
                # Calculate delay
                regular_start = df_regular['dt'].iloc[0]
                delay_hours = (first_label_date - regular_start).total_seconds() / 3600
                delay_days = delay_hours / 24
                
                print(f"  ⏱️ Labeling delay: {delay_hours:.1f} hours ({delay_days:.1f} days)")
                
                # Store results
                results.append({
                    'pair': pair,
                    'regular_start': regular_start,
                    'indicators_ready': first_complete_date,
                    'labels_start': first_label_date,
                    'delay_hours': delay_hours,
                    'delay_days': delay_days,
                    'warmup_minutes': warmup_hours,
                    'viable_trades': viable_labels,
                    'total_labels': total_labels
                })
                
            except Exception as e:
                print(f"  ❌ Institutional table error: {e}")
        
        except Exception as e:
            print(f"❌ Error analyzing {pair}: {e}")
    
    conn.close()
    
    # Summary analysis
    if results:
        print(f"\n📈 SUMMARY ANALYSIS")
        print("=" * 70)
        
        df_results = pd.DataFrame(results)
        
        print(f"Delay Statistics:")
        print(f"  Minimum delay: {df_results['delay_days'].min():.1f} days")
        print(f"  Maximum delay: {df_results['delay_days'].max():.1f} days")
        print(f"  Average delay: {df_results['delay_days'].mean():.1f} days")
        print(f"  Median delay: {df_results['delay_days'].median():.1f} days")
        
        # Find pairs with significant delays
        high_delay = df_results[df_results['delay_days'] > 30]  # More than 30 days
        if not high_delay.empty:
            print(f"\n⚠️ PAIRS WITH SIGNIFICANT DELAYS (>30 days):")
            for _, row in high_delay.iterrows():
                print(f"  {row['pair']}: {row['delay_days']:.1f} days delay")
        
        # Show the pattern
        print(f"\n📊 LABELING START DATES:")
        for _, row in df_results.iterrows():
            print(f"  {row['pair']}: {row['labels_start'].strftime('%Y-%m-%d')} "
                  f"({row['viable_trades']:,} viable trades)")
        
        # Find the earliest common start date
        latest_start = df_results['labels_start'].max()
        earliest_end = df_results['labels_start'].min()  # This should be similar across pairs
        
        print(f"\n🎯 FOR XGBOOST TRAINING:")
        print(f"  Latest start date: {latest_start.strftime('%Y-%m-%d')}")
        print(f"  Recommended common start: {latest_start.strftime('%Y-%m-%d')}")
        print(f"  This ensures all pairs have institutional labels")
        
        # Calculate how much data you'd have with alignment
        common_start = latest_start
        present = pd.to_datetime("2025-07-20")
        aligned_days = (present - common_start).days
        
        print(f"  Aligned period duration: {aligned_days} days ({aligned_days/365.25:.1f} years)")
        
        if aligned_days >= 365:
            print(f"  ✅ EXCELLENT: Sufficient data for reliable training")
        elif aligned_days >= 180:
            print(f"  ⚠️ ACCEPTABLE: Limited but usable for training")
        else:
            print(f"  ❌ INSUFFICIENT: Too little data for reliable training")

if __name__ == "__main__":
    investigate_labeling_delays()