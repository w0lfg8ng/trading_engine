#!/usr/bin/env python3
"""
Aggressive Database Corruption Fix
Removes all corrupted binary data from numeric columns
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import shutil

def aggressive_corruption_fix(db_path="institutional_trading_history.db"):
    """Aggressively fix all database corruption"""
    
    print("🔧 AGGRESSIVE DATABASE CORRUPTION FIX")
    print("=" * 50)
    
    # Create backup
    backup_path = f"institutional_trading_history_backup_aggressive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    print(f"📁 Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Fix trades table - most critical
        print("\n🔧 Fixing trades table...")
        trades_df = pd.read_sql("SELECT * FROM trades", conn)
        
        if not trades_df.empty:
            print(f"Found {len(trades_df)} trades")
            
            # Numeric columns that might be corrupted
            numeric_cols = ['entry_price', 'exit_price', 'quantity', 'position_value', 
                          'leverage', 'pnl', 'net_pnl', 'fees', 'hold_time', 
                          'confidence', 'risk_score']
            
            corrupted_rows = set()
            
            for col in numeric_cols:
                if col in trades_df.columns:
                    # Try to convert to numeric
                    original_values = trades_df[col].copy()
                    numeric_values = pd.to_numeric(original_values, errors='coerce')
                    
                    # Find rows where conversion failed (but original wasn't null)
                    corrupted_mask = numeric_values.isna() & original_values.notna()
                    corrupted_in_col = corrupted_mask.sum()
                    
                    if corrupted_in_col > 0:
                        print(f"  ⚠️ Found {corrupted_in_col} corrupted values in {col}")
                        corrupted_rows.update(trades_df[corrupted_mask].index.tolist())
            
            if corrupted_rows:
                print(f"  🗑️ Removing {len(corrupted_rows)} corrupted trade rows...")
                
                # Get IDs of corrupted rows
                corrupted_ids = trades_df.iloc[list(corrupted_rows)]['id'].tolist()
                
                # Delete corrupted rows
                for trade_id in corrupted_ids:
                    conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
                
                print(f"  ✅ Removed {len(corrupted_rows)} corrupted trades")
            else:
                print("  ✅ No corruption found in trades table")
        
        # Fix signals table
        print("\n🔧 Fixing signals table...")
        signals_df = pd.read_sql("SELECT * FROM signals", conn)
        
        if not signals_df.empty:
            print(f"Found {len(signals_df)} signals")
            
            numeric_cols = ['price', 'confidence', 'risk_score', 'rsi', 'macd', 
                          'macd_signal', 'vwap', 'bb_upper', 'bb_lower', 
                          'ema_9', 'ema_21', 'trend_strength']
            
            corrupted_rows = set()
            
            for col in numeric_cols:
                if col in signals_df.columns:
                    original_values = signals_df[col].copy()
                    numeric_values = pd.to_numeric(original_values, errors='coerce')
                    
                    corrupted_mask = numeric_values.isna() & original_values.notna()
                    corrupted_in_col = corrupted_mask.sum()
                    
                    if corrupted_in_col > 0:
                        print(f"  ⚠️ Found {corrupted_in_col} corrupted values in {col}")
                        corrupted_rows.update(signals_df[corrupted_mask].index.tolist())
            
            if corrupted_rows:
                print(f"  🗑️ Removing {len(corrupted_rows)} corrupted signal rows...")
                
                corrupted_ids = signals_df.iloc[list(corrupted_rows)]['id'].tolist()
                
                for signal_id in corrupted_ids:
                    conn.execute("DELETE FROM signals WHERE id = ?", (signal_id,))
                
                print(f"  ✅ Removed {len(corrupted_rows)} corrupted signals")
            else:
                print("  ✅ No corruption found in signals table")
        
        # Fix equity table
        print("\n🔧 Fixing equity table...")
        equity_df = pd.read_sql("SELECT * FROM equity", conn)
        
        if not equity_df.empty:
            numeric_cols = ['total_equity', 'cash', 'positions_value', 'unrealized_pnl', 'portfolio_var']
            corrupted_rows = set()
            
            for col in numeric_cols:
                if col in equity_df.columns:
                    original_values = equity_df[col].copy()
                    numeric_values = pd.to_numeric(original_values, errors='coerce')
                    
                    corrupted_mask = numeric_values.isna() & original_values.notna()
                    corrupted_in_col = corrupted_mask.sum()
                    
                    if corrupted_in_col > 0:
                        print(f"  ⚠️ Found {corrupted_in_col} corrupted values in {col}")
                        corrupted_rows.update(equity_df[corrupted_mask].index.tolist())
            
            if corrupted_rows:
                print(f"  🗑️ Removing {len(corrupted_rows)} corrupted equity rows...")
                
                corrupted_ids = equity_df.iloc[list(corrupted_rows)]['id'].tolist()
                
                for equity_id in corrupted_ids:
                    conn.execute("DELETE FROM equity WHERE id = ?", (equity_id,))
                
                print(f"  ✅ Removed {len(corrupted_rows)} corrupted equity entries")
            else:
                print("  ✅ No corruption found in equity table")
        
        # Commit all changes
        conn.commit()
        
        # Vacuum database
        print("\n🧹 Cleaning up database...")
        conn.execute("VACUUM")
        conn.commit()
        
        print(f"\n✅ Aggressive corruption fix completed!")
        print(f"📁 Backup saved as: {backup_path}")
        
    except Exception as e:
        print(f"❌ Error during aggressive fix: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        conn.close()

if __name__ == "__main__":
    aggressive_corruption_fix()