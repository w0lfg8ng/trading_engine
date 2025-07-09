import sqlite3
import pandas as pd
import sys
import numpy as np
from datetime import datetime

def analyze_pair_trades(db_path, pair_to_analyze):
    """Extract and analyze all trades for a specific trading pair from live trading database"""
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    
    # Query trades for the specific pair
    query = f"""
    SELECT 
        id,
        timestamp,
        pair,
        side,
        entry_price,
        exit_price,
        quantity,
        position_value,
        leverage,
        entry_time,
        exit_time,
        pnl,
        net_pnl,
        fees,
        hold_time,
        confidence,
        risk_score,
        exit_reason,
        market_regime,
        risk_tier,
        is_paper,
        partial_exit
    FROM 
        trades
    WHERE 
        pair = '{pair_to_analyze}'
        AND is_paper = 1
    ORDER BY 
        entry_time
    """
    
    # Load data into DataFrame
    trades_df = pd.read_sql(query, conn)
    
    # Convert timestamp columns to datetime
    for col in ['timestamp', 'entry_time', 'exit_time']:
        if col in trades_df.columns:
            trades_df[col] = pd.to_datetime(trades_df[col])
    
    # Close connection
    conn.close()
    
    if trades_df.empty:
        print(f"No trades found for {pair_to_analyze}")
        print("Available pairs in database:")
        # Show available pairs
        conn = sqlite3.connect(db_path)
        pairs_query = "SELECT DISTINCT pair FROM trades WHERE is_paper = 1 ORDER BY pair"
        pairs_df = pd.read_sql(pairs_query, conn)
        conn.close()
        if not pairs_df.empty:
            for pair in pairs_df['pair']:
                print(f"  - {pair}")
        return
    
    # Print detailed information
    print(f"\n=== DETAILED TRADE ANALYSIS FOR {pair_to_analyze.upper()} ===")
    print(f"Total trades: {len(trades_df)}")
    
    # Calculate performance metrics
    winning_trades = trades_df[trades_df['net_pnl'] > 0]
    losing_trades = trades_df[trades_df['net_pnl'] <= 0]
    
    win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
    profit_factor = abs(winning_trades['net_pnl'].sum() / losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 and losing_trades['net_pnl'].sum() != 0 else float('inf')
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Winning Trades: {len(winning_trades)}")
    print(f"Losing Trades: {len(losing_trades)}")
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Best Trade: ${trades_df['net_pnl'].max():.2f}")
    print(f"Worst Trade: ${trades_df['net_pnl'].min():.2f}")
    print(f"Total P&L: ${trades_df['net_pnl'].sum():.2f}")
    print(f"Average Hold Time: {trades_df['hold_time'].mean():.1f} minutes")
    
    # Risk tier analysis
    if 'risk_tier' in trades_df.columns and trades_df['risk_tier'].notna().any():
        print(f"\n🎚️ RISK TIER BREAKDOWN:")
        risk_analysis = trades_df.groupby('risk_tier').agg({
            'net_pnl': ['count', 'sum', 'mean'],
            'hold_time': 'mean',
            'confidence': 'mean'
        }).round(2)
        print(risk_analysis)
    
    # Market regime analysis
    if 'market_regime' in trades_df.columns and trades_df['market_regime'].notna().any():
        print(f"\n🌊 MARKET REGIME BREAKDOWN:")
        regime_analysis = trades_df.groupby('market_regime').agg({
            'net_pnl': ['count', 'sum', 'mean'],
            'hold_time': 'mean'
        }).round(2)
        print(regime_analysis)
    
    # Exit reason analysis
    if 'exit_reason' in trades_df.columns and trades_df['exit_reason'].notna().any():
        print(f"\n🚪 EXIT REASON BREAKDOWN:")
        exit_analysis = trades_df.groupby('exit_reason').agg({
            'net_pnl': ['count', 'sum', 'mean']
        }).round(2)
        print(exit_analysis)
    
    # Side analysis
    print(f"\n📈 LONG vs SHORT BREAKDOWN:")
    side_analysis = trades_df.groupby('side').agg({
        'net_pnl': ['count', 'sum', 'mean'],
        'hold_time': 'mean'
    }).round(2)
    print(side_analysis)
    
    # Partial vs full exit analysis
    if 'partial_exit' in trades_df.columns:
        print(f"\n📊 PARTIAL vs FULL EXIT BREAKDOWN:")
        partial_analysis = trades_df.groupby('partial_exit').agg({
            'net_pnl': ['count', 'sum', 'mean'],
            'hold_time': 'mean'
        }).round(2)
        print(partial_analysis)
    
    print(f"\n📋 TRADE-BY-TRADE BREAKDOWN:")
    print("-" * 150)
    print(f"{'ID':<4} {'Entry Time':<20} {'Exit Time':<20} {'Side':<6} {'Entry':<10} {'Exit':<10} {'Pos Val':<10} {'Lev':<4} {'P&L':<8} {'P&L%':<8} {'Fees':<8} {'Hold':<8} {'Conf':<6} {'Risk':<6} {'Regime':<12} {'Exit Reason':<15}")
    print("-" * 150)
    
    # Sort by net P&L for better visualization
    sorted_trades = trades_df.sort_values('net_pnl', ascending=False)
    
    for _, trade in sorted_trades.iterrows():
        entry_time_str = trade['entry_time'].strftime('%m/%d %H:%M') if pd.notna(trade['entry_time']) else 'N/A'
        exit_time_str = trade['exit_time'].strftime('%m/%d %H:%M') if pd.notna(trade['exit_time']) else 'N/A'
        
        print(f"{trade['id']:<4} {entry_time_str:<20} {exit_time_str:<20} {trade['side']:<6} "
              f"{trade['entry_price']:<10.4f} {trade['exit_price']:<10.4f} ${trade['position_value']:<9.0f} "
              f"{trade['leverage']:<4.1f} ${trade['net_pnl']:<7.2f} {trade['pnl']:<7.2f}% "
              f"${trade['fees']:<7.2f} {trade['hold_time']:<7.1f}m {trade['confidence']:<6.2f} "
              f"{str(trade['risk_tier']):<6} {str(trade['market_regime'])[:11]:<12} {str(trade['exit_reason']):<15}")
    
    print("-" * 150)
    
    # Calculate manually verifiable metrics
    total_position_value = trades_df['position_value'].sum()
    total_fees = trades_df['fees'].sum()
    
    print(f"\n📈 SUMMARY:")
    print(f"Total Position Value: ${total_position_value:.2f}")
    print(f"Total Fees: ${total_fees:.2f}")
    print(f"Total Net P&L: ${trades_df['net_pnl'].sum():.2f}")
    print(f"Fee Percentage: {(total_fees / total_position_value * 100):.3f}%")
    
    # Time-based analysis
    if len(trades_df) > 1:
        trades_df['date'] = trades_df['entry_time'].dt.date
        daily_performance = trades_df.groupby('date')['net_pnl'].sum().sort_index()
        
        print(f"\n📅 DAILY PERFORMANCE:")
        for date, pnl in daily_performance.items():
            print(f"{date}: ${pnl:.2f}")
    
    # Save to CSV for further analysis
    output_file = f"{pair_to_analyze}_live_trades_detailed.csv"
    trades_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed trade data saved to {output_file}")

def show_database_overview(db_path):
    """Show overview of the entire trading database"""
    conn = sqlite3.connect(db_path)
    
    # Get all pairs and their trade counts
    pairs_query = """
    SELECT 
        pair,
        COUNT(*) as trade_count,
        SUM(net_pnl) as total_pnl,
        AVG(net_pnl) as avg_pnl,
        AVG(CASE WHEN net_pnl > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate
    FROM trades 
    WHERE is_paper = 1
    GROUP BY pair 
    ORDER BY total_pnl DESC
    """
    
    pairs_df = pd.read_sql(pairs_query, conn)
    
    print("\n=== DATABASE OVERVIEW ===")
    print(f"Available trading pairs and their performance:")
    print("-" * 70)
    print(f"{'Pair':<12} {'Trades':<8} {'Total P&L':<12} {'Avg P&L':<10} {'Win Rate':<10}")
    print("-" * 70)
    
    for _, row in pairs_df.iterrows():
        print(f"{row['pair']:<12} {row['trade_count']:<8} ${row['total_pnl']:<11.2f} ${row['avg_pnl']:<9.2f} {row['win_rate']:<9.1f}%")
    
    print("-" * 70)
    print(f"Total Trades: {pairs_df['trade_count'].sum()}")
    print(f"Total P&L: ${pairs_df['total_pnl'].sum():.2f}")
    
    conn.close()

if __name__ == "__main__":
    # Default database path for live trading engine
    db_path = "institutional_trading_history.db"
    
    if len(sys.argv) < 2:
        # Show database overview first
        show_database_overview(db_path)
        print("\nUsage: python analyze_pair_trades_live.py <pair_symbol>")
        print("Example: python analyze_pair_trades_live.py btcusdt")
        pair_to_analyze = input("\nEnter pair to analyze: ").lower()
    else:
        pair_to_analyze = sys.argv[1].lower()
    
    if pair_to_analyze:
        analyze_pair_trades(db_path, pair_to_analyze)