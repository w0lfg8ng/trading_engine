#!/usr/bin/env python3
"""
Live Trading Results Analyzer
Comprehensive analysis of the institutional live trading results from SQLite database
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Optional seaborn import for better plotting style
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("📊 Note: seaborn not installed. Charts will use basic matplotlib styling.")
    print("    To get better-looking charts, install with: pip install seaborn")

class LiveTradingAnalyzer:
    def __init__(self, db_path="institutional_trading_history.db"):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Connect to the trading database"""
        self.conn = sqlite3.connect(self.db_path)
        
    def disconnect(self):
        """Disconnect from the database"""
        if self.conn:
            self.conn.close()
            
    def get_all_trades(self):
        """Get all paper trades from the database"""
        query = """
        SELECT * FROM trades 
        WHERE is_paper = 1 
        ORDER BY entry_time
        """
        return pd.read_sql(query, self.conn, parse_dates=['timestamp', 'entry_time', 'exit_time'])
    
    def get_equity_history(self):
        """Get equity history from the database"""
        query = "SELECT * FROM equity ORDER BY timestamp"
        return pd.read_sql(query, self.conn, parse_dates=['timestamp'])
    
    def get_signals_history(self):
        """Get signals history from the database"""
        query = "SELECT * FROM signals ORDER BY timestamp"
        return pd.read_sql(query, self.conn, parse_dates=['timestamp'])
    
    def get_market_regimes(self):
        """Get market regime history from the database"""
        query = "SELECT * FROM market_regimes ORDER BY timestamp"
        return pd.read_sql(query, self.conn, parse_dates=['timestamp'])
    
    def analyze_overall_performance(self):
        """Analyze overall trading performance"""
        trades_df = self.get_all_trades()
        
        if trades_df.empty:
            print("❌ No trades found in database")
            
            # Check for open positions (MUST be BEFORE return)
            print("\n🔄 CHECKING OPEN POSITIONS:")
            cursor = self.conn.cursor()  # ← Use class connection
            
            cursor.execute("SELECT * FROM equity ORDER BY timestamp DESC LIMIT 5")
            recent_equity = cursor.fetchall()

            if recent_equity:
                print("Recent equity entries found - positions may be open")
                latest_equity = recent_equity[0]
                print(f"Latest equity: ${latest_equity[2]:.2f}")
            else:
                print("No equity data found")

            # Check signals vs trades
            cursor.execute("SELECT COUNT(*) FROM signals WHERE acted_upon = 1")
            acted_signals = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM trades")
            completed_trades = cursor.fetchone()[0]

            print(f"Signals acted upon: {acted_signals}")
            print(f"Completed trades: {completed_trades}")
            print(f"Open positions: {acted_signals - completed_trades}")
            
            return  # ← RETURN GOES AFTER THE DEBUGGING CODE
        
        # This only runs when trades exist
        print("🚀 LIVE TRADING PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Basic metrics
        total_trades = len(trades_df)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] <= 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        total_profit = trades_df['net_pnl'].sum()
        avg_profit = trades_df['net_pnl'].mean()
        avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum() / 
                           trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].sum()) if losing_trades > 0 else float('inf')
        
        # Time-based metrics
        if total_trades > 0:
            first_trade = trades_df['entry_time'].min()
            last_trade = trades_df['entry_time'].max()
            trading_period = (last_trade - first_trade).days
            avg_hold_time = trades_df['hold_time'].mean()
        else:
            trading_period = 0
            avg_hold_time = 0
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total Profit: ${total_profit:.2f}")
        print(f"Average Profit per Trade: ${avg_profit:.2f}")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average Hold Time: {avg_hold_time:.1f} minutes")
        print(f"Trading Period: {trading_period} days")
        
        if total_trades > 0:
            print(f"Best Trade: ${trades_df['net_pnl'].max():.2f}")
            print(f"Worst Trade: ${trades_df['net_pnl'].min():.2f}")
        
        return trades_df
    
    def analyze_by_pair(self, trades_df):
        """Analyze performance by trading pair"""
        print(f"\n📈 PAIR-BY-PAIR ANALYSIS:")
        
        pair_analysis = trades_df.groupby('pair').agg({
            'net_pnl': ['count', 'sum', 'mean'],
            'pnl': 'mean',
            'hold_time': 'mean',
            'fees': 'sum',
            'confidence': 'mean',
            'risk_score': 'mean'
        }).round(3)
        
        # Calculate win rates
        win_rates = trades_df.groupby('pair').apply(
            lambda x: len(x[x['net_pnl'] > 0]) / len(x) * 100 if len(x) > 0 else 0
        ).round(2)
        
        print("-" * 100)
        print(f"{'Pair':<12} {'Trades':<8} {'Total P&L':<12} {'Avg P&L':<10} {'Win Rate':<10} {'Avg Hold':<10} {'Total Fees':<12}")
        print("-" * 100)
        
        for pair in pair_analysis.index:
            trades_count = int(pair_analysis.loc[pair, ('net_pnl', 'count')])
            total_pnl = pair_analysis.loc[pair, ('net_pnl', 'sum')]
            avg_pnl = pair_analysis.loc[pair, ('net_pnl', 'mean')]
            avg_hold = pair_analysis.loc[pair, ('hold_time', 'mean')]
            total_fees = pair_analysis.loc[pair, ('fees', 'sum')]
            win_rate = win_rates[pair]
            
            print(f"{pair:<12} {trades_count:<8} ${total_pnl:<11.2f} ${avg_pnl:<9.2f} {win_rate:<9.1f}% {avg_hold:<9.1f}m ${total_fees:<11.2f}")
        
        print("-" * 100)
        
        # Best and worst performing pairs
        pair_totals = trades_df.groupby('pair')['net_pnl'].sum().sort_values(ascending=False)
        print(f"\n🏆 Best Performing Pair: {pair_totals.index[0]} (${pair_totals.iloc[0]:.2f})")
        if len(pair_totals) > 1:
            print(f"💸 Worst Performing Pair: {pair_totals.index[-1]} (${pair_totals.iloc[-1]:.2f})")
    
    def analyze_risk_tiers(self, trades_df):
        """Analyze performance by risk tier"""
        if 'risk_tier' not in trades_df.columns or trades_df['risk_tier'].isna().all():
            print(f"\n❌ No risk tier data available")
            return
        
        print(f"\n🎚️ RISK TIER ANALYSIS:")
        
        risk_analysis = trades_df.groupby('risk_tier').agg({
            'net_pnl': ['count', 'sum', 'mean'],
            'hold_time': 'mean',
            'confidence': 'mean',
            'risk_score': 'mean',
            'leverage': 'mean'
        }).round(3)
        
        # Calculate win rates by risk tier
        risk_win_rates = trades_df.groupby('risk_tier').apply(
            lambda x: len(x[x['net_pnl'] > 0]) / len(x) * 100 if len(x) > 0 else 0
        ).round(2)
        
        print(risk_analysis)
        print(f"\nWin Rates by Risk Tier:")
        for tier, win_rate in risk_win_rates.items():
            print(f"{tier}: {win_rate:.1f}%")
    
    def analyze_market_regimes(self, trades_df):
        """Analyze performance by market regime"""
        if 'market_regime' not in trades_df.columns or trades_df['market_regime'].isna().all():
            print(f"\n❌ No market regime data available")
            return
        
        print(f"\n🌊 MARKET REGIME ANALYSIS:")
        
        regime_analysis = trades_df.groupby('market_regime').agg({
            'net_pnl': ['count', 'sum', 'mean'],
            'hold_time': 'mean',
            'confidence': 'mean'
        }).round(3)
        
        print(regime_analysis)
        
        # Regime win rates
        regime_win_rates = trades_df.groupby('market_regime').apply(
            lambda x: len(x[x['net_pnl'] > 0]) / len(x) * 100 if len(x) > 0 else 0
        ).round(2)
        
        print(f"\nWin Rates by Market Regime:")
        for regime, win_rate in regime_win_rates.items():
            print(f"{regime}: {win_rate:.1f}%")
    
    def analyze_exit_reasons(self, trades_df):
        """Analyze performance by exit reason"""
        if 'exit_reason' not in trades_df.columns or trades_df['exit_reason'].isna().all():
            print(f"\n❌ No exit reason data available")
            return
        
        print(f"\n🚪 EXIT REASON ANALYSIS:")
        
        exit_analysis = trades_df.groupby('exit_reason').agg({
            'net_pnl': ['count', 'sum', 'mean'],
            'hold_time': 'mean'
        }).round(3)
        
        print(exit_analysis)
        
        # Exit reason win rates
        exit_win_rates = trades_df.groupby('exit_reason').apply(
            lambda x: len(x[x['net_pnl'] > 0]) / len(x) * 100 if len(x) > 0 else 0
        ).round(2)
        
        print(f"\nWin Rates by Exit Reason:")
        for reason, win_rate in exit_win_rates.items():
            print(f"{reason}: {win_rate:.1f}%")
    
    def analyze_signals_quality(self):
        """Analyze signal generation and quality"""
        signals_df = self.get_signals_history()
        
        if signals_df.empty:
            print(f"\n❌ No signals data available")
            return
        
        print(f"\n📡 SIGNAL ANALYSIS:")
        
        total_signals = len(signals_df)
        acted_signals = len(signals_df[signals_df['acted_upon'] == 1])
        signal_conversion_rate = acted_signals / total_signals * 100 if total_signals > 0 else 0
        
        print(f"Total Signals Generated: {total_signals}")
        print(f"Signals Acted Upon: {acted_signals}")
        print(f"Signal Conversion Rate: {signal_conversion_rate:.2f}%")
        
        # Signal distribution by pair
        signal_by_pair = signals_df['pair'].value_counts()
        print(f"\nSignals by Pair:")
        for pair, count in signal_by_pair.items():
            acted_count = len(signals_df[(signals_df['pair'] == pair) & (signals_df['acted_upon'] == 1)])
            conversion = acted_count / count * 100 if count > 0 else 0
            print(f"{pair}: {count} signals, {acted_count} acted upon ({conversion:.1f}%)")
        
        # Average signal quality - handle potential data corruption
        try:
            # Try to convert confidence to numeric, handling corrupted data
            signals_df['confidence_clean'] = pd.to_numeric(signals_df['confidence'], errors='coerce')
            signals_df['risk_score_clean'] = pd.to_numeric(signals_df['risk_score'], errors='coerce')
            
            # Calculate averages, excluding NaN values
            valid_confidence = signals_df['confidence_clean'].dropna()
            valid_risk_score = signals_df['risk_score_clean'].dropna()
            
            if len(valid_confidence) > 0:
                avg_confidence = valid_confidence.mean()
                print(f"\nAverage Signal Confidence: {avg_confidence:.3f} (from {len(valid_confidence)}/{total_signals} valid values)")
            else:
                print(f"\n⚠️ No valid confidence data found (possible data corruption)")
            
            if len(valid_risk_score) > 0:
                avg_risk_score = valid_risk_score.mean()
                print(f"Average Risk Score: {avg_risk_score:.3f} (from {len(valid_risk_score)}/{total_signals} valid values)")
            else:
                print(f"⚠️ No valid risk score data found (possible data corruption)")
                
            # Check for data corruption
            corrupted_confidence = total_signals - len(valid_confidence)
            corrupted_risk_score = total_signals - len(valid_risk_score)
            
            if corrupted_confidence > 0 or corrupted_risk_score > 0:
                print(f"\n⚠️ DATA CORRUPTION DETECTED:")
                if corrupted_confidence > 0:
                    print(f"   {corrupted_confidence} corrupted confidence values")
                if corrupted_risk_score > 0:
                    print(f"   {corrupted_risk_score} corrupted risk_score values")
                print(f"   Consider cleaning your database or restarting the trading engine")
                
        except Exception as e:
            print(f"\n⚠️ Error analyzing signal quality (possible data corruption): {e}")
            print(f"   This may indicate database corruption in the signals table")
    
    def analyze_equity_curve(self):
        """Analyze equity curve and drawdowns"""
        equity_df = self.get_equity_history()
        
        if equity_df.empty:
            print(f"\n❌ No equity data available")
            return
        
        print(f"\n💰 EQUITY CURVE ANALYSIS:")
        
        initial_equity = equity_df['total_equity'].iloc[0]
        final_equity = equity_df['total_equity'].iloc[-1]
        max_equity = equity_df['total_equity'].max()
        min_equity = equity_df['total_equity'].min()
        
        total_return = (final_equity - initial_equity) / initial_equity * 100
        max_drawdown = (max_equity - min_equity) / max_equity * 100
        
        print(f"Initial Equity: ${initial_equity:.2f}")
        print(f"Final Equity: ${final_equity:.2f}")
        print(f"Peak Equity: ${max_equity:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        
        # Calculate Sharpe ratio (annualized)
        equity_df['returns'] = equity_df['total_equity'].pct_change()
        if equity_df['returns'].std() > 0:
            sharpe_ratio = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(365 * 24 * 60)  # Assuming minute-level data
            print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
    
    def show_top_trades(self, trades_df, n=10):
        """Show best and worst trades"""
        print(f"\n🏆 TOP {n} WINNING TRADES:")
        top_winners = trades_df.nlargest(n, 'net_pnl')[
            ['pair', 'side', 'entry_time', 'net_pnl', 'pnl', 'hold_time', 'exit_reason', 'confidence']
        ]
        
        for _, trade in top_winners.iterrows():
            print(f"{trade['pair']} {trade['side'].upper()}: ${trade['net_pnl']:.2f} ({trade['pnl']*100:.2f}%) "
                  f"in {trade['hold_time']:.1f}m - {trade['exit_reason']} (conf: {trade['confidence']:.3f})")
        
        print(f"\n💸 TOP {n} LOSING TRADES:")
        top_losers = trades_df.nsmallest(n, 'net_pnl')[
            ['pair', 'side', 'entry_time', 'net_pnl', 'pnl', 'hold_time', 'exit_reason', 'confidence']
        ]
        
        for _, trade in top_losers.iterrows():
            print(f"{trade['pair']} {trade['side'].upper()}: ${trade['net_pnl']:.2f} ({trade['pnl']*100:.2f}%) "
                  f"in {trade['hold_time']:.1f}m - {trade['exit_reason']} (conf: {trade['confidence']:.3f})")
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        try:
            self.connect()
            
            # Main analysis
            trades_df = self.analyze_overall_performance()
            
            if trades_df is not None and not trades_df.empty:
                self.analyze_by_pair(trades_df)
                self.analyze_risk_tiers(trades_df)
                self.analyze_market_regimes(trades_df)
                self.analyze_exit_reasons(trades_df)
                self.show_top_trades(trades_df)
            
            # Additional analyses
            self.analyze_signals_quality()
            self.analyze_equity_curve()
            
            print(f"\n✅ Analysis complete!")
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.disconnect()
    
    def create_performance_charts(self):
        """Create performance visualization charts"""
        try:
            self.connect()
            
            # Get data
            trades_df = self.get_all_trades()
            equity_df = self.get_equity_history()
            
            if trades_df.empty:
                print("❌ No trade data for charting")
                return
            
            # Set up the plotting style
            if HAS_SEABORN:
                plt.style.use('seaborn-v0_8')
            else:
                plt.style.use('default')
                # Apply some basic styling improvements
                plt.rcParams['figure.facecolor'] = 'white'
                plt.rcParams['axes.grid'] = True
                plt.rcParams['grid.alpha'] = 0.3
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Live Trading Performance Dashboard', fontsize=16)
            
            # 1. Equity Curve
            if not equity_df.empty:
                axes[0, 0].plot(equity_df['timestamp'], equity_df['total_equity'], linewidth=2)
                axes[0, 0].set_title('Equity Curve')
                axes[0, 0].set_ylabel('Total Equity ($)')
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. P&L Distribution
            axes[0, 1].hist(trades_df['net_pnl'], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(trades_df['net_pnl'].mean(), color='red', linestyle='--', label=f'Mean: ${trades_df["net_pnl"].mean():.2f}')
            axes[0, 1].set_title('P&L Distribution')
            axes[0, 1].set_xlabel('Net P&L ($)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Performance by Pair
            pair_performance = trades_df.groupby('pair')['net_pnl'].sum().sort_values()
            axes[1, 0].barh(range(len(pair_performance)), pair_performance.values)
            axes[1, 0].set_yticks(range(len(pair_performance)))
            axes[1, 0].set_yticklabels(pair_performance.index)
            axes[1, 0].set_title('Total P&L by Pair')
            axes[1, 0].set_xlabel('Total P&L ($)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Win Rate by Pair
            win_rates = trades_df.groupby('pair').apply(
                lambda x: len(x[x['net_pnl'] > 0]) / len(x) * 100
            ).sort_values()
            
            axes[1, 1].barh(range(len(win_rates)), win_rates.values)
            axes[1, 1].set_yticks(range(len(win_rates)))
            axes[1, 1].set_yticklabels(win_rates.index)
            axes[1, 1].set_title('Win Rate by Pair')
            axes[1, 1].set_xlabel('Win Rate (%)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the chart
            chart_filename = f"live_trading_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
            print(f"\n📊 Performance charts saved to: {chart_filename}")
            
            # Show the plot
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating charts: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.disconnect()

def main():
    """Main function to run the analysis"""
    print("🚀 Starting Live Trading Analysis...")
    
    # Check if database exists
    db_path = "institutional_trading_history.db"
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        print("Make sure you've run the live trading engine first!")
        return
    
    # Create analyzer instance
    analyzer = LiveTradingAnalyzer(db_path)
    
    # Generate comprehensive report
    analyzer.generate_performance_report()
    
    # Ask if user wants to create charts
    try:
        create_charts = input("\n📊 Create performance charts? (y/n): ").lower().strip()
        if create_charts == 'y':
            analyzer.create_performance_charts()
    except KeyboardInterrupt:
        print("\n👋 Analysis complete!")

if __name__ == "__main__":
    main()