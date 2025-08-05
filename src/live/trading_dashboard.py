#!/usr/bin/env python3
"""
Streamlit Live Trading Dashboard
Professional web-based monitoring interface for the institutional trading engine
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import requests
import numpy as np
import os

def create_kill_switch_signal():
    """Create a signal file that the trading engine will check for shutdown"""
    try:
        with open("EMERGENCY_STOP.signal", "w") as f:
            f.write(f"EMERGENCY_STOP_REQUESTED_{datetime.now().isoformat()}")
        return True
    except Exception as e:
        st.error(f"Failed to create emergency stop signal: {e}")
        return False

def check_trading_engine_process():
    """Check if trading engine process is running"""
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('7.6.25_live_trading_engine_v1.py' in str(cmd) for cmd in cmdline):
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    except ImportError:
        st.warning("psutil not installed - cannot check process status")
        return None

def emergency_shutdown():
    """Execute emergency shutdown procedure"""
    try:
        # Step 1: Create signal file
        signal_created = create_kill_switch_signal()
        
        # Step 2: Try to find and terminate the process
        pid = check_trading_engine_process()
        process_terminated = False
        
        if pid:
            try:
                import psutil
                proc = psutil.Process(pid)
                proc.terminate()
                process_terminated = True
                st.success(f"Trading engine process (PID: {pid}) terminated")
            except Exception as e:
                st.warning(f"Could not terminate process: {e}")
        
        # Step 3: Try to close positions via database flag
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO circuit_breakers 
                (timestamp, type, reason, value, threshold, pair, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                'emergency_shutdown',
                'Dashboard emergency stop activated',
                1.0,
                1.0,
                'ALL',
                1
            ))
            conn.commit()
            conn.close()
            st.success("Emergency shutdown signal sent via database")
        except Exception as e:
            st.error(f"Failed to set database emergency flag: {e}")
        
        return signal_created or process_terminated
    except Exception as e:
        st.error(f"Emergency shutdown failed: {e}")
        return False

@st.cache_data(ttl=300)  # Cache for 5 minutes
def run_live_analyzer():
    """Run the live trading analyzer and return formatted results"""
    try:
        # We'll recreate the key analyzer functions here to avoid import issues
        def analyze_performance():
            try:
                conn = sqlite3.connect(DB_PATH)
                
                # Get all trades
                trades_df = pd.read_sql("SELECT * FROM trades WHERE is_paper = 1 ORDER BY entry_time", conn)
                
                if trades_df.empty:
                    return {"status": "no_trades", "message": "No completed trades found"}
                
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
                first_trade = trades_df['entry_time'].min()
                last_trade = trades_df['entry_time'].max()
                trading_period = (pd.to_datetime(last_trade) - pd.to_datetime(first_trade)).days
                avg_hold_time = trades_df['hold_time'].mean()
                
                # Best and worst trades
                best_trade = trades_df['net_pnl'].max()
                worst_trade = trades_df['net_pnl'].min()
                
                # Pair analysis
                pair_analysis = trades_df.groupby('pair').agg({
                    'net_pnl': ['count', 'sum', 'mean'],
                    'hold_time': 'mean'
                }).round(2)
                
                # Exit reason analysis
                exit_analysis = trades_df.groupby('exit_reason').agg({
                    'net_pnl': ['count', 'sum', 'mean']
                }).round(2)
                
                # Win rates by various categories
                pair_win_rates = trades_df.groupby('pair').apply(
                    lambda x: len(x[x['net_pnl'] > 0]) / len(x) * 100 if len(x) > 0 else 0
                ).round(1)
                
                exit_win_rates = trades_df.groupby('exit_reason').apply(
                    lambda x: len(x[x['net_pnl'] > 0]) / len(x) * 100 if len(x) > 0 else 0
                ).round(1)
                
                conn.close()
                
                return {
                    "status": "success",
                    "overall": {
                        "total_trades": total_trades,
                        "winning_trades": winning_trades,
                        "losing_trades": losing_trades,
                        "win_rate": win_rate,
                        "total_profit": total_profit,
                        "avg_profit": avg_profit,
                        "avg_win": avg_win,
                        "avg_loss": avg_loss,
                        "profit_factor": profit_factor,
                        "trading_period": trading_period,
                        "avg_hold_time": avg_hold_time,
                        "best_trade": best_trade,
                        "worst_trade": worst_trade
                    },
                    "pair_analysis": pair_analysis,
                    "exit_analysis": exit_analysis,
                    "pair_win_rates": pair_win_rates,
                    "exit_win_rates": exit_win_rates
                }
                
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        return analyze_performance()
        
    except Exception as e:
        return {"status": "error", "message": f"Analyzer failed: {str(e)}"}

def display_analyzer_results(results):
    """Display the analyzer results in the dashboard"""
    if results["status"] == "no_trades":
        st.info("📊 No completed trades to analyze yet")
        return
    elif results["status"] == "error":
        st.error(f"❌ Analyzer Error: {results['message']}")
        return
    
    overall = results["overall"]
    
    # Overall Performance Section
    st.markdown("### 📊 Live Trading Performance Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", overall["total_trades"])
        st.metric("Win Rate", f"{overall['win_rate']:.1f}%")
    
    with col2:
        st.metric("Total P&L", f"${overall['total_profit']:.2f}")
        st.metric("Profit Factor", f"{overall['profit_factor']:.2f}")
    
    with col3:
        st.metric("Avg Win", f"${overall['avg_win']:.2f}")
        st.metric("Avg Loss", f"${overall['avg_loss']:.2f}")
    
    with col4:
        st.metric("Best Trade", f"${overall['best_trade']:.2f}")
        st.metric("Worst Trade", f"${overall['worst_trade']:.2f}")
    
    # Pair Performance
    if 'pair_analysis' in results and not results['pair_analysis'].empty:
        st.markdown("### 📈 Pair Performance")
        
        # Create a simplified pair performance table
        pair_data = []
        for pair in results['pair_analysis'].index:
            trades_count = int(results['pair_analysis'].loc[pair, ('net_pnl', 'count')])
            total_pnl = results['pair_analysis'].loc[pair, ('net_pnl', 'sum')]
            avg_pnl = results['pair_analysis'].loc[pair, ('net_pnl', 'mean')]
            win_rate = results['pair_win_rates'].get(pair, 0)
            
            pair_data.append({
                "Pair": pair.upper(),
                "Trades": trades_count,
                "Total P&L": f"${total_pnl:.2f}",
                "Avg P&L": f"${avg_pnl:.2f}",
                "Win Rate": f"{win_rate:.1f}%"
            })
        
        pair_df = pd.DataFrame(pair_data)
        st.dataframe(pair_df, use_container_width=True)
    
    # Exit Reason Analysis
    if 'exit_analysis' in results and not results['exit_analysis'].empty:
        st.markdown("### 🚪 Exit Reason Analysis")
        
        exit_data = []
        for reason in results['exit_analysis'].index:
            trades_count = int(results['exit_analysis'].loc[reason, ('net_pnl', 'count')])
            total_pnl = results['exit_analysis'].loc[reason, ('net_pnl', 'sum')]
            avg_pnl = results['exit_analysis'].loc[reason, ('net_pnl', 'mean')]
            win_rate = results['exit_win_rates'].get(reason, 0)
            
            exit_data.append({
                "Exit Reason": reason.replace('_', ' ').title(),
                "Trades": trades_count,
                "Total P&L": f"${total_pnl:.2f}",
                "Avg P&L": f"${avg_pnl:.2f}",
                "Win Rate": f"{win_rate:.1f}%"
            })
        
        exit_df = pd.DataFrame(exit_data)
        st.dataframe(exit_df, use_container_width=True)

# Page configuration
st.set_page_config(
    page_title="🚀 Trading Engine Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #1f1f1f 0%, #2d2d2d 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00ff88;
    }
    .status-green { color: #4CAF50; }
    .status-yellow { color: #FF9800; }
    .status-red { color: #f44336; }
    .big-font { font-size: 2rem !important; }
    .positive-value { color: #4CAF50 !important; }
    .negative-value { color: #f44336 !important; }
    .neutral-value { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

# Database configuration
DB_PATH = "institutional_trading_history.db"

# Trading pairs
PAIRS = [
    "btcusdt", "ethusdt", "solusdt", "xrpusdt", 
    "adausdt", "ltcusdt", "dotusdt", "linkusdt", "avaxusdt"
]

@st.cache_data(ttl=30)
def check_database_connection():
    """Check if database is accessible"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1 FROM equity LIMIT 1")
        conn.close()
        return True
    except:
        return False

@st.cache_data(ttl=30)
def check_api_access():
    """Check Kraken API accessibility"""
    try:
        response = requests.get("https://api.kraken.com/0/public/Time", timeout=5)
        return response.status_code == 200
    except:
        return False

@st.cache_data(ttl=30)
def get_latest_equity():
    """Get latest equity information"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM equity ORDER BY timestamp DESC LIMIT 1", conn)
        conn.close()
        
        if not df.empty:
            return df.iloc[0]
        return None
    except:
        return None

@st.cache_data(ttl=30)
def check_trading_engine_status():
    """Check if trading engine is actually running based on recent activity"""
    try:
        # Check multiple indicators to be more accurate
        latest_equity = get_latest_equity()
        
        # Get recent signal activity
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MAX(timestamp) FROM signals 
            WHERE timestamp >= datetime('now', '-10 minutes')
        """)
        latest_signal = cursor.fetchone()[0]
        conn.close()
        
        current_time = datetime.now()
        
        # Check equity updates (must be very recent)
        equity_status = 'red'
        if latest_equity is not None:
            last_equity_time = pd.to_datetime(latest_equity['timestamp'])
            equity_diff = current_time - last_equity_time.replace(tzinfo=None)
            
            if equity_diff.total_seconds() < 180:  # Less than 3 minutes (stricter)
                equity_status = 'green'
            elif equity_diff.total_seconds() < 600:  # Less than 10 minutes
                equity_status = 'yellow'
        
        # Check signal activity (must be very recent)
        signal_status = 'red'
        if latest_signal:
            last_signal_time = pd.to_datetime(latest_signal)
            signal_diff = current_time - last_signal_time.replace(tzinfo=None)
            
            if signal_diff.total_seconds() < 180:  # Less than 3 minutes (stricter)
                signal_status = 'green'
            elif signal_diff.total_seconds() < 600:  # Less than 10 minutes
                signal_status = 'yellow'
        
        # Engine is only "Running" if BOTH equity and signals are recent
        if equity_status == 'green' and signal_status == 'green':
            return 'green', 'Running'
        elif equity_status in ['green', 'yellow'] or signal_status in ['green', 'yellow']:
            return 'yellow', 'Stale'
        else:
            return 'red', 'Stopped'
            
    except Exception as e:
        return 'red', 'Error'
def get_equity_history(hours=24):
    """Get equity history for charts"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cutoff_time = datetime.now() - timedelta(hours=hours)
        query = """
        SELECT timestamp, total_equity, unrealized_pnl 
        FROM equity 
        WHERE timestamp >= ? 
        ORDER BY timestamp
        """
        df = pd.read_sql(query, conn, params=[cutoff_time.isoformat()])
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def get_daily_trades():
    """Get today's completed trades"""
    try:
        conn = sqlite3.connect(DB_PATH)
        today = datetime.now().date()
        query = """
        SELECT * FROM trades 
        WHERE DATE(timestamp) = ? 
        ORDER BY timestamp DESC
        """
        df = pd.read_sql(query, conn, params=[today.isoformat()])
        conn.close()
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def get_daily_trading_activities():
    """Get all trading activities today (opens, closes, signals acted upon)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        today = datetime.now().date()
        
        # Try to get from multiple sources
        activities = []
        
        # 1. Get completed trades
        try:
            trades_query = """
            SELECT timestamp, pair, 'completed_trade' as activity_type, side, net_pnl
            FROM trades 
            WHERE DATE(timestamp) = ?
            """
            trades_df = pd.read_sql(trades_query, conn, params=[today.isoformat()])
            activities.append(trades_df)
        except:
            pass
        
        # 2. Get signals that were acted upon
        try:
            signals_query = """
            SELECT timestamp, pair, 'signal_action' as activity_type, signal as side, 0 as net_pnl
            FROM signals 
            WHERE DATE(timestamp) = ? AND acted_upon = 1
            """
            signals_df = pd.read_sql(signals_query, conn, params=[today.isoformat()])
            activities.append(signals_df)
        except:
            pass
        
        # 3. Check for position changes table if it exists
        try:
            positions_query = """
            SELECT timestamp, pair, 'position_change' as activity_type, 
                   CASE WHEN position_size > 0 THEN 'buy' ELSE 'sell' END as side,
                   0 as net_pnl
            FROM positions 
            WHERE DATE(timestamp) = ?
            """
            positions_df = pd.read_sql(positions_query, conn, params=[today.isoformat()])
            activities.append(positions_df)
        except:
            pass
        
        conn.close()
        
        # Combine all activities
        if activities:
            combined_df = pd.concat(activities, ignore_index=True)
            return combined_df.drop_duplicates()
        else:
            return pd.DataFrame()
            
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def check_kraken_api_status():
    """Check actual Kraken API status"""
    try:
        response = requests.get("https://api.kraken.com/0/public/Time", timeout=5)
        if response.status_code == 200:
            return 'green', 'Online'
        else:
            return 'red', 'Offline'
    except:
        return 'red', 'Offline'

@st.cache_data(ttl=300)
def check_model_status():
    """Check if models are actually loaded"""
    try:
        model_dir = "../../models/xgboost_regime_specific/"
        if not os.path.exists(model_dir):
            return 'red', 'Missing'
        
        # Count available models
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_xgb_model.pkl')]
        if len(model_files) >= 30:  # Should have ~35 models
            return 'green', f'Loaded ({len(model_files)})'
        elif len(model_files) >= 20:
            return 'yellow', f'Partial ({len(model_files)})'
        else:
            return 'red', f'Few ({len(model_files)})'
    except:
        return 'red', 'Error'

@st.cache_data(ttl=60)
def check_circuit_breakers():
    """Check circuit breaker status from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM circuit_breakers WHERE is_active = 1")
        active_breakers = cursor.fetchone()[0]
        conn.close()
        
        if active_breakers == 0:
            return 'green', 'Normal'
        else:
            return 'red', f'{active_breakers} Active'
    except:
        return 'yellow', 'Unknown'

@st.cache_data(ttl=60)
def check_cash_status():
    """Check cash status from latest equity"""
    try:
        latest_equity = get_latest_equity()
        if latest_equity is None:
            return 'red', 'No Data'
        
        cash = latest_equity.get('cash', 0)
        if cash > 10000:  # Healthy cash level
            return 'green', f'${cash:,.0f}'
        elif cash > 5000:
            return 'yellow', f'${cash:,.0f}'
        else:
            return 'red', f'${cash:,.0f}'
    except:
        return 'red', 'Error'

@st.cache_data(ttl=60)
def get_open_positions_count():
    """Get count of currently open positions"""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Check for positions that were opened but not closed
        cursor = conn.cursor()
        
        # Get signals that were acted upon but check if they have corresponding closes
        cursor.execute("""
            SELECT COUNT(DISTINCT pair) 
            FROM signals 
            WHERE acted_upon = 1 
            AND timestamp >= datetime('now', '-24 hours')
        """)
        
        recent_opens = cursor.fetchone()[0] or 0
        
        # Get completed trades
        cursor.execute("""
            SELECT COUNT(*) 
            FROM trades 
            WHERE timestamp >= datetime('now', '-24 hours')
        """)
        
        completed_trades = cursor.fetchone()[0] or 0
        conn.close()
        
        # Estimate open positions (this is approximate)
        estimated_open = max(0, recent_opens - completed_trades)
        return estimated_open
        
    except Exception as e:
        return 0

@st.cache_data(ttl=60)
def get_market_regimes():
    """Get latest market regimes from signals table since regimes aren't stored separately"""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Get the most recent signal for each pair, which includes regime info
        query = """
        SELECT DISTINCT pair, 
               CASE 
                   WHEN confidence > 0.8 THEN 'high_confidence'
                   WHEN confidence > 0.6 THEN 'medium_confidence' 
                   WHEN confidence > 0.4 THEN 'low_confidence'
                   ELSE 'uncertain'
               END as regime_quality,
               confidence,
               timestamp
        FROM signals 
        WHERE timestamp IN (
            SELECT MAX(timestamp) 
            FROM signals 
            GROUP BY pair
        )
        ORDER BY pair
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        if not df.empty:
            return df.set_index('pair')['regime_quality'].to_dict()
        else:
            # Return default regimes if no signals
            return {pair: "monitoring" for pair in PAIRS}
    except Exception as e:
        # Return default regimes if error
        return {pair: "monitoring" for pair in PAIRS}

@st.cache_data(ttl=60)
def get_actual_regime_models():
    """Get actual regime models currently being used by the trading engine"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Get the most recent regime for each pair from market_regimes table
        query = """
        SELECT pair, regime, volatility, timestamp
        FROM market_regimes 
        WHERE timestamp IN (
            SELECT MAX(timestamp) 
            FROM market_regimes 
            GROUP BY pair
        )
        ORDER BY pair
        """
        
        regimes_df = pd.read_sql(query, conn)
        
        result = {}
        
        if not regimes_df.empty:
            # We have actual regime data from the trading engine
            for _, row in regimes_df.iterrows():
                pair = row['pair']
                regime = row['regime']
                volatility = row.get('volatility', 0)
                timestamp = row['timestamp']
                
                # Format the model name as the engine uses it
                model_name = f"{pair.upper()}_{regime}"
                
                result[pair] = {
                    'model': model_name,
                    'regime_type': regime,
                    'confidence': f"Vol: {volatility:.4f}",
                    'last_update': timestamp,
                    'source': 'Trading Engine',
                    'status': 'Active'
                }
        else:
            # Fallback: try to get from signals table
            signal_query = """
            SELECT pair, market_regime, confidence, timestamp
            FROM signals 
            WHERE market_regime IS NOT NULL 
            AND timestamp IN (
                SELECT MAX(timestamp) 
                FROM signals 
                WHERE market_regime IS NOT NULL
                GROUP BY pair
            )
            ORDER BY pair
            """
            
            signals_df = pd.read_sql(signal_query, conn)
            
            for pair in PAIRS:
                pair_data = signals_df[signals_df['pair'] == pair]
                
                if not pair_data.empty:
                    latest = pair_data.iloc[0]
                    regime = latest['market_regime']
                    confidence = latest['confidence']
                    timestamp = latest['timestamp']
                    
                    result[pair] = {
                        'model': f"{pair.upper()}_{regime}",
                        'regime_type': regime,
                        'confidence': f"{confidence:.3f}",
                        'last_update': timestamp,
                        'source': 'Signal Data',
                        'status': 'Estimated'
                    }
                else:
                    # No data available for this pair
                    result[pair] = {
                        'model': f"{pair.upper()}_unknown",
                        'regime_type': "unknown",
                        'confidence': 'No Data',
                        'last_update': 'Unknown',
                        'source': 'Default',
                        'status': 'Missing'
                    }
        
        conn.close()
        return result
        
    except Exception as e:
        # Return default models if everything fails
        return {pair: {
            'model': f"{pair.upper()}_error",
            'regime_type': "error",
            'confidence': 'Error',
            'last_update': 'Error',
            'source': 'Error',
            'status': 'Error'
        } for pair in PAIRS}

@st.cache_data(ttl=300)
def get_model_file_status():
    """Check which model files are actually available"""
    try:
        model_dir = "../../models/xgboost_regime_specific/"
        if not os.path.exists(model_dir):
            return {}
        
        available_models = {}
        
        # Get all model files
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_xgb_model.pkl')]
        
        for model_file in model_files:
            # Parse filename to get pair and regime
            # Format: PAIRUSDT_regime_type_xgb_model.pkl
            base_name = model_file.replace('_xgb_model.pkl', '')
            parts = base_name.split('_')
            
            if len(parts) >= 3:
                pair_part = parts[0]  # e.g., XBTUSDT
                regime_part = '_'.join(parts[1:])  # e.g., bull_high_vol
                
                # Convert back to lowercase for consistency
                pair_key = pair_part.lower().replace('xbt', 'btc')  # Handle BTC special case
                
                if pair_key not in available_models:
                    available_models[pair_key] = []
                
                available_models[pair_key].append(regime_part)
        
        return available_models
        
    except Exception as e:
        return {}

@st.cache_data(ttl=300)
def get_script_execution_times():
    """Get last execution times for various scripts"""
    scripts_info = {}
    
    # Define script patterns and their log locations
    script_patterns = {
        "Live Trading Analyzer": {
            "description": "Analyzes live trading performance",
            "log_pattern": None,  # Manual execution
            "status": "Manual"
        },
        "ML Pipeline": {
            "description": "Updates ML models",
            "log_pattern": "../../logs/ml_pipeline_*.log",
            "status": "Automated"
        },
        "Data Quality Check": {
            "description": "Validates data integrity", 
            "log_pattern": "../../logs/feature_engineering.log",
            "status": "Automated"
        },
        "Daily Data Update": {
            "description": "Updates market data",
            "log_pattern": "../../logs/enhanced_daily_update_*.log", 
            "status": "Automated"
        }
    }
    
    for script_name, info in script_patterns.items():
        try:
            if info["log_pattern"] is None:
                # Manual scripts - check if analyzer database was updated recently
                if "Analyzer" in script_name:
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        cursor.execute("SELECT MAX(timestamp) FROM trades")
                        last_trade = cursor.fetchone()[0]
                        conn.close()
                        
                        if last_trade:
                            last_time = pd.to_datetime(last_trade)
                            scripts_info[script_name] = {
                                "last_run": last_time.strftime('%Y-%m-%d %H:%M'),
                                "status": info["status"],
                                "description": info["description"]
                            }
                        else:
                            scripts_info[script_name] = {
                                "last_run": "Never",
                                "status": info["status"], 
                                "description": info["description"]
                            }
                    except:
                        scripts_info[script_name] = {
                            "last_run": "Unknown",
                            "status": "Error",
                            "description": info["description"]
                        }
                continue
            
            # For log-based scripts, find the most recent log file
            import glob
            log_files = glob.glob(info["log_pattern"])
            
            if log_files:
                # Get the most recent log file
                latest_log = max(log_files, key=os.path.getmtime)
                mod_time = datetime.fromtimestamp(os.path.getmtime(latest_log))
                
                scripts_info[script_name] = {
                    "last_run": mod_time.strftime('%Y-%m-%d %H:%M'),
                    "status": info["status"],
                    "description": info["description"]
                }
            else:
                scripts_info[script_name] = {
                    "last_run": "Never",
                    "status": "Missing",
                    "description": info["description"]
                }
                
        except Exception as e:
            scripts_info[script_name] = {
                "last_run": "Error",
                "status": "Error", 
                "description": info["description"]
            }
    
    return scripts_info

@st.cache_data(ttl=30)
def get_recent_signals(hours=24):
    """Get recent signals"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cutoff_time = datetime.now() - timedelta(hours=hours)
        query = """
        SELECT pair, signal, confidence, risk_score, timestamp, acted_upon
        FROM signals 
        WHERE timestamp >= ? 
        ORDER BY timestamp DESC
        LIMIT 50
        """
        df = pd.read_sql(query, conn, params=[cutoff_time.isoformat()])
        conn.close()
        
        # Ensure we return a DataFrame with expected columns even if empty
        if df.empty:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['pair', 'signal', 'confidence', 'risk_score', 'timestamp', 'acted_upon'])
        
        return df
    except Exception as e:
        # Return empty DataFrame with expected columns on error
        return pd.DataFrame(columns=['pair', 'signal', 'confidence', 'risk_score', 'timestamp', 'acted_upon'])

def get_status_color(status):
    """Get color for status indicators"""
    colors = {
        'green': '🟢',
        'yellow': '🟡', 
        'red': '🔴'
    }
    return colors.get(status, '🟡')

def format_currency_delta(current_value, initial_value):
    """Format currency delta with proper color coding"""
    delta = current_value - initial_value
    if delta >= 0:
        return f"+${delta:,.2f}"
    else:
        return f"-${abs(delta):,.2f}"

def format_colored_metric(value, is_currency=True, show_sign=True):
    """Format metric with appropriate color based on positive/negative value"""
    if pd.isna(value) or value == 0:
        color_class = "neutral-value"
        if is_currency:
            display_value = "$0.00"
        else:
            display_value = "0.00%"
    elif value > 0:
        color_class = "positive-value"
        if is_currency:
            display_value = f"+${value:,.2f}" if show_sign else f"${value:,.2f}"
        else:
            display_value = f"+{value:.2f}%" if show_sign else f"{value:.2f}%"
    else:
        color_class = "negative-value"
        if is_currency:
            display_value = f"-${abs(value):,.2f}"
        else:
            display_value = f"{value:.2f}%"
    
    return f'<span class="{color_class}">{display_value}</span>'

def main():
    # Title
    st.markdown("# 🚀 INSTITUTIONAL TRADING ENGINE DASHBOARD")
    st.markdown("---")
    
    # Auto-refresh
    placeholder = st.empty()
    
    with placeholder.container():
        # System Status Section
        # Emergency Kill Switch
        col_title, col_kill = st.columns([3, 1])
        with col_title:
            st.markdown("## 🔧 System Health Status")
        with col_kill:
            if st.button("🚨 EMERGENCY STOP", type="primary", help="Force stop trading engine and close all positions"):
                # Create confirmation dialog using session state
                st.session_state.show_kill_confirmation = True

        # Kill switch confirmation dialog
        if st.session_state.get('show_kill_confirmation', False):
            with st.container():
                st.error("⚠️ EMERGENCY SHUTDOWN CONFIRMATION")
                st.write("**This will:**")
                st.write("- Immediately stop the trading engine")
                st.write("- Force close all open positions")
                st.write("- Activate emergency circuit breakers")
        
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("✅ YES, STOP NOW", type="primary"):
                        if emergency_shutdown():
                            st.success("🛑 Emergency shutdown initiated!")
                            st.balloons()
                        else:
                            st.error("❌ Emergency shutdown failed!")
                        st.session_state.show_kill_confirmation = False
                        st.rerun()
        
                with col2:
                    if st.button("❌ Cancel"):
                        st.session_state.show_kill_confirmation = False
                        st.rerun()
        
                with col3:
                    st.write("")  # Spacer
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Check system status
        db_status = 'green' if check_database_connection() else 'red'
        api_status = 'green' if check_api_access() else 'red'
        
        # Check engine status based on recent equity updates
        latest_equity = get_latest_equity()
        engine_status = 'red'
        data_quality = 'red'
        
        if latest_equity is not None:
            last_update_time = pd.to_datetime(latest_equity['timestamp'])
            time_diff = datetime.now() - last_update_time.replace(tzinfo=None)
            
            if time_diff.total_seconds() < 300:  # Less than 5 minutes
                engine_status = 'green'
                data_quality = 'green'
            elif time_diff.total_seconds() < 900:  # Less than 15 minutes
                engine_status = 'yellow'
                data_quality = 'yellow'
        
        # Get all status checks
        engine_status, engine_value = check_trading_engine_status()
        api_status, api_value = check_kraken_api_status()
        model_status, model_value = check_model_status()
        cb_status, cb_value = check_circuit_breakers()
        cash_status, cash_value = check_cash_status()
        open_positions_count = get_open_positions_count()
        
        with col1:
            st.metric(
                label=f"{get_status_color(engine_status)} Trading Engine",
                value=engine_value
            )
            st.metric(
                label=f"{get_status_color(db_status)} Database",
                value="Connected" if db_status == 'green' else "Offline"
            )
        
        with col2:
            st.metric(
                label=f"{get_status_color(api_status)} Kraken API",
                value=api_value
            )
            st.metric(
                label=f"{get_status_color(data_quality)} Data Quality",
                value="Fresh" if data_quality == 'green' else "Stale" if data_quality == 'yellow' else "Old"
            )
        
        with col3:
            st.metric(
                label=f"{get_status_color(model_status)} Model Status",
                value=model_value
            )
            st.metric(
                label=f"{get_status_color(cb_status)} Circuit Breakers",
                value=cb_value
            )
        
        with col4:
            st.metric(
                label="📍 Open Positions",
                value=str(open_positions_count),
                help="Currently open trading positions"
            )
            st.metric(
                label=f"{get_status_color(cash_status)} Cash Status",
                value=cash_value
            )
        
        # Script Execution Times Section
        st.markdown("### 🕒 Script Execution Status")
        
        script_times = get_script_execution_times()
        
        # Create columns for script status display
        if script_times:
            script_cols = st.columns(len(script_times))
            
            for idx, (script_name, info) in enumerate(script_times.items()):
                with script_cols[idx]:
                    # Determine status color based on how recent the execution was
                    last_run = info.get('last_run', 'Never')
                    status = info.get('status', 'Unknown')
                    
                    if last_run == 'Never' or last_run == 'Error':
                        status_color = 'red'
                        status_icon = '🔴'
                    elif status == 'Manual':
                        status_color = 'yellow'
                        status_icon = '🟡'
                    else:
                        # Check if automated script ran recently (within 24 hours)
                        try:
                            if last_run != 'Unknown':
                                last_time = datetime.strptime(last_run, '%Y-%m-%d %H:%M')
                                time_diff = datetime.now() - last_time
                                if time_diff.total_seconds() < 86400:  # 24 hours
                                    status_color = 'green'
                                    status_icon = '🟢'
                                else:
                                    status_color = 'yellow'
                                    status_icon = '🟡'
                            else:
                                status_color = 'yellow'
                                status_icon = '🟡'
                        except:
                            status_color = 'yellow'
                            status_icon = '🟡'
                    
                    st.metric(
                        label=f"{status_icon} {script_name}",
                        value=last_run,
                        help=info.get('description', 'No description available')
                    )
        
        st.markdown("---")
        
        # Performance Metrics Section
        st.markdown("## 📊 Performance Metrics")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        # Get performance data
        daily_trades = get_daily_trades()
        daily_activities = get_daily_trading_activities()
        
        # Get total trades for all-time metrics
        try:
            conn = sqlite3.connect(DB_PATH)
            total_trades_df = pd.read_sql("SELECT * FROM trades", conn)
            conn.close()
            total_pnl = total_trades_df['net_pnl'].sum() if not total_trades_df.empty else 0
            total_completed_trades = len(total_trades_df)
        except:
            total_pnl = 0
            total_completed_trades = 0
        
        if latest_equity is not None:
            current_equity = latest_equity['total_equity']
            initial_capital = 15000  # From your config
            delta_amount = current_equity - initial_capital
            total_return = delta_amount / initial_capital * 100
            
            with col1:
                # Fixed equity display with proper delta handling
                st.metric(
                    label="💰 Current Equity",
                    value=f"${current_equity:,.2f}",
                    delta=delta_amount,  # Let streamlit handle the formatting and color
                    delta_color="normal"  # Use normal color rules (red for negative, green for positive)
                )
            
            with col2:
                st.metric(
                    label="📈 Total Return",
                    value=f"{total_return:.2f}%",
                    delta=f"{total_return:.2f}%",
                    delta_color="normal"
                )
            
            with col3:
                portfolio_var = latest_equity.get('portfolio_var', 0) * 100
                var_color = "normal" if portfolio_var < 2.5 else "inverse"  # Red if VaR too high
                st.metric(
                    label="⚠️ Portfolio VaR",
                    value=f"{portfolio_var:.2f}%",
                    delta=None,
                    help="Risk measure - lower is better"
                )
        
        # Use trading activities for count, completed trades for P&L
        total_activities = len(daily_activities) if not daily_activities.empty else 0
        
        if not daily_trades.empty:
            winning_trades = len(daily_trades[daily_trades['net_pnl'] > 0])
            win_rate = winning_trades / len(daily_trades) * 100 if len(daily_trades) > 0 else 0
            daily_pnl = daily_trades['net_pnl'].sum()
            
            with col4:
                st.metric(
                    label="🎯 Completed Trades",
                    value=f"Today: {len(daily_trades)} | Total: {total_completed_trades}",
                    delta=f"Win Rate: {win_rate:.1f}%" if len(daily_trades) > 0 else "No trades today",
                    help="Completed trades = full open + close cycle"
                )
            
            with col5:
                st.metric(
                    label="💵 Today's P&L",
                    value=f"${daily_pnl:.2f}",
                    delta=daily_pnl,
                    delta_color="normal"
                )
            
            with col6:
                st.metric(
                    label="💰 Total P&L",
                    value=f"${total_pnl:.2f}",
                    delta=total_pnl,
                    delta_color="normal"
                )
        else:
            with col4:
                st.metric(
                    label="🎯 Today's Activities", 
                    value=str(total_activities),
                    delta="No completed trades yet",
                    help="All trading activities: signals, opens, closes"
                )
            with col5:
                st.metric(label="💵 Today's P&L", value="$0.00")
        
        st.markdown("---")
        
        # Charts Section
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Time period selector for equity curve
            time_periods = {
                "24 Hours": 24,
                "48 Hours": 48, 
                "4 Days (96h)": 96,
                "8 Days (192h)": 192,
                "2 Weeks": 336,
                "1 Month": 720
            }
            
            selected_period = st.selectbox(
                "📈 Equity Curve Time Period:",
                options=list(time_periods.keys()),
                index=0,  # Default to 24 hours
                key="equity_period_selector"
            )
            
            hours = time_periods[selected_period]
            equity_df = get_equity_history(hours)
            
            if not equity_df.empty:
                # Calculate performance statistics for the period
                start_equity = equity_df['total_equity'].iloc[0]
                end_equity = equity_df['total_equity'].iloc[-1]
                max_equity = equity_df['total_equity'].max()
                min_equity = equity_df['total_equity'].min()
                period_return = (end_equity - start_equity) / start_equity * 100
                max_drawdown = (max_equity - min_equity) / max_equity * 100
                
                # Create the chart
                fig = go.Figure()
                
                # Main equity line
                fig.add_trace(go.Scatter(
                    x=equity_df['timestamp'],
                    y=equity_df['total_equity'],
                    mode='lines',
                    name='Total Equity',
                    line=dict(color='#00ff88', width=3),
                    hovertemplate='<b>%{y:$,.2f}</b><br>%{x}<extra></extra>'
                ))
                
                # Add initial capital reference line
                fig.add_hline(
                    y=15000, 
                    line_dash="dash", 
                    line_color="orange", 
                    opacity=0.7,
                    annotation_text="Initial Capital ($15,000)",
                    annotation_position="top right"
                )
                
                # Add max equity line
                fig.add_hline(
                    y=max_equity,
                    line_dash="dot",
                    line_color="green",
                    opacity=0.5,
                    annotation_text=f"Peak: ${max_equity:,.2f}",
                    annotation_position="top left"
                )
                
                # Add performance stats as annotations
                fig.add_annotation(
                    x=0.02, y=0.98,
                    xref="paper", yref="paper",
                    text=f"<b>Period Stats:</b><br>"
                         f"Return: {period_return:+.2f}%<br>"
                         f"Max Drawdown: {max_drawdown:.2f}%<br>"
                         f"Peak: ${max_equity:,.2f}<br>"
                         f"Low: ${min_equity:,.2f}",
                    showarrow=False,
                    align="left",
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor="white",
                    borderwidth=1,
                    font=dict(color="white", size=10)
                )
                
                # Chart layout
                fig.update_layout(
                    title=f"Equity Curve - {selected_period}",
                    height=350,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=40, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    title_font=dict(size=16, color='white')
                )
                
                # Axis styling
                fig.update_xaxes(
                    gridcolor='#444',
                    showgrid=True,
                    title="Time"
                )
                fig.update_yaxes(
                    gridcolor='#444',
                    showgrid=True,
                    title="Equity ($)",
                    tickformat="$,.0f"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance summary below chart
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Period Return", f"{period_return:+.2f}%")
                with col_b:
                    st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                with col_c:
                    st.metric("Peak Equity", f"${max_equity:,.2f}")
                with col_d:
                    st.metric("Period Low", f"${min_equity:,.2f}")
                    
            else:
                st.info(f"No equity data available for {selected_period.lower()}")
        
        with chart_col2:
            # Time period selector for signal quality (same as equity curve)
            signal_periods = {
                "24 Hours": 24,
                "48 Hours": 48, 
                "4 Days (96h)": 96,
                "8 Days (192h)": 192,
                "2 Weeks": 336,
                "1 Month": 720
            }
            
            selected_signal_period = st.selectbox(
                "🎯 Signal Quality Time Period:",
                options=list(signal_periods.keys()),
                index=0,  # Default to 24 hours
                key="signal_period_selector"
            )
            
            signal_hours = signal_periods[selected_signal_period]
            signals_df = get_recent_signals(signal_hours)
            
            if not signals_df.empty and 'confidence' in signals_df.columns:
                # Create signal quality distribution
                fig = px.histogram(
                    signals_df, 
                    x='confidence',
                    nbins=20,
                    title=f'Signal Confidence Distribution - {selected_signal_period}',
                    color_discrete_sequence=['#00ff88']
                )
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=40, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    title_font=dict(size=16, color='white')
                )
                fig.update_xaxes(gridcolor='#444', title="Confidence Score")
                fig.update_yaxes(gridcolor='#444', title="Count")
                st.plotly_chart(fig, use_container_width=True)
            
                # Signal stats below chart
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    avg_confidence = signals_df['confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                with col_b:
                    high_conf_signals = len(signals_df[signals_df['confidence'] > 0.7])
                    st.metric("High Conf (>0.7)", str(high_conf_signals))
                with col_c:
                    acted_signals = len(signals_df[signals_df['acted_upon'] == 1]) if 'acted_upon' in signals_df.columns else 0
                    st.metric("Acted Upon", str(acted_signals))
            else:
                st.info(f"No signal data available for {selected_signal_period.lower()}")
        
        st.markdown("---")
        
        # Positions and Market Regimes
        st.markdown("## 💼 Market Regimes & Position Status")
        
        regimes = get_market_regimes()
        regime_models = get_actual_regime_models()
        available_models = get_model_file_status()
        recent_signals = get_recent_signals(24)
        
        # Create enhanced regime status table
        regime_data = []
        for pair in PAIRS:
            # Get actual regime model info
            model_info = regime_models.get(pair, {})
            current_model = model_info.get('model', f"{pair}-unknown")
            regime_type = model_info.get('regime_type', 'unknown')
            confidence = model_info.get('confidence', 'N/A')
            last_update = model_info.get('last_update', 'Unknown')
            source = model_info.get('source', 'Unknown')
            status = model_info.get('status', 'Unknown')
            
            # Check if model file exists
            available_regimes = available_models.get(pair, [])
            model_exists = regime_type in available_regimes
            model_status = "✅ Available" if model_exists else "❌ Missing"
            
            # Get recent signal info - WITH PROPER ERROR HANDLING
            recent_activity = "No recent signals"  # Default value
        
            # Check if recent_signals DataFrame has data and required columns
            if not recent_signals.empty and 'pair' in recent_signals.columns:
                try:
                    pair_signals = recent_signals[recent_signals['pair'] == pair]
                    if not pair_signals.empty:
                        latest_signal = pair_signals.iloc[0]
                        signal_type = latest_signal['signal']
                        signal_confidence = latest_signal['confidence']
                        recent_activity = f"{signal_type.upper()} ({signal_confidence:.3f})"
                except Exception as e:
                    recent_activity = "Error reading signals"
        
        # Format regime type for display
            
            # Format regime type for display
            regime_display = regime_type.replace('_', ' ').title()
            
            # Format last update time
            try:
                if last_update not in ['Unknown', 'Error']:
                    update_time = pd.to_datetime(last_update)
                    formatted_time = update_time.strftime('%H:%M:%S')
                else:
                    formatted_time = last_update
            except:
                formatted_time = 'Error'
            
            regime_data.append({
                "Pair": pair.upper(),
                "Active Model": current_model,
                "Regime Type": regime_display,
                "Model Status": model_status,
                "Engine Status": status,
                "Confidence/Vol": confidence,
                "Last Update": formatted_time,
                "Data Source": source,
                "Recent Signal": recent_activity
            })
        
        regime_df = pd.DataFrame(regime_data)
        
        # Display the enhanced table
        st.dataframe(
            regime_df, 
            use_container_width=True,
            column_config={
                "Model Status": st.column_config.TextColumn(
                    help="Whether the model file exists and is available"
                ),
                "Confidence": st.column_config.TextColumn(
                    help="Model confidence or data quality indicator"
                ),
                "Data Source": st.column_config.TextColumn(
                    help="Source of regime information"
                )
            }
        )
        
        # Recent Trades Table
        if not daily_trades.empty:
            st.markdown("### 📋 Completed Trades (Today)")
            
            # Format trades for display
            display_trades = daily_trades[['timestamp', 'pair', 'side', 'entry_price', 'exit_price', 'net_pnl', 'exit_reason']].copy()
            display_trades['timestamp'] = pd.to_datetime(display_trades['timestamp']).dt.strftime('%H:%M:%S')
            display_trades['net_pnl'] = display_trades['net_pnl'].apply(lambda x: f"${x:.2f}")
            display_trades['entry_price'] = display_trades['entry_price'].apply(lambda x: f"${x:.4f}")
            display_trades['exit_price'] = display_trades['exit_price'].apply(lambda x: f"${x:.4f}")
            
            st.dataframe(display_trades, use_container_width=True)
        
        # Trading Activities Table
        if not daily_activities.empty and len(daily_activities) > len(daily_trades):
            st.markdown("### 🔄 All Trading Activities (Today)")
            
            # Format activities for display
            display_activities = daily_activities[['timestamp', 'pair', 'activity_type', 'side']].copy()
            display_activities['timestamp'] = pd.to_datetime(display_activities['timestamp']).dt.strftime('%H:%M:%S')
            display_activities['activity_type'] = display_activities['activity_type'].str.replace('_', ' ').str.title()
            display_activities = display_activities.sort_values('timestamp', ascending=False)
            
            st.dataframe(display_activities.head(20), use_container_width=True)
            
        if daily_trades.empty and daily_activities.empty:
            st.info("No trading activity today")
        
        # Live Trading Analysis Section
        st.markdown("---")
        st.markdown("## 📊 Live Trading Analysis")

        col_refresh, col_auto = st.columns([1, 3])
        with col_refresh:
            if st.button("🔄 Run Analysis", help="Analyze current trading performance"):
                st.cache_data.clear()
                st.rerun()

        with col_auto:
            auto_refresh = st.checkbox("Auto-refresh analysis", value=True, help="Automatically refresh analysis data")

        if auto_refresh or st.session_state.get('force_analysis', False):
            with st.spinner("🔍 Analyzing trading performance..."):
                analyzer_results = run_live_analyzer()
                display_analyzer_results(analyzer_results)
        else:
            st.info("👆 Click 'Run Analysis' to see detailed trading performance")

        st.session_state.force_analysis = False
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
        
        with col2:
            st.markdown(f"**Database:** {DB_PATH}")
        
        with col3:
            col3a, col3b = st.columns(2)
            with col3a:
                if st.button("🔄 Refresh Data"):
                    st.cache_data.clear()
                    st.rerun()
            with col3b:
                if st.button("📊 Run Analysis"):
                    st.session_state.force_analysis = True
                    st.cache_data.clear()
                    st.rerun()

# Auto-refresh setup
if __name__ == "__main__":
    # Add auto-refresh every 30 seconds
    st.markdown("""
    <script>
        setTimeout(function(){
            window.parent.location.reload();
        }, 30000);
    </script>
    """, unsafe_allow_html=True)
    
    main()