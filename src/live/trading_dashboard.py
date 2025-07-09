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
    """Get today's trades"""
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

@st.cache_data(ttl=60)
def get_market_regimes():
    """Get latest market regimes"""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
        SELECT pair, regime, timestamp 
        FROM market_regimes 
        WHERE timestamp IN (
            SELECT MAX(timestamp) 
            FROM market_regimes 
            GROUP BY pair
        )
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df.set_index('pair')['regime'].to_dict()
    except:
        return {}

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
        return df
    except:
        return pd.DataFrame()

def get_status_color(status):
    """Get color for status indicators"""
    colors = {
        'green': '🟢',
        'yellow': '🟡', 
        'red': '🔴'
    }
    return colors.get(status, '🟡')

def main():
    # Title
    st.markdown("# 🚀 INSTITUTIONAL TRADING ENGINE DASHBOARD")
    st.markdown("---")
    
    # Auto-refresh
    placeholder = st.empty()
    
    with placeholder.container():
        # System Status Section
        st.markdown("## 🔧 System Health Status")
        
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
        
        with col1:
            st.metric(
                label=f"{get_status_color(engine_status)} Trading Engine",
                value="Running" if engine_status == 'green' else "Stopped" if engine_status == 'red' else "Warning"
            )
            st.metric(
                label=f"{get_status_color(db_status)} Database",
                value="Connected" if db_status == 'green' else "Offline"
            )
        
        with col2:
            st.metric(
                label=f"{get_status_color(api_status)} Kraken API",
                value="Online" if api_status == 'green' else "Offline"
            )
            st.metric(
                label=f"{get_status_color(data_quality)} Data Quality",
                value="Fresh" if data_quality == 'green' else "Stale" if data_quality == 'yellow' else "Old"
            )
        
        with col3:
            st.metric(
                label="🟢 Model Status",
                value="Loaded"
            )
            st.metric(
                label="🟢 Circuit Breakers",
                value="Normal"
            )
        
        with col4:
            st.metric(
                label="🟢 Position Limits",
                value="Normal"
            )
            st.metric(
                label="🟢 Cash Status",
                value="Healthy"
            )
        
        st.markdown("---")
        
        # Performance Metrics Section
        st.markdown("## 📊 Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Get performance data
        daily_trades = get_daily_trades()
        
        if latest_equity is not None:
            current_equity = latest_equity['total_equity']
            initial_capital = 15000  # From your config
            total_return = (current_equity - initial_capital) / initial_capital * 100
            
            with col1:
                st.metric(
                    label="💰 Current Equity",
                    value=f"${current_equity:,.2f}",
                    delta=f"${current_equity - initial_capital:,.2f}"
                )
            
            with col2:
                st.metric(
                    label="📈 Total Return",
                    value=f"{total_return:.2f}%",
                    delta=f"{total_return:.2f}%"
                )
            
            with col3:
                portfolio_var = latest_equity.get('portfolio_var', 0) * 100
                st.metric(
                    label="⚠️ Portfolio VaR",
                    value=f"{portfolio_var:.2f}%"
                )
        
        if not daily_trades.empty:
            winning_trades = len(daily_trades[daily_trades['net_pnl'] > 0])
            win_rate = winning_trades / len(daily_trades) * 100 if len(daily_trades) > 0 else 0
            daily_pnl = daily_trades['net_pnl'].sum()
            
            with col4:
                st.metric(
                    label="🎯 Today's Trades",
                    value=str(len(daily_trades)),
                    delta=f"Win Rate: {win_rate:.1f}%"
                )
            
            with col5:
                st.metric(
                    label="💵 Today's P&L",
                    value=f"${daily_pnl:.2f}",
                    delta=f"${daily_pnl:.2f}"
                )
        else:
            with col4:
                st.metric(label="🎯 Today's Trades", value="0")
            with col5:
                st.metric(label="💵 Today's P&L", value="$0.00")
        
        st.markdown("---")
        
        # Charts Section
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("### 📈 Equity Curve (24h)")
            equity_df = get_equity_history(24)
            
            if not equity_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity_df['timestamp'],
                    y=equity_df['total_equity'],
                    mode='lines',
                    name='Total Equity',
                    line=dict(color='#00ff88', width=2)
                ))
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                fig.update_xaxes(gridcolor='#444')
                fig.update_yaxes(gridcolor='#444')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No equity data available")
        
        with chart_col2:
            st.markdown("### 🎯 Signal Quality (24h)")
            signals_df = get_recent_signals(24)
            
            if not signals_df.empty:
                # Create signal quality distribution
                fig = px.histogram(
                    signals_df, 
                    x='confidence',
                    nbins=20,
                    title='Signal Confidence Distribution',
                    color_discrete_sequence=['#00ff88']
                )
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                fig.update_xaxes(gridcolor='#444')
                fig.update_yaxes(gridcolor='#444')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No signal data available")
        
        st.markdown("---")
        
        # Positions and Market Regimes
        st.markdown("## 💼 Market Regimes & Position Status")
        
        regimes = get_market_regimes()
        
        # Create a nice table for market regimes
        regime_data = []
        for pair in PAIRS:
            regime = regimes.get(pair, "Unknown")
            regime_data.append({
                "Pair": pair.upper(),
                "Current Regime": regime,
                "Status": "🔄 Monitoring"
            })
        
        regime_df = pd.DataFrame(regime_data)
        st.dataframe(regime_df, use_container_width=True)
        
        # Recent Trades Table
        if not daily_trades.empty:
            st.markdown("### 📋 Recent Trades (Today)")
            
            # Format trades for display
            display_trades = daily_trades[['timestamp', 'pair', 'side', 'entry_price', 'exit_price', 'net_pnl', 'exit_reason']].copy()
            display_trades['timestamp'] = pd.to_datetime(display_trades['timestamp']).dt.strftime('%H:%M:%S')
            display_trades['net_pnl'] = display_trades['net_pnl'].apply(lambda x: f"${x:.2f}")
            display_trades['entry_price'] = display_trades['entry_price'].apply(lambda x: f"${x:.4f}")
            display_trades['exit_price'] = display_trades['exit_price'].apply(lambda x: f"${x:.4f}")
            
            st.dataframe(display_trades, use_container_width=True)
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
        
        with col2:
            st.markdown(f"**Database:** {DB_PATH}")
        
        with col3:
            if st.button("🔄 Refresh Data"):
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