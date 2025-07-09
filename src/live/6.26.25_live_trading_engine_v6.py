"""
Advanced Institutional-Style Cryptocurrency Live Trading Engine
Based on successful backtest strategy with enhanced portfolio management

Features:
- Pair-specific configurations and professional risk management
- Market regime detection and dynamic position sizing
- Advanced technical indicator filtering and portfolio analytics
- Database logging and performance tracking
- Circuit breaker system for risk management
"""

import os
import time
import pandas as pd
import numpy as np
import ccxt
import joblib
import dotenv
import ta
from datetime import datetime, timedelta
from collections import defaultdict, deque
import random
import sqlite3
import json
import logging
import sys
import traceback
from scipy import stats
import xgboost as xgb

# === Configure Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("institutional_trading_engine.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("institutional_trading_engine")

def safe_float(value):
    """Safely convert any numeric value to Python float, handling edge cases"""
    if value is None:
        return None
    
    # Handle pandas/numpy NaN
    if pd.isna(value):
        return None
    
    # Handle numpy types
    if hasattr(value, 'item'):  # numpy scalars have .item() method
        value = value.item()
    
    # Handle infinity
    if np.isinf(value):
        return None
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def safe_int(value):
    """Safely convert any value to Python int"""
    if value is None:
        return None
    
    if pd.isna(value):
        return None
    
    if hasattr(value, 'item'):
        value = value.item()
    
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def safe_str(value):
    """Safely convert any value to string"""
    if value is None:
        return None
    
    if pd.isna(value):
        return None
    
    return str(value)

# === Database Setup ===
DB_FILE = "institutional_trading_history.db"

def setup_database(reset=False):
    """Initialize the SQLite database for trade logging and performance tracking"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Drop existing tables if reset is True
    if reset:
        cursor.execute("DROP TABLE IF EXISTS trades")
        cursor.execute("DROP TABLE IF EXISTS signals") 
        cursor.execute("DROP TABLE IF EXISTS equity")
        cursor.execute("DROP TABLE IF EXISTS circuit_breakers")
        cursor.execute("DROP TABLE IF EXISTS market_regimes")
        logger.info("Reset database: All existing tables dropped")
    
    # Create trades table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        pair TEXT,
        side TEXT,
        entry_price REAL,
        exit_price REAL,
        quantity REAL,
        position_value REAL,
        leverage REAL,
        entry_time TEXT,
        exit_time TEXT,
        pnl REAL,
        net_pnl REAL,
        fees REAL,
        hold_time REAL,
        confidence REAL,
        risk_score REAL,
        exit_reason TEXT,
        market_regime TEXT,
        risk_tier TEXT,
        is_paper INTEGER,
        partial_exit INTEGER
    )
    ''')
    
    # Create signals table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        pair TEXT,
        signal TEXT,
        price REAL,
        confidence REAL,
        risk_score REAL,
        rsi REAL,
        macd REAL,
        macd_signal REAL,
        vwap REAL,
        bb_upper REAL,
        bb_lower REAL,
        ema_9 REAL,
        ema_21 REAL,
        trend_strength REAL,
        market_regime TEXT,
        acted_upon INTEGER
    )
    ''')
    
    # Create equity table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS equity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        total_equity REAL,
        cash REAL,
        positions_value REAL,
        unrealized_pnl REAL,
        portfolio_var REAL
    )
    ''')
    
    # Create circuit breaker table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS circuit_breakers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        type TEXT,
        reason TEXT,
        value REAL,
        threshold REAL,
        pair TEXT,
        is_active INTEGER
    )
    ''')
    
    # Create market regimes table to track regime transitions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_regimes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        pair TEXT,
        regime TEXT,
        volatility REAL,
        correlation REAL,
        trend_strength REAL
    )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database setup complete")

# === Load API Keys from .env ===
dotenv.load_dotenv()
api_key = os.getenv("KRAKEN_API_KEY")
api_secret = os.getenv("KRAKEN_API_SECRET")

kraken = ccxt.kraken({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
})

# === Global Settings ===

# === REGIME DETECTION PARAMETERS (from successful backtest) ===
TREND_WINDOW_DAYS = 60      # 60 days for trend detection
VOLATILITY_WINDOW_DAYS = 30 # 30 days for volatility calculation

# Volatility thresholds (based on daily returns)
HIGH_VOLATILITY_THRESHOLD = 0.035   # 3.5% daily volatility
LOW_VOLATILITY_THRESHOLD = 0.015    # 1.5% daily volatility

# Trend thresholds (based on 60-day price change)
BULL_TREND_THRESHOLD = 0.20         # 20% gain over 60 days
BEAR_TREND_THRESHOLD = -0.20        # -20% loss over 60 days

PAIRS = {
    "btcusdt": "BTC/USDT",      # maps to XBTUSDT
    "ethusdt": "ETH/USDT", 
    "solusdt": "SOL/USDT",
    "xrpusdt": "XRP/USDT",
    "adausdt": "ADA/USDT",
    "ltcusdt": "LTC/USDT",
    "dotusdt": "DOT/USDT",
    "linkusdt": "LINK/USDT",    # ✅ Add back (was in backtest)
    "avaxusdt": "AVAX/USDT",    # ✅ Add back (was in backtest)
    # Remove these - no models exist:
    # "xmrusdt": "XMR/USDT",
    # "dogeusdt": "DOGE/USDT",
}

# === Capital and Portfolio Settings ===
initial_capital = 15000
max_concurrent_trades = 5  # EMERGENCY FIX: Reduce from 5 to 2 to limit replacement frequency

# === Professional Portfolio Risk Management Settings ===
# VaR (Value at Risk) settings
MAX_PORTFOLIO_VAR = 0.035  # Maximum portfolio Value-at-Risk (as fraction of capital)
VAR_CONFIDENCE_LEVEL = 0.95  # VaR confidence level (95%)

# Position sizing and leverage
BASE_POSITION_SIZE_LIMIT = 0.25  # Maximum position size as fraction of capital
MAX_CONCENTRATED_CAPITAL_PCT = 0.60  # Maximum percentage of capital in similar assets
MAX_TOTAL_LEVERAGE = 12  # Maximum portfolio-wide leverage
TARGET_PORTFOLIO_SHARPE = 2.0  # Target portfolio Sharpe ratio

# Signal quality thresholds
risk_score_threshold = 0.42  # Increased from 0.35 to be more selective
# Updated constants for better replacement behavior
MIN_REPLACEMENT_IMPROVEMENT = 1.25  # Increased from 1.10 to 1.25 (25% improvement required)
MIN_HOLD_TIME_FOR_REPLACEMENT = 30  # Minimum 30 minutes before allowing replacement
EXCEPTIONAL_SIGNAL_REPLACEMENT_THRESHOLD = 0.95  # Very high confidence needed for quick replacement
EXCEPTIONAL_SIGNAL_THRESHOLD = 0.85  # Threshold for exceptional signals (used in position sizing)

# Pair-specific risk thresholds based on performance
pair_risk_thresholds = {
    "dotusdt": 0.35,   # Excellent performer in backtest
    "adausdt": 0.35,   # Excellent performer in backtest
    "ethusdt": 0.40,
    "xrpusdt": 0.45,   # Top performer but higher threshold
    "ltcusdt": 0.45,
    "linkusdt": 0.45,  # ✅ ADD
    "solusdt": 0.45,
    "btcusdt": 0.40,
    "avaxusdt": 0.20,  # ✅ ADD
    # REMOVE xmrusdt and dogeusdt entries
}

# Risk tier settings (for dynamic position sizing)
RISK_TIERS = {
    "high": {"leverage_cap": 5, "max_position_pct": 0.20, "confidence_threshold": 0.85, "score_threshold": 0.70},
    "medium": {"leverage_cap": 3, "max_position_pct": 0.15, "confidence_threshold": 0.70, "score_threshold": 0.55},
    "low": {"leverage_cap": 2, "max_position_pct": 0.10, "confidence_threshold": 0.60, "score_threshold": 0.45},
    "minimal": {"leverage_cap": 1, "max_position_pct": 0.05, "confidence_threshold": 0.50, "score_threshold": 0.35}
}

# Market regime detection
VOLATILITY_LOOKBACK_WINDOW = 20  # Window for volatility calculation
CORRELATION_LOOKBACK_WINDOW = 60  # Window for correlation calculation
HIGH_VOLATILITY_THRESHOLD = 0.03  # Daily volatility threshold for high volatility regime

# === Technical Indicators Settings ===
# Adjusted indicator weights based on performance
use_rsi_filter = True
use_macd_filter = True
use_vwap_filter = True
use_bbands_filter = True
use_ema_filter = True  # Added EMA crossover filter
use_trend_filter = True  # New filter for trend strength

rsi_weight = 0.4         # Decreased from 0.5
macd_weight = 0.8        # Increased from 0.7
vwap_weight = 0.8        # Increased from 0.7
bbands_weight = 0.4      # Decreased from 0.5
ema_weight = 0.8         # Increased from 0.6
trend_strength_weight = 0.6  # New indicator

# Relaxed RSI thresholds for entry
rsi_oversold = 40  # Changed from 30
rsi_overbought = 60  # Changed from 70

# === Fee Structure ===
# === REALISTIC KRAKEN FEE STRUCTURE ===
# Trading fees (actual Kraken rates for $0+ volume tier)
TRADING_FEES = {
    "maker_fee": 0.0025,  # 0.25% - when you add liquidity
    "taker_fee": 0.0040,  # 0.40% - when you take liquidity (use this for market orders)
}

# Margin fees per Kraken documentation (use conservative high-end rates)
MARGIN_FEES = {
    # BTC gets preferential rates
    "BTC": {
        "open_fee": 0.0002,      # 0.02% (high end of 0.01-0.02% range)
        "rollover_fee": 0.0002,  # 0.02% per 4-hour period
    },
    # All other cryptos
    "DEFAULT": {
        "open_fee": 0.0004,      # 0.04% (high end of 0.02-0.04% range)
        "rollover_fee": 0.0004,  # 0.04% per 4-hour period
    },
    "rollover_period_hours": 4,  # Kraken charges every 4 hours
}

# Exact Kraken maximum leverage limits
KRAKEN_MAX_LEVERAGE = {
    "btcusdt": 5,     # Bitcoin: up to 5x
    "ethusdt": 5,     # Ethereum: up to 5x
    "solusdt": 3,     # Solana: up to 3x
    "xrpusdt": 5,     # XRP: up to 5x
    "adausdt": 3,     # Cardano: up to 3x
    "ltcusdt": 3,     # Litecoin: up to 3x
    "dotusdt": 3,     # Polkadot: up to 3x
    "linkusdt": 3,    # ✅ ADD Chainlink: up to 3x
    "avaxusdt": 3,    # ✅ ADD Avalanche: up to 3x
    # REMOVE xmrusdt and dogeusdt entries
}

# === Position Management ===
max_position_size = 3000
stop_loss_pct = 0.04  # Reduced from 5% to 4%
take_profit_pct_first_half = 0.025  # Exit 50% at 2.5% profit
take_profit_pct_full = 0.05  # Exit remaining at 5% profit
break_even_profit_pct = 0.02  # Move stop to break-even when profit reaches 2%
max_holding_minutes = 18 * 60  # Extended from 12 hours to 18 hours

# === Liquidity Management ===
# Dynamic volume percentage based on pair liquidity
volume_pct_per_pair = {
    "btcusdt": 0.10, "ethusdt": 0.08, "solusdt": 0.05,
    "xrpusdt": 0.05, "adausdt": 0.04,
    "ltcusdt": 0.05, "dotusdt": 0.04, "linkusdt": 0.03,  # ✅ ADD
    "avaxusdt": 0.02,  # ✅ ADD
    # REMOVE xmrusdt and dogeusdt entries
}

# === Leverage Settings ===
# Adjusted leverage settings based on performance
max_leverage_per_pair = KRAKEN_MAX_LEVERAGE.copy()
max_leverage_per_pair.update({
    "dotusdt": 4,    # Increased from 3
    "adausdt": 4,    # Increased from 3
    "ethusdt": 4,    # Decreased from 5
    "xrpusdt": 3,    # Decreased from 5
    "ltcusdt": 2,    # Decreased from 3
    "linkusdt": 2,   # ✅ ADD
    "solusdt": 2,    # Decreased from 3
    "btcusdt": 4,    # Conservative for BTC
    "avaxusdt": 2,   # ✅ ADD
    # REMOVE xmrusdt and dogeusdt entries
})

# === Dynamic capital allocation by pair performance ===
pair_allocation_weight = {
    "dotusdt": 1.5,   # Top performers get more allocation
    "adausdt": 1.5,
    "ethusdt": 1.0,   # Standard allocation
    "xrpusdt": 0.75,  # Reduce allocation for poor performers
    "ltcusdt": 0.75,
    "linkusdt": 0.75, # ✅ ADD
    "solusdt": 0.5,   # Significantly reduce allocation
    "btcusdt": 1.0,   # Standard allocation
    "avaxusdt": 0.8,  # ✅ ADD
    # REMOVE xmrusdt and dogeusdt entries
}

# === Slippage Settings ===
slippage_per_pair = {
    "btcusdt": 0.0002, "ethusdt": 0.0003, "solusdt": 0.0005,
    "xrpusdt": 0.0010, "adausdt": 0.0012,
    "ltcusdt": 0.0004, "dotusdt": 0.0010, "linkusdt": 0.0006,  # ✅ ADD
    "avaxusdt": 0.0020,  # ✅ ADD
    # REMOVE xmrusdt and dogeusdt entries
}

# === Time-Based Filters ===
# Hours when crypto tends to be more active (UTC)
active_hours = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23]

# === Circuit Breaker Settings ===
# Global circuit breaker settings
daily_loss_limit_pct = 0.05  # 5% daily loss limit
drawdown_pause_pct = 0.15    # Pause trading after 15% drawdown
pause_duration_minutes = 30  # 30 minute pause when a circuit breaker is triggered

# === Models and state management ===
REPLACEMENT_LOOKBACK_WINDOW = 50  # Number of recent signals to keep for replacement logic
open_positions = defaultdict(dict)
active_circuit_breakers = {}
signal_queues = defaultdict(deque)
trading_paused_until = None
partial_exits = {}  # Track positions with partial exits
signal_quality_history = deque(maxlen=REPLACEMENT_LOOKBACK_WINDOW)  # Recent signal quality history

# Global tracking
current_day = datetime.utcnow().date()
cash = initial_capital
total_equity_history = []
price_data = {}  # For tracking volatility and market regimes

# Daily statistics tracking
daily_stats = {
    "start_capital": initial_capital,
    "current_capital": initial_capital,
    "trades_today": 0,
    "winning_trades_today": 0,
    "losing_trades_today": 0,
    "avg_win_today": 0,
    "avg_loss_today": 0,
}

# === Helper Functions for Portfolio Management ===

def get_margin_fees_for_pair(pair):
    """Get margin fees for a specific trading pair"""
    if "btc" in pair.lower() or "xbt" in pair.lower():
        return MARGIN_FEES["BTC"]
    else:
        return MARGIN_FEES["DEFAULT"]

def calculate_realistic_fees(pair, position_value, entry_price, leverage, held_minutes):
    """
    Calculate realistic Kraken-style fees for a position using exact Kraken rates
    """
    # Get pair-specific margin fees
    margin_rates = get_margin_fees_for_pair(pair)
    
    # Trading fees (use taker rate for market orders in live trading)
    entry_fee = position_value * TRADING_FEES["taker_fee"]
    exit_fee = position_value * TRADING_FEES["taker_fee"]
    
    # Margin opening fee (charged once on position opening)
    margin_open_fee = position_value * margin_rates["open_fee"]
    
    # Rollover fees (charged every 4 hours)
    rollover_periods = max(1, int(held_minutes // (MARGIN_FEES["rollover_period_hours"] * 60)))
    total_position_value = position_value * leverage  # Full exposure for rollover calculation
    rollover_fee = rollover_periods * total_position_value * margin_rates["rollover_fee"]
    
    total_fees = entry_fee + exit_fee + margin_open_fee + rollover_fee
    
    return {
        "entry_fee": entry_fee,
        "exit_fee": exit_fee, 
        "margin_open_fee": margin_open_fee,
        "rollover_fee": rollover_fee,
        "total_fees": total_fees,
        "rollover_periods": rollover_periods,
        "fee_breakdown": f"Trading: ${entry_fee + exit_fee:.2f}, Margin: ${margin_open_fee:.2f}, Rollover: ${rollover_fee:.2f}"
    }

def calculate_position_sharpe(position_data, lookback=20):
    """Calculate the Sharpe ratio of a position based on recent performance"""
    if 'recent_returns' not in position_data or len(position_data['recent_returns']) < 5:
        return 0
    
    returns = position_data['recent_returns'][-lookback:]
    if not returns or np.std(returns) == 0:
        return 0
        
    return np.mean(returns) / np.std(returns) * np.sqrt(365 * 24 * 60)  # Annualized

def estimate_position_var(position, price_volatility, confidence=0.95):
    """Estimate Value-at-Risk for a position using parametric method"""
    position_value = position["position_value"]
    leverage = position["leverage"]
    
    # Z-score for the given confidence level (e.g., 1.645 for 95%)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    # Daily VaR = Position Size * Leverage * Volatility * Z-score
    daily_var = position_value * leverage * price_volatility * z
    
    return daily_var

def calculate_portfolio_var(positions, price_data, correlations=None):
    """Calculate portfolio Value-at-Risk considering correlations between assets"""
    if not positions:
        return 0
    
    # If correlations not provided, assume 0.5 correlation between all assets
    if correlations is None:
        correlations = np.ones((len(positions), len(positions))) * 0.5
        np.fill_diagonal(correlations, 1.0)
    
    # Extract position values and volatilities
    position_values = []
    position_vars = []
    
    for pair, position in positions.items():
        if not position:
            continue
            
        position_value = position["position_value"]
        leverage = position["leverage"]
        
        # Get volatility for this pair (default to 2% if not available)
        volatility = price_data.get(pair, {}).get("volatility", 0.02)
        
        position_values.append(position_value * leverage)
        position_vars.append(estimate_position_var(position, volatility))
    
    if not position_values:
        return 0
    
    # Calculate portfolio VaR using correlation matrix
    portfolio_var = 0
    for i in range(len(position_vars)):
        for j in range(len(position_vars)):
            portfolio_var += position_vars[i] * position_vars[j] * correlations[i][j]
    
    return np.sqrt(portfolio_var)

def detect_market_regime(df_recent, lookback=VOLATILITY_LOOKBACK_WINDOW):
    """
    Detect market regime - SAME AS SUCCESSFUL BACKTEST
    """
    if len(df_recent) < TREND_WINDOW_DAYS * 24 * 60:  # Not enough data
        return "range_normal_vol"  # Default regime
    
    df = df_recent.copy()
    
    # Calculate windows in minutes (assuming 1-minute data)
    trend_window_minutes = TREND_WINDOW_DAYS * 24 * 60
    vol_window_minutes = VOLATILITY_WINDOW_DAYS * 24 * 60
    
    # Calculate rolling trend (60-day price change)
    if len(df) >= trend_window_minutes:
        price_change_60d = (df['close'].iloc[-1] / df['close'].iloc[-trend_window_minutes] - 1)
    else:
        price_change_60d = 0
    
    # Calculate rolling volatility (30-day)
    if len(df) >= vol_window_minutes:
        returns = df['close'].pct_change().dropna()
        volatility_30d = returns.iloc[-vol_window_minutes:].std() * np.sqrt(1440)  # Daily volatility
    else:
        volatility_30d = 0.02  # Default
    
    # Determine trend direction
    if price_change_60d >= BULL_TREND_THRESHOLD:
        trend_type = 'bull'
    elif price_change_60d <= BEAR_TREND_THRESHOLD:
        trend_type = 'bear'
    else:
        trend_type = 'range'
    
    # Determine volatility level
    if volatility_30d >= HIGH_VOLATILITY_THRESHOLD:
        vol_type = 'high_vol'
    elif volatility_30d <= LOW_VOLATILITY_THRESHOLD:
        vol_type = 'low_vol'
    else:
        vol_type = 'normal_vol'
    
    # Combine into regime
    regime = f"{trend_type}_{vol_type}"
    return regime

def calculate_risk_adjusted_score(signal, regime):
    """Calculate risk-adjusted score for a signal considering market regime - ENHANCED"""
    base_score = signal["confidence"] * signal["score"]
    
    # Adjust based on market regime
    # Updated regime multipliers to match new regime detection system
    regime_multipliers = {
        "bull_high_vol": 0.9,       # Bullish but high volatility - slightly cautious
        "bull_normal_vol": 1.2,     # Ideal trending conditions
        "bull_low_vol": 1.1,        # Bullish with low volatility
        "bear_high_vol": 0.7,       # Bearish and volatile - very cautious
        "bear_normal_vol": 0.8,     # Bearish conditions - cautious
        "bear_low_vol": 0.9,        # Bearish but stable
        "range_high_vol": 0.8,      # High volatility sideways - cautious
        "range_normal_vol": 1.0,    # Normal sideways market
        "range_low_vol": 1.1,       # Low volatility range - slightly favorable
        # Fallback for old regime names
        "high_volatility": 0.8,
        "strong_trend": 1.2,
        "downtrend": 0.9,
        "low_volatility": 1.1,
        "normal": 1.0
    }
    
    multiplier = regime_multipliers.get(regime, 1.0)
    
    # Additional adjustments for signal characteristics
    if signal.get("features"):
        # Favor counter-trend in oversold/overbought conditions
        if "rsi" in signal["features"] and signal["features"]["rsi"] is not None:
            rsi = signal["features"]["rsi"]
            if signal["side"] == "buy" and rsi < 30:
                multiplier *= 1.15
            elif signal["side"] == "sell" and rsi > 70:
                multiplier *= 1.15
        
        # Boost for strong MACD signals
        if "macd" in signal["features"] and "macd_signal" in signal["features"]:
            macd = signal["features"]["macd"]
            macd_signal = signal["features"]["macd_signal"]
            if macd is not None and macd_signal is not None:
                macd_strength = abs(macd - macd_signal) / abs(macd_signal) if macd_signal != 0 else 0
                if macd_strength > 0.1:  # Strong MACD divergence
                    multiplier *= 1.1
        
        # Boost for trend alignment
        if "trend_strength" in signal["features"] and signal["features"]["trend_strength"] is not None:
            trend_strength = signal["features"]["trend_strength"]
            if (signal["side"] == "buy" and trend_strength > 0.05) or \
               (signal["side"] == "sell" and trend_strength < -0.05):
                multiplier *= 1.1
    
    return base_score * multiplier

def determine_risk_tier(signal, regime, recent_performance=None):
    """Determine which risk tier a signal belongs to based on quality and market conditions"""
    confidence = signal["confidence"]
    score = signal["score"]
    
    # Adjust thresholds based on market regime
    regime_adjustments = {
        "high_volatility": -0.1,    # Lower thresholds in high volatility
        "strong_trend": 0.05,       # Higher thresholds in strong trends
        "downtrend": -0.05,         # Lower thresholds in downtrends
        "low_volatility": 0.0,      # Normal thresholds in low volatility
        "normal": 0.0               # Normal conditions
    }
    
    adjustment = regime_adjustments.get(regime, 0.0)
    
    # Check recent performance if available
    perf_bonus = 0.0
    if recent_performance and recent_performance.get("sharpe") and recent_performance.get("sharpe") > 1.5:
        perf_bonus = 0.05
    
    # Determine tier
    for tier_name, tier_config in RISK_TIERS.items():
        adj_conf_threshold = tier_config["confidence_threshold"] + adjustment - perf_bonus
        adj_score_threshold = tier_config["score_threshold"] + adjustment - perf_bonus
        
        if confidence >= adj_conf_threshold and score >= adj_score_threshold:
            return tier_name
    
    # Default to minimal if no tier matches
    return "minimal"

def calculate_dynamic_position_size(signal, cash_available, pair_key, recent_volatility, regime, open_positions):
    """Calculate position size dynamically based on signal quality, volatility, and portfolio state"""
    # Get risk tier for sizing
    risk_tier = determine_risk_tier(signal, regime)
    tier_config = RISK_TIERS[risk_tier]
    
    # Validate cash value to prevent NaN calculations
    if not cash_available or cash_available <= 0 or np.isnan(cash_available):
        cash_available = initial_capital  # Use initial capital as fallback
        logger.warning(f"Invalid cash value detected, using initial capital instead")
    
    # Base position size from risk tier
    base_size_pct = tier_config["max_position_pct"]
    
    # Apply pair-specific allocation weight
    pair_weight = pair_allocation_weight.get(pair_key, 1.0)
    base_size_pct *= pair_weight
    
    # Adjust based on volatility
    vol_adj = 1.0
    if np.isnan(recent_volatility):
        recent_volatility = 0.02  # Default if NaN
    
    if recent_volatility > 0.03:  # High volatility
        vol_adj = 0.7
    elif recent_volatility < 0.01:  # Low volatility
        vol_adj = 1.2
    
    # Adjust based on portfolio concentration
    pairs_in_same_class = get_correlated_pairs(pair_key, 0.7)
    current_exposure_in_class = sum(
        pos.get("position_value", 0) for p, pos in open_positions.items() 
        if pos and p in pairs_in_same_class
    )
    concentration_ratio = current_exposure_in_class / cash_available if cash_available > 0 else 0
    
    # Reduce size if approaching concentration limit
    concentration_adj = 1.0
    if concentration_ratio > MAX_CONCENTRATED_CAPITAL_PCT * 0.7:
        # Scale down as we approach the limit
        concentration_adj = max(0.3, 1.0 - (concentration_ratio / MAX_CONCENTRATED_CAPITAL_PCT))
    
    # Adjust for exceptional signals
    exceptional_adj = 1.0
    if signal["confidence"] > EXCEPTIONAL_SIGNAL_THRESHOLD and signal["score"] > 0.6:
        exceptional_adj = 1.3  # 30% boost for exceptional signals
    
    # Calculate leverage cap
    leverage_cap = min(
        tier_config["leverage_cap"],
        max_leverage_per_pair.get(pair_key, 2)
    )
    
    # Calculate final position size
    position_size_pct = base_size_pct * vol_adj * concentration_adj * exceptional_adj
    max_position_value = cash_available * min(position_size_pct, BASE_POSITION_SIZE_LIMIT)
    
    # Ensure we don't exceed the maximum leverage across the entire portfolio
    current_portfolio_leverage = sum(pos.get("leverage", 1) * pos.get("position_value", 0) / cash 
                                   for pos in open_positions.values() if pos)
    available_leverage = max(0, MAX_TOTAL_LEVERAGE - current_portfolio_leverage)
    
    if available_leverage < leverage_cap:
        leverage_cap = max(1, available_leverage)
    
    # Enforce absolute maximum position size
    max_position_value = min(max_position_value, max_position_size)
    
    # Add a sanity check to prevent absurd position sizes
    if max_position_value > 100000 or np.isnan(max_position_value) or np.isinf(max_position_value):
        logger.warning(f"Position size {max_position_value} exceeds reasonable limits. Capping at {max_position_size}.")
        max_position_value = max_position_size
    
    return max_position_value, leverage_cap, risk_tier

def get_correlated_pairs(pair_key, threshold=0.7):
    """Get list of pairs that are highly correlated with the given pair"""
    # Simplified implementation - in production you would use actual correlation data
    # For cryptocurrencies, we can use these simplified groupings
    btc_related = ["btcusdt"]
    eth_related = ["ethusdt"]
    alt_major = ["solusdt", "adausdt", "dotusdt", "xmrusdt"]
    alt_minor = ["ltcusdt", "xrpusdt", "dogeusdt"]
    
    if pair_key in btc_related:
        return btc_related
    elif pair_key in eth_related:
        return eth_related
    elif pair_key in alt_major:
        return alt_major
    elif pair_key in alt_minor:
        return alt_minor
    else:
        return [pair_key]  # Only itself

def get_adjusted_stop_loss(base_stop_loss, held_minutes):
    """Get time-adjusted stop loss - tightens as position ages"""
    # Tighten stop loss by 0.5% for every 3 hours held
    time_adjustment = min(0.02, (held_minutes / 180) * 0.005)
    return max(0.01, base_stop_loss - time_adjustment)

def should_replace_position(new_signal, open_positions, price_data):
    """Determine if a new signal should replace an existing position - IMPROVED"""
    # If we're under the position limit, no need to replace
    if len(open_positions) < max_concurrent_trades:
        return None
    
    # Calculate new signal quality
    regime = detect_market_regime(price_data.get(new_signal["pair"], pd.DataFrame()))
    new_signal_quality = calculate_risk_adjusted_score(new_signal, regime)
    
    # Don't replace positions unless the new signal is exceptional
    if new_signal_quality < 0.8:  # EMERGENCY FIX: Raised from 0.6 to 0.8 (much higher quality required)
        return None
    
    # Get all open positions with their quality scores and hold times
    position_candidates = []
    current_time = datetime.utcnow()
    
    for pair, position in open_positions.items():
        if not position:
            continue
        
        # Calculate how long the position has been held
        entry_time = position["entry_time"]
        held_minutes = (current_time - entry_time).total_seconds() / 60
        
        # Calculate how long the position has been held
        entry_time = position["entry_time"]
        held_minutes = (current_time - entry_time).total_seconds() / 60
    
        # EMERGENCY FIX: Protect winning positions from replacement
        try:
            # Use the price data that's already available
            current_price = price_data.get(pair, {}).get('price', position["entry_price"])
            entry_price = position["entry_price"]
            side = position["side"]
        
            if side == "buy":
                current_pnl_pct = (current_price - entry_price) / entry_price
            else:
                current_pnl_pct = (entry_price - current_price) / entry_price
            
            # Don't replace winning positions unless signal is exceptional
            if current_pnl_pct > 0.01:  # Position is winning more than 1%
                if new_signal["confidence"] < 0.9:  # Not exceptional signal
                    continue  # Skip this position - don't replace winning positions
                
        except Exception as e:
            logger.warning(f"Could not calculate PnL for {pair}: {e}")
            current_pnl_pct = 0
        
        # Don't replace positions that are too new (unless exceptional signal)
        if held_minutes < MIN_HOLD_TIME_FOR_REPLACEMENT:
            if new_signal["confidence"] < EXCEPTIONAL_SIGNAL_REPLACEMENT_THRESHOLD:
                continue  # Skip this position for replacement
        
        # Calculate position quality based on entry parameters and current performance
        entry_quality = position["confidence"] * position["score"]
        
        # Get current performance
        try:
            symbol = PAIRS[pair]
            ticker = kraken.fetch_ticker(symbol)
            current_price = ticker['last']
            
            entry_price = position["entry_price"]
            side = position["side"]
            
            # Calculate current PnL
            if side == "buy":
                current_pnl_pct = (current_price - entry_price) / entry_price
            else:
                current_pnl_pct = (entry_price - current_price) / entry_price
            
            # Adjust quality based on current performance
            performance_adjustment = 1.0 + (current_pnl_pct * 2)  # Boost for winning positions
            current_quality = entry_quality * performance_adjustment
            
            # Add Sharpe ratio if available
            if "recent_returns" in position and len(position["recent_returns"]) >= 5:
                sharpe = calculate_position_sharpe(position)
                # Blend entry quality with current performance and Sharpe
                current_quality = 0.3 * entry_quality + 0.4 * performance_adjustment + 0.3 * max(0, sharpe / 2)
            
            position_candidates.append((pair, position, current_quality, held_minutes, current_pnl_pct))
            
        except Exception as e:
            logger.error(f"Error calculating performance for {pair}: {e}")
            # If we can't calculate performance, assume it's neutral
            position_candidates.append((pair, position, entry_quality, held_minutes, 0.0))
    
    if not position_candidates:
        return None
    
    # Sort by quality (lowest first), but consider hold time and PnL
    def replacement_score(candidate):
        pair, position, quality, held_minutes, pnl_pct = candidate
        
        # Base score is inverse quality (lower quality = higher replacement score)
        score = 1.0 / max(quality, 0.1)
        
        # Bonus for losing positions
        if pnl_pct < -0.02:  # Losing more than 2%
            score *= 1.5
        
        # Penalty for winning positions
        if pnl_pct > 0.02:  # Winning more than 2%
            score *= 0.5
        
        # Slight bonus for older positions (but not the primary factor)
        if held_minutes > 60:  # More than 1 hour
            score *= 1.1
        
        return score
    
    # Sort by replacement score (highest score = best candidate for replacement)
    position_candidates.sort(key=replacement_score, reverse=True)
    
    # Check if new signal is significantly better than the worst position
    best_candidate = position_candidates[0]
    worst_position_quality = best_candidate[2]
    
    # Apply stricter improvement threshold
    required_improvement = MIN_REPLACEMENT_IMPROVEMENT
    
    # Require even higher improvement for winning positions
    if best_candidate[4] > 0.01:  # Position is winning
        required_improvement = MIN_REPLACEMENT_IMPROVEMENT * 1.5
    
    # Allow easier replacement for losing positions
    if best_candidate[4] < -0.03:  # Position is losing significantly
        required_improvement = MIN_REPLACEMENT_IMPROVEMENT * 1.5
    
    if new_signal_quality > worst_position_quality * required_improvement:
        logger.info(f"Position replacement criteria met:")
        logger.info(f"  New signal quality: {new_signal_quality:.3f}")
        logger.info(f"  Worst position quality: {worst_position_quality:.3f}")
        logger.info(f"  Required improvement: {required_improvement:.2f}x")
        logger.info(f"  Candidate for replacement: {best_candidate[0]} (PnL: {best_candidate[4]*100:.2f}%, held: {best_candidate[3]:.1f}m)")
        
        return best_candidate[0]  # Return pair to replace
    
    return None

# === Data Handling and Model Functions ===

def load_regime_model(pair, regime):
    """Load a specific regime model for prediction"""
    # Convert pair name format
    pair_mapping = {
        "btcusdt": "XBTUSDT",
        "ethusdt": "ETHUSDT", 
        "solusdt": "SOLUSDT",
        "xrpusdt": "XRPUSDT",
        "adausdt": "ADAUSDT",
        "ltcusdt": "LTCUSDT",
        "dotusdt": "DOTUSDT",
        "xmrusdt": "XMRUSDT",
        "dogeusdt": "DOGEUSDT"
    }
    
    model_pair = pair_mapping.get(pair, pair.upper())
    
    model_filename = f"{model_pair}_{regime}_xgb_model.pkl"
    encoder_filename = f"{model_pair}_{regime}_encoder.pkl"
    features_filename = f"{model_pair}_{regime}_features.pkl"
    
    model_dir = "../../models/xgboost_regime_specific/"
    model_path = os.path.join(model_dir, model_filename)
    encoder_path = os.path.join(model_dir, encoder_filename)
    features_path = os.path.join(model_dir, features_filename)
    
    try:
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        features = joblib.load(features_path)
        return model, encoder, features
    except Exception as e:
        logger.error(f"Error loading regime model for {pair}-{regime}: {e}")
        return None, None, None

def fetch_live_ohlcv(symbol, timeframe='1m', lookback_minutes=400):
    """Fetch recent OHLCV data from the exchange"""
    try:
        since = int((time.time() - lookback_minutes * 60) * 1000)
        ohlcv = kraken.fetch_ohlcv(symbol, timeframe, since=since)
        
        if not ohlcv:
            logger.warning(f"Empty OHLCV data received for {symbol}")
            return pd.DataFrame()
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        return pd.DataFrame()

def calculate_features(df):
    """Calculate technical indicators and features for prediction"""
    df = df.copy()
    
    # Calculate RSI
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    
    # Calculate MACD
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Calculate EMAs
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_26'] = ta.trend.EMAIndicator(close=df['close'], window=26).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['ema_200'] = ta.trend.EMAIndicator(close=df['close'], window=200).ema_indicator()
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    
    # Calculate VWAP
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close'])/3).cumsum() / df['volume'].cumsum()
    
    # Calculate ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    
    # Calculate trend strength
    lookback = 30  # 30-minute lookback for trend calculation
    df['trend_strength'] = 0.0
    for i in range(lookback, len(df)):
        prices = df['close'].iloc[i-lookback:i].values
        x = np.arange(lookback)
        slope, _, _, _, _ = stats.linregress(x, prices)
        # Normalize slope by average price
        avg_price = np.mean(prices)
        df.loc[i, 'trend_strength'] = slope / avg_price * 100 if avg_price > 0 else 0
    
    # Additional features
    df['ema_trend'] = (df['ema_12'] > df['ema_26']).astype(int)
    df['volume_spike'] = (df['volume'] > df['volume'].rolling(window=20).mean() * 1.5).astype(int)
    df['pct_change_5'] = df['close'].pct_change(5) * 100
    df['pct_change_15'] = df['close'].pct_change(15) * 100
    df['volume_change'] = df['volume'].pct_change() * 100
    
    # Add missing features that the models expect
    df['id'] = range(len(df))  # Simple row ID
    df['trade_count'] = 1  # Placeholder, or implement actual trade counting logic

    # Bollinger Band position
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Additional percentage changes
    df['pct_change_60'] = df['close'].pct_change(60) * 100
    df['pct_change_240'] = df['close'].pct_change(240) * 100  
    df['pct_change_1440'] = df['close'].pct_change(1440) * 100

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # Williams %R
    df['williams_r'] = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()

    # Rate of Change (ROC)
    df['roc_60'] = ta.momentum.ROCIndicator(close=df['close'], window=60).roc()
    df['roc_240'] = ta.momentum.ROCIndicator(close=df['close'], window=240).roc()
    df['roc_1440'] = ta.momentum.ROCIndicator(close=df['close'], window=1440).roc()

    # Additional EMAs that might be missing
    df['ema_100'] = ta.trend.EMAIndicator(close=df['close'], window=100).ema_indicator()

    # Volume-based indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Price-based features
    df['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Add the remaining missing features that models expect
    df['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()
    
    # Realized volatility features (24h and 7d)
    returns = df['close'].pct_change()
    df['realized_vol_24h'] = returns.rolling(window=min(1440, len(df))).std() * np.sqrt(1440)
    df['realized_vol_7d'] = returns.rolling(window=min(10080, len(df))).std() * np.sqrt(1440)
    
    # Volatility breakout
    atr_avg = df['atr_14'].rolling(window=min(20, len(df))).mean()
    df['vol_breakout'] = (df['atr_14'] > 1.5 * atr_avg).astype(int)
    
    # ATR percentile
    df['atr_percentile'] = df['atr_14'].rolling(window=min(100, len(df))).rank(pct=True)
    
    # Volume indicators
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).money_flow_index()
    df['ad_line'] = ta.volume.AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).acc_dist_index()
    
    # Volume rate of change
    df['volume_roc'] = ta.momentum.ROCIndicator(close=df['volume'], window=14).roc()
    
    # Additional volume ratio (different calculation)
    vol_ma_20 = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / vol_ma_20
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['european_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
    df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Market structure features
    high_20 = df['high'].rolling(window=20).max()
    low_20 = df['low'].rolling(window=20).min()
    df['near_high'] = (df['close'] > 0.95 * high_20).astype(int)
    df['near_low'] = (df['close'] < 1.05 * low_20).astype(int)
    df['range_position'] = (df['close'] - low_20) / (high_20 - low_20)
    df['ema_alignment'] = ((df['ema_12'] > df['ema_26']) & (df['ema_50'] > df['ema_200'])).astype(int)
    
    # Additional missing indicators
    df['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()
    
    # Realized volatility features
    df['realized_vol_24h'] = df['close'].pct_change().rolling(window=1440).std() * np.sqrt(1440)  # 24h
    df['realized_vol_7d'] = df['close'].pct_change().rolling(window=10080).std() * np.sqrt(10080)  # 7d
    
    # Volatility breakout
    df['vol_breakout'] = (df['realized_vol_24h'] > df['realized_vol_24h'].rolling(window=168).mean()).astype(int)
    
    # ATR percentile
    df['atr_percentile'] = df['atr_14'].rolling(window=100).rank(pct=True)
    
    # Volume indicators
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).money_flow_index()
    df['ad_line'] = ta.volume.AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).acc_dist_index()
    
    # Volume rate of change
    df['volume_roc'] = df['volume'].pct_change(periods=10)
    
    # Volume ratio (different from your existing volume_ratio)
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(window=50).mean()
    
    # === Multi-timeframe features (5-minute) ===
    # Only calculate if we have enough data
    if len(df) >= 300:  # Need at least 5 hours of 1-minute data
        try:
            # Create datetime column for resampling
            df['datetime_for_resample'] = df['timestamp']
            
            # Resample to 5-minute for higher timeframe indicators  
            df_5m = df.set_index('datetime_for_resample').resample('5min').agg({
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(df_5m) >= 200:  # Need enough 5-minute data
                # Calculate 5m RSI
                df_5m['rsi_5m'] = ta.momentum.RSIIndicator(close=df_5m['close'], window=14).rsi()
                
                # Calculate 5m EMA trend
                df_5m['ema_50_5m'] = ta.trend.EMAIndicator(close=df_5m['close'], window=50).ema_indicator()
                df_5m['ema_200_5m'] = ta.trend.EMAIndicator(close=df_5m['close'], window=200).ema_indicator()
                df_5m['ema_trend_5m'] = (df_5m['ema_50_5m'] > df_5m['ema_200_5m']).astype(int)
                
                # Merge back to 1-minute data
                df_original_index = df.index
                df = df.set_index('datetime_for_resample')
                df = df.merge(df_5m[['rsi_5m', 'ema_trend_5m']], 
                             left_index=True, right_index=True, how='left')
                df = df.fillna(method='ffill')  # Forward fill higher timeframe data
                df = df.reset_index(drop=True)
                df.index = df_original_index
            else:
                # Not enough 5m data, set defaults
                df['rsi_5m'] = 50.0  # Neutral RSI
                df['ema_trend_5m'] = 0   # Neutral trend
            
            # Drop the temporary datetime column
            if 'datetime_for_resample' in df.columns:
                df.drop(columns=['datetime_for_resample'], inplace=True)
                
        except Exception as e:
            logger.warning(f"Error calculating multi-timeframe features: {e}")
            # Set default values if calculation fails
            df['rsi_5m'] = 50.0  # Neutral RSI
            df['ema_trend_5m'] = 0   # Neutral trend
    else:
        # Not enough data for multi-timeframe analysis, set defaults
        df['rsi_5m'] = 50.0  # Neutral RSI
        df['ema_trend_5m'] = 0   # Neutral trend
    
    # Clean data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)  # Forward fill remaining NaNs
    df.fillna(0, inplace=True)  # Fill any remaining NaNs with 0
    
    return df

def update_price_data(pair_key, df):
    """Update price data for a pair, including volatility and market regime"""
    global price_data
    
    if df.empty or len(df) < VOLATILITY_LOOKBACK_WINDOW:
        return
    
    # Calculate recent volatility
    recent_df = df.iloc[-VOLATILITY_LOOKBACK_WINDOW:]
    returns = recent_df['close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(1440)  # Convert to daily volatility
    
    # Detect market regime
    market_regime = detect_market_regime(recent_df)
    
    # Store data
    price_data[pair_key] = {
        "volatility": volatility,
        "df": recent_df,
        "market_regime": market_regime,
        "last_update": datetime.utcnow()
    }
    
    # Log to database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO market_regimes 
    (timestamp, pair, regime, volatility, correlation, trend_strength)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        safe_str(datetime.utcnow().isoformat()),
        safe_str(pair_key),
        safe_str(market_regime),
        safe_float(volatility),
        safe_float(0.0),  # Placeholder for correlation
        safe_float(recent_df['trend_strength'].iloc[-1] if 'trend_strength' in recent_df.columns else 0.0)
    ))
    conn.commit()
    conn.close()
        
    logger.info(f"Market regime transition for {pair_key}: {market_regime} (volatility: {volatility:.4f})")

def generate_trading_signal(pair_key, symbol):
    """Generate a trading signal for a specific pair"""
    global price_data, signal_quality_history
    
    try:
        # Fetch and process data
        df = fetch_live_ohlcv(symbol)
        if df.empty:
            logger.warning(f"Unable to generate signal for {pair_key}: No data received")
            return None
        
        # Calculate features
        df = calculate_features(df)
        
        # Update price data
        update_price_data(pair_key, df)
        
        # Get latest data point
        latest = df.iloc[-1]
        
        # Extract features for prediction
        features = df.iloc[-1:].copy()
        
        # Add dummy future_return column if needed by certain models
        if "future_return" not in features.columns:
            features["future_return"] = 0  # Dummy value
        
        # Detect current market regime FIRST
        market_regime = detect_market_regime(df)

        # Load regime-specific model
        regime_model, regime_encoder, regime_features = load_regime_model(pair_key, market_regime)

        if regime_model is None:
            logger.warning(f"No regime model available for {pair_key}-{market_regime}")
            return None

        # DEBUG: Log feature comparison
        logger.info(f"Model expects {len(regime_features)} features: {regime_features[:10]}...")  # First 10
        available_features = [col for col in regime_features if col in features.columns]
        missing_features = [col for col in regime_features if col not in features.columns]
        logger.info(f"Available: {len(available_features)}, Missing: {len(missing_features)}")
        if missing_features:
            logger.info(f"Missing features: {missing_features[:10]}...")  # First 10 missing

        # Use regime-specific model
        try:
            # Prepare features for regime model
            available_features = [col for col in regime_features if col in features.columns]
    
            if len(available_features) < len(regime_features) * 0.8:
                logger.warning(f"Missing too many features for regime {market_regime}, skipping signal")
                return None
    
            # Clean data for prediction
            prediction_data = features[available_features].copy()
            prediction_data = prediction_data.replace([np.inf, -np.inf], np.nan)
            prediction_data = prediction_data.fillna(prediction_data.median())
    
            # Convert to XGBoost DMatrix
            dtest = xgb.DMatrix(prediction_data)
    
            # Get predictions
            regime_probabilities = regime_model.predict(dtest)
            regime_predictions = np.argmax(regime_probabilities, axis=1)
            confidence = regime_probabilities.max(axis=1)[0]
    
            # Convert prediction back to label
            prediction = regime_encoder.inverse_transform(regime_predictions)[0]
    
            logger.info(f"Using regime model {pair_key}-{market_regime} with confidence {confidence:.3f}")
    
        except Exception as e:
            logger.error(f"Error using regime model: {e}")
            return None
        
        # Calculate technical score with MORE OR CONDITIONS (relaxed)
        score = 0
        
        # 1. RSI Filter
        if use_rsi_filter and "rsi_14" in latest:
            if (latest["rsi_14"] < rsi_oversold and prediction == "buy") or \
               (latest["rsi_14"] > rsi_overbought and prediction == "sell"):
                score += rsi_weight
        
        # 2. MACD Filter
        if use_macd_filter and "macd" in latest and "macd_signal" in latest:
            if (latest["macd"] > latest["macd_signal"] and prediction == "buy") or \
               (latest["macd"] < latest["macd_signal"] and prediction == "sell"):
                score += macd_weight
        
        # 3. VWAP Filter
        if use_vwap_filter and "vwap" in latest:
            if (prediction == "buy" and latest["close"] < latest["vwap"]) or \
               (prediction == "sell" and latest["close"] > latest["vwap"]):
                score += vwap_weight
        
        # 4. Bollinger Bands Filter
        if use_bbands_filter and "bb_upper" in latest and "bb_lower" in latest:
            if (prediction == "buy" and latest["close"] < latest["bb_lower"]) or \
               (prediction == "sell" and latest["close"] > latest["bb_upper"]):
                score += bbands_weight
        
        # 5. EMA Crossover Filter
        if use_ema_filter and "ema_9" in df.columns and "ema_21" in df.columns and len(df) > 1:
            prev = df.iloc[-2]
            # Bullish crossover: EMA9 crosses above EMA21
            bullish_crossover = prev["ema_9"] <= prev["ema_21"] and latest["ema_9"] > latest["ema_21"]
            # Bearish crossover: EMA9 crosses below EMA21
            bearish_crossover = prev["ema_9"] >= prev["ema_21"] and latest["ema_9"] < latest["ema_21"]
            
            if (prediction == "buy" and bullish_crossover) or \
               (prediction == "sell" and bearish_crossover):
                score += ema_weight
        
        # 6. Trend Strength Filter
        if use_trend_filter and "trend_strength" in latest:
            trend_str = latest["trend_strength"]
            if pd.notna(trend_str):
                # Positive trend strength favors buys, negative favors sells
                if (prediction == "buy" and trend_str > 0.02) or \
                   (prediction == "sell" and trend_str < -0.02):
                    score += trend_strength_weight
        
        # 7. Time-based boost - increase score during active hours
        current_hour = datetime.utcnow().hour
        time_boost = 0.2 if current_hour in active_hours else 0
        score += time_boost
        
        # Get current market regime
        market_regime = price_data.get(pair_key, {}).get("market_regime", "normal")
        
        # Create features dict for signal
        signal_features = {
            "rsi": latest.get("rsi_14"),
            "macd": latest.get("macd"),
            "macd_signal": latest.get("macd_signal"),
            "vwap": latest.get("vwap"),
            "bb_upper": latest.get("bb_upper"),
            "bb_lower": latest.get("bb_lower"),
            "ema_9": latest.get("ema_9"),
            "ema_21": latest.get("ema_21"),
            "trend_strength": latest.get("trend_strength"),
            "atr": latest.get("atr_14")
        }
        
        # Log signal to database regardless of threshold
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO signals 
        (timestamp, pair, signal, price, confidence, risk_score, rsi, macd, macd_signal, 
        vwap, bb_upper, bb_lower, ema_9, ema_21, trend_strength, market_regime, acted_upon)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(),
            safe_str(pair_key),
            safe_str(prediction),
            safe_float(latest["close"]),
            safe_float(confidence),
            safe_float(score),
            safe_float(latest.get("rsi_14")),
            safe_float(latest.get("macd")),
            safe_float(latest.get("macd_signal")),
            safe_float(latest.get("vwap")),
            safe_float(latest.get("bb_upper")),
            safe_float(latest.get("bb_lower")),
            safe_float(latest.get("ema_9")),
            safe_float(latest.get("ema_21")),
            safe_float(latest.get("trend_strength")),
            safe_str(market_regime),
            safe_int(0)  # Not acted upon yet
        ))
        conn.commit()
        conn.close()
        
        # Check pair-specific risk threshold
        pair_threshold = pair_risk_thresholds.get(pair_key, risk_score_threshold)
        
        # Check if signal meets threshold criteria
        if score >= pair_threshold and prediction in ["buy", "sell"]:
            # Create and return trading signal object
            signal = {
                "pair": pair_key,
                "symbol": symbol,
                "side": prediction,
                "price": latest["close"],
                "confidence": confidence,
                "score": score,
                "timestamp": datetime.utcnow(),
                "features": signal_features,
                "market_regime": market_regime
            }
            
            # Store signal quality for comparisons later
            signal_quality_history.append((pair_key, calculate_risk_adjusted_score(signal, market_regime)))
            
            return signal
        
        return None
    
    except Exception as e:
        logger.error(f"Error generating signal for {pair_key}: {e}")
        logger.error(traceback.format_exc())
        return None

# === Position Management Functions ===

def check_circuit_breakers():
    """Check if any circuit breakers should be activated - FIXED VERSION"""
    global trading_paused_until, active_circuit_breakers, daily_stats
    
    current_time = datetime.utcnow()
    
    # Reset expired circuit breakers
    for cb_id in list(active_circuit_breakers.keys()):
        if active_circuit_breakers[cb_id]["expires_at"] <= current_time:
            logger.info(f"Circuit breaker {cb_id} has expired")
            
            # Update database
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE circuit_breakers SET is_active = 0 WHERE id = ?", 
                (cb_id,)
            )
            conn.commit()
            conn.close()
            
            # Remove from active list
            del active_circuit_breakers[cb_id]
    
    # Check if any active circuit breakers exist
    if active_circuit_breakers:
        latest_expiry = max(cb["expires_at"] for cb in active_circuit_breakers.values())
        trading_paused_until = latest_expiry
        return True
    
    # Check daily loss limit using PROPER equity calculation
    current_equity = calculate_total_equity()
    daily_pnl = current_equity - daily_stats["start_capital"]
    daily_pnl_pct = daily_pnl / daily_stats["start_capital"] if daily_stats["start_capital"] > 0 else 0
    
    # Log current status
    logger.info(f"Daily PnL check: Current Equity: ${current_equity:.2f}, Start Capital: ${daily_stats['start_capital']:.2f}, PnL: {daily_pnl_pct*100:.2f}%")
    
    # Only trigger circuit breaker for ACTUAL losses (not calculation errors)
    if daily_pnl_pct <= -daily_loss_limit_pct and abs(daily_pnl) > 100:  # At least $100 loss
        expiry_time = current_time + timedelta(minutes=pause_duration_minutes)
        cb_id = activate_circuit_breaker("daily_loss", "Daily loss limit reached", daily_pnl_pct, -daily_loss_limit_pct, None, expiry_time)
        trading_paused_until = expiry_time
        return True
    
    # Check maximum concurrent trades (soft limit)
    if len(open_positions) >= max_concurrent_trades:
        logger.info(f"Maximum concurrent trades reached ({len(open_positions)}/{max_concurrent_trades})")
        # This doesn't pause trading, just prevents new positions
        return True
    
    # Check portfolio VaR with realistic bounds
    if len(open_positions) > 0:
        portfolio_var = calculate_portfolio_var(open_positions, price_data) / current_equity if current_equity > 0 else 0
        
        # Only trigger if VaR is reasonable (not due to calculation error)
        if portfolio_var > MAX_PORTFOLIO_VAR and portfolio_var < 0.5:  # Cap at 50% to avoid false triggers
            expiry_time = current_time + timedelta(minutes=pause_duration_minutes)
            cb_id = activate_circuit_breaker("var_limit", "Portfolio VaR limit exceeded", portfolio_var, MAX_PORTFOLIO_VAR, None, expiry_time)
            trading_paused_until = expiry_time
            return True
    
    # Check drawdown from peak with improved logic
    if total_equity_history and len(total_equity_history) > 10:  # Need some history
        # Get recent equity values (last 100 points)
        recent_equity = [eq[1] for eq in total_equity_history[-100:]]
        peak_equity = max(recent_equity)
        
        # Only consider significant peaks (at least 5% above initial capital)
        if peak_equity > initial_capital * 1.05:
            drawdown_pct = (peak_equity - current_equity) / peak_equity
            
            # Only trigger for real drawdowns (not calculation errors)
            if drawdown_pct >= drawdown_pause_pct and drawdown_pct < 0.8:  # Cap at 80%
                logger.warning(f"Drawdown detected: Peak ${peak_equity:.2f}, Current ${current_equity:.2f}, Drawdown {drawdown_pct*100:.2f}%")
                expiry_time = current_time + timedelta(minutes=pause_duration_minutes)
                cb_id = activate_circuit_breaker("drawdown", "Drawdown limit reached", drawdown_pct, drawdown_pause_pct, None, expiry_time)
                trading_paused_until = expiry_time
                return True
    
    trading_paused_until = None
    return False

# === IMPROVED DAILY STATS TRACKING ===

def update_daily_stats():
    """Update daily statistics with proper equity tracking"""
    global daily_stats
    
    current_equity = calculate_total_equity()
    daily_stats["current_capital"] = current_equity
    
    # Update win/loss tracking from database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Get today's trades
    today = datetime.utcnow().date().isoformat()
    cursor.execute("""
        SELECT COUNT(*), SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END), 
               AVG(CASE WHEN net_pnl > 0 THEN net_pnl ELSE NULL END),
               AVG(CASE WHEN net_pnl <= 0 THEN net_pnl ELSE NULL END)
        FROM trades 
        WHERE DATE(timestamp) = ? AND is_paper = 1
    """, (today,))
    
    result = cursor.fetchone()
    if result:
        daily_stats["trades_today"] = result[0] or 0
        daily_stats["winning_trades_today"] = result[1] or 0
        daily_stats["losing_trades_today"] = daily_stats["trades_today"] - daily_stats["winning_trades_today"]
        daily_stats["avg_win_today"] = result[2] or 0
        daily_stats["avg_loss_today"] = result[3] or 0
    
    conn.close()

def activate_circuit_breaker(cb_type, reason, value, threshold, pair, expiry_time):
    """Activate a circuit breaker and log it to the database"""
    global active_circuit_breakers
    
    # Log to database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO circuit_breakers 
    (timestamp, type, reason, value, threshold, pair, is_active)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        safe_str(datetime.utcnow().isoformat()),
        safe_str(cb_type),
        safe_str(reason),
        safe_float(value),
        safe_float(threshold),
        safe_str(pair),
        safe_int(1)
    ))
    conn.commit()
    cb_id = cursor.lastrowid
    conn.close()
    
    # Add to active circuit breakers
    active_circuit_breakers[cb_id] = {
        "type": cb_type,
        "reason": reason,
        "value": value,
        "threshold": threshold,
        "pair": pair,
        "expires_at": expiry_time
    }
    
    logger.warning(f"Circuit breaker activated: {reason}. Trading paused until {expiry_time}")
    return cb_id

def check_exit_conditions():
    """Check exit conditions for all open positions"""
    global cash, open_positions, daily_stats, partial_exits
    
    current_time = datetime.utcnow()
    
    # Iterate through all open positions
    for pair_key in list(open_positions.keys()):
        position = open_positions[pair_key]
        
        # Skip if no position data
        if not position:
            continue
        
        # Fetch current price
        try:
            symbol = PAIRS[pair_key]
            ticker = kraken.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Apply slippage to exit price
            exit_price = current_price * (1 - slippage_per_pair[pair_key]) if position["side"] == "buy" else current_price * (1 + slippage_per_pair[pair_key])
            
            # Calculate position metrics
            entry_price = position["entry_price"]
            entry_time = position["entry_time"]
            side = position["side"]
            leverage = position["leverage"]
            position_value = position["position_value"]
            held_minutes = (current_time - entry_time).total_seconds() / 60
            
            # Update recent returns history (for Sharpe calculation)
            if current_price != position.get("last_price", entry_price):
                # Calculate instantaneous return
                if side == "buy":
                    current_return = (current_price - position.get("last_price", entry_price)) / position.get("last_price", entry_price)
                else:
                    current_return = (position.get("last_price", entry_price) - current_price) / position.get("last_price", entry_price)
                
                # Add to recent returns history
                if "recent_returns" not in position:
                    position["recent_returns"] = []
                
                position["recent_returns"].append(current_return)
                if len(position["recent_returns"]) > 20:  # Keep only recent returns
                    position["recent_returns"].pop(0)
                
                # Update last price
                position["last_price"] = current_price
            
            # Calculate current PnL
            if side == "buy":
                pnl = (exit_price - entry_price) / entry_price
                # Update trailing high
                position["highest_price"] = max(position.get("highest_price", entry_price), exit_price)
                # Calculate drawdown from highest
                drawdown = (exit_price - position["highest_price"]) / position["highest_price"]
            else:  # sell
                pnl = (entry_price - exit_price) / entry_price
                # Update trailing low
                position["lowest_price"] = min(position.get("lowest_price", entry_price), exit_price)
                # Calculate drawdown from lowest (for short positions)
                drawdown = (position["lowest_price"] - exit_price) / position["lowest_price"]
            
            # Track if this position has already had a partial exit
            has_partial_exit = pair_key in partial_exits
            
            # Get time-adjusted stop loss
            adjusted_stop_loss = get_adjusted_stop_loss(stop_loss_pct, held_minutes)
            
            # Check if profit is enough for break-even stop
            reached_break_even = pnl >= break_even_profit_pct
            
            # Calculate dynamic trailing stop based on ATR
            atr_value = position["features"].get('atr', entry_price * 0.01) # Default if missing
            atr_multiplier = 2.0  # Adjust based on risk preference
            atr_based_stop = atr_value * atr_multiplier / entry_price
            
            # Determine the effective stop loss (most conservative)
            effective_stop_loss = adjusted_stop_loss
            if reached_break_even:
                # If we've reached break-even profit, move stop to break-even
                effective_stop_loss = -0.001  # Small buffer below break-even
            
            # Check exit conditions
            should_exit_full = False
            should_exit_partial = False
            exit_reason = ""
            
            # Full exit conditions
            if abs(drawdown) >= effective_stop_loss:
                should_exit_full = True
                exit_reason = "stop_loss"
            elif not has_partial_exit and pnl >= take_profit_pct_first_half:
                should_exit_partial = True
                exit_reason = "partial_take_profit" 
            elif has_partial_exit and pnl >= take_profit_pct_full:
                should_exit_full = True
                exit_reason = "take_profit"
            elif held_minutes >= max_holding_minutes:
                should_exit_full = True
                exit_reason = "time_limit"
            
            # Check for reversal signals if available
            latest_signals = {}
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT signal, confidence FROM signals WHERE pair = ? ORDER BY timestamp DESC LIMIT 1", 
                (pair_key,)
            )
            result = cursor.fetchone()
            conn.close()
            
            if result:
                latest_signal, latest_confidence = result
                if ((side == "buy" and latest_signal == "sell") or 
                    (side == "sell" and latest_signal == "buy")) and latest_confidence >= 0.75:
                    should_exit_full = True
                    exit_reason = "reversal_signal"
            
            # Handle partial exit
            if should_exit_partial and not has_partial_exit:
                # Calculate partial position size (50%)
                partial_position_value = position_value * 0.5
                remaining_position_value = position_value - partial_position_value
    
                # Calculate fees for partial exit
                # Calculate realistic fees for partial exit
                partial_fee_breakdown = calculate_realistic_fees(pair_key, partial_position_value, entry_price, leverage, held_minutes)
                exit_fee = partial_fee_breakdown["exit_fee"]
    
                # Calculate position size and PnL with proper validation
                partial_position_size = partial_position_value / entry_price if entry_price != 0 else 0
    
                # Safety check to prevent unrealistic values
                if np.isnan(partial_position_size) or np.isinf(partial_position_size) or partial_position_size <= 0:
                    logger.warning(f"Invalid partial position size for {pair_key}: {partial_position_size}, setting to 0")
                    partial_position_size = 0
    
                # Calculate price difference with caps for realism
                price_diff = exit_price - entry_price if side == "buy" else entry_price - exit_price
                if abs(price_diff) / entry_price > 0.10:  # More than 10% move is suspicious for short timeframes
                    logger.warning(f"Extreme price movement in partial exit: {price_diff / entry_price * 100:.2f}%. Limiting impact.")
                    price_diff = np.sign(price_diff) * entry_price * 0.10  # Cap at 10% move
    
                # Calculate PnL with leverage
                partial_gross_pnl = partial_position_size * price_diff * leverage
    
                # Validate PnL calculation
                if np.isnan(partial_gross_pnl) or np.isinf(partial_gross_pnl):
                    logger.error(f"Invalid partial PnL calculation for {pair_key}: entry={entry_price}, exit={exit_price}, leverage={leverage}")
                    partial_gross_pnl = 0
    
                # Cap PnL at a reasonable multiple of position value to prevent cascading issues
                max_reasonable_pnl = partial_position_value * leverage * 0.10  # Max 10% return per trade
                if abs(partial_gross_pnl) > max_reasonable_pnl:
                    logger.warning(f"Partial PnL {partial_gross_pnl} exceeds reasonable limit. Capping at {max_reasonable_pnl}")
                    partial_gross_pnl = np.sign(partial_gross_pnl) * max_reasonable_pnl
    
                partial_net_pnl = partial_gross_pnl - exit_fee
    
                # Execute partial exit order
                try:
                    # ===============================================
                    # COMMENT OUT ACTUAL TRADING EXECUTION
                    # Uncomment the following section to enable real trading
                    # ===============================================
                    
                    """
                    # Close 50% of the position
                    if side == "buy":
                        order = kraken.create_market_sell_order(symbol, partial_position_size)
                    else:
                        order = kraken.create_market_buy_order(symbol, partial_position_size)
                    
                    logger.info(f"Partial exit order executed: {order}")
                    """
                    
                    # Update cash (margin trading: return partial margin fee + PnL)
                    margin_rates = get_margin_fees_for_pair(pair_key)
                    partial_margin_returned = partial_position_value * margin_rates["open_fee"]
                    cash += partial_margin_returned + partial_net_pnl
    
                    # Update remaining position value
                    position["position_value"] = remaining_position_value
    
                    # Mark this position as having a partial exit
                    partial_exits[pair_key] = {
                        "exit_time": current_time,
                        "exit_price": exit_price,
                        "exit_value": partial_position_value,
                        "pnl": pnl,
                        "net_pnl": partial_net_pnl
                    }
    
                    # Record partial exit as a trade to properly track win/loss statistics
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute('''
                    INSERT INTO trades 
                    (timestamp, pair, side, entry_price, exit_price, quantity, position_value, leverage,
                    entry_time, exit_time, pnl, net_pnl, fees, hold_time, confidence, risk_score, exit_reason, 
                    market_regime, risk_tier, is_paper, partial_exit)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        current_time.isoformat(),
                        pair_key,
                        side,
                        entry_price,
                        exit_price,
                        partial_position_size,
                        partial_position_value,
                        leverage,
                        entry_time.isoformat(),
                        current_time.isoformat(),
                        pnl * 100,  # Store as percentage
                        partial_net_pnl,
                        exit_fee,
                        held_minutes,
                        position["confidence"],
                        position["score"],
                        "partial_take_profit",
                        position.get("market_regime", "unknown"),
                        position.get("risk_tier", "unknown"),
                        1,  # Is paper trade
                        1   # Is partial exit
                    ))
                    conn.commit()
                    conn.close()
    
                    # Update daily stats
                    daily_stats["trades_today"] += 1
                    if partial_net_pnl > 0:
                        daily_stats["winning_trades_today"] += 1
                    else:
                        daily_stats["losing_trades_today"] += 1
                    daily_stats["current_capital"] = cash
    
                    logger.info(f"{pair_key} | 📊 PARTIAL EXIT {side.upper()} @ {exit_price:.4f} | Entry: {entry_price:.4f} | " 
                          f"PnL: ${partial_net_pnl:.2f} ({pnl*100:.2f}%) | Held: {held_minutes:.1f}m")
                
                except Exception as e:
                    logger.error(f"Error executing partial exit for {pair_key}: {e}")
                    logger.error(traceback.format_exc())
                
            # Handle full position exit
            elif should_exit_full:
                # Calculate fees using realistic Kraken rates
                fee_breakdown = calculate_realistic_fees(pair_key, position_value, entry_price, leverage, held_minutes)
                fees = fee_breakdown["total_fees"]
    
                # Calculate PnL with proper validation
                position_size = position_value / entry_price if entry_price != 0 else 0

                # Safety check to prevent unrealistic values
                if np.isnan(position_size) or np.isinf(position_size) or position_size <= 0:
                    logger.warning(f"Invalid position size for {pair_key}: {position_size}, setting to 0")
                    position_size = 0

                # Calculate price difference with caps for realism
                price_diff = exit_price - entry_price if side == "buy" else entry_price - exit_price
                if abs(price_diff) / entry_price > 0.10:  # More than 10% move is suspicious for short timeframes
                    logger.warning(f"Extreme price movement detected: {price_diff / entry_price * 100:.2f}%. Limiting impact.")
                    price_diff = np.sign(price_diff) * entry_price * 0.10  # Cap at 10% move

                # Calculate PnL with leverage
                gross_pnl = position_size * price_diff * leverage
                
                # Validate PnL calculation with a simple percentage check
                expected_pnl_pct = (exit_price - entry_price) / entry_price if side == "buy" else (entry_price - exit_price) / entry_price
                expected_pnl = position_size * entry_price * expected_pnl_pct * leverage
                if abs(gross_pnl - expected_pnl) > position_value * 0.01:  # More than 1% difference
                    print(f"Warning: PnL calculation discrepancy detected. Calculated: {gross_pnl:.2f}, Expected: {expected_pnl:.2f}")
                    # Use the expected PnL if the discrepancy is significant
                    if abs(gross_pnl - expected_pnl) > position_value * 0.05:  # 5% difference
                        gross_pnl = expected_pnl
                        print(f"PnL calculation fixed to {gross_pnl:.2f}")

                # Validate PnL calculation
                if np.isnan(gross_pnl) or np.isinf(gross_pnl):
                    logger.error(f"Invalid PnL calculation for {pair_key}: entry={entry_price}, exit={exit_price}, leverage={leverage}")
                    gross_pnl = 0

                # Cap PnL at a reasonable multiple of position value to prevent cascading issues
                max_reasonable_pnl = position_value * leverage * 0.10  # Max 10% return per trade
                if abs(gross_pnl) > max_reasonable_pnl:
                    logger.warning(f"PnL {gross_pnl} exceeds reasonable limit. Capping at {max_reasonable_pnl}")
                    gross_pnl = np.sign(gross_pnl) * max_reasonable_pnl

                net_pnl = gross_pnl - fees
                
                try:
                    # ===============================================
                    # COMMENT OUT ACTUAL TRADING EXECUTION
                    # Uncomment the following section to enable real trading
                    # ===============================================
                    
                    """
                    # Execute the exit order
                    if side == "buy":
                        order = kraken.create_market_sell_order(symbol, position_size)
                    else:
                        order = kraken.create_market_buy_order(symbol, position_size)
                    
                    logger.info(f"Exit order executed: {order}")
                    """
                    
                    # Update cash (margin trading: return margin fee + PnL)
                    # Return margin using realistic fee structure
                    margin_rates = get_margin_fees_for_pair(pair_key)
                    margin_returned = position_value * margin_rates["open_fee"]
                    cash += margin_returned + net_pnl
                    
                    # Record trade in database
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute('''
                    INSERT INTO trades 
                    (timestamp, pair, side, entry_price, exit_price, quantity, position_value, leverage,
                    entry_time, exit_time, pnl, net_pnl, fees, hold_time, confidence, risk_score, exit_reason, 
                    market_regime, risk_tier, is_paper, partial_exit)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        safe_str(current_time.isoformat()),
                        safe_str(pair_key),
                        safe_str(side),
                        safe_float(entry_price),
                        safe_float(exit_price),
                        safe_float(position_size),
                        safe_float(position_value),
                        safe_float(leverage),
                        safe_str(entry_time.isoformat()),
                        safe_str(current_time.isoformat()),
                        safe_float(pnl * 100),  # Store as percentage
                        safe_float(net_pnl),
                        safe_float(fees),
                        safe_float(held_minutes),
                        safe_float(position["confidence"]),
                        safe_float(position["score"]),
                        safe_str(exit_reason),
                        safe_str(position.get("market_regime", "unknown")),
                        safe_str(position.get("risk_tier", "unknown")),
                        safe_int(1),  # Is paper trade
                        safe_int(1 if has_partial_exit else 0)
                    ))
                    conn.commit()
                    conn.close()
                    
                    # Update daily stats
                    daily_stats["trades_today"] += 1
                    if net_pnl > 0:
                        daily_stats["winning_trades_today"] += 1
                    else:
                        daily_stats["losing_trades_today"] += 1
                    daily_stats["current_capital"] = cash
                    
                    logger.info(f"{pair_key} | 📉 CLOSED {side.upper()} @ {exit_price:.4f} | Entry: {entry_price:.4f} | " 
                          f"PnL: ${net_pnl:.2f} ({pnl*100:.2f}%) | Held: {held_minutes:.1f}m | Reason: {exit_reason}")
                    
                    # Remove position from tracking
                    del open_positions[pair_key]
                    if pair_key in partial_exits:
                        del partial_exits[pair_key]
                
                except Exception as e:
                    logger.error(f"Error executing exit for {pair_key}: {e}")
                    logger.error(traceback.format_exc())
                
        except Exception as e:
            logger.error(f"Error checking exit conditions for {pair_key}: {e}")
            logger.error(traceback.format_exc())

# Add the new function here - AFTER check_exit_conditions function
def enforce_position_limit():
    """Ensure we don't exceed the maximum number of concurrent positions"""
    global open_positions, max_concurrent_trades, cash
    
    # If we're under the limit, no action needed
    if len(open_positions) <= max_concurrent_trades:
        return
    
    # We have too many positions, close the worst performing ones
    excess_count = len(open_positions) - max_concurrent_trades
    logger.warning(f"Position limit exceeded: {len(open_positions)}/{max_concurrent_trades}. Closing {excess_count} positions.")
    
    # Calculate performance metrics for all positions
    position_performance = []
    for pair, position in open_positions.items():
        if not position:
            continue
        
        # Get current price
        try:
            symbol = PAIRS[pair]
            ticker = kraken.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate performance
            entry_price = position["entry_price"]
            side = position["side"]
            
            if side == "buy":
                performance = (current_price - entry_price) / entry_price
            else:
                performance = (entry_price - current_price) / entry_price
            
            # Add to list
            position_performance.append((pair, performance))
        except Exception as e:
            logger.error(f"Error calculating performance for {pair}: {e}")
            # If we can't calculate performance, assume it's bad
            position_performance.append((pair, -1.0))
    
    # Sort by performance (worst first)
    position_performance.sort(key=lambda x: x[1])
    
    # Close the worst performing positions
    for i in range(excess_count):
        if i < len(position_performance):
            pair_to_close = position_performance[i][0]
            logger.info(f"Enforcing position limit: Closing {pair_to_close} with performance {position_performance[i][1]:.2f}")
            
            # Set up an immediate forced exit
            position = open_positions[pair_to_close]
            if position:
                try:
                    # Get current price for exit
                    symbol = PAIRS[pair_to_close]
                    ticker = kraken.fetch_ticker(symbol)
                    exit_price = ticker['last']
                    
                    # Apply slippage
                    exit_price *= (1 - slippage_per_pair[pair_to_close]) if position["side"] == "buy" else (1 + slippage_per_pair[pair_to_close])
                    
                    # Calculate position metrics
                    entry_price = position["entry_price"]
                    entry_time = position["entry_time"]
                    position_side = position["side"]
                    leverage = position["leverage"]
                    position_value = position["position_value"]
                    held_minutes = (datetime.utcnow() - entry_time).total_seconds() / 60
                    
                    # Calculate PnL
                    if position_side == "buy":
                        pnl = (exit_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - exit_price) / entry_price
                    
                    # Calculate fees
                    position_size = position_value / entry_price if entry_price > 0 else 0
                    # Calculate realistic fees for forced close
                    fee_breakdown = calculate_realistic_fees(pair_to_close, position_value, entry_price, leverage, held_minutes)
                    fees = fee_breakdown["total_fees"]
                    
                    # Calculate gross and net PnL
                    price_diff = exit_price - entry_price if position_side == "buy" else entry_price - exit_price
                    gross_pnl = position_size * price_diff * leverage
                    net_pnl = gross_pnl - fees
                    
                    # Update cash (margin trading: return margin fee + PnL)
                    # Return margin using realistic fee structure
                    margin_rates = get_margin_fees_for_pair(pair_to_close)
                    margin_returned = position_value * margin_rates["open_fee"]
                    cash += margin_returned + net_pnl
                    
                    # Record trade in database
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute('''
                    INSERT INTO trades 
                    (timestamp, pair, side, entry_price, exit_price, quantity, position_value, leverage,
                    entry_time, exit_time, pnl, net_pnl, fees, hold_time, confidence, risk_score, exit_reason, 
                    market_regime, risk_tier, is_paper, partial_exit)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        safe_str(datetime.utcnow().isoformat()),
                        safe_str(pair_to_close),
                        safe_str(position_side),
                        safe_float(entry_price),
                        safe_float(exit_price),
                        safe_float(position_size),
                        safe_float(position_value),
                        safe_float(leverage),
                        safe_str(entry_time.isoformat()),
                        safe_str(datetime.utcnow().isoformat()),
                        safe_float(pnl * 100),
                        safe_float(net_pnl),
                        safe_float(fees),
                        safe_float(held_minutes),
                        safe_float(position["confidence"]),
                        safe_float(position["score"]),
                        safe_str("position_limit_enforced"),
                        safe_str(position.get("market_regime", "unknown")),
                        safe_str(position.get("risk_tier", "unknown")),
                        safe_int(1),  # Is paper trade
                        safe_int(0)   # Not partial exit
                    ))
                    conn.commit()
                    conn.close()
                    
                    # Log the forced exit
                    logger.info(f"{pair_to_close} | 📉 CLOSED (ENFORCED) {position_side.upper()} @ {exit_price:.4f} | " 
                          f"PnL: ${net_pnl:.2f} ({pnl*100:.2f}%) | Held: {held_minutes:.1f}m")
                    
                    # Remove the position
                    del open_positions[pair_to_close]
                    if pair_to_close in partial_exits:
                        del partial_exits[pair_to_close]
                
                except Exception as e:
                    logger.error(f"Error enforcing position limit for {pair_to_close}: {e}")
                    logger.error(traceback.format_exc())

# === FIXED POSITION MANAGEMENT FUNCTIONS ===

def enter_position(signal):
    """Process a trading signal and enter a position - FIXED VERSION"""
    global cash, open_positions, daily_stats
    
    pair_key = signal["pair"]
    symbol = signal["symbol"]
    side = signal["side"]
    price = signal["price"]
    confidence = signal["confidence"]
    score = signal["score"]
    features = signal.get("features", {})
    market_regime = signal.get("market_regime", "normal")
    
    # Mark signal as acted upon in database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE signals SET acted_upon = 1 WHERE pair = ? ORDER BY timestamp DESC LIMIT 1", 
        (pair_key,)
    )
    conn.commit()
    conn.close()
    
    # Skip if a position is already open for this pair
    if pair_key in open_positions and open_positions[pair_key]:
        logger.info(f"Skipping {side} signal for {pair_key}: position already open")
        return False
    
    # Check if we need to replace an existing position
    if len(open_positions) >= max_concurrent_trades:
        position_to_replace = should_replace_position(signal, open_positions, price_data)
        
        if position_to_replace:
            # Close the position first
            success = force_close_position(position_to_replace, "replaced")
            if not success:
                logger.error(f"Failed to close position {position_to_replace} for replacement")
                return False
        else:
            logger.info(f"Skipping {side} signal for {pair_key}: maximum positions reached")
            return False
    
    # Calculate position sizing
    recent_volatility = price_data.get(pair_key, {}).get("volatility", 0.02)
    position_value, leverage, risk_tier = calculate_dynamic_position_size(
        signal, cash, pair_key, recent_volatility, market_regime, open_positions
    )
    
    # Skip if position too small
    if position_value < 50:
        logger.info(f"Skipping {side} signal for {pair_key}: position value too small (${position_value:.2f})")
        return False
    
    # Check if we have enough cash for margin fee
    # Calculate realistic margin fees
    margin_rates = get_margin_fees_for_pair(pair_key)
    margin_fee_needed = position_value * margin_rates["open_fee"]
    if cash < margin_fee_needed:
        logger.warning(f"Insufficient cash for margin fee on {pair_key}: need ${margin_fee_needed:.2f}, have ${cash:.2f}")
        return False
    
    # Apply slippage to entry price
    execution_price = price * (1 + slippage_per_pair[pair_key]) if side == "buy" else price * (1 - slippage_per_pair[pair_key])
    
    # Calculate quantity
    quantity = position_value / execution_price
    
    try:
        # === PAPER TRADING EXECUTION ===
        # For live trading, uncomment the actual kraken order execution
        
        # MARGIN TRADING: Only deduct margin fee from cash (not full position value)
        # MARGIN TRADING: Only deduct margin fee from cash (not full position value)
        margin_rates = get_margin_fees_for_pair(pair_key)
        margin_fee = position_value * margin_rates["open_fee"]
        cash -= margin_fee
        # Note: In margin trading, we don't deduct the full position_value from cash
        
        # Create position record
        open_positions[pair_key] = {
            "entry_time": datetime.utcnow(),
            "entry_price": execution_price,
            "position_value": position_value,
            "quantity": quantity,
            "side": side,
            "leverage": leverage,
            "confidence": confidence,
            "score": score, 
            "highest_price": execution_price,
            "lowest_price": execution_price,
            "features": features,
            "market_regime": market_regime,
            "risk_tier": risk_tier,
            "recent_returns": [],
            "last_price": execution_price
        }
        
        logger.info(f"{pair_key} | 📈 OPENED {side.upper()} @ {execution_price:.4f} | " 
                  f"Conf: {confidence:.2f} | Score: {score:.2f} | "
                  f"Size: ${position_value:.2f} | Lev: {leverage}x | Tier: {risk_tier}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error entering position for {pair_key}: {e}")
        logger.error(traceback.format_exc())
        return False

def force_close_position(pair_key, exit_reason):
    """Force close a position immediately - helper function"""
    global cash, open_positions, daily_stats
    
    if pair_key not in open_positions or not open_positions[pair_key]:
        return False
    
    position = open_positions[pair_key]
    
    try:
        # Get current price
        symbol = PAIRS[pair_key]
        ticker = kraken.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # Apply slippage
        exit_price = current_price * (1 - slippage_per_pair[pair_key]) if position["side"] == "buy" else current_price * (1 + slippage_per_pair[pair_key])
        
        # Calculate metrics
        entry_price = position["entry_price"]
        entry_time = position["entry_time"]
        side = position["side"]
        leverage = position["leverage"]
        position_value = position["position_value"]
        quantity = position.get("quantity", position_value / entry_price)
        held_minutes = (datetime.utcnow() - entry_time).total_seconds() / 60
        
        # Calculate PnL
        if side == "buy":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
        
        # Calculate fees
        # Calculate realistic fees for force close
        fee_breakdown = calculate_realistic_fees(pair_key, position_value, entry_price, leverage, held_minutes)
        total_fees = fee_breakdown["total_fees"]
        
        # Calculate gross and net PnL
        gross_pnl = position_value * pnl_pct * leverage
        net_pnl = gross_pnl - total_fees
        
        # Update cash (margin trading: return margin fee + PnL)
        # Return margin using realistic fee structure
        margin_rates = get_margin_fees_for_pair(pair_key)
        margin_returned = position_value * margin_rates["open_fee"]
        cash += margin_returned + net_pnl
        
        # Record trade
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO trades 
        (timestamp, pair, side, entry_price, exit_price, quantity, position_value, leverage,
        entry_time, exit_time, pnl, net_pnl, fees, hold_time, confidence, risk_score, exit_reason, 
        market_regime, risk_tier, is_paper, partial_exit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            safe_str(datetime.utcnow().isoformat()),
            safe_str(pair_key),
            safe_str(side),
            safe_float(entry_price),
            safe_float(exit_price),
            safe_float(quantity),
            safe_float(position_value),
            safe_float(leverage),
            safe_str(entry_time.isoformat()),
            safe_str(datetime.utcnow().isoformat()),
            safe_float(pnl_pct * 100),  # Store as percentage
            safe_float(net_pnl),
            safe_float(total_fees),
            safe_float(held_minutes),
            safe_float(position["confidence"]),
            safe_float(position["score"]),
            safe_str(exit_reason),
            safe_str(position.get("market_regime", "unknown")),
            safe_str(position.get("risk_tier", "unknown")),
            safe_int(1),  # Is paper trade
            safe_int(0)   # Not partial exit
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"{pair_key} | 📉 CLOSED {side.upper()} @ {exit_price:.4f} | Entry: {entry_price:.4f} | " 
              f"PnL: ${net_pnl:.2f} ({pnl_pct*100:.2f}%) | Held: {held_minutes:.1f}m | Reason: {exit_reason}")
        
        # Remove position
        del open_positions[pair_key]
        if pair_key in partial_exits:
            del partial_exits[pair_key]
        
        return True
        
    except Exception as e:
        logger.error(f"Error force closing position {pair_key}: {e}")
        return False

def check_exit_conditions():
    """Check exit conditions for all open positions - IMPROVED VERSION"""
    global cash, open_positions, daily_stats, partial_exits
    
    current_time = datetime.utcnow()
    
    for pair_key in list(open_positions.keys()):
        position = open_positions[pair_key]
        
        if not position:
            continue
        
        try:
            # Get current price
            symbol = PAIRS[pair_key]
            ticker = kraken.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Update position tracking
            position["last_price"] = current_price
            
            # Calculate metrics
            entry_price = position["entry_price"]
            entry_time = position["entry_time"]
            side = position["side"]
            leverage = position["leverage"]
            position_value = position["position_value"]
            held_minutes = (current_time - entry_time).total_seconds() / 60
            
            # Apply slippage for exit
            exit_price = current_price * (1 - slippage_per_pair[pair_key]) if side == "buy" else current_price * (1 + slippage_per_pair[pair_key])
            
            # Calculate current PnL
            if side == "buy":
                pnl_pct = (exit_price - entry_price) / entry_price
                # Update trailing high
                position["highest_price"] = max(position.get("highest_price", entry_price), exit_price)
                drawdown = (exit_price - position["highest_price"]) / position["highest_price"]
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
                # Update trailing low
                position["lowest_price"] = min(position.get("lowest_price", entry_price), exit_price)
                drawdown = (position["lowest_price"] - exit_price) / position["lowest_price"]
            
            # Update recent returns
            if "recent_returns" not in position:
                position["recent_returns"] = []
            
            last_price = position.get("last_price_for_returns", entry_price)
            if current_price != last_price:
                if side == "buy":
                    return_pct = (current_price - last_price) / last_price
                else:
                    return_pct = (last_price - current_price) / last_price
                
                position["recent_returns"].append(return_pct)
                if len(position["recent_returns"]) > 20:
                    position["recent_returns"].pop(0)
                
                position["last_price_for_returns"] = current_price
            
            # Check exit conditions
            has_partial_exit = pair_key in partial_exits
            adjusted_stop_loss = get_adjusted_stop_loss(stop_loss_pct, held_minutes)
            reached_break_even = pnl_pct >= break_even_profit_pct
            
            effective_stop_loss = adjusted_stop_loss
            if reached_break_even:
                effective_stop_loss = -0.001  # Move to break-even
            
            # Exit condition checks
            should_exit_full = False
            should_exit_partial = False
            exit_reason = ""
            
            if abs(drawdown) >= effective_stop_loss:
                should_exit_full = True
                exit_reason = "stop_loss"
            elif not has_partial_exit and pnl_pct >= take_profit_pct_first_half:
                should_exit_partial = True
                exit_reason = "partial_take_profit"
            elif has_partial_exit and pnl_pct >= take_profit_pct_full:
                should_exit_full = True
                exit_reason = "take_profit"
            elif held_minutes >= max_holding_minutes:
                should_exit_full = True
                exit_reason = "time_limit"
            
            # Execute exits
            if should_exit_partial:
                execute_partial_exit(pair_key, position, exit_price, current_time)
            elif should_exit_full:
                force_close_position(pair_key, exit_reason)
                
        except Exception as e:
            logger.error(f"Error checking exit conditions for {pair_key}: {e}")

def execute_partial_exit(pair_key, position, exit_price, current_time):
    """Execute a partial exit (50% of position)"""
    global cash, partial_exits
    
    # Calculate partial metrics
    original_position_value = position["position_value"]
    partial_position_value = original_position_value * 0.5
    remaining_position_value = original_position_value - partial_position_value
    
    entry_price = position["entry_price"]
    side = position["side"]
    leverage = position["leverage"]
    
    # Calculate PnL
    if side == "buy":
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - exit_price) / entry_price
    
    # Calculate fees and PnL
    exit_fee = partial_position_value * TRADING_FEES["taker_fee"]
    gross_pnl = partial_position_value * pnl_pct * leverage
    net_pnl = gross_pnl - exit_fee
    
    # Update cash (margin trading: return partial margin fee + PnL)  
    margin_rates = get_margin_fees_for_pair(pair_key)
    margin_rates = get_margin_fees_for_pair(pair_key)
    partial_margin_returned = partial_position_value * margin_rates["open_fee"]
    cash += partial_margin_returned + net_pnl
    
    # Update position
    position["position_value"] = remaining_position_value
    
    # Mark as partial exit
    partial_exits[pair_key] = {
        "exit_time": current_time,
        "exit_price": exit_price,
        "exit_value": partial_position_value,
        "pnl_pct": pnl_pct,
        "net_pnl": net_pnl
    }
    
    # Record partial exit trade
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO trades 
    (timestamp, pair, side, entry_price, exit_price, quantity, position_value, leverage,
    entry_time, exit_time, pnl, net_pnl, fees, hold_time, confidence, risk_score, exit_reason, 
    market_regime, risk_tier, is_paper, partial_exit)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        safe_str(current_time.isoformat()),
        safe_str(pair_key),
        safe_str(side),
        safe_float(entry_price),
        safe_float(exit_price),
        safe_float(partial_position_value / entry_price),  # quantity calculation
        safe_float(partial_position_value),
        safe_float(leverage),
        safe_str(position["entry_time"].isoformat()),  # use position's entry_time
        safe_str(current_time.isoformat()),
        safe_float(pnl_pct * 100),  # Store as percentage
        safe_float(net_pnl),
        safe_float(exit_fee),
        safe_float((current_time - position["entry_time"]).total_seconds() / 60),  # held_minutes calculation
        safe_float(position["confidence"]),
        safe_float(position["score"]),
        safe_str("partial_take_profit"),
        safe_str(position.get("market_regime", "unknown")),
        safe_str(position.get("risk_tier", "unknown")),
        safe_int(1),  # Is paper trade
        safe_int(1)   # Is partial exit
    ))
    conn.commit()
    conn.close()
    
    logger.info(f"{pair_key} | 📊 PARTIAL EXIT {side.upper()} @ {exit_price:.4f} | " 
              f"PnL: ${net_pnl:.2f} ({pnl_pct*100:.2f}%) | 50% of position closed")

# === Portfolio Management and Analytics Functions ===

# Make sure this function adds position values:
def calculate_total_equity():
    """Calculate total equity including cash and open positions with unrealized PnL"""
    global cash, open_positions
    
    # Start with cash
    total = cash
    
    # Add position values AND unrealized PnL for each position
    for pair_key, position in open_positions.items():
        if not position:
            continue
            
        try:
            # Get current price
            symbol = PAIRS[pair_key]
            ticker = kraken.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate position metrics
            entry_price = position["entry_price"]
            side = position["side"]
            leverage = position["leverage"]
            position_value = position["position_value"]
            
            # Calculate position size (quantity)
            position_size = position_value / entry_price
            
            # Calculate current market value of the position
            current_market_value = position_size * current_price
            
            # Calculate PnL
            if side == "buy":
                # For long positions: current value - initial investment
                unrealized_pnl = (current_market_value - position_value) * leverage
            else:  # sell/short
                # For short positions: initial investment - current value  
                unrealized_pnl = (position_value - current_market_value) * leverage
            
            # Add unrealized PnL only (in margin trading, position_value is notional)
            total += unrealized_pnl
            
        except Exception as e:
            logger.error(f"Error calculating equity for {pair_key}: {e}")
            # If we can't get current price, just add the position value
            total += position.get("position_value", 0)
    
    return total

def reconcile_equity():
    """Reconcile equity calculations to prevent drift - ENHANCED VERSION"""
    global cash, open_positions, total_equity_history
    
    # Calculate expected equity from trades
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(net_pnl) FROM trades WHERE is_paper = 1")
    total_realized_pnl = cursor.fetchone()[0] or 0
    conn.close()
    
    # Expected equity = initial capital + realized PnL from completed trades
    expected_equity_from_trades = initial_capital + total_realized_pnl
    
    # Calculate current margin fees in use (THIS IS THE FIX)
    current_margin_fees = sum(p.get("position_value", 0) * get_margin_fees_for_pair(pair_key)["open_fee"] for pair_key, p in open_positions.items() if p)
    
    # Calculate unrealized PnL
    unrealized_pnl = calculate_unrealized_pnl()
    
    # Current total equity (margin trading)
    current_equity = cash + unrealized_pnl
    
    # Expected equity should be: initial capital + realized PnL + unrealized PnL - margin fees in use
    expected_equity = initial_capital + total_realized_pnl + unrealized_pnl - current_margin_fees
    
    # Check for significant deviation (more than 1% of initial capital)
    deviation = abs(current_equity - expected_equity)
    if deviation > initial_capital * 0.01:
        logger.warning(f"Equity reconciliation needed:")
        logger.warning(f"  Current Equity: ${current_equity:.2f}")
        logger.warning(f"  Expected Equity: ${expected_equity:.2f}")
        logger.warning(f"  Deviation: ${deviation:.2f}")
        logger.warning(f"  Cash: ${cash:.2f}")
        logger.warning(f"  Margin Fees in Use: ${current_margin_fees:.2f}")  # FIXED LINE 1
        logger.warning(f"  Unrealized PnL: ${unrealized_pnl:.2f}")
        logger.warning(f"  Realized PnL: ${total_realized_pnl:.2f}")
        
        # Adjust cash to reconcile (this should be rare)
        required_cash = expected_equity_from_trades - current_margin_fees  # FIXED LINE 2
        if abs(cash - required_cash) > deviation * 0.5:  # Only adjust if cash is significantly off
            logger.warning(f"Adjusting cash from ${cash:.2f} to ${required_cash:.2f} to reconcile equity")
            cash = required_cash
        
        return expected_equity
    
    return current_equity

def calculate_unrealized_pnl():
    """Calculate ONLY the unrealized PnL component (gains/losses)"""
    unrealized_pnl_total = 0
    
    for pair_key, position in open_positions.items():
        if not position:
            continue
            
        try:
            # Get current price
            symbol = PAIRS[pair_key]
            ticker = kraken.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate position metrics
            entry_price = position["entry_price"]
            side = position["side"]
            leverage = position["leverage"]
            position_value = position["position_value"]
            
            # Calculate position size
            position_size = position_value / entry_price
            
            # Calculate PnL percentage
            if side == "buy":
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # sell/short
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Apply leverage to PnL
            unrealized_pnl = position_value * pnl_pct * leverage
            unrealized_pnl_total += unrealized_pnl
            
        except Exception as e:
            logger.error(f"Error calculating unrealized PnL for {pair_key}: {e}")
    
    return unrealized_pnl_total

def update_equity_log():
    """Update the equity log with current values - FIXED VERSION"""
    global cash, open_positions, total_equity_history
    
    # Calculate margin fees in use (in margin trading, this is what we actually "invested")
    margin_fees_in_use = sum(p.get("position_value", 0) * get_margin_fees_for_pair(pair_key)["open_fee"] for pair_key, p in open_positions.items() if p)
    
    # Calculate unrealized PnL (gains/losses only)
    unrealized_pnl = calculate_unrealized_pnl()
    
    # Calculate total equity (cash + gains/losses - margin trading doesn't "invest" position_value)
    total_equity = cash + unrealized_pnl
    
    # Calculate portfolio VaR if we have open positions
    portfolio_var = 0
    if open_positions:
        portfolio_var = calculate_portfolio_var(open_positions, price_data) / total_equity if total_equity > 0 else 0
    
    # Log to database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO equity 
    (timestamp, total_equity, cash, positions_value, unrealized_pnl, portfolio_var)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        safe_str(datetime.utcnow().isoformat()),
        safe_float(total_equity),
        safe_float(cash),
        safe_float(margin_fees_in_use),  # Track margin fees instead of position values
        safe_float(unrealized_pnl),
        safe_float(portfolio_var)
    ))
    conn.commit()
    conn.close()
    
    # Update equity history
    total_equity_history.append((datetime.utcnow(), total_equity))
    
    # Trim history if it gets too long
    if len(total_equity_history) > 10000:
        total_equity_history = total_equity_history[-10000:]
    
    # Enhanced logging
    logger.info(f"Current Equity: ${total_equity:.2f} | Cash: ${cash:.2f} | " 
              f"Margin Fees: ${margin_fees_in_use:.2f} | Unrealized PnL: ${unrealized_pnl:.2f} | VaR: {portfolio_var*100:.2f}%")
    
    return total_equity

def check_day_change():
    """Check if the day has changed and reset daily stats"""
    global current_day, daily_stats
    
    now = datetime.utcnow().date()
    
    if now != current_day:
        logger.info(f"Day changed from {current_day} to {now}")
        
        # Report previous day stats
        win_rate = daily_stats["winning_trades_today"] / daily_stats["trades_today"] if daily_stats["trades_today"] > 0 else 0
        daily_pnl = daily_stats["current_capital"] - daily_stats["start_capital"]
        daily_pnl_pct = daily_pnl / daily_stats["start_capital"] * 100 if daily_stats["start_capital"] > 0 else 0
        
        logger.info(f"Daily Summary ({current_day}):")
        logger.info(f"Trades: {daily_stats['trades_today']} | Win Rate: {win_rate*100:.2f}%")
        logger.info(f"PnL: ${daily_pnl:.2f} ({daily_pnl_pct:.2f}%)")
        
        # Reset daily stats
        daily_stats = {
            "start_capital": calculate_total_equity(),
            "current_capital": calculate_total_equity(),
            "trades_today": 0,
            "winning_trades_today": 0,
            "losing_trades_today": 0,
            "avg_win_today": 0,
            "avg_loss_today": 0,
        }
        
        # Update current day
        current_day = now

def queue_signal(signal):
    """Add a signal to the queue with randomized delay"""
    global signal_queues
    
    pair_key = signal["pair"]
    
    # Add random delay (1-3 minutes) as in backtest
    delay_minutes = random.randint(1, 3)
    execute_time = datetime.utcnow() + timedelta(minutes=delay_minutes)
    
    # Add to queue
    signal_queues[pair_key].append({
        "signal": signal,
        "execute_time": execute_time
    })
    
    logger.info(f"Queued {signal['side']} signal for {pair_key} with {delay_minutes}m delay (execute at {execute_time})")

def process_signal_queues():
    """Process all queued signals that are ready to execute"""
    global signal_queues
    
    current_time = datetime.utcnow()
    
    # Process each pair's queue
    for pair_key in list(signal_queues.keys()):
        # Skip if trading is paused
        if trading_paused_until and current_time < trading_paused_until:
            continue
            
        # Process signals ready for execution
        queue = signal_queues[pair_key]
        while queue and queue[0]["execute_time"] <= current_time:
            # Get the next signal
            next_signal = queue.popleft()["signal"]
            
            # Try to enter position
            enter_position(next_signal)

def generate_performance_summary():
    """Generate a performance summary from the database"""
    conn = sqlite3.connect(DB_FILE)
    
    # Get all completed trades
    trades_df = pd.read_sql("SELECT * FROM trades", conn)
    
    if trades_df.empty:
        logger.info("No trades completed yet")
        conn.close()
        return
    
    # Convert timestamp columns to datetime
    for col in ['timestamp', 'entry_time', 'exit_time']:
        if col in trades_df.columns:
            trades_df[col] = pd.to_datetime(trades_df[col])
    
    # Calculate performance metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
    losing_trades = len(trades_df[trades_df['net_pnl'] <= 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    total_profit = trades_df['net_pnl'].sum()
    avg_profit = trades_df['net_pnl'].mean()
    avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].mean() if losing_trades > 0 else 0
    
    profit_factor = abs(trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum() / trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].sum()) if losing_trades > 0 else float('inf')
    
    # Average holding time
    avg_hold_time = trades_df['hold_time'].mean()
    
    # Sharpe ratio from equity curve
    equity_df = pd.read_sql("SELECT * FROM equity ORDER BY timestamp", conn)
    if not equity_df.empty:
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df.set_index('timestamp', inplace=True)
        equity_df['returns'] = equity_df['total_equity'].pct_change()
        sharpe_ratio = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(365 * 24 * 60) if equity_df['returns'].std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Exit reason analysis
    exit_reasons = trades_df['exit_reason'].value_counts()
    
    # Get pair performance
    pair_performance = trades_df.groupby('pair').agg({
        'net_pnl': 'sum',
        'id': 'count',
        'pnl': 'mean'
    }).rename(columns={'id': 'trades', 'pnl': 'avg_pnl_pct'}).sort_values('net_pnl', ascending=False).reset_index()
    
    # Market regime analysis
    regime_performance = trades_df.groupby('market_regime').agg({
        'net_pnl': 'sum',
        'id': 'count',
        'net_pnl': ['sum', 'mean']
    })
    
    # Risk tier analysis
    tier_performance = trades_df.groupby('risk_tier').agg({
        'net_pnl': 'sum',
        'id': 'count',
        'net_pnl': ['sum', 'mean']
    })
    
    # Monthly performance breakdown
    trades_df['month'] = trades_df['exit_time'].dt.strftime('%Y-%m')
    monthly_performance = trades_df.groupby('month').agg({
        'net_pnl': 'sum',
        'id': 'count'
    }).rename(columns={'id': 'trades'}).sort_index()
    
    # Partial vs full exit analysis
    partial_vs_full = trades_df.groupby('partial_exit').agg({
        'net_pnl': ['sum', 'mean'],
        'id': 'count'
    })
    
    # Log performance summary
    logger.info("\n=== INSTITUTIONAL TRADING PERFORMANCE SUMMARY ===")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Win Rate: {win_rate*100:.2f}%")
    logger.info(f"Total Profit: ${total_profit:.2f}")
    logger.info(f"Avg Profit per Trade: ${avg_profit:.2f}")
    logger.info(f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
    logger.info(f"Profit Factor: {profit_factor:.2f}")
    logger.info(f"Avg Hold Time: {avg_hold_time:.2f} minutes")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    logger.info("\nExit Reason Breakdown:")
    for reason, count in exit_reasons.items():
        logger.info(f"{reason}: {count} ({count/total_trades*100:.2f}%)")
    
    logger.info("\nPair Performance:")
    for _, row in pair_performance.iterrows():
        logger.info(f"{row['pair']}: ${row['net_pnl']:.2f} ({row['trades']} trades, {row['avg_pnl_pct']*100:.2f}% avg return)")
    
    # Log monthly performance
    logger.info("\nMonthly Performance:")
    for month, row in monthly_performance.iterrows():
        logger.info(f"{month}: ${row['net_pnl']:.2f} ({row['trades']} trades)")
    
    # Generate recommendations for parameter tuning
    best_pairs = pair_performance.head(3)['pair'].tolist()
    worst_pairs = pair_performance.tail(3)['pair'].tolist()
    
    logger.info("\nStrategy Optimization Recommendations:")
    logger.info(f"Top Performing Pairs: {', '.join(best_pairs)}")
    logger.info(f"Consider increasing allocation weight for: {', '.join(best_pairs)}")
    logger.info(f"Underperforming Pairs: {', '.join(worst_pairs)}")
    logger.info(f"Consider decreasing allocation weight for: {', '.join(worst_pairs)}")
    
    # Most profitable exit reasons
    if not exit_reasons.empty and len(trades_df) > 0:
        exit_pnl = trades_df.groupby('exit_reason')['net_pnl'].mean().sort_values(ascending=False)
        most_profitable_exit = exit_pnl.index[0] if len(exit_pnl) > 0 else "none"
        logger.info(f"Most profitable exit strategy: {most_profitable_exit} (${exit_pnl.iloc[0]:.2f} avg PnL)")
    
    conn.close()

def analyze_market_regimes():
    """Analyze the distribution and performance of market regimes"""
    conn = sqlite3.connect(DB_FILE)
    
    # Get market regime data
    regimes_df = pd.read_sql("SELECT * FROM market_regimes ORDER BY timestamp", conn)
    trades_df = pd.read_sql("SELECT * FROM trades", conn)
    
    if regimes_df.empty or trades_df.empty:
        logger.info("Not enough data to analyze market regimes")
        conn.close()
        return
    
    # Convert timestamp columns to datetime
    regimes_df['timestamp'] = pd.to_datetime(regimes_df['timestamp'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    
    # Analyze regime distribution
    regime_counts = regimes_df['regime'].value_counts()
    
    # Combine with trade performance
    regime_performance = trades_df.groupby('market_regime').agg({
        'net_pnl': ['sum', 'mean', 'count'],
        'pnl': ['mean']
    })
    
    logger.info("\n=== MARKET REGIME ANALYSIS ===")
    logger.info("Regime Distribution:")
    for regime, count in regime_counts.items():
        percentage = count / len(regimes_df) * 100
        logger.info(f"{regime}: {count} occurrences ({percentage:.2f}%)")
    
    logger.info("\nRegime Performance:")
    for regime, stats in regime_performance.iterrows():
        if pd.notna(regime) and isinstance(stats, pd.Series):
            trades_count = stats[('net_pnl', 'count')] if ('net_pnl', 'count') in stats else 0
            total_pnl = stats[('net_pnl', 'sum')] if ('net_pnl', 'sum') in stats else 0
            avg_pnl = stats[('net_pnl', 'mean')] if ('net_pnl', 'mean') in stats else 0
            avg_pct = stats[('pnl', 'mean')] if ('pnl', 'mean') in stats else 0
            
            logger.info(f"{regime}: {trades_count} trades, ${total_pnl:.2f} total PnL, "
                      f"${avg_pnl:.2f} avg PnL, {avg_pct*100:.2f}% avg return")
    
    conn.close()

def analyze_risk_tiers():
    """Analyze performance by risk tier"""
    conn = sqlite3.connect(DB_FILE)
    trades_df = pd.read_sql("SELECT * FROM trades", conn)
    
    if trades_df.empty:
        logger.info("No trades completed yet")
        conn.close()
        return
    
    # Risk tier performance
    tier_performance = trades_df.groupby('risk_tier').agg({
        'net_pnl': ['sum', 'mean'],
        'pnl': ['mean'],
        'id': 'count'
    })
    
    logger.info("\n=== RISK TIER ANALYSIS ===")
    
    for tier, stats in tier_performance.iterrows():
        if pd.notna(tier) and isinstance(stats, pd.Series):
            count = stats[('id', 'count')] if ('id', 'count') in stats else 0
            total_pnl = stats[('net_pnl', 'sum')] if ('net_pnl', 'sum') in stats else 0
            avg_pnl = stats[('net_pnl', 'mean')] if ('net_pnl', 'mean') in stats else 0
            avg_pct = stats[('pnl', 'mean')] if ('pnl', 'mean') in stats else 0
            
            logger.info(f"{tier}: {count} trades, ${total_pnl:.2f} total PnL, "
                      f"${avg_pnl:.2f} avg PnL, {avg_pct*100:.2f}% avg return")
    
    conn.close()

# === Main Trading Loop ===
def main(reset_db=False):
    """Main trading loop for the institutional trading engine"""
    global cash, daily_stats
    
    logger.info("=== Starting Institutional-Style Cryptocurrency Trading Engine ===")
    logger.info(f"Initial Capital: ${initial_capital}")
    logger.info(f"Trading pairs: {list(PAIRS.keys())}")
    
    # Setup database with reset option
    setup_database(reset=reset_db)
    
    # If resetting, set cash to initial_capital
    # In main(), after the check for reset_db=True:
    if reset_db:
        cash = initial_capital
        open_positions.clear()  # Clear all open positions
        active_circuit_breakers.clear()  # Reset circuit breakers
        signal_queues.clear()  # Clear signal queues
        partial_exits.clear()  # Clear partial exits
        signal_quality_history.clear()  # Clear signal history
        total_equity_history.clear()  # Clear equity history
        # Reset price data
        price_data.clear()
        logger.info("Reset all trading state for clean restart")
    
    # Initial equity update
    update_equity_log()
    
    # Initialize reconciliation timer
    main.last_reconcile_time = datetime.utcnow()
    
    # Set initial daily stats
    daily_stats = {
        "start_capital": initial_capital,
        "current_capital": initial_capital,
        "trades_today": 0,
        "winning_trades_today": 0,
        "losing_trades_today": 0,
        "avg_win_today": 0,
        "avg_loss_today": 0,
    }
    
    # Performance summary generation interval (every 6 hours)
    last_summary_time = datetime.utcnow()
    summary_interval = timedelta(hours=6)
    
    # Market regime analysis interval (every 12 hours)
    last_regime_analysis_time = datetime.utcnow()
    regime_analysis_interval = timedelta(hours=12)
    
    # Risk tier analysis interval (every 24 hours)
    last_tier_analysis_time = datetime.utcnow()
    tier_analysis_interval = timedelta(hours=24)
    
    try:
        while True:
            current_time = datetime.utcnow()
            
            # Reconcile equity every hour to prevent drift
            if not hasattr(main, 'last_reconcile_time') or (current_time - main.last_reconcile_time).total_seconds() >= 3600:
                main.last_reconcile_time = current_time
                reconcile_equity()
            
            # Check if the day has changed
            check_day_change()
            
            # Add this line to enforce the position limit
            enforce_position_limit()
            
            # Check for and log circuit breakers
            is_trading_paused = check_circuit_breakers()
            if is_trading_paused:
                logger.info(f"Trading is paused until {trading_paused_until}")
            
            # Update equity log (every 5 minutes)
            if not total_equity_history or (current_time - total_equity_history[-1][0]).total_seconds() >= 300:
                update_equity_log()
            
            # Process existing positions (check for exits)
            check_exit_conditions()
            
            # Process queued signals
            process_signal_queues()
            
            # Generate performance summary periodically
            if (current_time - last_summary_time) >= summary_interval:
                generate_performance_summary()
                last_summary_time = current_time
            
            # Analyze market regimes periodically
            if (current_time - last_regime_analysis_time) >= regime_analysis_interval:
                analyze_market_regimes()
                last_regime_analysis_time = current_time
            
            # Analyze risk tiers periodically
            if (current_time - last_tier_analysis_time) >= tier_analysis_interval:
                analyze_risk_tiers()
                last_tier_analysis_time = current_time
            
            # Skip signal generation if trading is paused
            if is_trading_paused:
                time.sleep(60)  # Check again in 1 minute
                continue
            
            # Generate signals for each pair
            for pair_key, symbol in PAIRS.items():
                try:
                    # Skip if we already have an open position for this pair
                    if pair_key in open_positions and open_positions[pair_key]:
                        continue
                    
                    # Generate signal
                    signal = generate_trading_signal(pair_key, symbol)
                    
                    # Queue valid signals
                    if signal:
                        queue_signal(signal)
                
                except Exception as e:
                    logger.error(f"Error processing {pair_key}: {e}")
                    logger.error(traceback.format_exc())
            
            # Sleep for 1 minute before next iteration
            logger.info(f"Completed market check @ {datetime.utcnow()} UTC | "
                      f"Open positions: {len(open_positions)}/{max_concurrent_trades} | "
                      f"Cash: ${cash:.2f}")
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Trading engine stopped by user")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Generate final performance summary
        generate_performance_summary()
        
        # Log current positions at shutdown
        if open_positions:
            logger.info("Open positions at shutdown:")
            for pair, position in open_positions.items():
                if position:
                    logger.info(f"{pair}: {position['side']} @ {position['entry_price']:.4f}, "
                              f"Size: ${position['position_value']:.2f}, Leverage: {position['leverage']}x")
        
        logger.info("Institutional trading engine shutdown complete")

def print_welcome_message():
    """Print welcome message with parameter settings"""
    print("\n==================================================")
    print("  INSTITUTIONAL-STYLE CRYPTOCURRENCY TRADING ENGINE")
    print("==================================================")
    print("\nAdvanced Features:")
    print("- Dynamic position sizing based on risk tiers")
    print("- Market regime detection and adaptation")
    print("- Portfolio-level VaR controls")
    print("- Advanced technical indicator filtering")
    print("- Partial profit taking and trailing stops")
    print(f"\nTrading {len(PAIRS)} pairs with ${initial_capital:.2f} initial capital")
    print(f"Max concurrent positions: {max_concurrent_trades}")
    print(f"Base risk score threshold: {risk_score_threshold}")
    print(f"Portfolio VaR limit: {MAX_PORTFOLIO_VAR*100:.1f}%")
    print("\nPress Ctrl+C to stop the trading engine")
    print("==================================================\n")

if __name__ == "__main__":
    print_welcome_message()
    main(reset_db=True)