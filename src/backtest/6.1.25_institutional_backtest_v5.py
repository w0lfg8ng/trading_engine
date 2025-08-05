# Advanced Institutional-Style Cryptocurrency Backtest - Enhanced Version (FIXED)
# Implementing live trading engine improvements with parallel processing

import pandas as pd
import xgboost as xgb
import sqlite3
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import deque, defaultdict
import random
from datetime import timedelta, datetime
import multiprocessing as mp
from functools import partial
import sklearn
import filelock
import signal
import sys
import gc
from scipy import stats

# Set scikit-learn to use a single thread per process to avoid nested parallelism
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
sklearn.set_config(working_memory=1024, assume_finite=True)

# Add these imports at the top after the existing imports
import sys
sys.path.append('../../src/training')  # Add path to training modules

# === REGIME DETECTION PARAMETERS (from training) ===
TREND_WINDOW_DAYS = 60      # 60 days for trend detection
VOLATILITY_WINDOW_DAYS = 30 # 30 days for volatility calculation

# Volatility thresholds (based on daily returns)
HIGH_VOLATILITY_THRESHOLD = 0.035   # 3.5% daily volatility
LOW_VOLATILITY_THRESHOLD = 0.015    # 1.5% daily volatility

# Trend thresholds (based on 60-day price change)
BULL_TREND_THRESHOLD = 0.20         # 20% gain over 60 days
BEAR_TREND_THRESHOLD = -0.20        # -20% loss over 60 days

# === REGIME MODEL LOADING (ADD THIS SECTION HERE) ===
def load_regime_model_safe(pair, regime, model_dir="models/xgboost_regime_specific/"):
    """
    Safely load a regime model with validation
    Returns None if model is incomplete or missing
    """
    model_path = f"{model_dir}/{pair}_{regime}_xgb_model.pkl"
    encoder_path = f"{model_dir}/{pair}_{regime}_encoder.pkl"
    features_path = f"{model_dir}/{pair}_{regime}_features.pkl"
    
    # Check all 3 files exist
    if not all(os.path.exists(p) for p in [model_path, encoder_path, features_path]):
        print(f"⚠️  Missing files for {pair}_{regime}")
        return None, None, None
    
    try:
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        features = joblib.load(features_path)
        
        # Validate loaded objects
        if model is None or encoder is None or not features:
            print(f"⚠️  Invalid model data for {pair}_{regime}")
            return None, None, None
            
        return model, encoder, features
    except Exception as e:
        print(f"❌ Failed to load {pair}_{regime}: {e}")
        return None, None, None

def detect_market_regime(df, price_col='close'):
    """
    Detect market regime for each row in the dataframe - SAME AS TRAINING
    Returns: regime string for each row
    """
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    
    # Calculate windows in minutes (assuming 1-minute data)
    trend_window_minutes = TREND_WINDOW_DAYS * 24 * 60
    vol_window_minutes = VOLATILITY_WINDOW_DAYS * 24 * 60
    
    # Initialize regime column
    df['market_regime'] = 'range_normal_vol'
    
    # Calculate rolling trend (60-day price change)
    df['price_change_60d'] = df[price_col].pct_change(periods=trend_window_minutes)
    
    # Calculate rolling volatility (30-day)
    df['returns'] = df[price_col].pct_change()
    df['volatility_30d'] = df['returns'].rolling(window=vol_window_minutes).std() * np.sqrt(1440)  # Daily volatility
    
    for i in range(len(df)):
        if pd.isna(df.loc[i, 'price_change_60d']) or pd.isna(df.loc[i, 'volatility_30d']):
            continue
            
        trend = df.loc[i, 'price_change_60d']
        volatility = df.loc[i, 'volatility_30d']
        
        # Determine trend direction
        if trend >= BULL_TREND_THRESHOLD:
            trend_type = 'bull'
        elif trend <= BEAR_TREND_THRESHOLD:
            trend_type = 'bear'
        else:
            trend_type = 'range'
        
        # Determine volatility level
        if volatility >= HIGH_VOLATILITY_THRESHOLD:
            vol_type = 'high_vol'
        elif volatility <= LOW_VOLATILITY_THRESHOLD:
            vol_type = 'low_vol'
        else:
            vol_type = 'normal_vol'
        
        # Combine into regime
        df.loc[i, 'market_regime'] = f"{trend_type}_{vol_type}"
    
    # Clean up temporary columns
    df.drop(['price_change_60d', 'returns', 'volatility_30d'], axis=1, inplace=True)
    
    return df

def load_regime_model(pair, regime):
    """Load a specific regime model for prediction"""
    # Convert pair name format (btcusd -> XBTUSDT)
    pair_mapping = {
        "btcusd": "XBTUSDT",
        "ethusd": "ETHUSDT", 
        "solusd": "SOLUSDT",
        "xrpusd": "XRPUSDT",
        "adausd": "ADAUSDT",
        "ltcusd": "LTCUSDT",
        "dotusd": "DOTUSDT",
        "xmrusd": "XMRUSDT",
        "dogeusd": "DOGEUSD"
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
        print(f"Error loading regime model for {pair}-{regime}: {e}")
        return None, None, None

# === DATABASE AND PAIRS SETTINGS ===
DB_FILE = "data/kraken_v2.db"

# REMOVED MATICUSD - No longer available on Kraken
PAIRS = [
    "XBTUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "LTCUSDT", "DOTUSDT", "LINKUSDT", "AVAXUSDT"
]

# === Capital and Portfolio Settings ===
initial_capital = 15000
max_concurrent_trades = 5  # Maximum concurrent open positions

# === Professional Portfolio Risk Management Settings ===
# VaR (Value at Risk) settings - UPDATED FROM LIVE ENGINE
MAX_PORTFOLIO_VAR = 0.035  # Increased from 0.025 to 0.035 (live engine value)
VAR_CONFIDENCE_LEVEL = 0.95  # VaR confidence level (95%)

# Position sizing and leverage
BASE_POSITION_SIZE_LIMIT = 0.25  # Maximum position size as fraction of capital
MAX_CONCENTRATED_CAPITAL_PCT = 0.60  # Maximum percentage of capital in similar assets
MAX_TOTAL_LEVERAGE = 12  # Maximum portfolio-wide leverage
TARGET_PORTFOLIO_SHARPE = 2.0  # Target portfolio Sharpe ratio

# Signal quality thresholds
risk_score_threshold = 0.42  # Increased from 0.35 to be more selective
# UPDATED REPLACEMENT LOGIC FROM LIVE ENGINE
MIN_REPLACEMENT_IMPROVEMENT = 1.25  # Increased from 1.10 to 1.25 (25% improvement required)
MIN_HOLD_TIME_FOR_REPLACEMENT = 30  # NEW: Minimum 30 minutes before allowing replacement
EXCEPTIONAL_SIGNAL_REPLACEMENT_THRESHOLD = 0.85  # NEW: Very high confidence needed for quick replacement
EXCEPTIONAL_SIGNAL_THRESHOLD = 0.85  # Threshold for exceptional signals (used in position sizing)
REPLACEMENT_LOOKBACK_WINDOW = 50  # Reduced from 100 to 50 (live engine value)

# Pair-specific risk thresholds based on performance - REMOVED MATICUSD
pair_risk_thresholds = {
    "DOTUSDT": 0.35,
    "ADAUSDT": 0.35,
    "ETHUSDT": 0.40,
    "XRPUSDT": 0.45,  # Higher threshold for poor performers
    "LTCUSDT": 0.45,
    "LINKUSDT": 0.45,
    "SOLUSDT": 0.45,
    "XBTUSDT": 0.40,
    "AVAXUSDT": 0.20,
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
# Trading fees (actual Kraken rates for $0+ volume tier)
TRADING_FEES = {
    "maker_fee": 0.0025,  # 0.25% - when you add liquidity
    "taker_fee": 0.0040,  # 0.40% - when you take liquidity (use this for backtesting)
}

# Margin fees per Kraken documentation (use conservative high-end rates)
MARGIN_FEES = {
    # BTC gets preferential rates
    "BTC": {
        "open_fee": 0.0002,      # 0.02% (high end of 0.01-0.02% range)
        "rollover_fee": 0.0002,  # 0.02% per 4-hour period (high end of 0.01-0.02% range)
    },
    # All other cryptos
    "DEFAULT": {
        "open_fee": 0.0004,      # 0.04% (high end of 0.02-0.04% range)
        "rollover_fee": 0.0004,  # 0.04% per 4-hour period (high end of 0.02-0.04% range)
    },
    "rollover_period_hours": 4,  # Kraken charges every 4 hours
}

# Exact Kraken maximum leverage limits
KRAKEN_MAX_LEVERAGE = {
    "XBTUSDT": 5,     # Bitcoin: up to 5x
    "ETHUSDT": 5,     # Ethereum: up to 5x
    "SOLUSDT": 3,     # Solana: up to 3x
    "XRPUSDT": 5,     # XRP: up to 5x
    "ADAUSDT": 3,     # Cardano: up to 3x
    "LTCUSDT": 3,     # Litecoin: up to 3x
    "DOTUSDT": 3,     # Polkadot: up to 3x
    "LINKUSDT": 3,    # Chainlink: up to 3x
    "AVAXUSDT": 3,    # Avalanche: up to 3x
}

# === Position Management ===
max_position_size = 3000
stop_loss_pct = 0.04  # Reduced from 5% to 4%
take_profit_pct_first_half = 0.025  # Exit 50% at 2.5% profit
take_profit_pct_full = 0.05  # Exit remaining at 5% profit
break_even_profit_pct = 0.02  # Move stop to break-even when profit reaches 2%
max_holding_minutes = 18 * 60  # Extended from 12 hours to 18 hours

# === Liquidity Management ===
# Dynamic volume percentage based on pair liquidity - REMOVED MATICUSD
volume_pct_per_pair = {
    "XBTUSDT": 0.10, "ETHUSDT": 0.08, "SOLUSDT": 0.05,
    "XRPUSDT": 0.05, "ADAUSDT": 0.04,
    "LTCUSDT": 0.05, "DOTUSDT": 0.04, "LINKUSDT": 0.03, 
    "AVAXUSDT": 0.02,
}

# === Leverage Settings ===
# Adjusted leverage settings based on performance - REMOVED MATICUSD
max_leverage_per_pair = KRAKEN_MAX_LEVERAGE.copy()

# Conservative overrides for risk management (optional - comment out for max leverage)
max_leverage_per_pair.update({
    "XBTUSDT": 3,    # Use 3x instead of 5x max for safety
    "ETHUSDT": 3,    # Use 3x instead of 5x max for safety
    "SOLUSDT": 2,    # Use 2x instead of 3x max for safety
    "XRPUSDT": 3,    # Use 3x instead of 5x max for safety
    "ADAUSDT": 2,    # Keep at 2x max
    "LTCUSDT": 2,    # Use 2x instead of 3x max for safety
    "DOTUSDT": 2,    # Use 2x instead of 3x max for safety
    "LINKUSDT": 2,   # Use 2x instead of 3x max for safety
    "AVAXUSDT": 2,   # Use 2x instead of 3x max for safety
})

# === Dynamic capital allocation by pair performance ===
# REMOVED MATICUSD - Updated weights
pair_allocation_weight = {
    "DOTUSDT": 1.5,   # Promoted to top performer
    "ADAUSDT": 1.5,   # Promoted to top performer
    "ETHUSDT": 1.0,   # Standard allocation
    "XRPUSDT": 0.75,  # Reduce allocation for poor performers
    "LTCUSDT": 0.75,
    "LINKUSDT": 0.75,
    "SOLUSDT": 0.5,   # Significantly reduce allocation for worst performer
    "XBTUSDT": 1.0,   # Standard allocation
    "AVAXUSDT": 0.8,  # Slightly reduced allocation
}

# === ENHANCED SLIPPAGE SETTINGS ===
# Slippage values based on actual Kraken trading experience
# These represent ONE-WAY slippage (entry OR exit), so round-trip cost is 2x
# Values calibrated for typical position sizes ($500-$3000)
# Higher values for less liquid pairs, lower for major pairs
# 
# Expected round-trip slippage costs:
# BTC/ETH: ~0.04-0.06% (most liquid)
# Major alts: ~0.08-0.20% 
# Minor alts: ~0.40% (least liquid)
#
# CRITICAL: These match live trading engine exactly for comparison
# === Slippage Settings ===
# REMOVED MATICUSD
slippage_per_pair = {
    "XBTUSDT": 0.0002, "ETHUSDT": 0.0003, "SOLUSDT": 0.0005,
    "XRPUSDT": 0.0010, "ADAUSDT": 0.0012,
    "LTCUSDT": 0.0004, "DOTUSDT": 0.0010, "LINKUSDT": 0.0006,
    "AVAXUSDT": 0.0020,
}

# === Time-Based Filters ===
# Hours when crypto tends to be more active (UTC)
active_hours = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23]

# === Helper Functions for Portfolio Management ===

def get_margin_fees_for_pair(pair):
    """Get margin fees for a specific trading pair"""
    if "XBT" in pair or "BTC" in pair:
        return MARGIN_FEES["BTC"]
    else:
        return MARGIN_FEES["DEFAULT"]  # All crypto pairs

def calculate_realistic_fees(pair, position_value, entry_price, leverage, held_minutes):
    """
    Calculate realistic Kraken-style fees for a position using exact Kraken rates
    """
    # Get pair-specific margin fees
    margin_rates = get_margin_fees_for_pair(pair)
    
    # Trading fees (use taker rate for market orders in backtesting)
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

def detect_current_market_regime(df_recent, lookback=VOLATILITY_LOOKBACK_WINDOW):
    """Detect current market regime based on volatility and trend"""
    if len(df_recent) < lookback:
        return "normal"  # Default regime
    
    # Calculate recent volatility
    returns = df_recent['close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(1440)  # Convert to daily volatility (1440 minutes in a day)
    
    # Calculate trend strength
    price_change = (df_recent['close'].iloc[-1] / df_recent['close'].iloc[-lookback] - 1)
    normalized_change = price_change / volatility if volatility > 0 else 0
    
    # Determine regime
    if volatility > HIGH_VOLATILITY_THRESHOLD:
        return "high_volatility"
    elif normalized_change > 1.5:
        return "strong_trend"
    elif normalized_change < -1.5:
        return "downtrend"
    elif volatility < 0.01:
        return "low_volatility"
    else:
        return "normal"

def calculate_risk_adjusted_score(signal, regime):
    """Calculate risk-adjusted score for a signal considering market regime - ENHANCED FROM LIVE ENGINE"""
    base_score = signal["confidence"] * signal["score"]
    
    # UPDATED: Use live engine regime multipliers (more conservative)
    regime_multipliers = {
        "high_volatility": 0.8,     # Live engine: Less conservative than backtest (was 0.7)
        "strong_trend": 1.2,        # Live engine: Less aggressive than backtest (was 1.3)
        "downtrend": 0.9,           # Live engine: Less conservative than backtest (was 0.8)
        "low_volatility": 1.1,      # Live engine: Less aggressive than backtest (was 1.2)
        "normal": 1.0               # Normal conditions
    }
    
    multiplier = regime_multipliers.get(regime, 1.0)
    
    # ENHANCED: Additional adjustments for signal characteristics (FROM LIVE ENGINE)
    if signal.get("features"):
        # Favor counter-trend in oversold/overbought conditions
        if "rsi" in signal["features"] and signal["features"]["rsi"] is not None:
            rsi = signal["features"]["rsi"]
            if signal["side"] == "buy" and rsi < 30:
                multiplier *= 1.15  # Enhanced from 1.2 to 1.15
            elif signal["side"] == "sell" and rsi > 70:
                multiplier *= 1.15  # Enhanced from 1.2 to 1.15
        
        # NEW: Boost for strong MACD signals (FROM LIVE ENGINE)
        if "macd" in signal["features"] and "macd_signal" in signal["features"]:
            macd = signal["features"]["macd"]
            macd_signal = signal["features"]["macd_signal"]
            if macd is not None and macd_signal is not None:
                macd_strength = abs(macd - macd_signal) / abs(macd_signal) if macd_signal != 0 else 0
                if macd_strength > 0.1:  # Strong MACD divergence
                    multiplier *= 1.1
        
        # NEW: Boost for trend alignment (FROM LIVE ENGINE)
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

def calculate_dynamic_position_size(signal, cash, pair_key, recent_volatility, regime, open_positions):
    """Calculate position size dynamically based on signal quality, volatility, and portfolio state"""
    # Get risk tier for sizing
    risk_tier = determine_risk_tier(signal, regime)
    tier_config = RISK_TIERS[risk_tier]
    
    # Validate cash value to prevent NaN calculations
    if not cash or cash <= 0 or np.isnan(cash):
        cash = initial_capital  # Use initial capital as fallback
        print(f"Warning: Invalid cash value detected, using initial capital instead")
    
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
    concentration_ratio = current_exposure_in_class / cash if cash > 0 else 0
    
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
    max_position_value = cash * min(position_size_pct, BASE_POSITION_SIZE_LIMIT)
    
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
        print(f"Warning: Position size {max_position_value} exceeds reasonable limits. Capping at {max_position_size}.")
        max_position_value = max_position_size
    
    return max_position_value, leverage_cap, risk_tier

def get_correlated_pairs(pair_key, threshold=0.7):
    """Get list of pairs that are highly correlated with the given pair"""
    # Simplified implementation - in production you would use actual correlation data
    # For cryptocurrencies, we can use these simplified groupings
    btc_related = ["XBTUSDT"]
    eth_related = ["ETHUSDT"]
    alt_major = ["SOLUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
    alt_minor = ["LTCUSDT", "XRPUSDT", "AVAXUSDT"]
    
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
    """Determine if a new signal should replace an existing position - ENHANCED FROM LIVE ENGINE"""
    # If we're under the position limit, no need to replace
    if len(open_positions) < max_concurrent_trades:
        return None
    
    # Calculate new signal quality
    regime = detect_current_market_regime(price_data.get(new_signal["pair"], pd.DataFrame()))
    new_signal_quality = calculate_risk_adjusted_score(new_signal, regime)
    
    # NEW: Don't replace positions unless the new signal is exceptional (FROM LIVE ENGINE)
    if new_signal_quality < 0.6:  # Minimum quality threshold for replacement
        return None
    
    # Get all open positions with their quality scores and hold times
    position_candidates = []
    current_time = datetime.utcnow()  # Simulate current time in backtest
    
    for pair, position in open_positions.items():
        if not position:
            continue
        
        # NEW: Calculate how long the position has been held (FROM LIVE ENGINE)
        entry_time = position["entry_time"]
        held_minutes = (current_time - entry_time).total_seconds() / 60
        
        # NEW: Don't replace positions that are too new unless exceptional signal (FROM LIVE ENGINE)
        if held_minutes < MIN_HOLD_TIME_FOR_REPLACEMENT:
            if new_signal["confidence"] < EXCEPTIONAL_SIGNAL_REPLACEMENT_THRESHOLD:
                continue  # Skip this position for replacement
        
        # Calculate position quality based on entry parameters and current performance
        entry_quality = position["confidence"] * position["score"]
        
        # ENHANCED: Get current performance and add more sophisticated scoring (FROM LIVE ENGINE)
        current_quality = entry_quality
        current_pnl_pct = 0  # Default
        
        # Add Sharpe ratio if available
        if "recent_returns" in position and len(position["recent_returns"]) >= 5:
            sharpe = calculate_position_sharpe(position)
            # NEW: Blend entry quality with current performance and Sharpe (FROM LIVE ENGINE)
            performance_adjustment = 1.0 + (current_pnl_pct * 2)  # Boost for winning positions
            current_quality = 0.3 * entry_quality + 0.4 * performance_adjustment + 0.3 * max(0, sharpe / 2)
        
        position_candidates.append((pair, position, current_quality, held_minutes, current_pnl_pct))
    
    if not position_candidates:
        return None
    
    # NEW: Sort by replacement score with enhanced logic (FROM LIVE ENGINE)
    def replacement_score(candidate):
        pair, position, quality, held_minutes, pnl_pct = candidate
        
        # Base score is inverse quality (lower quality = higher replacement score)
        score = 1.0 / max(quality, 0.1)
        
        # NEW: Bonus for losing positions (FROM LIVE ENGINE)
        if pnl_pct < -0.02:  # Losing more than 2%
            score *= 1.5
        
        # NEW: Penalty for winning positions (FROM LIVE ENGINE)
        if pnl_pct > 0.02:  # Winning more than 2%
            score *= 0.5
        
        # NEW: Slight bonus for older positions (FROM LIVE ENGINE)
        if held_minutes > 60:  # More than 1 hour
            score *= 1.1
        
        return score
    
    # Sort by replacement score (highest score = best candidate for replacement)
    position_candidates.sort(key=replacement_score, reverse=True)
    
    # Check if new signal is significantly better than the worst position
    best_candidate = position_candidates[0]
    worst_position_quality = best_candidate[2]
    
    # NEW: Apply stricter improvement threshold (FROM LIVE ENGINE)
    required_improvement = MIN_REPLACEMENT_IMPROVEMENT  # Now 1.25 instead of 1.10
    
    # NEW: Require even higher improvement for winning positions (FROM LIVE ENGINE)
    if best_candidate[4] > 0.01:  # Position is winning
        required_improvement = MIN_REPLACEMENT_IMPROVEMENT * 1.5
    
    # NEW: Allow easier replacement for losing positions (FROM LIVE ENGINE)
    if best_candidate[4] < -0.03:  # Position is losing significantly
        required_improvement = MIN_REPLACEMENT_IMPROVEMENT * 0.8
    
    if new_signal_quality > worst_position_quality * required_improvement:
        print(f"Position replacement criteria met:")
        print(f"  New signal quality: {new_signal_quality:.3f}")
        print(f"  Worst position quality: {worst_position_quality:.3f}")
        print(f"  Required improvement: {required_improvement:.2f}x")
        print(f"  Candidate for replacement: {best_candidate[0]} (PnL: {best_candidate[4]*100:.2f}%, held: {best_candidate[3]:.1f}m)")
        
        return best_candidate[0]  # Return pair to replace
    
    return None

# Simplified approach: Use trade-based equity calculation only
# This avoids the cross-pair pricing complexity entirely
# CORRECTED VERSION - Fix the double-counting bug
def calculate_backtest_equity(cash, trades, open_positions):
    """
    Calculate equity using trade-based approach to avoid cross-pair pricing errors.
    For margin trading: cash represents our available margin + realized PnL.
    """
    # Calculate total realized PnL from completed trades
    realized_pnl = sum(t["net_pnl"] for t in trades)
    
    # Calculate unrealized PnL for open positions
    unrealized_pnl = 0
    for pair, position in open_positions.items():
        if not position:
            continue
            
        # For margin positions, we only track the PnL, not the full position value
        # The position_value represents the margin requirement, not the total exposure
        try:
            # This is a simplified approach - in the backtest we can't get real-time prices
            # for cross-pair calculations, so we use a conservative estimate
            unrealized_pnl += 0  # Conservative: assume positions are at break-even
        except:
            unrealized_pnl += 0
    
    # Total equity = initial capital + realized PnL + unrealized PnL + remaining cash above initial
    total_equity = initial_capital + realized_pnl + unrealized_pnl + max(0, cash - initial_capital)
    
    return total_equity

def verify_cash_flow(cash, trades, iteration):
    """Verify cash flow makes sense"""
    realized_pnl = sum(t["net_pnl"] for t in trades)
    expected_cash_from_trades = initial_capital + realized_pnl
    
    # For margin trading, cash can differ from trade PnL due to margin requirements
    # But it shouldn't be wildly different
    if abs(cash - expected_cash_from_trades) > initial_capital * 2:
        print(f"Warning at iteration {iteration}: Cash flow discrepancy")
        print(f"  Cash: ${cash:.2f}")
        print(f"  Expected from trades: ${expected_cash_from_trades:.2f}")
        print(f"  Difference: ${cash - expected_cash_from_trades:.2f}")
        return False
    return True

# === Signal Handling for Backtest ===
def run_backtest_for_pair(pair):
    """Run institutional-style backtest for a single trading pair"""
    from scipy import stats
    
    print(f"\n{'='*50}")
    print(f"Running ENHANCED institutional backtest for {pair.upper()}...")
    print(f"{'='*50}")

    start_time_pair = datetime.now()
    print(f"[{start_time_pair}] Starting enhanced backtest for {pair.upper()}")

    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql(f"SELECT * FROM features_{pair}_1m_institutional", conn, parse_dates=["timestamp"])
        conn.close()
    except Exception as e:
        print(f"Error loading data for {pair}: {e}")
        return {
            "pair": pair,
            "error": str(e),
            "success": False
        }

    # Note: Regime-specific models will be loaded dynamically during backtest
    # This allows us to switch models based on detected market regime
    print(f"Will load regime-specific XGBoost models dynamically for {pair}")

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Features will be determined dynamically by each regime model
    # This section is no longer needed as models contain their own feature lists
    print(f"Using dynamic feature selection based on regime-specific models")
    
    # Detect market regimes for the entire dataset
    print(f"Detecting market regimes for {pair}...")
    df = detect_market_regime(df)
    
    # Get regime distribution
    regime_counts = df['market_regime'].value_counts()
    print(f"Regime distribution for {pair}:")
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Initialize prediction columns
    df["predicted"] = "hold"
    df["confidence"] = 0.0
    df["regime_model_used"] = "none"
    
    # Load and apply regime-specific models
    successful_regimes = 0
    failed_regimes = 0
    
    for regime in regime_counts.index:
        regime_mask = df['market_regime'] == regime
        regime_data = df[regime_mask].copy()
        
        if len(regime_data) == 0:
            continue
            
        # NEW: Safe regime-specific model loading
        regime_model, regime_encoder, regime_features = load_regime_model_safe(pair, regime)
        
        if regime_model is None:
            print(f"  ⚠️ No model available for {pair}-{regime}, using default predictions")
            failed_regimes += 1
            continue
        
        # Prepare features (using same columns as training)
        available_features = [col for col in regime_features if col in regime_data.columns]
        if len(available_features) != len(regime_features):
            missing = set(regime_features) - set(available_features)
            print(f"  ⚠️ Missing features for {regime}: {missing}")
        
        if len(available_features) < len(regime_features) * 0.8:  # Need at least 80% of features
            print(f"  ⚠️ Too many missing features for {regime}, skipping")
            failed_regimes += 1
            continue
            
        # Make predictions for this regime
        try:
            # Clean data before prediction (same as training)
            prediction_data = regime_data[available_features].copy()
            
            # Replace inf with NaN
            prediction_data = prediction_data.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN with median values
            prediction_data = prediction_data.fillna(prediction_data.median())
            
            # Convert DataFrame to DMatrix (required for xgb.train models)
            dtest = xgb.DMatrix(prediction_data)
            
            # Get probabilities (xgb.train models return probabilities by default)
            regime_probabilities = regime_model.predict(dtest)
            
            # Get class predictions by taking argmax
            regime_predictions = np.argmax(regime_probabilities, axis=1)
            
            # Get confidence scores (max probability)
            regime_confidences = regime_probabilities.max(axis=1)
            
            # Convert predictions back to labels
            regime_labels = regime_encoder.inverse_transform(regime_predictions)
            
            # Update main dataframe
            df.loc[regime_mask, "predicted"] = regime_labels
            df.loc[regime_mask, "confidence"] = regime_confidences
            df.loc[regime_mask, "regime_model_used"] = regime
            
            successful_regimes += 1
            print(f"  ✅ {regime}: {len(regime_data):,} predictions made")
            
        except Exception as e:
            print(f"  ❌ Error making predictions for {regime}: {e}")
            failed_regimes += 1
    
    print(f"Regime model summary: {successful_regimes} successful, {failed_regimes} failed")
    print(f"Final predictions: {df['predicted'].value_counts().to_dict()}")
    print(f"Confidence range: {df['confidence'].min():.2f} to {df['confidence'].max():.2f}")
    
    # Use institutional labels as target (already calculated during training)
    target_col = "institutional_label_12h"
    if target_col not in df.columns:
        print(f"Warning: {target_col} not found, using backup labeling")
        # Fallback to simple labeling if institutional labels missing
        df[target_col] = "hold"
    
    # Free up memory
    gc.collect()  # Force garbage collection before feature selection

    # Define critical features to keep and drop others to save memory
    critical_features = ["open", "high", "low", "close", "volume"]
    if "rsi_14" in df.columns:
        critical_features.append("rsi_14")
    if "macd" in df.columns:
        critical_features.append("macd")
    if "macd_signal" in df.columns:
        critical_features.append("macd_signal")
    if "vwap" in df.columns:
        critical_features.append("vwap")
    if "bb_upper" in df.columns:
        critical_features.append("bb_upper")
    if "bb_lower" in df.columns:
        critical_features.append("bb_lower")

    # Remove non-critical features
    for col in df.columns:
        if col not in critical_features and col not in ["timestamp", "predicted", "confidence"]:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
    
    # Now calculate additional features AFTER making predictions
    # Add hour of day
    df['hour'] = df['timestamp'].dt.hour
    df['is_active_hour'] = df['hour'].isin(active_hours).astype(int)
    
    # Calculate ATR if not present
    if 'atr_14' not in df.columns:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr_14'] = true_range.rolling(14).mean()

    # Calculate additional EMAs for crossover signals
    if 'ema_9' not in df.columns and 'close' in df.columns:
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    
    if 'ema_21' not in df.columns and 'close' in df.columns:
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    # Calculate trend strength
    if 'trend_strength' not in df.columns and 'close' in df.columns:
        # Simple trend strength using linear regression slope
        lookback = 30  # 30-minute lookback for trend calculation
        df['trend_strength'] = 0.0
        for i in range(lookback, len(df)):
            prices = df['close'].iloc[i-lookback:i].values
            x = np.arange(lookback)
            slope, _, _, _, _ = stats.linregress(x, prices)
            # Normalize slope by average price
            avg_price = np.mean(prices)
            df.loc[i, 'trend_strength'] = slope / avg_price * 100 if avg_price > 0 else 0

    # Initialize tracking variables
    cash = initial_capital
    equity_curve = []
    open_positions = {}  # Dictionary mapping pair -> position details
    trades = []
    total_fees_paid = 0
    delay_queue = deque()
    signal_count = 0
    processed_signals = 0
    signal_quality_history = deque(maxlen=REPLACEMENT_LOOKBACK_WINDOW)  # Recent signal quality history
    partial_exits = {}  # Track positions with partial exits

    # Initialize price data for VaR calculations
    price_data = {}

    # DIAGNOSTIC: Check price data quality
    print(f"Price range for {pair}: Min: {df['close'].min():.6f}, Max: {df['close'].max():.6f}")
    print(f"Price standard deviation: {df['close'].std():.6f}")
    print(f"First 5 prices: {df['close'].head().tolist()}")
    print(f"Total data points: {len(df)}")

    for i, row in df.iterrows():
        
        timestamp = row["timestamp"]
        current_price = row["close"]

        # Update price data for VaR calculations
        if i >= VOLATILITY_LOOKBACK_WINDOW:
            recent_df = df.iloc[i-VOLATILITY_LOOKBACK_WINDOW:i+1]
            returns = recent_df['close'].pct_change().dropna()
            price_data[pair] = {
                "volatility": returns.std(),
                "df": recent_df
            }
        else:
            # Default if not enough data
            price_data[pair] = {
                "volatility": 0.02,  # Default 2% volatility
                "df": df.iloc[:i+1] if i > 0 else pd.DataFrame()
            }

        # Process delayed signals
        while delay_queue and delay_queue[0]["execute_time"] <= timestamp:
            delayed = delay_queue.popleft()
            delayed_row = delayed["row"]
            delayed_price = delayed_row["close"]
            delayed_predicted = delayed_row["predicted"]
            delayed_confidence = delayed_row["confidence"]
            processed_signals += 1

            # Extract features for the signal
            signal_features = {
                "rsi": delayed_row.get("rsi_14"),
                "macd": delayed_row.get("macd"),
                "macd_signal": delayed_row.get("macd_signal"),
                "vwap": delayed_row.get("vwap"),
                "bb_upper": delayed_row.get("bb_upper"),
                "bb_lower": delayed_row.get("bb_lower"),
                "ema_9": delayed_row.get("ema_9"),
                "ema_21": delayed_row.get("ema_21"),
                "trend_strength": delayed_row.get("trend_strength")
            }

            # Detect market regime
            market_regime = detect_current_market_regime(price_data[pair]["df"])

            # Update open positions with current price info
            for pos_pair, position in list(open_positions.items()):
                if not position:
                    continue
                
                # Skip if this is the same pair we're processing a new signal for
                if pos_pair == pair:
                    continue
                
                # Update position metrics with current price
                entry_price = position["entry_price"]
                position_side = position["side"]
                
                # Current pricing for existing position
                if pos_pair == pair:  # If it's the same pair we're analyzing
                    update_price = delayed_price
                else:
                    # In a real backtest, you would have the current prices for all pairs
                    # Here we'll use a simplified approach
                    update_price = entry_price  # Placeholder
                
                # Calculate current return
                if position_side == "buy":
                    current_return = (update_price - entry_price) / entry_price
                else:
                    current_return = (entry_price - update_price) / entry_price
                
                # Update recent returns history
                position["recent_returns"].append(current_return)
                if len(position["recent_returns"]) > 20:  # Keep only recent returns
                    position["recent_returns"].pop(0)
                
                # Update highest/lowest price seen
                if position_side == "buy":
                    position["highest_price"] = max(position["highest_price"], update_price)
                else:
                    position["lowest_price"] = min(position["lowest_price"], update_price)

            # Exit Logic - First check existing position for this pair
            if pair in open_positions and open_positions[pair]:
                position = open_positions[pair]
                entry_price = position["entry_price"]
                entry_time = position["entry_time"]
                side = position["side"]
                leverage = position["leverage"]
                held_minutes = (timestamp - entry_time).total_seconds() / 60
                position_value = position["position_value"]
                
                # Track if this position has already had a partial exit
                has_partial_exit = pair in partial_exits
                
                # Enhanced slippage application for position exit (from live engine)
                # Apply slippage to exit price
                exit_price = delayed_price * (1 - slippage_per_pair[pair]) if side == "buy" else delayed_price * (1 + slippage_per_pair[pair])

                # Add execution delay for exits (matching live engine behavior)
                exit_delay_minutes = random.randint(1, 2)  # Slightly faster for exits
                exit_timestamp = delayed_row['timestamp'] + timedelta(minutes=exit_delay_minutes)

                # Calculate total round-trip slippage cost
                entry_slippage = slippage_per_pair[pair]
                exit_slippage = slippage_per_pair[pair]
                total_slippage_cost = (entry_slippage + exit_slippage) * 100  # Convert to percentage

                # Log slippage impact for analysis (ONLY ONCE PER TRADE)
                # DEBUG: Verify slippage is working  
                exit_slippage_cost = abs(exit_price - delayed_price) / delayed_price
                # print(f"DEBUG EXIT: Original price: {delayed_price:.10f}, Slipped price: {exit_price:.10f}, Difference: {exit_slippage_cost*100:.6f}%")
                
                if side == "buy":
                    pnl = (exit_price - entry_price) / entry_price
                    # Update trailing high
                    position["highest_price"] = max(position["highest_price"], exit_price)
                    # Calculate drawdown from highest
                    drawdown = (exit_price - position["highest_price"]) / position["highest_price"]
                else:  # sell
                    pnl = (entry_price - exit_price) / entry_price
                    # Update trailing low
                    position["lowest_price"] = min(position["lowest_price"], exit_price)
                    # Calculate drawdown from lowest (for short positions)
                    drawdown = (position["lowest_price"] - exit_price) / position["lowest_price"]
                
                # Get time-adjusted stop loss
                adjusted_stop_loss = get_adjusted_stop_loss(stop_loss_pct, held_minutes)
                
                # Check if profit is enough for break-even stop
                reached_break_even = pnl >= break_even_profit_pct
                
                # Calculate dynamic trailing stop based on ATR
                atr_value = delayed_row.get('atr_14', delayed_price * 0.01)  # Default if missing
                atr_multiplier = 2.0  # Adjust based on risk preference
                atr_based_stop = atr_value * atr_multiplier / delayed_price
                
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
                elif (side == "buy" and delayed_predicted == "sell" and delayed_confidence >= 0.75) or \
                     (side == "sell" and delayed_predicted == "buy" and delayed_confidence >= 0.75):
                    should_exit_full = True
                    exit_reason = "reversal_signal"
                
                # Handle partial exit
                if should_exit_partial and not has_partial_exit:
                    # Calculate partial position size (50%)
                    partial_position_value = position_value * 0.5
                    remaining_position_value = position_value - partial_position_value
    
                    # Calculate fees for partial exit
                    # Calculate realistic fees for partial exit
                    partial_fees = calculate_realistic_fees(pair, partial_position_value, entry_price, leverage, held_minutes)
                    exit_fee = partial_fees["exit_fee"]
                    # Note: margin and rollover fees are already paid on the full position
    
                    # Calculate position size and PnL with proper validation
                    partial_position_size = partial_position_value / entry_price if entry_price != 0 else 0
    
                    # Safety check to prevent unrealistic values
                    if np.isnan(partial_position_size) or np.isinf(partial_position_size) or partial_position_size <= 0:
                        print(f"Warning: Invalid partial position size for {pair}: {partial_position_size}, setting to 0")
                        partial_position_size = 0
    
                    # Calculate price difference with caps for realism
                    price_diff = exit_price - entry_price if side == "buy" else entry_price - exit_price
                    if abs(price_diff) / entry_price > 0.10:  # More than 10% move is suspicious for short timeframes
                        print(f"Warning: Extreme price movement in partial exit: {price_diff / entry_price * 100:.2f}%. Limiting impact.")
                        price_diff = np.sign(price_diff) * entry_price * 0.10  # Cap at 10% move
    
                    # Calculate PnL with leverage
                    partial_gross_pnl = partial_position_size * price_diff * leverage
    
                    # Validate PnL calculation
                    if np.isnan(partial_gross_pnl) or np.isinf(partial_gross_pnl):
                        print(f"Warning: Invalid partial PnL calculation for {pair}: entry={entry_price}, exit={exit_price}, leverage={leverage}")
                        partial_gross_pnl = 0
    
                    # Cap PnL at a reasonable multiple of position value to prevent cascading issues
                    max_reasonable_pnl = partial_position_value * leverage * 0.10  # Max 10% return per trade
                    if abs(partial_gross_pnl) > max_reasonable_pnl:
                        print(f"Warning: Partial PnL {partial_gross_pnl} exceeds reasonable limit. Capping at {max_reasonable_pnl}")
                        partial_gross_pnl = np.sign(partial_gross_pnl) * max_reasonable_pnl
    
                    partial_net_pnl = partial_gross_pnl - exit_fee
    
                    # Update cash
                    # Update cash (for margin trading: return margin + PnL)
                    margin_rates = get_margin_fees_for_pair(pair)
                    partial_margin_returned = partial_position_value * margin_rates["open_fee"]  
                    cash += partial_margin_returned + partial_net_pnl
    
                    # Update remaining position value
                    position["position_value"] = remaining_position_value
    
                    # Mark this position as having a partial exit
                    partial_exits[pair] = {
                        "exit_time": timestamp,
                        "exit_price": exit_price,
                        "exit_value": partial_position_value,
                        "pnl": pnl,
                        "net_pnl": partial_net_pnl
                    }
    
                    # Record partial exit as a trade to properly track win/loss statistics
                    partial_trade = {
                        "pair": pair,
                        "entry": position["entry_time"],
                        "exit": timestamp,
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "position_value": partial_position_value,
                        "leverage": leverage,
                        "pnl": pnl,
                        "net_pnl": partial_net_pnl,
                        "held": held_minutes,
                        "confidence": position["confidence"],
                        "exit_reason": "partial_take_profit",
                        "market_regime": position.get("market_regime", "unknown"),
                        "risk_tier": position.get("risk_tier", "unknown"),
                        "partial_exit": True
                    }
                    trades.append(partial_trade)
    
                    print(f"{pair} | 📊 PARTIAL EXIT {side.upper()} @ {timestamp} | Entry: {entry_price:.4f} → Exit: {exit_price:.4f} | " 
                        f"PnL: ${partial_net_pnl:.2f} ({pnl*100:.2f}%) | Held: {held_minutes:.1f}m")
                
                # Handle full position exit
                elif should_exit_full:
                    # Calculate fees
                    # Calculate realistic fees
                    fee_breakdown = calculate_realistic_fees(pair, position_value, entry_price, leverage, held_minutes)
                    fees = fee_breakdown["total_fees"]
    
                    # Calculate PnL with proper validation
                    position_size = position_value / entry_price if entry_price != 0 else 0

                    # Safety check to prevent unrealistic values
                    if np.isnan(position_size) or np.isinf(position_size) or position_size <= 0:
                        print(f"Warning: Invalid position size for {pair}: {position_size}, setting to 0")
                        position_size = 0

                    # Calculate price difference with caps for realism
                    price_diff = exit_price - entry_price if side == "buy" else entry_price - exit_price
                    if abs(price_diff) / entry_price > 0.10:  # More than 10% move is suspicious for short timeframes
                        print(f"Warning: Extreme price movement detected: {price_diff / entry_price * 100:.2f}%. Limiting impact.")
                        price_diff = np.sign(price_diff) * entry_price * 0.10  # Cap at 10% move

                    # Calculate PnL with leverage
                    gross_pnl = position_size * price_diff * leverage

                    # DEBUG: Verify PnL calculation uses slipped price
                    # print(f"DEBUG PnL: Entry: {entry_price:.6f}, Exit: {exit_price:.6f}, Price_diff: {price_diff:.6f}, Gross PnL: ${gross_pnl:.2f}")


                    # Validate PnL calculation
                    if np.isnan(gross_pnl) or np.isinf(gross_pnl):
                        print(f"Warning: Invalid PnL calculation for {pair}: entry={entry_price}, exit={exit_price}, leverage={leverage}")
                        gross_pnl = 0

                    # Cap PnL at a reasonable multiple of position value to prevent cascading issues
                    max_reasonable_pnl = position_value * leverage * 0.10  # Max 10% return per trade
                    if abs(gross_pnl) > max_reasonable_pnl:
                        print(f"Warning: PnL {gross_pnl} exceeds reasonable limit. Capping at {max_reasonable_pnl}")
                        gross_pnl = np.sign(gross_pnl) * max_reasonable_pnl

                    net_pnl = gross_pnl - fees
                    
                    # Update cash
                    # Update cash (for margin trading: return margin + PnL, not full position value)
                    margin_rates = get_margin_fees_for_pair(pair)
                    margin_returned = position_value * margin_rates["open_fee"]
                    cash += margin_returned + net_pnl
                    
                    # Record trade
                    trades.append({
                        "pair": pair,
                        "entry": entry_time,
                        "exit": timestamp,
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "position_value": position_value,
                        "leverage": leverage,
                        "pnl": pnl,
                        "net_pnl": net_pnl,
                        "held": held_minutes,
                        "confidence": position["confidence"],
                        "exit_reason": exit_reason,
                        "market_regime": position.get("market_regime", "unknown"),
                        "risk_tier": position.get("risk_tier", "unknown"),
                        "partial_exit": has_partial_exit
                    })
                    
                    print(f"{pair} | 📉 CLOSED {side.upper()} @ {timestamp} | Entry: {entry_price:.8f} → Exit: {exit_price:.8f} | " 
                        f"PnL: ${net_pnl:.2f} ({pnl*100:.2f}%) | Held: {held_minutes:.1f}m | Reason: {exit_reason}")
                    
                    # Remove position from tracking
                    del open_positions[pair]
                    if pair in partial_exits:
                        del partial_exits[pair]

            # Entry Logic - Only if no position is open for this pair
            if pair not in open_positions or not open_positions.get(pair):
                # Skip if confidence is too low
                if delayed_confidence < 0.4:
                    continue
                
                # Calculate technical score with MORE OR CONDITIONS (relaxed)
                score = 0
                
                # 1. RSI Filter (relaxed thresholds)
                if use_rsi_filter and "rsi_14" in delayed_row:
                    if (delayed_row["rsi_14"] < rsi_oversold and delayed_predicted == "buy") or \
                       (delayed_row["rsi_14"] > rsi_overbought and delayed_predicted == "sell"):
                        score += rsi_weight
                
                # 2. MACD Filter
                if use_macd_filter and "macd" in delayed_row and "macd_signal" in delayed_row:
                    # Buy when MACD crosses above signal, sell when crosses below
                    if (delayed_row["macd"] > delayed_row["macd_signal"] and delayed_predicted == "buy") or \
                       (delayed_row["macd"] < delayed_row["macd_signal"] and delayed_predicted == "sell"):
                        score += macd_weight
                
                # 3. VWAP Filter (correct logic for mean reversion)
                if use_vwap_filter and "vwap" in delayed_row:
                    # Buy when price is below VWAP (undervalued), sell when above VWAP (overvalued)
                    if (delayed_predicted == "buy" and delayed_price < delayed_row["vwap"]) or \
                       (delayed_predicted == "sell" and delayed_price > delayed_row["vwap"]):
                        score += vwap_weight
                
                # 4. Bollinger Bands Filter
                if use_bbands_filter and "bb_upper" in delayed_row and "bb_lower" in delayed_row:
                    if (delayed_predicted == "buy" and delayed_price < delayed_row["bb_lower"]) or \
                       (delayed_predicted == "sell" and delayed_price > delayed_row["bb_upper"]):
                        score += bbands_weight
                
                # 5. EMA Crossover Filter
                if use_ema_filter and "ema_9" in df.columns and "ema_21" in df.columns:
                    idx = df.index[df['timestamp'] == delayed_row['timestamp']]
                    if len(idx) > 0:
                        current_idx = idx[0]
                        if current_idx > 0:  # Make sure we have a previous candle to check
                            # Check EMA crossover
                            current_ema9 = df.at[current_idx, 'ema_9']
                            current_ema21 = df.at[current_idx, 'ema_21']
                            prev_ema9 = df.at[current_idx-1, 'ema_9']
                            prev_ema21 = df.at[current_idx-1, 'ema_21']
                            
                            # Bullish crossover: EMA9 crosses above EMA21
                            bullish_crossover = prev_ema9 <= prev_ema21 and current_ema9 > current_ema21
                            # Bearish crossover: EMA9 crosses below EMA21
                            bearish_crossover = prev_ema9 >= prev_ema21 and current_ema9 < current_ema21
                            
                            if (delayed_predicted == "buy" and bullish_crossover) or \
                               (delayed_predicted == "sell" and bearish_crossover):
                                score += ema_weight
                
                # 6. Trend Strength Filter (new)
                if use_trend_filter and "trend_strength" in delayed_row:
                    trend_str = delayed_row["trend_strength"]
                    if pd.notna(trend_str):
                        # Positive trend strength favors buys, negative favors sells
                        if (delayed_predicted == "buy" and trend_str > 0.02) or \
                           (delayed_predicted == "sell" and trend_str < -0.02):
                            score += trend_strength_weight
                
                # 7. Time-based boost - increase score during active hours
                hour = delayed_row['timestamp'].hour
                time_boost = 0.2 if hour in active_hours else 0
                score += time_boost

                # Count all potential signals for analysis
                signal_count += 1
                # Check pair-specific risk threshold
                pair_threshold = pair_risk_thresholds.get(pair, risk_score_threshold)
                # DEBUG: Add this logging AFTER pair_threshold is defined
                if signal_count <= 10:  # Only log first 10 signals to avoid spam
                    print(f"Signal #{signal_count}: pred={delayed_predicted}, conf={delayed_confidence:.2f}, score={score:.2f}, threshold={pair_threshold}")
                    print(f"  RSI: {signal_features.get('rsi')}, MACD: {signal_features.get('macd')}, VWAP: {signal_features.get('vwap')}")

                # Fix for the indentation error around lines 1102-1105
# Replace the problematic section with this corrected version:

                # Proceed if score exceeds threshold and we have a buy or sell signal
                if score >= pair_threshold and delayed_predicted in ["buy", "sell"]:
                    # Create signal object for position sizing
                    signal = {
                        "pair": pair,
                        "timestamp": timestamp,
                        "price": delayed_price, 
                        "side": delayed_predicted,
                        "confidence": delayed_confidence,
                        "score": score,
                        "features": signal_features
                    }
                    
                    # Detect market regime for this pair
                    market_regime = detect_current_market_regime(price_data[pair]["df"])
                    
                    # Check if we should replace an existing position - ENHANCED LOGIC
                    position_to_replace = None
                    if len(open_positions) >= max_concurrent_trades:
                        position_to_replace = should_replace_position(signal, open_positions, price_data)
                        
                        if position_to_replace:
                            # Close the position to be replaced
                            old_position = open_positions[position_to_replace]
                            
                            old_entry_price = old_position["entry_price"]
                            old_entry_time = old_position["entry_time"]
                            old_side = old_position["side"]
                            old_leverage = old_position["leverage"]
                            old_position_value = old_position["position_value"]
                            old_held_minutes = (timestamp - old_entry_time).total_seconds() / 60
                            
                            # Determine exit price (with slippage)
                            old_exit_price = delayed_price * (1 - slippage_per_pair[position_to_replace]) if old_side == "buy" else delayed_price * (1 + slippage_per_pair[position_to_replace])
                            # Enhanced slippage tracking for position replacement
                            replacement_slippage_cost = abs(old_exit_price - delayed_price) / delayed_price
                            replacement_total_slippage = (slippage_per_pair[position_to_replace] * 2) * 100  # Entry + Exit

                            print(f"Replacement exit slippage for {position_to_replace}: {replacement_slippage_cost*100:.3f}%, Total cost: {replacement_total_slippage:.3f}%")

                            # Track slippage impact on replacement decision
                            replacement_slippage_impact = old_position_value * (slippage_per_pair[position_to_replace] * 2)
                            print(f"Slippage cost for replacement: ${replacement_slippage_impact:.2f}")
                            
                            
                            if old_side == "buy":
                                old_pnl = (old_exit_price - old_entry_price) / old_entry_price
                            else:
                                old_pnl = (old_entry_price - old_exit_price) / old_entry_price
                            
                            # Calculate fees
                            # Calculate realistic fees for position being replaced
                            old_fee_breakdown = calculate_realistic_fees(position_to_replace, old_position_value, old_entry_price, old_leverage, old_held_minutes)
                            old_fees = old_fee_breakdown["total_fees"]
                            total_fees_paid += old_fees
                            
                            # Calculate PnL with leverage
                            old_position_size = old_position_value / old_entry_price
                            old_gross_pnl = old_position_size * (old_exit_price - old_entry_price) * old_leverage if old_side == "buy" else old_position_size * (old_entry_price - old_exit_price) * old_leverage
                            old_net_pnl = old_gross_pnl - old_fees
                            
                            # Update cash
                            # Update cash (for margin trading: return margin + PnL)
                            old_margin_rates = get_margin_fees_for_pair(position_to_replace)
                            old_margin_returned = old_position_value * old_margin_rates["open_fee"]
                            cash += old_margin_returned + old_net_pnl
                            
                            # Record trade
                            trades.append({
                                "pair": position_to_replace,
                                "entry": old_entry_time,
                                "exit": timestamp,
                                "side": old_side,
                                "entry_price": old_entry_price,
                                "exit_price": old_exit_price,
                                "position_value": old_position_value,
                                "leverage": old_leverage,
                                "pnl": old_pnl,
                                "net_pnl": old_net_pnl,
                                "held": old_held_minutes,
                                "confidence": old_position["confidence"],
                                "exit_reason": "replaced_by_stronger_signal",
                                "market_regime": old_position.get("market_regime", "unknown"),
                                "risk_tier": old_position.get("risk_tier", "unknown")
                            })
                            
                            print(f"{position_to_replace} | 🔄 REPLACED {old_side.upper()} @ {timestamp} | Entry: {old_entry_price:.4f} → Exit: {old_exit_price:.4f} | " 
                                  f"PnL: ${old_net_pnl:.2f} ({old_pnl*100:.2f}%) | Held: {old_held_minutes:.1f}m | Reason: replaced_by_stronger_signal")
                            
                            # Remove from open positions
                            del open_positions[position_to_replace]
                    
                    # Enhanced slippage application for position entry (from live engine)
                    # Apply slippage to entry price - BUY orders get worse fill (higher price), SELL orders get worse fill (lower price)
                    execution_price = delayed_price * (1 + slippage_per_pair[pair]) if delayed_predicted == "buy" else delayed_price * (1 - slippage_per_pair[pair])

                    # Add execution delay slippage (simulate 1-3 minute delays from live engine)
                    execution_delay_minutes = random.randint(1, 3)
                    execution_timestamp = delayed_row['timestamp'] + timedelta(minutes=execution_delay_minutes)

                    # Log slippage impact for analysis
                    slippage_cost = abs(execution_price - delayed_price) / delayed_price
                    if delayed_predicted == "buy":
                        slippage_direction = "negative"  # Paying higher price
                    else:
                        slippage_direction = "negative"  # Getting lower price

                    # print(f"Entry slippage for {pair}: {slippage_cost*100:.3f}% ({slippage_direction})")
                    
                    # Calculate position size using institutional approach
                    recent_volatility = price_data[pair]["volatility"]
                    position_value, leverage, risk_tier = calculate_dynamic_position_size(
                        signal, cash, pair, recent_volatility, market_regime, open_positions
                    )
                    
                    # Skip if position too small
                    if position_value < 50:
                        continue
                    
                    # Deduct margin fee from cash
                    # Calculate and deduct realistic margin requirement
                    margin_rates = get_margin_fees_for_pair(pair)
                    margin_open_fee = position_value * margin_rates["open_fee"]

                    # Deduct margin opening fee from cash
                    cash -= margin_open_fee

                    # Validate we have sufficient cash
                    if cash < 0:
                        print(f"Insufficient cash for {pair} position. Required: ${margin_open_fee:.2f}, Available: ${cash + margin_open_fee:.2f}")
                        cash += margin_open_fee  # Revert
                        continue
                    # Note: In margin trading, we don't deduct the full position_value from cash
                    
                    # Create new position with enhanced tracking
                    new_position = {
                        "entry_time": timestamp,
                        "entry_price": execution_price,
                        "position_value": position_value,
                        "side": delayed_predicted,
                        "leverage": leverage,
                        "confidence": delayed_confidence,
                        "score": score,
                        "highest_price": execution_price,
                        "lowest_price": execution_price,
                        "features": signal_features,
                        "risk_tier": risk_tier,
                        "market_regime": market_regime,
                        "recent_returns": []  # To track returns for Sharpe calculation
                    }
                    
                    open_positions[pair] = new_position
                    
                    print(f"{pair} | 📈 OPENED {delayed_predicted.upper()} @ {timestamp} | Price: {execution_price:.8f} | " 
                        f"Conf: {delayed_confidence:.2f} | Score: {score:.2f} | Size: ${position_value:.2f} | Lev: {leverage}x | Tier: {risk_tier}")

        # === FIXED EQUITY CALCULATION ===
        # Use ONLY trade-based equity - this is the accurate representation
        current_equity = initial_capital + sum(t["net_pnl"] for t in trades)

        # Sanity check - if unrealistic, log warning but continue
        if current_equity > initial_capital * 10 or current_equity < initial_capital * 0.1:
            if i % 1000 == 0:  # Log occasionally
                print(f"Warning: Extreme equity value detected: ${current_equity:.2f}")

        # Debug logging every 1000 iterations
        if i % 1000 == 0:
            verify_cash_flow(cash, trades, i)
            trade_pnl = sum(t["net_pnl"] for t in trades)
            positions_value = sum(p.get("position_value", 0) for p in open_positions.values() if p)
            print(f"Debug {i}: Cash: ${cash:.2f}, Trade PnL: ${trade_pnl:.2f}, "
                f"Positions Margin: ${positions_value:.2f}, Equity: ${current_equity:.2f}")

        equity_curve.append((timestamp, current_equity))

        # Queue new signal with randomized latency
        delay_minutes = random.randint(1, 3)
        delay_queue.append({
            "execute_time": timestamp + timedelta(minutes=delay_minutes),
            "row": row
        })

# Force close any open positions at the end of the backtest

        # Queue new signal with randomized latency
            
    # Force close any open positions at the end of the backtest
    for pos_pair, position in list(open_positions.items()):
        if not position:
            continue
            
        final_row = df.iloc[-1]
        final_price = final_row["close"]
        
        # Apply slippage to exit price
        exit_price = final_price * (1 - slippage_per_pair[pos_pair]) if position["side"] == "buy" else final_price * (1 + slippage_per_pair[pos_pair])
        # Enhanced slippage tracking for forced closes at end of backtest
        forced_close_slippage_cost = abs(exit_price - final_price) / final_price
        forced_close_total_slippage = (slippage_per_pair[pos_pair] * 2) * 100  # Entry + Exit

        print(f"Forced close slippage for {pos_pair}: {forced_close_slippage_cost*100:.3f}%, Total round-trip: {forced_close_total_slippage:.3f}%")

        # Calculate total slippage cost for this position
        total_position_slippage_cost = position_value * (slippage_per_pair[pos_pair] * 2)
        print(f"Total slippage cost for forced close: ${total_position_slippage_cost:.2f}")
        
        
        entry_price = position["entry_price"]
        entry_time = position["entry_time"]
        side = position["side"]
        leverage = position["leverage"]
        position_value = position["position_value"]
        held_minutes = (df.iloc[-1]["timestamp"] - entry_time).total_seconds() / 60
        
        if side == "buy":
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price
            
        # Calculate fees
        # Calculate realistic fees for forced close
        fee_breakdown = calculate_realistic_fees(pos_pair, position_value, entry_price, leverage, held_minutes)
        fees = fee_breakdown["total_fees"]
        total_fees_paid += fees
        
        # Calculate PnL with leverage
        position_size = position_value / entry_price
        gross_pnl = position_size * (exit_price - entry_price) * leverage if side == "buy" else position_size * (entry_price - exit_price) * leverage
        net_pnl = gross_pnl - fees
        
        # Update cash (for margin trading: return margin + PnL, not full position value)
        margin_rates = get_margin_fees_for_pair(pos_pair)
        margin_returned = position_value * margin_rates["open_fee"]
        cash += margin_returned + net_pnl
        
        # Record trade
        trades.append({
            "pair": pos_pair,
            "entry": entry_time,
            "exit": df.iloc[-1]["timestamp"],
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_value": position_value,
            "leverage": leverage,
            "pnl": pnl,
            "net_pnl": net_pnl,
            "held": held_minutes,
            "confidence": position["confidence"],
            "exit_reason": "end_of_data",
            "market_regime": position.get("market_regime", "unknown"),
            "risk_tier": position.get("risk_tier", "unknown"),
            "partial_exit": pos_pair in partial_exits
        })
        
        print(f"{pos_pair} | 📉 FORCE CLOSED {side.upper()} @ {df.iloc[-1]['timestamp']} | Entry: {entry_price:.4f} → Exit: {exit_price:.4f} | " 
              f"PnL: ${net_pnl:.2f} ({pnl*100:.2f}%) | Held: {held_minutes:.1f}m | Reason: end_of_data")
        
        # Remove from tracking
        del open_positions[pos_pair]

    # === Summary and Analysis ===
    
    # Convert equity curve to DataFrame
    df_equity = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
    
    # Create results directory
    os.makedirs("results/", exist_ok=True)
    
    # Protect against empty equity curve
    if not df_equity.empty and len(df_equity) > 1:
        # Ensure data is valid before plotting
        if df_equity["equity"].isna().any():
            print(f"Warning: NaN values found in equity curve for {pair}, cleaning data...")
            df_equity = df_equity.dropna(subset=["equity"])
            
        # Ensure we have valid data before plotting
        if len(df_equity) > 1 and not df_equity["equity"].isna().all():
            # Plot equity curve
            plt.figure(figsize=(14, 6))
            plt.plot(df_equity["timestamp"], df_equity["equity"])
            plt.title(f"ENHANCED FIXED Equity Curve - {pair.upper()}")
            plt.grid()
            plt.savefig(f"results/enhanced_fixed_equity_curve_{pair}.png")
            plt.close()
        else:
            print(f"Warning: Insufficient valid data points for equity curve plotting for {pair}")
    else:
        print(f"Warning: Empty equity curve for {pair}, skipping plot")
        
    # Plot drawdown curve
    plt.figure(figsize=(14, 6))
    rolling_max = df_equity["equity"].cummax()
    drawdown = (df_equity["equity"] - rolling_max) / rolling_max * 100
    plt.plot(df_equity["timestamp"], drawdown)
    plt.title(f"ENHANCED FIXED Drawdown Curve (%) - {pair.upper()}")
    plt.grid()
    plt.savefig(f"results/enhanced_fixed_drawdown_curve_{pair}.png")
    plt.close()
    
    # Calculate performance metrics
    final_equity = df_equity["equity"].iloc[-1] if not df_equity.empty else initial_capital
    total_return = (final_equity - initial_capital) / initial_capital
    
    # Calculate alternative return based on trade PnL instead of equity curve
    trade_based_profit = sum(t["net_pnl"] for t in trades)
    trade_based_return = trade_based_profit / initial_capital
    print(f"Trade-based Return: {trade_based_return * 100:.2f}%, Equity-based Return: {total_return * 100:.2f}%")

    # If the difference is extreme, use trade-based return as the source of truth
    if abs(total_return - trade_based_return) > 1.0:  # If difference is more than 100%
        print(f"Warning: Using trade-based return ({trade_based_return * 100:.2f}%) instead of equity-based return ({total_return * 100:.2f}%)")
        total_return = trade_based_return
        final_equity = initial_capital + trade_based_profit
    
    max_dd = (df_equity["equity"].cummax() - df_equity["equity"]).max() if not df_equity.empty else 0
    max_dd_pct = max_dd / df_equity["equity"].cummax().max() if not df_equity.empty and df_equity["equity"].cummax().max() > 0 else 0
    
    wins = [t for t in trades if t["net_pnl"] > 0]
    win_rate = len(wins) / len(trades) if trades else 0
    
    # Calculate Sharpe ratio
    if len(df_equity) > 1:
        try:
            returns = df_equity["equity"].pct_change().dropna()
            # Replace any inf values that might occur
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365 * 24 * 60)
            else:
                print(f"Warning: Insufficient valid returns data for {pair} Sharpe calculation")
                sharpe_ratio = 0
        except Exception as e:
            print(f"Error calculating Sharpe ratio for {pair}: {e}")
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0
        
    # Calculate profit factor
    total_gains = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    total_losses = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
    profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
    
    # Calculate average trade metrics
    avg_trade_pnl = sum(t["net_pnl"] for t in trades) / len(trades) if trades else 0
    avg_win = sum(t["net_pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["net_pnl"] for t in [t for t in trades if t["net_pnl"] <= 0]) / len([t for t in trades if t["net_pnl"] <= 0]) if [t for t in trades if t["net_pnl"] <= 0] else 0
    avg_hold_time = sum(t["held"] for t in trades) / len(trades) if trades else 0
    
    # Calculate signal efficiency
    signal_efficiency = len(trades) / signal_count if signal_count > 0 else 0
    
    # Analyze exit reasons
    exit_reasons = {}
    for trade in trades:
        reason = trade.get("exit_reason", "unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    
    # Analyze performance by market regime
    regime_performance = {}
    for trade in trades:
        regime = trade.get("market_regime", "unknown")
        if regime not in regime_performance:
            regime_performance[regime] = {"count": 0, "net_pnl": 0, "wins": 0}
        
        regime_performance[regime]["count"] += 1
        regime_performance[regime]["net_pnl"] += trade["net_pnl"]
        if trade["net_pnl"] > 0:
            regime_performance[regime]["wins"] += 1
    
    # Calculate win rates and avg PnL by regime
    for regime, stats in regime_performance.items():
        stats["win_rate"] = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
        stats["avg_pnl"] = stats["net_pnl"] / stats["count"] if stats["count"] > 0 else 0
    
    # Analyze performance by risk tier
    tier_performance = {}
    for trade in trades:
        tier = trade.get("risk_tier", "unknown")
        if tier not in tier_performance:
            tier_performance[tier] = {"count": 0, "net_pnl": 0, "wins": 0}
        
        tier_performance[tier]["count"] += 1
        tier_performance[tier]["net_pnl"] += trade["net_pnl"]
        if trade["net_pnl"] > 0:
            tier_performance[tier]["wins"] += 1
    
    # Calculate win rates and avg PnL by tier
    for tier, stats in tier_performance.items():
        stats["win_rate"] = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
        stats["avg_pnl"] = stats["net_pnl"] / stats["count"] if stats["count"] > 0 else 0
    
    # === COMPREHENSIVE SLIPPAGE IMPACT ANALYSIS ===
    print(f"\n🔍 SLIPPAGE IMPACT ANALYSIS for {pair.upper()}:")
    print("="*60)

    # Calculate total slippage cost across all trades
    total_slippage_cost = 0
    slippage_stats_by_pair = {}

    for trade in trades:
        trade_pair = trade.get("pair", pair)
        pair_slippage = slippage_per_pair.get(trade_pair, 0.001)  # Default 0.1%
    
        # Round-trip slippage cost (entry + exit)
        trade_slippage_cost = trade["position_value"] * (pair_slippage * 2)
        total_slippage_cost += trade_slippage_cost
    
        # Track by pair
        if trade_pair not in slippage_stats_by_pair:
            slippage_stats_by_pair[trade_pair] = {"cost": 0, "trades": 0}
    
        slippage_stats_by_pair[trade_pair]["cost"] += trade_slippage_cost
        slippage_stats_by_pair[trade_pair]["trades"] += 1

    # Calculate performance metrics with and without slippage
    total_profit_with_slippage = sum(t["net_pnl"] for t in trades)
    total_profit_without_slippage = total_profit_with_slippage + total_slippage_cost

    slippage_impact_pct = (total_slippage_cost / initial_capital) * 100
    profit_reduction_pct = (total_slippage_cost / abs(total_profit_without_slippage)) * 100 if total_profit_without_slippage != 0 else 0

    print(f"Total Slippage Cost: ${total_slippage_cost:.2f}")
    print(f"Slippage as % of Capital: {slippage_impact_pct:.2f}%")
    print(f"Profit Reduction due to Slippage: {profit_reduction_pct:.2f}%")
    print(f"Average Slippage per Trade: ${total_slippage_cost/len(trades):.2f}" if trades else "No trades")

    # Performance comparison
    return_with_slippage = (total_profit_with_slippage / initial_capital) * 100
    return_without_slippage = (total_profit_without_slippage / initial_capital) * 100

    print(f"\nPERFORMANCE COMPARISON:")
    print(f"Return WITH slippage: {return_with_slippage:.2f}%")
    print(f"Return WITHOUT slippage: {return_without_slippage:.2f}%")
    print(f"Performance degradation: {return_without_slippage - return_with_slippage:.2f}%")

    # Pair-specific slippage analysis
    if slippage_stats_by_pair:
        print(f"\nSLIPPAGE BY PAIR:")
        for pair_name, stats in slippage_stats_by_pair.items():
            avg_slippage_per_trade = stats["cost"] / stats["trades"] if stats["trades"] > 0 else 0
            pair_slippage_pct = (slippage_per_pair.get(pair_name, 0.001) * 2) * 100
            print(f"{pair_name}: ${stats['cost']:.2f} total, ${avg_slippage_per_trade:.2f} avg/trade ({pair_slippage_pct:.3f}% round-trip)")

    # High-slippage trade analysis
    high_slippage_threshold = initial_capital * 0.001  # 0.1% of capital
    high_slippage_trades = []

    for trade in trades:
        trade_pair = trade.get("pair", pair)
        pair_slippage = slippage_per_pair.get(trade_pair, 0.001)
        trade_slippage_cost = trade["position_value"] * (pair_slippage * 2)
    
        if trade_slippage_cost > high_slippage_threshold:
            high_slippage_trades.append({
                "pair": trade_pair,
                "position_value": trade["position_value"],
                "slippage_cost": trade_slippage_cost,
                "slippage_pct": (pair_slippage * 2) * 100
            })

    if high_slippage_trades:
        print(f"\nHIGH SLIPPAGE TRADES (>${high_slippage_threshold:.2f}+):")
        for hst in high_slippage_trades[:5]:  # Show top 5
            print(f"{hst['pair']}: ${hst['slippage_cost']:.2f} cost on ${hst['position_value']:.2f} position ({hst['slippage_pct']:.3f}%)")

    print("="*60)
    
    
    # Print detailed summary
    print(f"\n📊 ENHANCED FIXED Backtest Summary for {pair.upper()}:")
    summary_df = pd.DataFrame({
        "Metric": [
            "Total Return (%)",
            "Profit ($)",
            "Win Rate (%)",
            "Number of Trades",
            "Max Drawdown ($)",
            "Max Drawdown (%)",
            "Total Fees ($)",
            "Sharpe Ratio",
            "Profit Factor",
            "Avg Trade PnL ($)",
            "Avg Win ($)",
            "Avg Loss ($)",
            "Avg Hold Time (min)",
            "Signals Generated",
            "Signal Efficiency (%)"
        ],
        "Value": [
            f"{total_return * 100:.2f}",
            f"{final_equity - initial_capital:.2f}",
            f"{win_rate * 100:.2f}",
            len(trades),
            f"{max_dd:.2f}",
            f"{max_dd_pct * 100:.2f}",
            f"{total_fees_paid:.2f}",
            f"{sharpe_ratio:.2f}",
            f"{profit_factor:.2f}",
            f"{avg_trade_pnl:.2f}",
            f"{avg_win:.2f}",
            f"{avg_loss:.2f}",
            f"{avg_hold_time:.2f}",
            signal_count,
            f"{signal_efficiency * 100:.2f}"
        ]
    })
    print(summary_df.to_string(index=False))
    
    # Print exit reason analysis
    if exit_reasons:
        print(f"\n📊 Exit Reason Analysis for {pair.upper()}:")
        exit_reason_df = pd.DataFrame({
            "Exit Reason": list(exit_reasons.keys()),
            "Count": list(exit_reasons.values()),
            "Percentage": [f"{count/len(trades)*100:.2f}%" for count in exit_reasons.values()] if trades else []
        })
        print(exit_reason_df.to_string(index=False))
    
    # Print market regime analysis
    if regime_performance:
        print(f"\n📊 Market Regime Analysis for {pair.upper()}:")
        regime_df = pd.DataFrame({
            "Regime": list(regime_performance.keys()),
            "Count": [stats["count"] for stats in regime_performance.values()],
            "Net PnL": [f"${stats['net_pnl']:.2f}" for stats in regime_performance.values()],
            "Win Rate": [f"{stats['win_rate']*100:.2f}%" for stats in regime_performance.values()],
            "Avg PnL": [f"${stats['avg_pnl']:.2f}" for stats in regime_performance.values()]
        })
        print(regime_df.to_string(index=False))
    
    # Print risk tier analysis
    if tier_performance:
        print(f"\n📊 Risk Tier Analysis for {pair.upper()}:")
        tier_df = pd.DataFrame({
            "Risk Tier": list(tier_performance.keys()),
            "Count": [stats["count"] for stats in tier_performance.values()],
            "Net PnL": [f"${stats['net_pnl']:.2f}" for stats in tier_performance.values()],
            "Win Rate": [f"{stats['win_rate']*100:.2f}%" for stats in tier_performance.values()],
            "Avg PnL": [f"${stats['avg_pnl']:.2f}" for stats in tier_performance.values()]
        })
        print(tier_df.to_string(index=False))
    
    # Save trade log to CSV
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(f"results/enhanced_fixed_trades_{pair}.csv", index=False)
        
        # Save equity curve to CSV
        df_equity.to_csv(f"results/enhanced_fixed_equity_{pair}.csv", index=False)
    
    # Add completion timestamp logging
    end_time_pair = datetime.now()
    elapsed_time_pair = end_time_pair - start_time_pair
    print(f"[{end_time_pair}] Completed ENHANCED FIXED backtest for {pair.upper()} with {total_return * 100:.2f}% return in {elapsed_time_pair}")
    
    # Return results for cross-pair analysis
    return {
        "pair": pair,
        "return_pct": total_return * 100,
        "profit": final_equity - initial_capital,
        "win_rate": win_rate * 100,
        "trades": len(trades),
        "max_drawdown_pct": max_dd_pct * 100,
        "sharpe": sharpe_ratio,
        "profit_factor": profit_factor,
        "avg_trade_pnl": avg_trade_pnl,
        "exit_reasons": exit_reasons,
        "market_regimes": regime_performance,
        "risk_tiers": tier_performance,
        "success": True
    }

def signal_handler(sig, frame):
    """Handle Ctrl+C interruption"""
    print("\nCtrl+C detected. Gracefully shutting down worker processes...")
    sys.exit(0)

def analyze_portfolio_allocation(results):
    """Analyze allocation across portfolio based on backtest results"""
    if not results:
        return {}
    
    # Extract performance metrics
    performance_metrics = {}
    for result in results:
        if not result.get("success", False):
            continue
            
        pair = result["pair"]
        performance_metrics[pair] = {
            "return_pct": result["return_pct"],
            "sharpe": result["sharpe"],
            "profit_factor": result["profit_factor"],
            "win_rate": result["win_rate"],
            "drawdown": result["max_drawdown_pct"]
        }
    
    # Calculate normalized scores for each metric (higher is better)
    normalized_scores = {}
    
    # Get max and min values for normalization
    max_return = max([metrics["return_pct"] for metrics in performance_metrics.values()], default=0)
    min_return = min([metrics["return_pct"] for metrics in performance_metrics.values()], default=0)
    
    max_sharpe = max([metrics["sharpe"] for metrics in performance_metrics.values()], default=0)
    min_sharpe = min([metrics["sharpe"] for metrics in performance_metrics.values()], default=0)
    
    max_pf = max([metrics["profit_factor"] for metrics in performance_metrics.values()], default=1)
    min_pf = min([metrics["profit_factor"] for metrics in performance_metrics.values()], default=1)
    
    max_win_rate = max([metrics["win_rate"] for metrics in performance_metrics.values()], default=0)
    min_win_rate = min([metrics["win_rate"] for metrics in performance_metrics.values()], default=0)
    
    max_drawdown = max([metrics["drawdown"] for metrics in performance_metrics.values()], default=0)
    min_drawdown = min([metrics["drawdown"] for metrics in performance_metrics.values()], default=0)
    
    # Calculate normalized scores
    for pair, metrics in performance_metrics.items():
        # For return, sharpe, profit factor, win rate: higher is better
        return_score = (metrics["return_pct"] - min_return) / (max_return - min_return) if max_return > min_return else 0.5
        sharpe_score = (metrics["sharpe"] - min_sharpe) / (max_sharpe - min_sharpe) if max_sharpe > min_sharpe else 0.5
        pf_score = (metrics["profit_factor"] - min_pf) / (max_pf - min_pf) if max_pf > min_pf else 0.5
        win_rate_score = (metrics["win_rate"] - min_win_rate) / (max_win_rate - min_win_rate) if max_win_rate > min_win_rate else 0.5
        
        # For drawdown: lower is better, so invert the score
        drawdown_score = 1 - (metrics["drawdown"] - min_drawdown) / (max_drawdown - min_drawdown) if max_drawdown > min_drawdown else 0.5
        
        # Calculate combined score with weights
        combined_score = (
            return_score * 0.3 +      # 30% weight for return
            sharpe_score * 0.25 +     # 25% weight for sharpe
            pf_score * 0.2 +          # 20% weight for profit factor
            win_rate_score * 0.15 +   # 15% weight for win rate
            drawdown_score * 0.1      # 10% weight for drawdown
        )
        
        normalized_scores[pair] = combined_score
    
    # Calculate proportional allocation based on score
    total_score = sum(normalized_scores.values())
    
    recommended_allocation = {}
    for pair, score in normalized_scores.items():
        # Calculate allocation percentage (minimum 5%, maximum 30%)
        alloc_pct = min(max(5, (score / total_score) * 100 if total_score > 0 else 10), 30)
        recommended_allocation[pair] = round(alloc_pct, 1)
    
    # Normalize to ensure total is 100%
    total_alloc = sum(recommended_allocation.values())
    scaling_factor = 100 / total_alloc
    
    for pair in recommended_allocation:
        recommended_allocation[pair] = round(recommended_allocation[pair] * scaling_factor, 1)
    
    return recommended_allocation

def risk_tier_distribution_analysis(results):
    """Analyze the performance distribution across risk tiers"""
    if not results:
        return {}
    
    # Aggregate risk tier performance across all pairs
    all_tiers = {}
    
    for result in results:
        if not result.get("success", False) or "risk_tiers" not in result:
            continue
        
        for tier, stats in result["risk_tiers"].items():
            if tier not in all_tiers:
                all_tiers[tier] = {"count": 0, "net_pnl": 0, "wins": 0}
            
            all_tiers[tier]["count"] += stats["count"]
            all_tiers[tier]["net_pnl"] += stats["net_pnl"]
            all_tiers[tier]["wins"] += stats["wins"]
    
    # Calculate aggregated metrics
    for tier, stats in all_tiers.items():
        stats["win_rate"] = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
        stats["avg_pnl"] = stats["net_pnl"] / stats["count"] if stats["count"] > 0 else 0
    
    return all_tiers

def market_regime_distribution_analysis(results):
    """Analyze the performance distribution across market regimes"""
    if not results:
        return {}
    
    # Aggregate market regime performance across all pairs
    all_regimes = {}
    
    for result in results:
        if not result.get("success", False) or "market_regimes" not in result:
            continue
        
        for regime, stats in result["market_regimes"].items():
            if regime not in all_regimes:
                all_regimes[regime] = {"count": 0, "net_pnl": 0, "wins": 0}
            
            all_regimes[regime]["count"] += stats["count"]
            all_regimes[regime]["net_pnl"] += stats["net_pnl"]
            all_regimes[regime]["wins"] += stats["wins"]
    
    # Calculate aggregated metrics
    for regime, stats in all_regimes.items():
        stats["win_rate"] = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
        stats["avg_pnl"] = stats["net_pnl"] / stats["count"] if stats["count"] > 0 else 0
    
    return all_regimes

def exit_reason_analysis(results):
    """Analyze the performance by exit reason across all pairs"""
    if not results:
        return {}
    
    # Aggregate exit reason counts and PnL
    exit_reasons = {}
    
    for result in results:
        if not result.get("success", False) or "exit_reasons" not in result:
            continue
        
        # Get trades for this pair
        pair = result["pair"]
        trades_file = f"results/enhanced_fixed_trades_{pair}.csv"
        
        try:
            trades_df = pd.read_csv(trades_file)
            
            # Group by exit reason
            for reason in result["exit_reasons"].keys():
                reason_trades = trades_df[trades_df["exit_reason"] == reason]
                
                if reason not in exit_reasons:
                    exit_reasons[reason] = {
                        "count": 0, 
                        "net_pnl": 0, 
                        "wins": 0, 
                        "losses": 0,
                        "avg_held_minutes": 0,
                        "total_held_minutes": 0
                    }
                
                exit_reasons[reason]["count"] += len(reason_trades)
                exit_reasons[reason]["net_pnl"] += reason_trades["net_pnl"].sum()
                exit_reasons[reason]["wins"] += len(reason_trades[reason_trades["net_pnl"] > 0])
                exit_reasons[reason]["losses"] += len(reason_trades[reason_trades["net_pnl"] <= 0])
                exit_reasons[reason]["total_held_minutes"] += reason_trades["held"].sum()
        except Exception as e:
            print(f"Error analyzing exit reasons for {pair}: {e}")
    
    # Calculate additional metrics
    for reason, stats in exit_reasons.items():
        stats["win_rate"] = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
        stats["avg_pnl"] = stats["net_pnl"] / stats["count"] if stats["count"] > 0 else 0
        stats["avg_held_minutes"] = stats["total_held_minutes"] / stats["count"] if stats["count"] > 0 else 0
    
    return exit_reasons

# === Main Function ===
if __name__ == "__main__":
    # Use multiprocessing to run backtests in parallel
    start_time = datetime.now()
    print(f"Starting parallel ENHANCED FIXED institutional backtest at {start_time}")
    print("🔧 Testing live trading engine improvements against original backtest with FIXED equity calculation...")
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   
    
    # Calculate optimal number of processes - leave 1 core free for system operations
    n_cores = max(1, mp.cpu_count() - 1)
    print(f"Using {n_cores} CPU cores out of {mp.cpu_count()} available")

    # Set up the process pool with the adjusted number of CPU cores
    with mp.Pool(processes=n_cores) as pool:
        # Run all backtests in parallel
        results = pool.map(run_backtest_for_pair, PAIRS)
    
    # Filter out any failed results
    successful_results = [r for r in results if r.get("success", False)]
    
    # === Cross-Pair Analysis ===
    if successful_results:
        print("\n📊 ENHANCED FIXED Cross-Pair Performance Comparison:")
        cross_pair_df = pd.DataFrame([
            {
                "Pair": r["pair"],
                "Return (%)": f"{r['return_pct']:.2f}",
                "Profit ($)": f"{r['profit']:.2f}",
                "Win Rate (%)": f"{r['win_rate']:.2f}",
                "Trades": r["trades"],
                "Max DD (%)": f"{r['max_drawdown_pct']:.2f}",
                "Sharpe": f"{r['sharpe']:.2f}",
                "Profit Factor": f"{r['profit_factor']:.2f}",
                "Avg PnL ($)": f"{r['avg_trade_pnl']:.2f}"
            } for r in successful_results
        ])
        
        # Sort by return percentage
        cross_pair_df = cross_pair_df.sort_values("Return (%)", key=lambda x: pd.to_numeric(x, errors='coerce'), ascending=False)
        print(cross_pair_df.to_string(index=False))
        
        # Save overall results to CSV
        cross_pair_df.to_csv("results/enhanced_fixed_cross_pair_performance.csv", index=False)
        
        # Create combined performance chart
        plt.figure(figsize=(14, 8))
        
        # Convert string percentages to float
        returns = [float(r["return_pct"]) for r in successful_results]
        pairs = [r["pair"] for r in successful_results]
        
        # Sort by return
        sorted_data = sorted(zip(pairs, returns), key=lambda x: x[1], reverse=True)
        sorted_pairs, sorted_returns = zip(*sorted_data) if sorted_data else ([], [])
        
        if len(sorted_pairs) > 0:
            bars = plt.bar(sorted_pairs, sorted_returns)
            
            # Color bars by performance
            for i, bar in enumerate(bars):
                if sorted_returns[i] > 0:
                    bar.set_color("green")
                else:
                    bar.set_color("red")
            
            plt.title("ENHANCED FIXED Total Return (%) by Cryptocurrency")
            plt.grid(axis="y", alpha=0.3)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.savefig("results/enhanced_fixed_cross_pair_returns.png")
            plt.close()
            
            # Calculate average performance
            avg_return = sum(r["return_pct"] for r in successful_results) / len(successful_results)
            avg_win_rate = sum(r["win_rate"] for r in successful_results) / len(successful_results)
            avg_profit_factor = sum(r["profit_factor"] for r in successful_results) / len(successful_results)
            avg_sharpe = sum(r["sharpe"] for r in successful_results) / len(successful_results)
            
            print(f"\n📊 ENHANCED FIXED Average Performance Across All Pairs:")
            print(f"Average Return: {avg_return:.2f}%")
            print(f"Average Win Rate: {avg_win_rate:.2f}%")
            print(f"Average Profit Factor: {avg_profit_factor:.2f}")
            print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
            
            if len(successful_results) > 0:
                # Sort by return
                sorted_results = sorted(successful_results, key=lambda x: x["return_pct"], reverse=True)
                print(f"Best Performing Pair: {sorted_results[0]['pair']} ({sorted_results[0]['return_pct']:.2f}%)")
                print(f"Worst Performing Pair: {sorted_results[-1]['pair']} ({sorted_results[-1]['return_pct']:.2f}%)")
        
        # === Advanced Institutional Analysis ===
        
        # 1. Portfolio Allocation Analysis
        recommended_allocation = analyze_portfolio_allocation(successful_results)
        
        print("\n📊 ENHANCED FIXED Recommended Portfolio Allocation:")
        alloc_df = pd.DataFrame({
            "Pair": list(recommended_allocation.keys()),
            "Allocation (%)": list(recommended_allocation.values())
        })
        alloc_df = alloc_df.sort_values("Allocation (%)", ascending=False)
        print(alloc_df.to_string(index=False))
        
        # Create allocation pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(alloc_df["Allocation (%)"], labels=alloc_df["Pair"], autopct='%1.1f%%', 
                startangle=90, shadow=True)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title("ENHANCED FIXED Recommended Portfolio Allocation")
        plt.savefig("results/enhanced_fixed_recommended_allocation.png")
        plt.close()
        
        # 2. Risk Tier Analysis
        risk_tier_stats = risk_tier_distribution_analysis(successful_results)
        
        if risk_tier_stats:
            print("\n📊 ENHANCED FIXED Risk Tier Performance Analysis:")
            tier_df = pd.DataFrame({
                "Risk Tier": list(risk_tier_stats.keys()),
                "Count": [stats["count"] for stats in risk_tier_stats.values()],
                "Net PnL": [f"${stats['net_pnl']:.2f}" for stats in risk_tier_stats.values()],
                "Win Rate": [f"{stats['win_rate']*100:.2f}%" for stats in risk_tier_stats.values()],
                "Avg PnL": [f"${stats['avg_pnl']:.2f}" for stats in risk_tier_stats.values()]
            })
            tier_df = tier_df.sort_values("Count", ascending=False)
            print(tier_df.to_string(index=False))
            
            # Create risk tier bar chart
            plt.figure(figsize=(12, 6))
            tiers = [tier for tier in risk_tier_stats.keys()]
            counts = [stats["count"] for stats in risk_tier_stats.values()]
            avg_pnls = [stats["avg_pnl"] for stats in risk_tier_stats.values()]
            
            # Sort by tier level
            tier_order = ["high", "medium", "low", "minimal", "unknown"]
            sorted_data = sorted(zip(tiers, counts, avg_pnls), 
                                key=lambda x: tier_order.index(x[0]) if x[0] in tier_order else 999)
            sorted_tiers, sorted_counts, sorted_avg_pnls = zip(*sorted_data) if sorted_data else ([], [], [])
            
            x = np.arange(len(sorted_tiers))
            width = 0.35
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot count on primary y-axis
            bars1 = ax1.bar(x - width/2, sorted_counts, width, label='Count', color='skyblue')
            ax1.set_ylabel('Trade Count', color='skyblue')
            ax1.tick_params(axis='y', labelcolor='skyblue')
            
            # Plot average PnL on secondary y-axis
            ax2 = ax1.twinx()
            bars2 = ax2.bar(x + width/2, sorted_avg_pnls, width, label='Avg PnL ($)', color='orange')
            ax2.set_ylabel('Average PnL ($)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            
            # Set x-axis ticks and labels
            ax1.set_xticks(x)
            ax1.set_xticklabels(sorted_tiers)
            
            # Add a title and legend
            plt.title('ENHANCED FIXED Risk Tier Performance Analysis')
            fig.tight_layout()
            
            # Create a legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.savefig("results/enhanced_fixed_risk_tier_analysis.png")
            plt.close()
        
        # 3. Market Regime Analysis
        regime_stats = market_regime_distribution_analysis(successful_results)
        
        if regime_stats:
            print("\n📊 ENHANCED FIXED Market Regime Performance Analysis:")
            regime_df = pd.DataFrame({
                "Market Regime": list(regime_stats.keys()),
                "Count": [stats["count"] for stats in regime_stats.values()],
                "Net PnL": [f"${stats['net_pnl']:.2f}" for stats in regime_stats.values()],
                "Win Rate": [f"{stats['win_rate']*100:.2f}%" for stats in regime_stats.values()],
                "Avg PnL": [f"${stats['avg_pnl']:.2f}" for stats in regime_stats.values()]
            })
            regime_df = regime_df.sort_values("Count", ascending=False)
            print(regime_df.to_string(index=False))
            
            # Create market regime bar chart
            plt.figure(figsize=(12, 6))
            regimes = [regime for regime in regime_stats.keys()]
            counts = [stats["count"] for stats in regime_stats.values()]
            win_rates = [stats["win_rate"] * 100 for stats in regime_stats.values()]
            
            # Sort by count
            sorted_data = sorted(zip(regimes, counts, win_rates), key=lambda x: x[1], reverse=True)
            sorted_regimes, sorted_counts, sorted_win_rates = zip(*sorted_data) if sorted_data else ([], [], [])
            
            x = np.arange(len(sorted_regimes))
            width = 0.35
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot count on primary y-axis
            bars1 = ax1.bar(x - width/2, sorted_counts, width, label='Count', color='skyblue')
            ax1.set_ylabel('Trade Count', color='skyblue')
            ax1.tick_params(axis='y', labelcolor='skyblue')
            
            # Plot win rate on secondary y-axis
            ax2 = ax1.twinx()
            bars2 = ax2.bar(x + width/2, sorted_win_rates, width, label='Win Rate (%)', color='green')
            ax2.set_ylabel('Win Rate (%)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            
            # Set x-axis ticks and labels
            ax1.set_xticks(x)
            ax1.set_xticklabels(sorted_regimes)
            
            # Add a title
            plt.title('ENHANCED FIXED Market Regime Performance Analysis')
            fig.tight_layout()
            
            # Create a legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.savefig("results/enhanced_fixed_market_regime_analysis.png")
            plt.close()
        
        # 4. Exit Reason Analysis
        exit_reason_stats = exit_reason_analysis(successful_results)
        
        if exit_reason_stats:
            print("\n📊 ENHANCED FIXED Exit Reason Performance Analysis:")
            reason_df = pd.DataFrame({
                "Exit Reason": list(exit_reason_stats.keys()),
                "Count": [stats["count"] for stats in exit_reason_stats.values()],
                "Win Rate": [f"{stats['win_rate']*100:.2f}%" for stats in exit_reason_stats.values()],
                "Avg PnL": [f"${stats['avg_pnl']:.2f}" for stats in exit_reason_stats.values()],
                "Avg Hold Time": [f"{stats['avg_held_minutes']:.1f}m" for stats in exit_reason_stats.values()]
            })
            reason_df = reason_df.sort_values("Count", ascending=False)
            print(reason_df.to_string(index=False))
            
            # Create exit reason chart
            plt.figure(figsize=(14, 7))
            exit_reasons = [reason for reason in exit_reason_stats.keys()]
            counts = [stats["count"] for stats in exit_reason_stats.values()]
            avg_pnls = [stats["avg_pnl"] for stats in exit_reason_stats.values()]
            
            # Sort by count
            sorted_data = sorted(zip(exit_reasons, counts, avg_pnls), key=lambda x: x[1], reverse=True)
            sorted_reasons, sorted_counts, sorted_avg_pnls = zip(*sorted_data) if sorted_data else ([], [], [])
            
            x = np.arange(len(sorted_reasons))
            width = 0.35
            
            fig, ax1 = plt.subplots(figsize=(14, 7))
            
            # Plot count on primary y-axis
            bars1 = ax1.bar(x - width/2, sorted_counts, width, label='Count', color='skyblue')
            ax1.set_ylabel('Trade Count', color='skyblue')
            ax1.tick_params(axis='y', labelcolor='skyblue')
            
            # Plot average PnL on secondary y-axis
            ax2 = ax1.twinx()
            bars2 = ax2.bar(x + width/2, sorted_avg_pnls, width, label='Avg PnL ($)', color='orange')
            ax2.set_ylabel('Average PnL ($)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            
            # Set x-axis ticks and labels
            ax1.set_xticks(x)
            ax1.set_xticklabels(sorted_reasons, rotation=45, ha='right')
            
            # Add a title
            plt.title('ENHANCED FIXED Exit Reason Analysis')
            plt.subplots_adjust(bottom=0.25)  # Adjust to make room for rotated labels
            
            # Create a legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.savefig("results/enhanced_fixed_exit_reason_analysis.png")
            plt.close()
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"\nENHANCED FIXED institutional backtest completed successfully in {elapsed_time}!")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    
    # Final recommendations
    if successful_results:
        print("\n📈 ENHANCED FIXED Key Recommendations for Live Trading:")
        print("------------------------------------------------------------")
        print("1. Portfolio Allocation:")
        for pair, alloc in sorted(recommended_allocation.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {pair.upper()}: {alloc}% of capital")
        
        # Analyze best performing risk tiers if available
        best_tier = None
        best_tier_pnl = float('-inf')
        if risk_tier_stats:
            for tier, stats in risk_tier_stats.items():
                if stats["avg_pnl"] > best_tier_pnl and stats["count"] >= 5:
                    best_tier_pnl = stats["avg_pnl"]
                    best_tier = tier
        
        if best_tier:
            print(f"\n2. Risk Tier Focus:")
            print(f"   - Prioritize {best_tier} risk tier positions ({risk_tier_stats[best_tier]['win_rate']*100:.1f}% win rate)")
        
        # Analyze best market regimes if available
        best_regime = None
        best_regime_pnl = float('-inf')
        if regime_stats:
            for regime, stats in regime_stats.items():
                if stats["avg_pnl"] > best_regime_pnl and stats["count"] >= 5:
                    best_regime_pnl = stats["avg_pnl"]
                    best_regime = regime
        
        if best_regime:
            print(f"\n3. Market Regime Focus:")
            print(f"   - Capitalize on {best_regime} market conditions ({regime_stats[best_regime]['win_rate']*100:.1f}% win rate)")
        
        # Analyze best exit reasons if available
        best_exit = None
        best_exit_pnl = float('-inf')
        if exit_reason_stats:
            for reason, stats in exit_reason_stats.items():
                if stats["avg_pnl"] > best_exit_pnl and stats["count"] >= 5 and reason != "end_of_data":
                    best_exit_pnl = stats["avg_pnl"]
                    best_exit = reason
        
        if best_exit:
            print(f"\n4. Exit Strategy Optimization:")
            print(f"   - Focus on {best_exit} exits (${exit_reason_stats[best_exit]['avg_pnl']:.2f} avg PnL)")
        
        print("\n5. Top Performing Pairs (ENHANCED FIXED):")
        for r in sorted(successful_results, key=lambda x: x["return_pct"], reverse=True)[:3]:
            print(f"   - {r['pair'].upper()}: {r['return_pct']:.2f}% return, {r['win_rate']:.1f}% win rate, {r['profit_factor']:.2f} profit factor")
        
        print("\n🔬 COMPARE THESE RESULTS TO YOUR ORIGINAL BACKTEST (v2)!")
        print("   - If ENHANCED FIXED performs better → Keep live engine changes")
        print("   - If ENHANCED FIXED performs worse → Revert to original backtest parameters")
        print("   - If results are similar → Live engine equity bug was the main issue")