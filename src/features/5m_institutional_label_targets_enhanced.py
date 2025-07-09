# === institutional_label_targets_enhanced.py ===
# Enhanced version of label_targets_full.py with institutional trading rules
# Accounts for slippage, fees, position sizing, leverage, stops, and take profits

import pandas as pd
import sqlite3
import numpy as np
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# === ENHANCED CONFIG (From your institutional backtest) ===
DB_FILE = "../../data/kraken_v2.db"  # From src/features/ to data/

# Updated pairs to match your V2 data
PAIRS = [
    "XBTUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT",
    "LTCUSDT", "XRPUSDT", "LINKUSDT", "AVAXUSDT"
]

# === INSTITUTIONAL TRADING PARAMETERS ($14K AGGRESSIVE STRATEGY) ===
INITIAL_CAPITAL = 14000
BASE_POSITION_SIZE_LIMIT = 0.35  # Max 35% per trade (aggressive)
MAX_TOTAL_LEVERAGE = 8

# Confidence-based position sizing (AGGRESSIVE for $14K account)
POSITION_SIZING_TIERS = {
    0.85: {"position_pct": 0.35, "leverage": 6},  # Exceptional signals
    0.75: {"position_pct": 0.30, "leverage": 5},  # High confidence
    0.65: {"position_pct": 0.25, "leverage": 4},  # Good confidence  
    0.55: {"position_pct": 0.20, "leverage": 3},  # Medium confidence
    0.45: {"position_pct": 0.15, "leverage": 2},  # Low confidence
    0.00: {"position_pct": 0.10, "leverage": 1},  # Default minimum
}

# === TRADING COSTS (From your backtest) ===
# All your pairs use the same Kraken fees (0.40% taker)
TRADING_COSTS = {
    "XBTUSDT": {"slippage": 0.0002, "fee": 0.0040},  # 0.40% taker
    "ETHUSDT": {"slippage": 0.0003, "fee": 0.0040},  # 0.40% taker
    "SOLUSDT": {"slippage": 0.0005, "fee": 0.0040},  # 0.40% taker
    "ADAUSDT": {"slippage": 0.0012, "fee": 0.0040},  # 0.40% taker
    "DOTUSDT": {"slippage": 0.0010, "fee": 0.0040},  # 0.40% taker
    "LTCUSDT": {"slippage": 0.0004, "fee": 0.0040},  # 0.40% taker
    "XRPUSDT": {"slippage": 0.0010, "fee": 0.0040},  # 0.40% taker
    "LINKUSDT": {"slippage": 0.0008, "fee": 0.0040}, # 0.40% taker
    "AVAXUSDT": {"slippage": 0.0015, "fee": 0.0040}, # 0.40% taker
}

# === AGGRESSIVE TRADING RULES ===
STOP_LOSS_PCT = 0.06  # 6% stop loss (aggressive but safe)
TAKE_PROFIT_LEVELS = [0.06, 0.12, 0.25]  # 6%, 12%, 25%
TAKE_PROFIT_ALLOCATIONS = [0.30, 0.40, 0.30]  # 30%, 40%, 30% exits
MARGIN_OPEN_FEE = 0.0002
MAX_HOLD_HOURS = 48  # Maximum 48 hours per trade

# Minimum profit threshold after ALL costs
MIN_PROFIT_THRESHOLD = 0.012  # Must make at least 1.2% after all costs

def calculate_confidence_score(row: pd.Series) -> float:
    """
    Calculate confidence score based on technical indicators alignment
    This simulates what your ML model confidence would be
    """
    confidence = 0.5  # Base confidence
    
    # RSI confirmation
    if 'rsi_14' in row and pd.notna(row['rsi_14']):
        rsi = row['rsi_14']
        if rsi < 30 or rsi > 70:  # Extreme levels
            confidence += 0.15
        elif rsi < 40 or rsi > 60:  # Moderate levels
            confidence += 0.10
    
    # MACD confirmation
    if all(col in row for col in ['macd', 'macd_signal']) and all(pd.notna(row[col]) for col in ['macd', 'macd_signal']):
        macd_diff = abs(row['macd'] - row['macd_signal'])
        if macd_diff > 0.1:  # Strong MACD signal
            confidence += 0.15
        elif macd_diff > 0.05:  # Moderate MACD signal
            confidence += 0.10
    
    # Volume confirmation
    if 'volume' in row and 'volume_sma_20' in row:
        if pd.notna(row['volume']) and pd.notna(row['volume_sma_20']) and row['volume_sma_20'] > 0:
            volume_ratio = row['volume'] / row['volume_sma_20']
            if volume_ratio > 2.0:  # High volume
                confidence += 0.10
            elif volume_ratio > 1.5:  # Elevated volume
                confidence += 0.05
    
    # Bollinger Bands position
    if all(col in row for col in ['close', 'bb_upper', 'bb_lower']):
        if all(pd.notna(row[col]) for col in ['close', 'bb_upper', 'bb_lower']):
            bb_range = row['bb_upper'] - row['bb_lower']
            if bb_range > 0:
                bb_position = (row['close'] - row['bb_lower']) / bb_range
                if bb_position < 0.1 or bb_position > 0.9:  # Near extremes
                    confidence += 0.10
    
    # Cap confidence at reasonable maximum
    return min(0.90, confidence)

def get_position_sizing(confidence: float) -> Tuple[float, int]:
    """
    Get position size percentage and leverage based on confidence
    Returns: (position_size_pct, leverage)
    """
    for conf_threshold in sorted(POSITION_SIZING_TIERS.keys(), reverse=True):
        if confidence >= conf_threshold:
            tier = POSITION_SIZING_TIERS[conf_threshold]
            return tier["position_pct"], tier["leverage"]
    
    # Default to minimum if below all thresholds
    return 0.05, 1

def calculate_total_costs(pair: str, position_size_pct: float, leverage: int) -> float:
    """
    Calculate total round-trip trading costs as percentage
    """
    costs = TRADING_COSTS.get(pair, {"slippage": 0.001, "fee": 0.0026})
    
    # Round-trip costs
    slippage_cost = costs["slippage"] * 2  # Entry + Exit
    trading_fees = costs["fee"] * 2        # Entry + Exit  
    margin_cost = MARGIN_OPEN_FEE * leverage  # Margin fee scaled by leverage
    
    total_cost_pct = slippage_cost + trading_fees + margin_cost
    
    return total_cost_pct

def simulate_trade_outcome(df_window: pd.DataFrame, entry_idx: int, 
                          entry_price: float, side: str, leverage: int,
                          max_hours: int = MAX_HOLD_HOURS) -> Dict:
    """
    Simulate trade outcome using institutional trading rules
    Returns dictionary with trade metrics
    """
    if len(df_window) <= entry_idx:
        return {"return": 0, "exit_reason": "insufficient_data", "hold_hours": 0}
    
    # Initialize tracking
    remaining_position = 1.0  # 100% of position
    partial_exits = []
    max_profit = 0
    max_loss = 0
    
    # Simulate each hour after entry
    for i in range(entry_idx + 1, min(len(df_window), entry_idx + max_hours + 1)):
        current_price = df_window.iloc[i]['close']
        hours_held = i - entry_idx
        
        # Calculate current P&L
        if side == "buy":
            raw_return = (current_price - entry_price) / entry_price
        else:  # sell/short
            raw_return = (entry_price - current_price) / entry_price
        
        leveraged_return = raw_return * leverage
        
        # Track max profit/loss
        max_profit = max(max_profit, leveraged_return)
        max_loss = min(max_loss, leveraged_return)
        
        # Check stop loss (6% with time-based tightening)
        time_adjusted_stop = STOP_LOSS_PCT - min(0.02, (hours_held / 6) * 0.01)
        time_adjusted_stop = max(0.02, time_adjusted_stop)  # Never below 2%
        
        if leveraged_return <= -time_adjusted_stop:
            return {
                "return": leveraged_return,
                "exit_reason": "stop_loss",
                "hold_hours": hours_held,
                "max_profit": max_profit,
                "max_loss": max_loss
            }
        
        # Check take profit levels
        for j, target_pct in enumerate(TAKE_PROFIT_LEVELS):
            if leveraged_return >= target_pct and len(partial_exits) == j:
                # Hit this take profit level
                exit_allocation = TAKE_PROFIT_ALLOCATIONS[j]
                partial_exits.append({
                    "level": j,
                    "return": leveraged_return,
                    "allocation": exit_allocation,
                    "hours": hours_held
                })
                remaining_position -= exit_allocation
                
                # If this was the final target or we've exited most position
                if j == len(TAKE_PROFIT_LEVELS) - 1 or remaining_position <= 0.1:
                    # Calculate weighted average return
                    total_return = sum([
                        exit_data["return"] * exit_data["allocation"] 
                        for exit_data in partial_exits
                    ])
                    total_return += leveraged_return * remaining_position
                    
                    return {
                        "return": total_return,
                        "exit_reason": f"take_profit_level_{j}",
                        "hold_hours": hours_held,
                        "partial_exits": partial_exits,
                        "max_profit": max_profit,
                        "max_loss": max_loss
                    }
        
        # Check maximum hold time
        if hours_held >= max_hours:
            # Calculate return including any partial exits
            total_return = sum([
                exit_data["return"] * exit_data["allocation"] 
                for exit_data in partial_exits
            ])
            total_return += leveraged_return * remaining_position
            
            return {
                "return": total_return,
                "exit_reason": "time_limit",
                "hold_hours": hours_held,
                "max_profit": max_profit,
                "max_loss": max_loss
            }
    
    # End of data - force exit
    current_price = df_window.iloc[-1]['close']
    if side == "buy":
        final_return = (current_price - entry_price) / entry_price * leverage
    else:
        final_return = (entry_price - current_price) / entry_price * leverage
    
    # Include any partial exits
    total_return = sum([
        exit_data["return"] * exit_data["allocation"] 
        for exit_data in partial_exits
    ])
    total_return += final_return * remaining_position
    
    return {
        "return": total_return,
        "exit_reason": "end_of_data",
        "hold_hours": len(df_window) - entry_idx - 1,
        "max_profit": max_profit,
        "max_loss": max_loss
    }

def create_institutional_labels(df: pd.DataFrame, pair: str, 
                              lookforward_hours: int = 72) -> pd.DataFrame:
    """
    Create institutional labels that account for ALL trading costs and rules
    
    Args:
        df: DataFrame with OHLCV and features  
        pair: Trading pair (e.g., 'XBTUSDT')
        lookforward_hours: How many hours to simulate forward (72 = 3 days)
    
    Returns:
        DataFrame with institutional labels
    """
    print(f"Creating institutional labels for {pair}...")
    
    # Initialize new columns
    df = df.copy()
    df['institutional_label_1h'] = 'hold'
    df['institutional_label_4h'] = 'hold'
    df['institutional_label_12h'] = 'hold'
    df['confidence_score'] = 0.0
    df['position_size_pct'] = 0.0
    df['leverage_used'] = 1
    df['total_costs_pct'] = 0.0
    df['expected_return'] = 0.0
    df['net_return_after_costs'] = 0.0
    df['trade_viable'] = False
    df['exit_reason'] = 'not_tested'
    df['hold_hours'] = 0
    
    # Process each potential entry point
    total_rows = len(df) - lookforward_hours
    processed = 0
    
    for i in range(total_rows):
        current_row = df.iloc[i]
        
        # Calculate confidence score based on technical indicators
        confidence = calculate_confidence_score(current_row)
        
        # Skip if confidence too low for any trade
        if confidence < 0.45:  # Minimum threshold for institutional trading
            df.loc[i, 'confidence_score'] = confidence
            continue
        
        # Get position sizing based on confidence
        position_size_pct, leverage = get_position_sizing(confidence)
        
        # Calculate total trading costs
        total_costs = calculate_total_costs(pair, position_size_pct, leverage)
        
        # Test both long and short scenarios for different time horizons
        entry_price = current_row['close']
        
        # Test different time horizons (1h, 4h, 12h)
        for horizon_hours, label_col in [(2, 'institutional_label_2h'), 
                                       (8, 'institutional_label_8h'),
                                       (24, 'institutional_label_24h')]:
            
            if i + horizon_hours >= len(df):
                continue
                
            future_window = df.iloc[i:i + min(lookforward_hours, horizon_hours * 3)]
            
            best_return = -999
            best_label = 'hold'
            best_outcome = None
            
            # Test LONG position
            long_outcome = simulate_trade_outcome(
                future_window, 0, entry_price, "buy", leverage, horizon_hours
            )
            long_net_return = long_outcome["return"] - total_costs
            
            if long_net_return > best_return:
                best_return = long_net_return
                best_label = 'buy'
                best_outcome = long_outcome
            
            # Test SHORT position  
            short_outcome = simulate_trade_outcome(
                future_window, 0, entry_price, "sell", leverage, horizon_hours
            )
            short_net_return = short_outcome["return"] - total_costs
            
            if short_net_return > best_return:
                best_return = short_net_return
                best_label = 'sell'
                best_outcome = short_outcome
            
            # Only label as buy/sell if profitable after ALL costs
            if best_return > MIN_PROFIT_THRESHOLD:
                df.loc[i, label_col] = best_label
                if horizon_hours == 4:  # Use 4h as primary for metadata
                    df.loc[i, 'trade_viable'] = True
                    df.loc[i, 'expected_return'] = best_outcome["return"]
                    df.loc[i, 'net_return_after_costs'] = best_return
                    df.loc[i, 'exit_reason'] = best_outcome["exit_reason"]
                    df.loc[i, 'hold_hours'] = best_outcome["hold_hours"]
            else:
                df.loc[i, label_col] = 'hold'
        
        # Store metadata
        df.loc[i, 'confidence_score'] = confidence
        df.loc[i, 'position_size_pct'] = position_size_pct
        df.loc[i, 'leverage_used'] = leverage
        df.loc[i, 'total_costs_pct'] = total_costs
        
        processed += 1
        if processed % 10000 == 0:
            print(f"Processed {processed:,} / {total_rows:,} rows ({processed/total_rows*100:.1f}%)")
    
    return df

def analyze_institutional_labels(df: pd.DataFrame, pair: str) -> None:
    """
    Analyze the quality and distribution of institutional labels
    """
    print(f"\n📊 INSTITUTIONAL LABEL ANALYSIS FOR {pair}")
    print("=" * 60)
    
    # Analyze each time horizon
    for horizon in ['1h', '4h', '12h']:
        label_col = f'institutional_label_{horizon}'
        if label_col not in df.columns:
            continue
            
        print(f"\n📈 {horizon.upper()} Labels:")
        label_counts = df[label_col].value_counts()
        for label, count in label_counts.items():
            pct = count / len(df) * 100
            print(f"  {label.upper()}: {count:,} ({pct:.2f}%)")
    
    # Viable trades analysis (using 4h as primary)
    viable_trades = df[df['trade_viable'] == True]
    if len(viable_trades) > 0:
        print(f"\n💰 Viable Trades Analysis (4h horizon):")
        print(f"  Total viable trades: {len(viable_trades):,}")
        print(f"  Percentage of all data: {len(viable_trades)/len(df)*100:.2f}%")
        print(f"  Average confidence: {viable_trades['confidence_score'].mean():.3f}")
        print(f"  Average expected return: {viable_trades['expected_return'].mean()*100:.2f}%")
        print(f"  Average net return (after costs): {viable_trades['net_return_after_costs'].mean()*100:.2f}%")
        print(f"  Average leverage used: {viable_trades['leverage_used'].mean():.1f}x")
        print(f"  Average position size: {viable_trades['position_size_pct'].mean()*100:.1f}%")
        print(f"  Average hold time: {viable_trades['hold_hours'].mean():.1f} hours")
    
    # Cost analysis
    print(f"\n💸 Cost Analysis:")
    print(f"  Average total costs: {df['total_costs_pct'].mean()*100:.3f}%")
    print(f"  Max total costs: {df['total_costs_pct'].max()*100:.3f}%")
    print(f"  Min profit threshold: {MIN_PROFIT_THRESHOLD*100:.1f}%")
    
    # Exit reason analysis
    if len(viable_trades) > 0:
        print(f"\n🚪 Exit Reason Analysis:")
        exit_reasons = viable_trades['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            pct = count / len(viable_trades) * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")
    
    # Confidence distribution
    if len(viable_trades) > 0:
        print(f"\n🎯 Confidence Distribution (Viable Trades):")
        conf_bins = pd.cut(viable_trades['confidence_score'], 
                          bins=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                          labels=['0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
        conf_dist = conf_bins.value_counts().sort_index()
        for bin_range, count in conf_dist.items():
            pct = count / len(viable_trades) * 100 if len(viable_trades) > 0 else 0
            print(f"  {bin_range}: {count} ({pct:.1f}%)")

def main():
    """
    Main function to create institutional labels for all pairs
    """
    print("🏛️ INSTITUTIONAL LABELING FOR $14K AGGRESSIVE STRATEGY")
    print("=" * 70)
    print(f"Capital: ${INITIAL_CAPITAL:,}")
    print(f"Max Position Size: {BASE_POSITION_SIZE_LIMIT*100:.0f}%")
    print(f"Max Leverage: {MAX_TOTAL_LEVERAGE}x")
    print(f"Stop Loss: {STOP_LOSS_PCT*100:.0f}%")
    print(f"Take Profits: {[tp*100 for tp in TAKE_PROFIT_LEVELS]}%")
    print(f"Min Profit Threshold: {MIN_PROFIT_THRESHOLD*100:.1f}%")
    print("=" * 70)
    
    conn = sqlite3.connect(DB_FILE)
    
    for pair in PAIRS:
        try:
            # Load feature data
            table_name = f"features_{pair}_5m"
            print(f"\n🔄 Processing {pair}...")
            
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            print(f"Loaded {len(df):,} rows for {pair}")
            
            # Create institutional labels
            df_labeled = create_institutional_labels(df, pair, lookforward_hours=144)
            
            # Analyze label quality
            analyze_institutional_labels(df_labeled, pair)
            
            # Save labeled data
            output_table = f"features_{pair}_5m_institutional"
            df_labeled.to_sql(output_table, conn, if_exists="replace", index=False)
            print(f"✅ Saved to {output_table}")
            
        except Exception as e:
            print(f"❌ Error processing {pair}: {e}")
            continue
    
    conn.close()
    print("\n🎉 Institutional labeling completed!")
    print("\nNext Steps:")
    print("1. Train models on institutional labels")
    print("2. Create regime-specific models") 
    print("3. Run enhanced backtests")
    print("4. Deploy live trading system")

if __name__ == "__main__":
    main()