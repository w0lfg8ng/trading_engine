# regime_matrix_training_xgboost.py
"""
Train 81 XGBoost models for regime-specific cryptocurrency trading
9 Market Regimes × 9 Crypto Pairs = 81 Models

Regimes:
- Bull Market: High/Normal/Low Volatility
- Range Market: High/Normal/Low Volatility  
- Bear Market: High/Normal/Low Volatility
"""

import pandas as pd
import sqlite3
import xgboost as xgb
import joblib
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add after imports
print("🔍 GPU Detection:")
print(f"XGBoost version: {xgb.__version__}")
try:
    # Test GPU availability
    test_gpu = xgb.DMatrix(np.random.random((10, 5)), label=np.random.randint(0, 2, 10))
    gpu_params = {"tree_method": "hist", "device": "cuda:0"}
    test_model = xgb.train(gpu_params, test_gpu, num_boost_round=1, verbose_eval=False)
    print("✅ GPU acceleration is working!")
except Exception as e:
    print(f"❌ GPU not available: {e}")
    print("Will fall back to CPU training...")

# === CONFIG ===
DB_FILE = "../../data/kraken_v2.db"

# Updated pairs for V2 (USDT versions)
PAIRS = [
    "XBTUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT",
    "LTCUSDT", "XRPUSDT", "LINKUSDT", "AVAXUSDT"
]

# Target label from institutional labeling
TARGET = "institutional_label_12h"  # Use 12h horizon labels (optimal timeframe)

# Model output directory
MODEL_DIR = "../../models/xgboost_regime_specific/"

# === REGIME DETECTION PARAMETERS ===
TREND_WINDOW_DAYS = 60      # 60 days for trend detection
VOLATILITY_WINDOW_DAYS = 30 # 30 days for volatility calculation

# Volatility thresholds (based on daily returns)
HIGH_VOLATILITY_THRESHOLD = 0.035   # 3.5% daily volatility
LOW_VOLATILITY_THRESHOLD = 0.015    # 1.5% daily volatility

# Trend thresholds (based on 60-day price change)
BULL_TREND_THRESHOLD = 0.20         # 20% gain over 60 days
BEAR_TREND_THRESHOLD = -0.20        # -20% loss over 60 days

def detect_market_regime(df, price_col='close'):
    """
    Detect market regime for each row in the dataframe
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

def prepare_features(df):
    """
    Prepare features for training by excluding non-feature columns
    """
    # Columns to exclude from features
    exclude_cols = [
        'timestamp', 'pair', 'open', 'high', 'low', 'close', 'volume',
        'institutional_label_1h', 'institutional_label_4h', 'institutional_label_12h',
        'confidence_score', 'position_size_pct', 'leverage_used', 'total_costs_pct',
        'expected_return', 'net_return_after_costs', 'trade_viable', 'exit_reason',
        'hold_hours', 'market_regime',
        # Add these datetime columns that are causing the error
        'datetime', 'created_at'
    ]
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Additional safety check: exclude any remaining object/datetime columns
    feature_cols_clean = []
    for col in feature_cols:
        if df[col].dtype in ['object', 'datetime64[ns]', '<M8[ns]']:
            print(f"    ⚠️  Excluding {col} (dtype: {df[col].dtype})")
            continue
        feature_cols_clean.append(col)
    
    return feature_cols_clean

def train_regime_model(df_regime, pair, regime, feature_cols):
    """
    Train XGBoost model for a specific pair and regime
    """
    print(f"  Training {pair} - {regime} model...")
    
    # Filter valid labels
    df_clean = df_regime[df_regime[TARGET].isin(['buy', 'sell', 'hold'])].copy()
    
    if len(df_clean) < 1000:  # Minimum data requirement
        print(f"    ⚠️  Insufficient data for {pair}-{regime}: {len(df_clean)} rows")
        return None, None, None
    
    # Prepare features and target
    X = df_clean[feature_cols].copy()
    y = df_clean[TARGET]
    
    # === DATA CLEANING FOR XGBOOST ===
    print(f"    🧹 Cleaning data...")

    # Check for inf values before cleaning
    inf_counts = np.isinf(X).sum().sum()
    nan_counts = X.isna().sum().sum()  # ✅ Fixed: Use X.isna() instead of np.isna(X)
    print(f"    Found {inf_counts} inf values, {nan_counts} NaN values")

    # Replace inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Check for extremely large values and clip them
    for col in X.columns:
        if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Get reasonable bounds (99.9th percentile)
            valid_data = X[col][~X[col].isna()]  # ✅ Fixed: Use X[col].isna() instead of np.isna(X[col])
            if len(valid_data) > 0:
                upper_bound = valid_data.quantile(0.999)
                lower_bound = valid_data.quantile(0.001)
            
                # Clip extreme values
                X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)

    # Handle remaining NaN values
    # Option 1: Drop rows with any NaN (conservative)
    before_dropna = len(X)
    X_clean = X.dropna()
    y_clean = y.loc[X_clean.index]
    after_dropna = len(X_clean)

    print(f"    Dropped {before_dropna - after_dropna} rows with NaN ({(before_dropna - after_dropna)/before_dropna*100:.1f}%)")

    if len(X_clean) < 1000:  # Check again after cleaning
        print(f"    ⚠️  Insufficient data after cleaning for {pair}-{regime}: {len(X_clean)} rows")
        return None, None, None

    # Update variables
    X = X_clean
    y = y_clean

    # Encode target labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Check class distribution
    class_counts = pd.Series(y_encoded).value_counts()
    if len(class_counts) < 2:  # Need at least 2 classes
        print(f"    ⚠️  Only one class in {pair}-{regime}")
        return None, None, None

    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=3)

    # Use the last split for final training
    train_idx, val_idx = list(tscv.split(X))[-1]

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # Final safety check for inf/nan in splits
    if np.isinf(X_train).any().any() or X_train.isna().any().any():  # ✅ Fixed: Use X_train.isna()
        print(f"    ❌ Training data still contains inf/nan values!")
        return None, None, None
    
    if np.isinf(X_val).any().any() or X_val.isna().any().any():  # ✅ Fixed: Use X_val.isna()
        print(f"    ❌ Validation data still contains inf/nan values!")
        return None, None, None
    
    # NEW (XGBoost 3.0+) params - USE THIS:
    params = {
        "objective": "multi:softprob",
        "num_class": len(encoder.classes_),
        "eval_metric": "mlogloss",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "tree_method": "hist",            # ✅ NEW: Use hist + device
        "device": "cuda:0",               # ✅ NEW: Specify GPU device
        "verbosity": 0
    }
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dval, "eval")],
        num_boost_round=300,
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    # Validate model
    val_pred = model.predict(dval)
    val_pred_labels = np.argmax(val_pred, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(val_pred_labels == y_val)
    
    # Check for reasonable performance
    if accuracy < 0.4:  # At least better than random for 3 classes
        print(f"    ⚠️  Poor accuracy for {pair}-{regime}: {accuracy:.3f}")
    else:
        print(f"    ✅ {pair}-{regime}: {len(X_clean):,} samples, accuracy: {accuracy:.3f}")
    
    return model, encoder, feature_cols

def train_all_regime_models():
    """
    Train all 81 regime-specific models
    """
    print("🚀 STARTING REGIME-SPECIFIC MODEL TRAINING")
    print("=" * 60)
    print(f"Target: {TARGET}")
    print(f"Pairs: {len(PAIRS)}")
    print(f"Expected models: {len(PAIRS) * 9} (9 regimes per pair)")
    print("=" * 60)
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(DB_FILE)
    
    # Track results
    results = {
        'successful_models': 0,
        'failed_models': 0,
        'regime_distribution': {},
        'pair_results': {}
    }
    
    # Train models for each pair
    for pair in PAIRS:
        print(f"\n📊 Processing {pair}...")
        
        # Load data
        table_name = f"features_{pair}_1m_institutional"
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=["timestamp"])
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            print(f"  Loaded {len(df):,} rows")
            
        except Exception as e:
            print(f"  ❌ Failed to load data for {pair}: {e}")
            results['pair_results'][pair] = {'error': str(e)}
            continue
        
        # Detect market regimes
        print(f"  Detecting market regimes...")
        df = detect_market_regime(df)
        
        # Get regime distribution
        regime_counts = df['market_regime'].value_counts()
        print(f"  Regime distribution:")
        for regime, count in regime_counts.items():
            print(f"    {regime}: {count:,} ({count/len(df)*100:.1f}%)")
        
        # Prepare features
        feature_cols = prepare_features(df)
        print(f"  Using {len(feature_cols)} features")
        
        # Track pair results
        pair_success = 0
        pair_total = 0
        
        # Train model for each regime
        for regime in regime_counts.index:
            pair_total += 1
            
            # Filter data for this regime
            df_regime = df[df['market_regime'] == regime].copy()
            
            # Train model
            model, encoder, features = train_regime_model(df_regime, pair, regime, feature_cols)
            
            if model is not None:
                # Save model
                model_filename = f"{pair}_{regime}_xgb_model.pkl"
                encoder_filename = f"{pair}_{regime}_encoder.pkl"
                features_filename = f"{pair}_{regime}_features.pkl"
                
                model_path = os.path.join(MODEL_DIR, model_filename)
                encoder_path = os.path.join(MODEL_DIR, encoder_filename)
                features_path = os.path.join(MODEL_DIR, features_filename)
                
                # Save files
                joblib.dump(model, model_path)
                joblib.dump(encoder, encoder_path)
                joblib.dump(features, features_path)
                
                pair_success += 1
                results['successful_models'] += 1
                
                # Update regime distribution
                if regime not in results['regime_distribution']:
                    results['regime_distribution'][regime] = 0
                results['regime_distribution'][regime] += 1
                
            else:
                results['failed_models'] += 1
        
        results['pair_results'][pair] = {
            'successful': pair_success,
            'total': pair_total,
            'success_rate': pair_success / pair_total if pair_total > 0 else 0
        }
        
        print(f"  ✅ {pair}: {pair_success}/{pair_total} models trained successfully")
    
    conn.close()
    
    # Print final results
    print("\n" + "=" * 60)
    print("🎉 REGIME MODEL TRAINING COMPLETED")
    print("=" * 60)
    print(f"Successful models: {results['successful_models']}")
    print(f"Failed models: {results['failed_models']}")
    print(f"Success rate: {results['successful_models']/(results['successful_models']+results['failed_models'])*100:.1f}%")
    
    print(f"\nRegime distribution (successful models):")
    for regime, count in sorted(results['regime_distribution'].items()):
        print(f"  {regime}: {count} models")
    
    print(f"\nPair success rates:")
    for pair, stats in results['pair_results'].items():
        if 'error' not in stats:
            print(f"  {pair}: {stats['successful']}/{stats['total']} ({stats['success_rate']*100:.1f}%)")
        else:
            print(f"  {pair}: ERROR - {stats['error']}")
    
    return results

def load_regime_model(pair, regime):
    """
    Load a specific regime model for prediction
    """
    model_filename = f"{pair}_{regime}_xgb_model.pkl"
    encoder_filename = f"{pair}_{regime}_encoder.pkl"
    features_filename = f"{pair}_{regime}_features.pkl"
    
    model_path = os.path.join(MODEL_DIR, model_filename)
    encoder_path = os.path.join(MODEL_DIR, encoder_filename)
    features_path = os.path.join(MODEL_DIR, features_filename)
    
    try:
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        features = joblib.load(features_path)
        return model, encoder, features
    except Exception as e:
        print(f"Error loading model for {pair}-{regime}: {e}")
        return None, None, None

if __name__ == "__main__":
    # Run training
    results = train_all_regime_models()
    
    print(f"\n📁 Models saved to: {MODEL_DIR}")
    print(f"🚀 Ready for regime-switching backtesting!")