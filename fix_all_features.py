# fix_all_features.py
import pandas as pd
import numpy as np
import joblib, glob, os
from datetime import datetime

# --- Config ---
MODEL_FILE = 'models/btc_offline_lgb_recomputed_optuna.joblib'
FEATURES_PATTERN = 'features/BTCUSDT__*__features_recomputed_auto.parquet'
PRINT_EVERY = True

# --- Load model ---
model = joblib.load(MODEL_FILE)
feature_order = model.feature_name_ if hasattr(model,'feature_name_') else model.feature_name()

# --- Functions ---
def compute_indicators(df):
    df = df.copy()
    # EMA
    df['ema12'] = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
    # MACD
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    # RSI
    delta = df['close'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(14, min_periods=1).mean()
    roll_down = down.rolling(14, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi14'] = 100 - 100 / (1 + rs)
    # VWAP
    df['vwap14'] = (df['close'] * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-9)
    # ATR simple placeholder
    df['atr14'] = (df['high'] - df['low']).rolling(14, min_periods=1).mean()
    return df

def compute_lag_features(df):
    df = df.copy()
    for lag in [1,2,3,6,12]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'return_{lag}'] = df['close'].pct_change()  # <-- düzeltilmiş
    # rolling window features
    df['rolling_mean_14'] = df['close'].rolling(14, min_periods=1).mean()
    df['rolling_std_14'] = df['close'].rolling(14, min_periods=1).std().fillna(0)
    df['volatility'] = df['rolling_std_14'] / (df['rolling_mean_14'] + 1e-9)
    df['momentum14'] = df['close'] - df['close'].shift(14)
    df['vol_mean_14'] = df['volume'].rolling(14, min_periods=1).mean()
    df['vol_spike'] = df['volume'] / (df['vol_mean_14'] + 1e-9)
    df['ema_ratio'] = df['ema12'] / (df['ema26'] + 1e-9)
    df['price_high_low'] = df['high'] - df['low']
    df['atr14_simple'] = df['atr14'].fillna(0)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

# --- Process all parquet files ---
for ffile in glob.glob(FEATURES_PATTERN):
    df = pd.read_parquet(ffile)
    df = compute_indicators(df)
    df = compute_lag_features(df)

    # Add missing model features as 0
    missing = set(feature_order) - set(df.columns)
    for m in missing:
        df[m] = 0.0

    # Reorder columns to match model
    df = df[feature_order]

    # Save back
    df.to_parquet(ffile, index=False)
    if PRINT_EVERY:
        print(f"[{datetime.now()}] Fixed {ffile}, columns: {len(df.columns)}")

print("✅ All feature files recomputed and aligned with model.")
