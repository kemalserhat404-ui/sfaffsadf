# auto_retrain_validate_full.py
import pandas as pd
import numpy as np
import joblib
import os
from lightgbm import LGBMClassifier
from datetime import datetime

# --- CONFIG ---
INPUT_FEATURES = "features/BTC_USDT__1h__features.parquet"
OUTPUT_FEATURES = "features/BTC_USDT__1h__features_recomputed_auto.parquet"
OUTPUT_MODEL = "models/btc_offline_lgb_recomputed.joblib"
TARGET_COLUMN = "target"

print(f"[{datetime.now()}] Loading features: {INPUT_FEATURES}")
df = pd.read_parquet(INPUT_FEATURES)

# --- SORT & RESET ---
if 'timestamp' in df.columns:
    df = df.sort_values('timestamp').reset_index(drop=True)
df = df.reset_index(drop=True)

# --- RECOMPUTE FEATURES ---
lag_periods = [1,2,3,6,12]
for k in lag_periods:
    df[f'close_lag_{k}'] = df['close'].shift(k)
    df[f'return_{k}'] = df['close'] / df[f'close_lag_{k}'] - 1

df['return1'] = df['close'].pct_change(1)
df['rolling_mean_14'] = df['close'].rolling(14).mean()
df['rolling_std_14'] = df['close'].rolling(14).std()
df['volatility'] = df['rolling_std_14']
df['momentum14'] = df['close'] - df['close'].shift(14)
df['price_range'] = df['high'] - df['low']
df['vol_mean_14'] = df['volume'].rolling(14).mean()
df['vol_spike'] = df['volume'] / (df['vol_mean_14'] + 1e-9)
df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = df['ema12'] - df['ema26']
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']
df['ema_ratio'] = df['ema12'] / (df['ema26'] + 1e-9)
df['price_high_low'] = df['high'] - df['low']
df['atr14_simple'] = df['price_high_low'].rolling(14).mean()

# --- FILL/ DROP ---
df = df.ffill().bfill()
df.dropna(inplace=True)

# --- CREATE TARGET ---
if TARGET_COLUMN not in df.columns:
    df[TARGET_COLUMN] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)

print(f"[{datetime.now()}] Saved recomputed features -> {OUTPUT_FEATURES}")
df.to_parquet(OUTPUT_FEATURES, index=False)

# --- TRAIN / RETRAIN MODEL ---
feature_cols = [c for c in df.columns if c not in ['timestamp', TARGET_COLUMN]]
X = df[feature_cols].values
y = df[TARGET_COLUMN].values

print(f"[{datetime.now()}] Training LGBMClassifier...")
model = LGBMClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)
joblib.dump(model, OUTPUT_MODEL)
print(f"[{datetime.now()}] Saved model -> {OUTPUT_MODEL}")

# --- VALIDATE HOLDOUT ---
# Basit: son %20'u test olarak kullan
split_idx = int(len(df)*0.8)
X_test = X[split_idx:]
y_test = y[split_idx:]
acc = (model.predict(X_test) == y_test).mean()
print(f"[{datetime.now()}] Holdout test accuracy: {acc:.4f}")
