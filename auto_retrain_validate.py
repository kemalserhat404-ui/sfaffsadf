# auto_pipeline_full.py
import pandas as pd
import numpy as np
import os
from lightgbm import LGBMClassifier
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

# ---------- CONFIG ----------
FEATURES_INPUT = "features/BTC_USDT__1h__features.parquet"
FEATURES_OUTPUT = "features/BTC_USDT__1h__features_recomputed_auto.parquet"
MODEL_OUTPUT = "models/btc_offline_lgb_recomputed.joblib"
TARGET_COLUMN = "target"

# ---------- STEP 1: LOAD AND PREP DATA ----------
logging.info(f"Loading features: {FEATURES_INPUT}")
df = pd.read_parquet(FEATURES_INPUT)

# Sort by timestamp if exists
if 'timestamp' in df.columns:
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.drop(columns=['timestamp'], inplace=True)

df = df.reset_index(drop=True)

# Recompute / Standardize features
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

# Fill missing values
df = df.ffill().bfill()
df.dropna(inplace=True)

# Create target if missing
if TARGET_COLUMN not in df.columns:
    df[TARGET_COLUMN] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)

# Save recomputed features
df.to_parquet(FEATURES_OUTPUT, index=False)
logging.info(f"Saved recomputed features -> {FEATURES_OUTPUT}")

# ---------- STEP 2: TRAIN MODEL ----------
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN].values

# Make sure all features are numeric
for col in X.columns:
    if not np.issubdtype(X[col].dtype, np.number):
        X[col] = X[col].astype(float)

logging.info("Training LGBMClassifier...")
model = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
model.fit(X, y)
joblib.dump(model, MODEL_OUTPUT)
logging.info(f"[SAVE] Saved -> {MODEL_OUTPUT}")

# ---------- STEP 3: VALIDATE ----------
logging.info("Running validation on latest features...")
probs = model.predict_proba(X)[:,1]
preds = (probs > 0.5).astype(int)
accuracy = (preds == y).mean()
logging.info(f"[INFO] Validation accuracy: {accuracy:.4f}")

# ---------- STEP 4: OPTIONAL: NEW DATA FETCH ----------
# Buraya istediğin veri çekme kodunu ekleyebilirsin. Örneğin Binance API veya CSV çekme.
# Örnek placeholder:
# logging.info("Fetching new data from API...")
# new_data = fetch_binance_data(...)  # kullanıcıya özel fonksiyon
# new_data.to_parquet(FEATURES_INPUT, index=False)
# Sonrasında pipeline baştan çalışacak

logging.info("[DONE] Auto pipeline completed.")
