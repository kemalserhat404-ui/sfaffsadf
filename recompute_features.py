# recompute_features.py
import pandas as pd
import numpy as np
import os

IN = "features/BTC_USDT__1h__features.parquet"
OUT = "features/BTC_USDT__1h__features_recomputed.parquet"

print("[INFO] Loading:", IN)
df = pd.read_parquet(IN)

# Ensure sorted by timestamp
if 'timestamp' in df.columns:
    df = df.sort_values('timestamp').reset_index(drop=True)

df = df.reset_index(drop=True)

# --- Feature Engineering ---

# 1) Lags & Returns (multi-timeframe)
lag_periods = [1,2,3,6,12,24]
for k in lag_periods:
    df[f'close_lag_{k}'] = df['close'].shift(k)
    df[f'return_{k}'] = df['close'] / df[f'close_lag_{k}'] - 1

df['return1'] = df['close'].pct_change(1)

# 2) Rolling stats
rolling_windows = [14, 28]
for w in rolling_windows:
    df[f'rolling_mean_{w}'] = df['close'].rolling(w).mean()
    df[f'rolling_std_{w}'] = df['close'].rolling(w).std()

df['volatility'] = df['rolling_std_14']

# 3) Momentum & Price Range
df['momentum14'] = df['close'] - df['close'].shift(14)
df['price_range'] = df['high'] - df['low']

# 4) Volume features
df['vol_mean_14'] = df['volume'].rolling(14).mean()
df['vol_spike'] = df['volume'] / (df['vol_mean_14'] + 1e-9)

# 5) EMA & MACD
df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = df['ema12'] - df['ema26']
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']
df['ema_ratio'] = df['ema12'] / (df['ema26'] + 1e-9)

# 6) ATR-ish
df['price_high_low'] = df['high'] - df['low']
df['atr14_simple'] = df['price_high_low'].rolling(14).mean()

# 7) Multi-timeframe rolling returns
multi_windows = [1,3,6,12]
for w in multi_windows:
    df[f'return_{w}h'] = df['close'].pct_change(w)

# 8) Fill missing & drop remaining NaNs
df = df.ffill().bfill()
df.dropna(inplace=True)

# 9) Target column
if 'target' not in df.columns:
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)

print("[INFO] Rows after prep:", len(df))
print("[INFO] Saving to:", OUT)
df.to_parquet(OUT, index=False)
print("[DONE]")
