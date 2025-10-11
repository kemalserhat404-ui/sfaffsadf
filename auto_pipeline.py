# auto_pipeline.py
import os
import sys
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier

# CONFIG
RAW_FEATURES = "features/BTC_USDT__1h__features.parquet"
RECOMPUTED_FEATURES = "features/BTC_USDT__1h__features_recomputed.parquet"
MODEL_PATH = "models/btc_offline_lgb_recomputed.joblib"
VALIDATE_SPLIT = 0.2
RANDOM_SEED = 42

def recompute_features(input_path=RAW_FEATURES, output_path=RECOMPUTED_FEATURES):
    print("[INFO] Loading features:", input_path)
    df = pd.read_parquet(input_path)
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.reset_index(drop=True)
    
    # Basic lags and returns
    lag_periods = [1,2,3,6,12]
    for k in lag_periods:
        df[f'close_lag_{k}'] = df['close'].shift(k)
        df[f'return_{k}'] = df['close'] / df[f'close_lag_{k}'] - 1
    df['return1'] = df['close'].pct_change(1)

    # Rolling stats
    df['rolling_mean_14'] = df['close'].rolling(14).mean()
    df['rolling_std_14'] = df['close'].rolling(14).std()
    df['volatility'] = df['rolling_std_14']

    # Momentum and price range
    df['momentum14'] = df['close'] - df['close'].shift(14)
    df['price_range'] = df['high'] - df['low']

    # Volume spike
    df['vol_mean_14'] = df['volume'].rolling(14).mean()
    df['vol_spike'] = df['volume'] / (df['vol_mean_14'] + 1e-9)

    # EMA & MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['ema_ratio'] = df['ema12'] / (df['ema26'] + 1e-9)

    # ATR simple
    df['price_high_low'] = df['high'] - df['low']
    df['atr14_simple'] = df['price_high_low'].rolling(14).mean()

    df = df.ffill().bfill()
    
    # Target
    if 'target' not in df.columns:
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df.dropna(inplace=True)
    df.to_parquet(output_path, index=False)
    print("[DONE] Features recomputed and saved:", output_path)
    return df

def train_model(df):
    feature_cols = [c for c in df.columns if c not in ['timestamp','target']]
    X = df[feature_cols]
    y = df['target']

    split_idx = int(len(df)*(1-VALIDATE_SPLIT))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_SEED
    )

    print("[INFO] Fitting model...")
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"[SAVE] Model saved -> {MODEL_PATH}")

    # Validation
    val_acc = model.score(X_val, y_val)
    print(f"[INFO] Validation accuracy: {val_acc:.4f}")
    return model, feature_cols

def validate_model(model, feature_cols, df):
    X_test = df[feature_cols]
    y_test = df['target']
    try:
        probs = model.predict_proba(X_test)[:,1]
        acc = (model.predict(X_test) == y_test).mean()
        print(f"[INFO] Test accuracy: {acc:.4f}")
    except Exception as e:
        print("[ERROR] Validation failed:", e)

def main():
    df = recompute_features()
    model, feature_cols = train_model(df)
    validate_model(model, feature_cols, df)

if __name__ == "__main__":
    main()
