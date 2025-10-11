import time
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from binance.client import Client
from lightgbm import LGBMClassifier
import joblib

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = 'q4N9z7SD1ykbDfNdSFK9WxvsfK1bF3Pj8yI0NY046s2odgYwpRKIkmnfecXHYqui'
API_SECRET = 'KAyxq3HATSZkDnPFDlRDBDqb1jR4ROt0PCeVouiErTo0hdqdpLTlSL0c0YE0L5lA'
FEATURE_FILE = "features/BTC_USDT__1h__features.parquet"
FEATURE_FILE_RECOMPUTED = "features/BTC_USDT__1h__features_recomputed_auto.parquet"
MODEL_FILE = "models/btc_offline_lgb_recomputed.joblib"
HOLDOUT_RATIO = 0.1
SLEEP_SECONDS = 3600  # 1 saat

logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    level=logging.INFO
)

client = Client(API_KEY, API_SECRET)

def fetch_new_data():
    logging.info("Yeni veri çekiliyor...")
    klines = client.get_historical_klines(
        'BTCUSDT', Client.KLINE_INTERVAL_1HOUR, '1 day ago UTC'
    )
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def recompute_features(df):
    logging.info("Feature recompute başlatılıyor...")
    df = df.sort_values('timestamp').reset_index(drop=True) if 'timestamp' in df.columns else df
    df = df.reset_index(drop=True)

    # Lags & returns
    lag_periods = [1,2,3,6,12]
    for k in lag_periods:
        df[f'close_lag_{k}'] = df['close'].shift(k)
        df[f'return_{k}'] = df['close'] / df[f'close_lag_{k}'] - 1

    df['return1'] = df['close'].pct_change(1)

    # Rolling stats
    df['rolling_mean_14'] = df['close'].rolling(14).mean()
    df['rolling_std_14'] = df['close'].rolling(14).std()
    df['volatility'] = df['rolling_std_14']

    # Momentum & range
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
    df.dropna(inplace=True)

    # Target
    if 'target' not in df.columns:
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df.dropna(inplace=True)

    df.to_parquet(FEATURE_FILE_RECOMPUTED, index=False)
    logging.info(f"Saved recomputed features -> {FEATURE_FILE_RECOMPUTED}")
    return df

def retrain_validate(df):
    logging.info("Model retrain & validate başlatılıyor...")
    holdout_idx = int(len(df)*(1-HOLDOUT_RATIO))
    train_df = df.iloc[:holdout_idx]
    holdout_df = df.iloc[holdout_idx:]

    X_train = train_df.drop(['target','timestamp'] if 'timestamp' in train_df.columns else ['target'], axis=1)
    y_train = train_df['target'].values

    X_holdout = holdout_df.drop(['target','timestamp'] if 'timestamp' in holdout_df.columns else ['target'], axis=1)
    y_holdout = holdout_df['target'].values

    model = LGBMClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_holdout)
    acc = (y_pred == y_holdout).mean()
    logging.info(f"Holdout test accuracy: {acc:.4f}")

    joblib.dump(model, MODEL_FILE)
    logging.info(f"Saved model -> {MODEL_FILE}")

if __name__ == "__main__":
    while True:
        try:
            df = fetch_new_data()
            df = recompute_features(df)
            retrain_validate(df)
        except Exception as e:
            logging.error(f"Hata: {e}")
        logging.info(f"Sleeping {SLEEP_SECONDS/60:.0f} dakika...")
        time.sleep(SLEEP_SECONDS)
