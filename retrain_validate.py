# retrain_validate.py
import pandas as pd
import numpy as np
import joblib
import time
import logging
from lightgbm import LGBMClassifier
from datetime import datetime
from binance.client import Client

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = '...'
API_SECRET = '...'
FEATURE_FILE = "features/BTC_USDT__1h__features.parquet"
FEATURE_FILE_RECOMPUTED = "features/BTC_USDT__1h__features_recomputed_auto.parquet"
MODEL_FILE = "models/btc_offline_lgb_recomputed.joblib"
LOG_PATH = "logs/high_conf_predictions.csv"
TARGET_COLUMN = "target"
CONF_THRESHOLD = 0.7
SLEEP_SECONDS = 3600  # 1 saat

logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    level=logging.INFO
)

client = Client(API_KEY, API_SECRET)

# -----------------------------
# Fonksiyonlar
# -----------------------------
def fetch_new_data():
    logging.info("Yeni veri çekiliyor...")
    klines = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_1HOUR, '1 day ago UTC')
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df.reset_index()

def recompute_features(df):
    logging.info("Feature recompute başlatılıyor...")
    df = df.sort_values('timestamp').reset_index(drop=True)
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
    df = df.ffill().bfill()
    if 'target' not in df.columns:
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    df.to_parquet(FEATURE_FILE_RECOMPUTED, index=False)
    logging.info(f"Saved recomputed features -> {FEATURE_FILE_RECOMPUTED}")
    return df

def train_and_retrain(df):
    # Prepare data
    X = df.drop(columns=['timestamp','target'], errors='ignore')
    y = df['target'].values
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Load existing model or new
    try:
        model = joblib.load(MODEL_FILE)
        logging.info("Existing model loaded.")
    except:
        model = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42)
        logging.info("New model created.")

    # High-confidence retrain
    if hasattr(model, "predict_proba"):
        preds_proba = model.predict_proba(X)[:,1]
        X_high_conf = X[preds_proba >= CONF_THRESHOLD]
        y_high_conf = y[preds_proba >= CONF_THRESHOLD]
        if len(X_high_conf) > 10:
            model.fit(X_high_conf, y_high_conf)
            logging.info(f"High-confidence örneklerle model güncellendi: {len(X_high_conf)} örnek")
        else:
            logging.info("Yeterli yüksek-confidence örnek yok, retrain atlandı.")

    # Fit on full dataset
    model.fit(X, y)

    # Save model
    joblib.dump(model, MODEL_FILE)
    logging.info(f"Saved model -> {MODEL_FILE}")

    # Log high-confidence predictions
    preds_proba = model.predict_proba(X)[:,1]
    high_conf_preds = df.loc[preds_proba >= CONF_THRESHOLD].copy()
    high_conf_preds['pred_proba'] = preds_proba[preds_proba >= CONF_THRESHOLD]
    high_conf_preds['timestamp'] = df.loc[preds_proba >= CONF_THRESHOLD].get('timestamp', datetime.now())
    high_conf_preds.to_csv(LOG_PATH, mode='a', header=not pd.io.common.file_exists(LOG_PATH), index=False)
    logging.info(f"{len(high_conf_preds)} yüksek güvenli tahmin loglandı.")

if __name__ == "__main__":
    while True:
        try:
            df = fetch_new_data()
            df = recompute_features(df)
            train_and_retrain(df)
        except Exception as e:
            logging.error(f"Hata: {e}")
        logging.info(f"Sleeping {SLEEP_SECONDS/60:.0f} dakika...")
        time.sleep(SLEEP_SECONDS)
