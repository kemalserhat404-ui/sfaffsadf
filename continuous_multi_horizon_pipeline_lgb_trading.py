# continuous_multi_horizon_pipeline_lgb_trading.py
import os, time, logging, joblib, numpy as np, pandas as pd
from datetime import datetime, timedelta
from data_pipeline import fetch_ohlcv
from utils.indicators import compute_indicators
from models.train_xgb import train_xgb_model  # retrain fonksiyonu

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

MODEL_PATH = "models/btc_offline_lgb_recomputed_optuna.joblib"
PAIR = "BTC/USDT"
INTERVALS = ["5m", "15m", "30m", "1h", "4h"]
SLEEP_MIN = 60
HIGH_CONF_THRESH = 0.9
LOW_CONF_THRESH = 0.1
RETRAIN_DAYS = 7

def retrain_if_needed(model_path=MODEL_PATH, features_path=f"features/{PAIR.replace('/','')}__1h__features.parquet"):
    """Model eskiyse retrain et"""
    if not os.path.exists(model_path):
        logging.info(f"Model bulunamadı, yeniden eğitiliyor -> {model_path}")
        train_xgb_model(features_path, model_path)
        return

    mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
    if datetime.now() - mod_time > timedelta(days=RETRAIN_DAYS):
        logging.info(f"Model {RETRAIN_DAYS} günden eski, yeniden eğitiliyor...")
        train_xgb_model(features_path, model_path)
    else:
        logging.info(f"Model güncel, yeniden eğitim gerekmedi: {model_path}")

def safe_predict(model, df):
    """LightGBM için güvenli tahmin"""
    if df is None or len(df) == 0:
        raise ValueError("Empty features")
    X = df.select_dtypes(include=[np.number])
    model_features = model.booster_.feature_name()
    if X.shape[1] != len(model_features):
        X = X.reindex(columns=model_features, fill_value=0)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return y_pred, y_prob

def run_continuous():
    logging.info("Starting continuous multi-horizon trading pipeline")
    retrain_if_needed()
    model = joblib.load(MODEL_PATH)
    logging.info(f"Model loaded -> {MODEL_PATH}")
    logging.info(f"Model ID: {os.path.basename(MODEL_PATH)}")

    while True:
        for interval in INTERVALS:
            logging.info(f"Yeni veri çekiliyor... ({interval})")
            try:
                df = fetch_ohlcv(PAIR, interval, limit=500)
                if df is None or len(df) < 50:
                    logging.warning(f"Yetersiz veri ({interval}), atlanıyor.")
                    continue

                df_feat = compute_indicators(df)
                save_path = f"features/{PAIR.replace('/','')}__1h__features_recomputed_auto_{interval}.parquet"
                df_feat.to_parquet(save_path)
                logging.info(f"Saved recomputed features -> {save_path}")

                try:
                    y_pred, y_prob = safe_predict(model, df_feat.tail(1))
                    prob = float(np.max(y_prob))
                    pred_label = int(y_pred[0])
                    logging.info(f"[{interval}] Last pred: {pred_label}, prob: {prob:.3f}")

                    # Probability saturation uyarısı
                    if prob < LOW_CONF_THRESH or prob > HIGH_CONF_THRESH:
                        logging.warning(f"[{interval}] Probability saturation detected ({prob:.3f})")

                    # High-confidence trading sinyali
                    if prob > HIGH_CONF_THRESH:
                        logging.info(f"[{interval}] HIGH CONFIDENCE LONG SIGNAL (pred={pred_label}, prob={prob:.3f})")
                    elif prob < LOW_CONF_THRESH:
                        logging.info(f"[{interval}] HIGH CONFIDENCE SHORT SIGNAL (pred={pred_label}, prob={prob:.3f})")
                    else:
                        logging.info(f"[{interval}] No high-confidence signal.")

                except Exception as e:
                    logging.error(f"Prediction failed for {interval}: {e}")

            except Exception as e:
                logging.error(f"Veri çekme hatası ({interval}): {e}")

        logging.info(f"Sleeping {SLEEP_MIN} dakika...")
        time.sleep(SLEEP_MIN * 60)

if __name__ == "__main__":
    run_continuous()
