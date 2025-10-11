# continuous_multi_horizon_pipeline_lgb_fixed.py
import os, time, logging, joblib, numpy as np, pandas as pd
from datetime import datetime
from data_pipeline import fetch_ohlcv
from utils.indicators import compute_indicators

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

MODEL_PATH = "models/btc_offline_lgb_recomputed_optuna.joblib"
PAIR = "BTC/USDT"
INTERVALS = ["5m", "15m", "30m", "1h", "4h"]
SLEEP_MIN = 60
HIGH_CONF_THRESH = 0.9  # high confidence threshold
LOW_CONF_THRESH = 0.1   # low probability threshold

def safe_predict(model, df):
    """LightGBM için güvenli tahmin"""
    if df is None or len(df) == 0:
        raise ValueError("Empty features")
    X = df.select_dtypes(include=[np.number])
    # Eğer feature sayısı uyuşmazsa booster feature listesine göre reindex et
    model_features = model.booster_.feature_name()
    if X.shape[1] != len(model_features):
        X = X.reindex(columns=model_features, fill_value=0)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return y_pred, y_prob

def run_continuous():
    logging.info("Starting continuous multi-horizon pipeline")

    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model not found: {MODEL_PATH}")
        return

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
                    logging.info(f"[{interval}] Last pred: {int(y_pred[0])}, prob: {prob:.3f}")

                    if prob < LOW_CONF_THRESH or prob > HIGH_CONF_THRESH:
                        logging.warning(f"[{interval}] Probability saturation detected ({prob:.3f})")

                    if prob > HIGH_CONF_THRESH:
                        logging.info(f"[{interval}] High-confidence prediction detected.")
                    else:
                        logging.info(f"[{interval}] Not enough high-confidence samples for retrain.")

                except Exception as e:
                    logging.error(f"Prediction failed for {interval}: {e}")

            except Exception as e:
                logging.error(f"Veri çekme hatası ({interval}): {e}")

        logging.info(f"Sleeping {SLEEP_MIN} dakika...")
        time.sleep(SLEEP_MIN * 60)

if __name__ == "__main__":
    run_continuous()
