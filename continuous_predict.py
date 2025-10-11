import pandas as pd
import joblib
import time
import logging
from datetime import datetime

# --- CONFIG ---
MODEL_PATH = "models/btc_offline_lgb_recomputed.joblib"
FEATURE_PATH = "features/BTC_USDT__1h__features_recomputed_auto.parquet"
LOG_PATH = "logs/continuous_predictions.log"
CONF_THRESHOLD = 0.7
SLEEP_SECONDS = 600  # 10 dakika

# --- LOGGING SETUP ---
logging.basicConfig(filename=LOG_PATH, level=logging.INFO,
                    format='%(asctime)s | %(message)s')

# --- LOAD MODEL ---
model = joblib.load(MODEL_PATH)
logging.info("Model loaded: %s", MODEL_PATH)

def run_prediction():
    df = pd.read_parquet(FEATURE_PATH)
    X = df.drop(columns=['target'], errors='ignore')
    preds_proba = model.predict_proba(X)[:,1]  # upward movement probability
    preds_label = (preds_proba >= CONF_THRESHOLD).astype(int)
    
    # log only predictions above threshold
    for ts, p, l in zip(df['timestamp'], preds_proba, preds_label):
        if l == 1:
            logging.info("Time: %s | Predicted Up | Probability: %.4f", ts, p)

if __name__ == "__main__":
    logging.info("Continuous prediction started...")
    while True:
        try:
            run_prediction()
            time.sleep(SLEEP_SECONDS)
        except KeyboardInterrupt:
            logging.info("Continuous prediction stopped by user")
            break
        except Exception as e:
            logging.exception("Error during prediction: %s", e)
            time.sleep(SLEEP_SECONDS)
