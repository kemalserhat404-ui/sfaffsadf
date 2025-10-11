import pandas as pd
import joblib
import time
import logging
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/btc_offline_lgb_recomputed.joblib"
FEATURE_PATH = "features/BTC_USDT__1h__features_recomputed_auto.parquet"
LOG_PATH = "logs/high_conf_preds.csv"
CONF_THRESHOLD = 0.7
SLEEP_SECONDS = 3600  # 1 saat

logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    level=logging.INFO
)

def load_features(file_path):
    df = pd.read_parquet(file_path)
    return df

def fix_features(X, expected_features):
    """
    Modelin beklediği feature setine göre X'i sabitler.
    Eksik feature'lar 0 ile doldurulur.
    Feature sırası model ile aynı yapılır.
    """
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0.0
    X = X[expected_features]
    return X

if __name__ == "__main__":
    model = joblib.load(MODEL_PATH)
    expected_features = model.feature_name_  # eğitimde kullanılan feature listesi

    while True:
        try:
            logging.info("Yeni veri yükleniyor...")
            df = load_features(FEATURE_PATH)

            X = df.drop(columns=['target','timestamp'], errors='ignore')
            y = df['target'] if 'target' in df.columns else None

            # Feature set'i sabitle
            X = fix_features(X, expected_features)

            # Tahmin
            probs = model.predict_proba(X)[:,1]

            # Yüksek-confidence filtre
            high_conf_idx = probs >= CONF_THRESHOLD
            X_high_conf = X[high_conf_idx]
            y_high_conf = y[high_conf_idx] if y is not None else None

            # High-confidence retrain
            if y_high_conf is not None and len(X_high_conf) > 10:
                model.fit(X_high_conf, y_high_conf)
                joblib.dump(model, MODEL_PATH)
                logging.info(f"High-confidence örneklerle model güncellendi: {len(X_high_conf)} örnek")
            else:
                logging.info("Yeterli yüksek-confidence örnek yok, retrain atlandı.")

            # Loglama
            if len(X_high_conf) > 0:
                high_conf_preds = df.loc[high_conf_idx].copy()
                high_conf_preds['pred_proba'] = probs[high_conf_idx]
                high_conf_preds['timestamp'] = df.loc[high_conf_idx].get('timestamp', pd.Timestamp.now())
                high_conf_preds.to_csv(LOG_PATH, mode='a', header=not pd.io.common.file_exists(LOG_PATH), index=False)
                logging.info(f"{len(high_conf_preds)} yüksek-confidence tahmin loglandı.")

        except Exception as e:
            logging.error(f"Hata: {e}")

        logging.info(f"Sleeping {SLEEP_SECONDS/60:.0f} dakika...")
        time.sleep(SLEEP_SECONDS)
