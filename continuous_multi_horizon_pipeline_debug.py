# continuous_multi_horizon_pipeline_debug.py
from recompute_features_full import compute_indicators
import os, time, logging, joblib, traceback
from datetime import datetime
import numpy as np
import pandas as pd

# --- Repo helpers ---
try:
    from data_fetcher import fetch_ohlcv
except Exception:
    try:
        from data_pipeline import fetch_ohlcv
    except Exception:
        fetch_ohlcv = None

try:
    from utils.indicators import compute_indicators
except Exception:
    try:
        from recompute_features import recompute_features as compute_indicators
    except Exception:
        compute_indicators = None

# Config
MODEL_PATH = "models/btc_offline_lgb_recomputed_optuna.joblib"
FEATURE_DIR = "features"
PAIR = "BTCUSDT"
INTERVALS = ["5m", "15m", "30m", "1h", "4h"]

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def safe_reindex_for_model(X_df, expected_features):
    if expected_features is None:
        return X_df.fillna(0)
    X = X_df.reindex(columns=expected_features, fill_value=0)
    return X.fillna(0)

def run_debug():
    logging.info("=== Starting debug multi-horizon pipeline ===")
    
    # Load model + expected features
    payload = joblib.load(MODEL_PATH)
    if isinstance(payload, dict):
        model = payload.get("model") or payload
        expected_features = payload.get("features") or None
    else:
        model = payload
        expected_features = None
        if hasattr(model, "booster_"):
            expected_features = model.booster_.feature_name()
    
    logging.info(f"Model loaded -> {MODEL_PATH}")
    logging.info(f"Expected features ({len(expected_features) if expected_features else 0}): {expected_features}")

    for interval in INTERVALS:
        logging.info(f"[{interval}] fetching data...")
        df_raw = fetch_ohlcv(PAIR, interval, limit=600)
        if df_raw is None or df_raw.empty:
            logging.warning(f"[{interval}] no data returned.")
            continue

        df_feat = compute_indicators(df_raw.copy())
        if df_feat is None or len(df_feat) == 0:
            logging.warning(f"[{interval}] feature compute returned empty.")
            continue

        out_file = os.path.join(FEATURE_DIR, f"{PAIR}__{interval}__features_recomputed_auto.parquet")
        os.makedirs(FEATURE_DIR, exist_ok=True)
        df_feat.to_parquet(out_file)
        logging.info(f"[{interval}] Saved features -> {out_file}")

        # DEBUG: last row + feature check
        X_last = df_feat.tail(1).drop(columns=["target"], errors="ignore")
        logging.info(f"[{interval}] last row features:\n{X_last}")

        X_for_model = safe_reindex_for_model(X_last, expected_features)
        logging.info(f"[{interval}] features after reindex (missing filled 0):\n{X_for_model}")

        preds = model.predict(X_for_model)
        probs = model.predict_proba(X_for_model)
        pred_label = int(preds[0])
        prob_max = float(np.max(probs))
        logging.info(f"[{interval}] pred={pred_label} prob={prob_max:.3f}")

if __name__ == "__main__":
    run_debug()
