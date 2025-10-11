# continuous_multi_horizon_pipeline_full_fixed.py
"""
Continuous multi-horizon pipeline (LightGBM compatible)
- multi-interval fetch, full feature recompute
- safe prediction with expected feature reindex
- high-confidence logging
- heartbeat file for monitoring
- automatic retrain hook
"""
import os, time, json, logging, joblib, traceback
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# --- Local repo helpers ---
try:
    from data_fetcher import fetch_ohlcv
except:
    try:
        from data_pipeline import fetch_ohlcv
    except:
        fetch_ohlcv = None

try:
    from recompute_features_full import compute_indicators
except:
    compute_indicators = None

try:
    from models.train_xgb import train_xgb_model
except:
    train_xgb_model = None

# ---- CONFIG ----
MODEL_PATH = "models/btc_offline_lgb_recomputed_optuna.joblib"
FEATURE_DIR = "features"
PAIR = "BTCUSDT"
INTERVALS = ["5m", "15m", "30m", "1h", "4h"]
SLEEP_MINUTES = 60
HIGH_CONF_THRESH = 0.90
LOW_CONF_THRESH = 0.10
RETRAIN_DAYS = 7
HEARTBEAT_PATH = "run_heartbeat.txt"
HIGH_CONF_LOG = "logs/high_confidence_signals.csv"
MODEL_MANIFEST = MODEL_PATH + ".manifest.json"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ---- UTILITIES ----
def save_manifest(features_list, model_path=MODEL_PATH):
    manifest = {"model_path": os.path.basename(model_path),
                "saved_at": datetime.utcnow().isoformat(),
                "features": list(features_list)}
    with open(MODEL_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved manifest -> {MODEL_MANIFEST}")

def load_model_and_expected_features(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    payload = joblib.load(path)
    if isinstance(payload, dict):
        model = payload.get("model") or payload.get("estimator") or payload
        features = payload.get("features") or payload.get("feature_names")
        if features:
            return model, list(features)
    else:
        model = payload
    try:
        if hasattr(model, "booster_"):
            feat = model.booster_.feature_name()
            if feat:
                return model, list(feat)
    except: pass
    if os.path.exists(MODEL_MANIFEST):
        try:
            with open(MODEL_MANIFEST,"r",encoding="utf-8") as f:
                m = json.load(f)
                return model, m.get("features")
        except: pass
    return model, None

def ensure_numeric_df(df):
    df_num = df.select_dtypes(include=[np.number])
    if df_num.shape[1] != df.shape[1]:
        df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    return df

def safe_reindex_for_model(X_df, expected_features):
    if expected_features is None:
        return ensure_numeric_df(X_df)
    X = X_df.copy()
    X = X.reindex(columns=expected_features, fill_value=0)
    return ensure_numeric_df(X)

def heartbeat():
    with open(HEARTBEAT_PATH,"w",encoding="utf-8") as f:
        f.write(datetime.utcnow().isoformat())

def append_high_conf_log(row: dict):
    os.makedirs(os.path.dirname(HIGH_CONF_LOG) or ".", exist_ok=True)
    header = not os.path.exists(HIGH_CONF_LOG)
    pd.DataFrame([row]).to_csv(HIGH_CONF_LOG, mode="a", header=header, index=False)

def retrain_if_needed_hook(model_path=MODEL_PATH, features_dir=FEATURE_DIR):
    if not train_xgb_model: return
    if not os.path.exists(model_path):
        logging.info("Model missing, running trainer.")
        candidate = os.path.join(features_dir, f"{PAIR}__1h__features_recomputed.parquet")
        if os.path.exists(candidate):
            train_xgb_model(candidate, model_path)
    else:
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        if datetime.utcnow() - mod_time > timedelta(days=RETRAIN_DAYS):
            logging.info("Model outdated, retraining.")
            candidate = os.path.join(features_dir, f"{PAIR}__1h__features_recomputed.parquet")
            if os.path.exists(candidate):
                train_xgb_model(candidate, model_path)

# ---- MAIN LOOP ----
def run_continuous():
    logging.info("Starting continuous multi-horizon pipeline (full fixed)")
    try: retrain_if_needed_hook()
    except Exception as e: logging.warning(f"Retrain hook failed: {e}")

    try:
        model, expected_features = load_model_and_expected_features(MODEL_PATH)
        logging.info(f"Model loaded -> {MODEL_PATH}")
        if expected_features:
            logging.info(f"Expected feature count: {len(expected_features)}")
        else:
            logging.info("No expected feature list found; will infer at runtime.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

    loop_count = 0
    while True:
        loop_count += 1
        for interval in INTERVALS:
            logging.info(f"[{interval}] fetching...")
            try:
                if fetch_ohlcv is None:
                    raise RuntimeError("fetch_ohlcv not available")
                df_raw = fetch_ohlcv(f"{PAIR}", interval, limit=600)
                if df_raw is None or df_raw.empty:
                    logging.warning(f"[{interval}] no data returned, skipping.")
                    continue

                if compute_indicators is None:
                    raise RuntimeError("compute_indicators not available")
                df_feat = compute_indicators(df_raw.copy())
                if df_feat is None or len(df_feat)==0:
                    logging.warning(f"[{interval}] feature compute returned empty.")
                    continue

                # save snapshot
                out_file = os.path.join(FEATURE_DIR, f"{PAIR}__{interval}__features_recomputed_auto.parquet")
                os.makedirs(FEATURE_DIR, exist_ok=True)
                try: df_feat.to_parquet(out_file)
                except Exception as e: logging.warning(f"[{interval}] parquet save failed: {e}")
                logging.info(f"[{interval}] Saved features -> {out_file}")

                # predict last row
                X_last = df_feat.tail(1).drop(columns=["target"], errors="ignore")
                X_for_model = safe_reindex_for_model(X_last, expected_features)
                if X_for_model.shape[1]==0:
                    logging.error(f"[{interval}] No numeric features after reindex. Skipping.")
                    continue
                preds = model.predict(X_for_model)
                probs = model.predict_proba(X_for_model)
                pred_label = int(preds[0])
                prob_max = float(np.max(probs))
                logging.info(f"[{interval}] pred={pred_label} prob={prob_max:.3f}")

                # high-confidence log
                if prob_max >= HIGH_CONF_THRESH or prob_max <= LOW_CONF_THRESH:
                    now = datetime.utcnow().isoformat()
                    price = float(df_raw['close'].iloc[-1]) if 'close' in df_raw.columns else None
                    log_row = {"timestamp": now, "interval": interval, "pair": PAIR,
                               "pred": pred_label, "prob": prob_max, "price": price,
                               "model": os.path.basename(MODEL_PATH),
                               "row_index": df_feat.index[-1].isoformat() if hasattr(df_feat.index[-1],"isoformat") else str(df_feat.index[-1])}
                    for c in X_for_model.columns[:20]:
                        log_row[c] = float(X_for_model.iloc[0].get(c,0.0))
                    append_high_conf_log(log_row)
                    logging.info(f"[{interval}] High-conf signal logged (prob={prob_max:.3f})")

            except Exception as e:
                logging.error(f"[{interval}] Error: {e}")
                logging.debug(traceback.format_exc())

            time.sleep(1.0)

        heartbeat()
        if loop_count % 6 == 0:
            try: retrain_if_needed_hook()
            except Exception as e: logging.warning(f"Retrain hook failed in periodic check: {e}")

        logging.info(f"Sleeping {SLEEP_MINUTES} minutes...")
        time.sleep(SLEEP_MINUTES*60)

if __name__=="__main__":
    run_continuous()
