# models/predictor.py
import joblib
import pandas as pd
import numpy as np
import os

# tolerant import: compute_indicators veya add_common_indicators
_add_ind = None
try:
    from utils.indicators import add_common_indicators as _add_ind
except Exception:
    try:
        from utils.indicators import compute_indicators as _add_ind
    except Exception:
        _add_ind = None

def load_model_bundle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    bundle = joblib.load(path)
    # expected keys: 'model','scaler','features'
    model = bundle.get('model', bundle) if not isinstance(bundle, dict) else bundle.get('model')
    scaler = bundle.get('scaler') if isinstance(bundle, dict) else (bundle.get('scaler') if hasattr(bundle,'get') else None)
    features = bundle.get('features') if isinstance(bundle, dict) else (bundle.get('features') if hasattr(bundle,'get') else None)

    # backward compatibility: maybe saved as single estimator -> handle minimal case
    if model is None and hasattr(bundle, "predict_proba"):
        # bundle is the estimator itself
        model = bundle
        scaler = None
        features = None

    if model is None:
        raise ValueError("Model bulunamadı inside bundle.")

    return model, scaler, features

def prepare_X_from_df(df_raw, required_features):
    """
    Ensure returned X is a DataFrame with columns in same order as required_features.
    - If raw OHLCV is provided and indicator function exists, compute indicators.
    - If df already has required_features, simply pick & reorder.
    """
    df = df_raw.copy().reset_index(drop=True)

    # if raw ohlcv present and indicator function available -> compute
    raw_cols = set(['open','high','low','close','volume','timestamp'])
    if _add_ind is not None and raw_cols.intersection(set(df.columns)):
        df = _add_ind(df)

    # required_features may be None -> just return numeric DF
    if required_features is None:
        X = df.select_dtypes(include=[np.number])
        return X

    # check missing features
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        raise ValueError(f"Model bekliyor fakat dataframe içinde eksik sütunlar: {missing}")

    # select and ensure numeric dtype
    X = df[required_features].copy()
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] != len(required_features):
        # some features might be non-numeric -> try coerce
        X = df[required_features].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    return X

def predict_latest(model_path, ohlcv_or_features_df):
    """
    model_path: path to joblib bundle (with keys 'model','scaler','features')
    ohlcv_or_features_df: DataFrame (either raw OHLCV or already-featured)
    returns: (pred_int, prob_float)
    """
    model, scaler, features = load_model_bundle(model_path)

    X = prepare_X_from_df(ohlcv_or_features_df, features)

    # keep column order exactly as features if provided
    if features is not None:
        X = X[features]

    x_latest = X.tail(1)

    # apply scaler if present
    if scaler is not None:
        # if scaler expects feature names, ensure we pass DataFrame w/ same columns
        try:
            x_scaled = scaler.transform(x_latest)
        except Exception:
            # fallback: convert to numpy
            x_scaled = scaler.transform(x_latest.values)
    else:
        x_scaled = x_latest.values

    proba = None
    pred = None
    # predict_proba if available
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(x_scaled)[0][1])
    else:
        # if no probas, fallback to model.predict
        pred = int(model.predict(x_scaled)[0])
        proba = 1.0 if pred==1 else 0.0

    if pred is None:
        pred = int(model.predict(x_scaled)[0])

    return pred, proba

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/btc_usdt_1h_xgb.joblib")
    parser.add_argument("--file", default="features/BTC_USDT__1h__features.parquet")
    args = parser.parse_args()
    df = pd.read_parquet(args.file)
    print(predict_latest(args.model, df))
