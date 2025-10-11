import pandas as pd
import joblib
import argparse
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
from datetime import datetime, timedelta

def train_xgb_model(features_path, out_path):
    """Eğitim fonksiyonu — modeli ve feature listesini birlikte kaydeder."""
    df = pd.read_parquet(features_path)
    if "label" not in df.columns:
        raise ValueError("Veride 'label' sütunu yok — eğitim yapılamaz.")
    X = df.select_dtypes(include=[np.number]).drop(columns=["label"], errors="ignore")
    y = df["label"]

    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    sample_weights = y.map(dict(zip(classes, weights)))

    model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(classes),
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='mlogloss'
    )

    print(f"[TRAIN] Model eğitiliyor... (n={len(df)})")
    model.fit(X, y, sample_weight=sample_weights)
    bundle = {"model": model, "features": list(X.columns)}
    joblib.dump(bundle, out_path)
    print(f"[TRAIN] Model kaydedildi -> {out_path}")

def retrain_model_if_needed(model_path="models/btc_offline_lgb_recomputed_optuna.joblib",
                            features_path="features/BTC_USDT__1h__features.parquet",
                            retrain_days=7):
    """Model eskiyse otomatik retrain yapar."""
    if not os.path.exists(model_path):
        print(f"[INFO] Model bulunamadı, yeniden eğitiliyor -> {model_path}")
        train_xgb_model(features_path, model_path)
        return

    mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
    if datetime.now() - mod_time > timedelta(days=retrain_days):
        print(f"[INFO] Model {retrain_days} günden eski, yeniden eğitiliyor...")
        train_xgb_model(features_path, model_path)
    else:
        print(f"[INFO] Model güncel, yeniden eğitim gerekmedi: {model_path}")
