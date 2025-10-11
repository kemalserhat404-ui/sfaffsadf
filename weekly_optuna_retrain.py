# weekly_optuna_retrain.py
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
import optuna
import logging
from datetime import datetime

FEATURE_FILE = "features/BTC_USDT__1h__features_recomputed_auto.parquet"
MODEL_FILE = "models/btc_offline_lgb_recomputed_optuna.joblib"
TARGET_COLUMN = "target"

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

df = pd.read_parquet(FEATURE_FILE)
X = df.drop(columns=['timestamp', TARGET_COLUMN], errors='ignore')
y = df[TARGET_COLUMN]

def objective(trial):
    param = {
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'random_state': 42
    }
    model = LGBMClassifier(**param)
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = (y_pred == y).mean()
    return acc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
logging.info(f"Best params: {best_params}")

best_model = LGBMClassifier(**best_params)
best_model.fit(X, y)

joblib.dump(best_model, MODEL_FILE)
logging.info(f"Optimized model saved -> {MODEL_FILE}")
