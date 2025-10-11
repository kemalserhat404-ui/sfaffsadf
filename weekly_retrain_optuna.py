# weekly_retrain_optuna.py
import pandas as pd
import joblib
import logging
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# CONFIG
# -----------------------------
FEATURE_FILE = "features/BTC_USDT__1h__features_recomputed.parquet"
MODEL_FILE = "models/btc_offline_lgb_recomputed.joblib"
TARGET_COLUMN = "target"
N_TRIALS = 50  # Optuna deneme sayısı, artırılabilir

logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    level=logging.INFO
)

# -----------------------------
# LOAD FEATURES
# -----------------------------
logging.info(f"Loading features: {FEATURE_FILE}")
df = pd.read_parquet(FEATURE_FILE)
X = df.drop(columns=['target', 'timestamp'], errors='ignore')
y = df[TARGET_COLUMN].values

# Holdout split Optuna için
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# -----------------------------
# OPTUNA OBJECTIVE
# -----------------------------
def objective(trial):
    param = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }

    model = LGBMClassifier(**param)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return acc

# -----------------------------
# OPTUNA STUDY
# -----------------------------
logging.info("Starting Optuna optimization...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS)
best_params = study.best_trial.params
logging.info(f"Best trial params: {best_params}")

# -----------------------------
# TRAIN FINAL MODEL
# -----------------------------
logging.info("Training final model with best parameters...")
final_model = LGBMClassifier(**best_params)
final_model.fit(X, y)

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(final_model, MODEL_FILE)
logging.info(f"Saved optimized model -> {MODEL_FILE}")
