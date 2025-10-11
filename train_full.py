# train_full.py (GÜNCEL)
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from lightgbm import LGBMClassifier
import optuna
import warnings

warnings.filterwarnings("ignore")

# ------------- CONFIG -------------
FEATURE_FILE = "features/BTC_USDT__1h__features.parquet"
MODEL_OUT = "models/btc_offline_lgb.joblib"
N_SPLITS = 5
RANDOM_STATE = 42
N_TRIALS = 20  # ilk turda 20 yap; istersen artır
# ----------------------------------

def load_and_engineer(path):
    df = pd.read_parquet(path)
    # basic fill
    df = df.ffill().bfill()
    # target: next close higher than current
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    # feature engineering: lags, returns, ratios
    for lag in (1,2,3,6,12):
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'return_{lag}'] = df['close'] / (df[f'close_lag_{lag}'] + 1e-12) - 1
    df['ema_ratio'] = (df['ema12'] / (df['ema26'] + 1e-12))
    df['price_range'] = (df['high'] - df['low']) / (df['close'] + 1e-12)
    df['volatility'] = df['close'].rolling(14).std().bfill()  # deprecated fillna(method=...) avoided
    df.dropna(inplace=True)
    # drop non-numeric or unnecessary
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    return df

print("[INFO] Loading and engineering features...")
df = load_and_engineer(FEATURE_FILE)
feature_cols = [c for c in df.columns if c != 'target']
X = df[feature_cols]
y = df['target']

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

def objective(trial):
    params = {
        'n_estimators': 2000,
        'random_state': RANDOM_STATE,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 256),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'verbosity': -1,
    }
    accs = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        clf = LGBMClassifier(**params)
        # use eval_set + early stopping
        try:
            clf.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='binary_logloss',
                    early_stopping_rounds=50,
                    verbose=False)
        except TypeError:
            # bazı lgb sürümlerinde farklı signature olabilir -> fallback no early stopping
            clf.fit(X_train, y_train)
        preds = (clf.predict_proba(X_val)[:,1] > 0.5).astype(int)
        accs.append(accuracy_score(y_val, preds))
    return float(np.mean(accs))

print("[INFO] Starting Optuna study (this can take time)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
print("[INFO] Best params found:", study.best_params)

# Train final model on whole data with best params
best_params = study.best_params.copy()
best_params.update({
    'n_estimators': 2000,
    'random_state': RANDOM_STATE,
    'verbosity': -1,
})
print("[INFO] Training final LGBM on full dataset...")
final_clf = LGBMClassifier(**best_params)
# try with early stopping on a small split to avoid overfit,
split = int(len(X) * 0.95)
X_train_full, X_val_small = X.iloc[:split], X.iloc[split:]
y_train_full, y_val_small = y.iloc[:split], y.iloc[split:]
try:
    final_clf.fit(X_train_full, y_train_full,
                  eval_set=[(X_val_small, y_val_small)],
                  eval_metric='binary_logloss',
                  early_stopping_rounds=50,
                  verbose=50)
except TypeError:
    final_clf.fit(X_train_full, y_train_full)

# Save model and feature list
os.makedirs("models", exist_ok=True)
joblib.dump((final_clf, feature_cols), MODEL_OUT)
print("[INFO] Saved final model to", MODEL_OUT)

# Quick evaluation on holdout (last 5%)
X_hold, y_hold = X_val_small, y_val_small
preds = (final_clf.predict_proba(X_hold)[:,1] > 0.5).astype(int)
print("Holdout accuracy:", accuracy_score(y_hold, preds))
print("Precision:", precision_score(y_hold, preds, zero_division=0))
print("Recall:", recall_score(y_hold, preds, zero_division=0))
