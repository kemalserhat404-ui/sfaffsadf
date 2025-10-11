# analyze_model.py
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

clf, feature_cols = joblib.load("models/btc_offline_lgb.joblib")
df = pd.read_parquet("features/BTC_USDT__1h__features.parquet").ffill().bfill()
# --- Hedef sütunu kontrolü ---
if 'target' not in df.columns:
    print("[WARN] 'target' column missing, creating automatically...")
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# aynı engineering fonksiyonuyla hizala (kısaca)
for lag in (1,2,3,6,12):
    df[f'close_lag_{lag}'] = df['close'].shift(lag)
    df[f'return_{lag}'] = df['close'] / (df[f'close_lag_{lag}'] + 1e-12) - 1
df['ema_ratio'] = (df['ema12'] / (df['ema26'] + 1e-12))
df['price_range'] = (df['high'] - df['low']) / (df['close'] + 1e-12)
df['volatility'] = df['close'].rolling(14).std().bfill()
df.dropna(inplace=True)
df = df.drop(columns=['timestamp']) if 'timestamp' in df.columns else df

X = df[feature_cols]
y = df['target']

probs = clf.predict_proba(X)[:,1]
preds = (probs > 0.5).astype(int)

print("Confusion matrix:\n", confusion_matrix(y, preds))
print("\nClassification report:\n", classification_report(y, preds, zero_division=0))
try:
    print("ROC AUC:", roc_auc_score(y, probs))
except:
    pass

# Threshold optimization for F1 / precision-recall (print candidate thresholds)
prec, rec, thr = precision_recall_curve(y, probs)
f1 = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = np.argmax(f1)
print("\nBest threshold by F1:", thr[best_idx] if best_idx < len(thr) else 0.5, "F1:", f1[best_idx])

# Feature importance
importances = clf.feature_importances_
feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
print("\nTop features:")
for f,i in feat_imp[:20]:
    print(f, i)
