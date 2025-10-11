import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

FEATURE_SET = [
    'open','high','low','close','volume',
    'return1','ema12','ema26','macd','macd_signal','macd_hist',
    'rsi14','atr14','vwap14','obv','spread',
    'close_lag_1','return_1','close_lag_2','return_2',
    'close_lag_3','return_3','close_lag_6','return_6','close_lag_12','return_12',
    'rolling_mean_14','rolling_std_14','volatility','momentum14','price_range',
    'vol_mean_14','vol_spike','ema_ratio','price_high_low','atr14_simple'
]

FEATURE_FILE = "features/BTC_USDT__1h__features_v1.parquet"
MODEL_FILE = "models/btc_offline_lgb_recomputed.joblib"

print("[INFO] Loading data...")
df = pd.read_parquet(FEATURE_FILE)

X_test = df[FEATURE_SET].values
y_test = (df['close'].shift(-1) > df['close']).astype(int).values[:-1]
X_test = X_test[:-1]

print("[INFO] Loading model:", MODEL_FILE)
model = joblib.load(MODEL_FILE)

print("[INFO] Running predictions...")
probs = model.predict_proba(X_test)[:,1]
preds = (probs > 0.5).astype(int)
acc = accuracy_score(y_test, preds)
print("[INFO] Test accuracy:", round(acc, 4))
