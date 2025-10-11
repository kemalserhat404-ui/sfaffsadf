import pandas as pd
import lightgbm as lgb
import joblib

FEATURE_SET = [
    'open','high','low','close','volume',
    'return1','ema12','ema26','macd','macd_signal','macd_hist',
    'rsi14','atr14','vwap14','obv','spread',
    'close_lag_1','return_1','close_lag_2','return_2',
    'close_lag_3','return_3','close_lag_6','return_6','close_lag_12','return_12',
    'rolling_mean_14','rolling_std_14','volatility','momentum14','price_range',
    'vol_mean_14','vol_spike','ema_ratio','price_high_low','atr14_simple'
]

INPUT_FILE = "features/BTC_USDT__1h__features_v1.parquet"
OUTPUT_MODEL = "models/btc_offline_lgb_recomputed.joblib"

print("[INFO] Loading data...")
df = pd.read_parquet(INPUT_FILE)

X = df[FEATURE_SET].values
y = (df['close'].shift(-1) > df['close']).astype(int).values[:-1]
X = X[:-1]  # target ile hizalama

print("[INFO] Fitting model...")
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model.fit(X, y)
joblib.dump(model, OUTPUT_MODEL)
print("[SAVE] Saved ->", OUTPUT_MODEL)
