import pandas as pd
import os
from river import linear_model, preprocessing, metrics, compose, ensemble
import joblib

# -----------------------------
# CONFIG
# -----------------------------
DATA_FILE = r"C:\Users\digon\trading-bot-proto\features\BTC_USDT__1h__features.parquet"
MODEL_PATH = "models/online_bot1.joblib"

feature_cols = [
    'open','high','low','close','volume','return1',
    'ema12','ema26','macd','macd_signal','macd_hist','rsi14','atr14','vwap14','obv','spread',
    'close_lag_1','return_lag_1','close_lag_2','return_lag_2','close_lag_3','return_lag_3',
    'momentum14','price_range','volatility','vol_spike','ema_ratio','macd_cross','rsi_overbought','rsi_oversold'
]

# -----------------------------
# DATA LOADING & FEATURE ENGINEERING
# -----------------------------
df = pd.read_parquet(DATA_FILE)

# Lag features
df['close_lag_1'] = df['close'].shift(1)
df['return_lag_1'] = df['return1'].shift(1)
df['close_lag_2'] = df['close'].shift(2)
df['return_lag_2'] = df['return1'].shift(2)
df['close_lag_3'] = df['close'].shift(3)
df['return_lag_3'] = df['return1'].shift(3)

# Momentum
df['momentum14'] = df['close'] - df['close'].shift(14)

# Price range
df['price_range'] = df['high'] - df['low']

# Volatility
df['volatility'] = df['return1'].rolling(14).std()

# EMA ratio
df['ema_ratio'] = df['ema12'] / df['ema26']

# MACD cross
df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)

# RSI overbought/oversold
df['rsi_overbought'] = (df['rsi14'] > 70).astype(int)
df['rsi_oversold'] = (df['rsi14'] < 30).astype(int)

# Target
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Drop NA
df.dropna(inplace=True)

# Volume spike
df['vol_spike'] = (df['volume'] > df['volume'].rolling(14).mean() * 1.5).astype(int)


print(f"[INFO] Data loaded. Rows after prep: {len(df)}")
print(f"[INFO] Feature cols used: {feature_cols}")

# -----------------------------
# MODEL CREATION
# -----------------------------
def create_model():
    base_model = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression()
    )
    bag = ensemble.BaggingClassifier(base_model=base_model, n_models=10, seed=42)
    return bag

# -----------------------------
# STREAM TRAINING
# -----------------------------
def stream_train(model, df, feature_cols):
    acc_metric = metrics.Accuracy()
    prec_metric = metrics.Precision()
    rec_metric = metrics.Recall()

    correct = 0
    total = 0

    for i, row in df.iterrows():
        X = {col: row[col] for col in feature_cols}
        y = int(row['target'])

        y_pred = model.predict_one(X)
        if y_pred is None:
            y_pred = 0

        acc_metric.update(y, y_pred)
        prec_metric.update(y, y_pred)
        rec_metric.update(y, y_pred)

        model.learn_one(X, y)

        total += 1
        correct += (y_pred == y)

        if total % 50 == 0:
            print(f"[PROGRESS] {total}/{len(df)} | Acc: {acc_metric.get():.3f} | Prec: {prec_metric.get():.3f} | Rec: {rec_metric.get():.3f}")

        if total % 100 == 0:
            joblib.dump(model, MODEL_PATH)
            print(f"[SAVE] Model saved -> {MODEL_PATH}")

    print("=== FIN ===")
    print(f"Final Accuracy: {acc_metric.get():.3f}")
    print(f"Precision: {prec_metric.get():.3f}")
    print(f"Recall: {rec_metric.get():.3f}")
    joblib.dump(model, MODEL_PATH)
    print(f"[DONE] Model updated and saved at {MODEL_PATH}")

    return model, acc_metric.get(), prec_metric.get(), rec_metric.get(), correct, total

# -----------------------------
# MAIN
# -----------------------------
def main():
    if os.path.exists(MODEL_PATH):
        print("[INFO] Existing model found. Loading...")
        model = joblib.load(MODEL_PATH)
    else:
        print("[INFO] New model created.")
        model = create_model()

    model, acc, prec, rec, correct, total = stream_train(model, df[feature_cols + ["target"]], feature_cols)

if __name__ == "__main__":
    print("[INFO] Starting online orchestrator v3.2")
    main()
