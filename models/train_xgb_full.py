import pandas as pd
import numpy as np
import joblib
import talib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Ayarlar
FEATURE_FILE = "features/all_labeled_mapped.parquet"
MODEL_FILE = "models/btc_usdt_all_xgb.joblib"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 1️⃣ Veri yükleme
df = pd.read_parquet(FEATURE_FILE)
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# 2️⃣ Teknik göstergeler ekleme (opsiyonel)
df['SMA20'] = talib.SMA(df['close'], timeperiod=20)
df['EMA12'] = talib.EMA(df['close'], timeperiod=12)
df['RSI14'] = talib.RSI(df['close'], timeperiod=14)
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# 3️⃣ Özellik ve hedef belirleme
X = df.drop(columns=['label','timestamp'])
y = df['label']

# 4️⃣ Label encoding (gerekirse)
le = LabelEncoder()
y = le.fit_transform(y)

# 5️⃣ Eğitim / test bölme
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 6️⃣ Model oluşturma
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# 7️⃣ Modeli eğitme
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# 8️⃣ Modeli kaydetme
joblib.dump(model, MODEL_FILE)
print(f"Model kaydedildi: {MODEL_FILE}")

# 9️⃣ Basit kontrol
preds = model.predict(X_val)
print("İlk 10 tahmin:", preds[:10])
