# walk_forward_cv_fixed.py
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import numpy as np

DATA = "features/BTC_USDT__1h__features_recomputed.parquet"
df = pd.read_parquet(DATA)
if 'target' not in df.columns:
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df.dropna(inplace=True)

feature_cols = [c for c in df.columns if c not in ('timestamp','target')]
n_splits = 5
split_size = len(df)//(n_splits+1)
scores = []

for i in range(n_splits):
    train_end = split_size*(i+1)
    test_end = train_end + split_size
    train = df.iloc[:train_end]
    test = df.iloc[train_end:test_end]
    if len(test) < 20:
        break
    model = LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=32, random_state=42)
    model.fit(train[feature_cols], train['target'])
    preds = model.predict(test[feature_cols])
    score = accuracy_score(test['target'], preds)
    scores.append(score)
    print(f"Fold {i+1} acc: {score:.4f}")

print("Mean walk-forward acc:", np.mean(scores))
