# leakage_check.py
import pandas as pd
import numpy as np

df = pd.read_parquet("features/BTC_USDT__1h__features.parquet")
if 'target' not in df.columns:
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df.dropna(inplace=True)

# compute correlation of each feature with future close (shift -1)
future_close = df['close'].shift(-1)
features = [c for c in df.columns if c not in ('timestamp','target','close')]

corrs = []
for f in features:
    try:
        corr = df[f].corr(future_close)
    except Exception:
        corr = 0
    corrs.append((f, corr))

corrs_sorted = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
print("Top features correlated with future close (abs correlation):")
for f, c in corrs_sorted[:30]:
    print(f, c)
print("\nNote: high absolute corr with future_close may indicate leakage or extremely predictive feature. Investigate features with |corr|>0.6")
