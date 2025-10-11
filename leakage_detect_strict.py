# leakage_detect_strict.py
import pandas as pd
import numpy as np

df = pd.read_parquet("features/BTC_USDT__1h__features.parquet")
if 'target' not in df.columns:
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df.dropna(inplace=True)

future_close = df['close'].shift(-1)
features = [c for c in df.columns if c not in ('timestamp','target','close')]

print("Checking exact equality and very-high corr with future close...")
exact_matches = []
high_corr = []
for f in features:
    # exact match? (rare)
    try:
        if df[f].equals(future_close):
            exact_matches.append((f, "exact==future_close"))
    except Exception:
        pass
    # correlation
    try:
        corr = df[f].corr(future_close)
    except Exception:
        corr = 0
    if abs(corr) > 0.9:
        high_corr.append((f, corr))

print("Exact matches:", exact_matches)
print("Very high correlation (>0.9):")
for f, c in sorted(high_corr, key=lambda x: -abs(x[1])):
    print(f, c)

# additionally test if any column equals shifted close by k steps (-5..5)
print("\nCheck if feature equals close.shift(k) for k in -5..5")
matches = []
for f in features:
    for k in range(-5, 6):
        if k == 0: continue
        try:
            if df[f].equals(df['close'].shift(k)):
                matches.append((f, k))
        except Exception:
            pass

print("Shift matches (feature == close.shift(k)):", matches)
