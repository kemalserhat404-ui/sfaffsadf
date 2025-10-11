import pandas as pd
import glob
import os

features_dir = "features"

# timeframe’e göre horizon ve threshold
timeframes = {
    "1m": {"H": 12, "thr": 0.0002},  # 12 dakikalık horizon, 0.02%
    "5m": {"H": 12, "thr": 0.0005},  # 1 saatlik horizon, 0.05%
    "15m": {"H": 4, "thr": 0.0015},  # 1 saatlik horizon, 0.15%
    "1h": {"H": 4, "thr": 0.003},    # 4 saatlik horizon, 0.3%
}

files = glob.glob(os.path.join(features_dir, "*__features.parquet"))

for f in files:
    df = pd.read_parquet(f)
    base = os.path.basename(f)
    
    # timeframe belirle
    tf = "1h"  # default
    for key in timeframes:
        if key in base:
            tf = key
            break
    H = timeframes[tf]["H"]
    thr = timeframes[tf]["thr"]

    # label oluştur
    df['future_return'] = df['close'].shift(-H) / df['close'] - 1
    df['label'] = 0
    df.loc[df['future_return'] > thr, 'label'] = 1
    df.loc[df['future_return'] < -thr, 'label'] = -1
    df = df.drop(columns=['future_return'])
    
    out_path = f.replace("__features.parquet", "__labeled.parquet")
    df.to_parquet(out_path)
    print(f"Saved labeled: {out_path} rows: {len(df)}")
