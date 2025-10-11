import pandas as pd
import glob
import os
from utils.indicators import compute_indicators  # senin mevcut indicator fonksiyonun

data_processed_dir = "data/processed"
features_dir = "features"

files = glob.glob(os.path.join(data_processed_dir, "*.parquet"))

for f in files:
    df = pd.read_parquet(f)
    
    # timestamp yoksa atla
    if 'timestamp' not in df.columns:
        print("Atlandı (timestamp yok):", f)
        continue

    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # teknik göstergeleri hesapla
    df_feat = compute_indicators(df)
    
    # feature dosyası adı
    base = os.path.basename(f).replace(".parquet", "__features.parquet")
    out_path = os.path.join(features_dir, base)
    df_feat.to_parquet(out_path)
    print(f"Saved features: {out_path} rows: {len(df_feat)}")
