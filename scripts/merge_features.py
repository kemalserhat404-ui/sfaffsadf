import pandas as pd
import glob
import os

features_dir = "features"
all_features_path = os.path.join(features_dir, "all_features.parquet")

files = glob.glob(os.path.join(features_dir, "*__labeled.parquet"))

all_features = []

for f in files:
    df = pd.read_parquet(f)
    all_features.append(df)

all_features = pd.concat(all_features, ignore_index=True)
all_features.to_parquet(all_features_path)
print(f"{all_features_path} başarıyla oluşturuldu. Toplam satır: {len(all_features)}")
