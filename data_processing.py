import os
import glob
import pandas as pd
from utils.indicators import compute_indicators

def process_file(path):
    df = pd.read_csv(path)
    df_feat = compute_indicators(df)
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(path).replace(".csv", ".parquet")
    out_path = os.path.join(out_dir, base)
    df_feat.to_parquet(out_path, index=False)
    print(f"  -> kaydedildi: {out_path} satır: {len(df_feat)}")
    first_ts = df_feat['timestamp'].iloc[0] if len(df_feat)>0 else None
    return {
        "source": path,
        "output": out_path,
        "rows": len(df_feat),
        "first_ts": first_ts
    }

def main():
    csv_files = glob.glob("data/**/*.csv", recursive=True)
    print(f"Bulunan CSV sayısı: {len(csv_files)}")

    manifest = []
    for p in csv_files:
        if "signals" in str(p).lower():
            continue  # sinyal dosyalarını atla
        info = process_file(p)
        manifest.append(info)

    # manifest dosyası oluştur
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv("data/manifest.csv", index=False)
    print("Manifest güncellendi: data/manifest.csv")

if __name__ == "__main__":
    main()
