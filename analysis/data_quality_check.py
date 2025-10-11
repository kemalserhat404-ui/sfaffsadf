import pandas as pd
import os
import numpy as np

def analyze_parquet(file_path):
    df = pd.read_parquet(file_path)
    stats = {
        "file": os.path.basename(file_path),
        "rows": len(df),
        "cols": len(df.columns),
        "missing_ratio": df.isna().mean().mean(),
        "rsi_mean": df["rsi14"].mean() if "rsi14" in df.columns else np.nan,
        "macd_mean": df["macd"].mean() if "macd" in df.columns else np.nan,
        "return_std": df["return1"].std() if "return1" in df.columns else np.nan,
    }
    return stats

def main():
    base_dir = "data/processed"
    files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(".parquet")]
    results = [analyze_parquet(f) for f in files]
    summary = pd.DataFrame(results)
    print(summary)
    summary.to_csv("data/quality_summary.csv", index=False)
    print("\nKayıt: data/quality_summary.csv oluşturuldu.")

if __name__ == "__main__":
    main()
