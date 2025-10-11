# features/feature_engineering.py
import os
import glob
import pandas as pd
import numpy as np

# Ayarlar — gerekirse değiştirebilirsin
HORIZON_BARS = 5          # kaç bar sonrasını hedefleyeceğiz (ör: 5 bar ilerisi)
TARGET_THRESHOLD = 0.0    # future_return > threshold => target=1

def engineer_features_df(df, horizon_bars=HORIZON_BARS, thr=TARGET_THRESHOLD):
    """
    Tek bir parquet DataFrame için:
    - future_return ve binary target ekler
    - sayısal sütunları z-score ile normalize eder
    - NaN satırları düşürür
    """
    df = df.copy().reset_index(drop=True)

    if 'close' not in df.columns:
        raise ValueError("DataFrame içinde 'close' sütunu yok — işlem atlanacak.")

    # future return ve target
    df['future_return'] = df['close'].shift(-horizon_bars) / df['close'] - 1
    df['target'] = (df['future_return'] > thr).astype(int)

    # normalize edilecek numerik sütunlar (target ve future_return haric)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ('target','future_return')]

    # standart sapma sıfırsa 1 ile değiştir (bölme hatası önlemi)
    stds = df[num_cols].std().replace({0:1})
    means = df[num_cols].mean()
    df[num_cols] = (df[num_cols] - means) / stds

    df = df.dropna().reset_index(drop=True)
    return df

def main():
    os.makedirs('features', exist_ok=True)
    files = sorted(glob.glob('data/processed/*.parquet'))
    if not files:
        print("data/processed içinde işlenecek .parquet dosyası yok.")
        return

    processed_count = 0
    for p in files:
        # atlama kuralları
        bn = os.path.basename(p).lower()
        if "signals" in bn or "unknown" in bn:
            print("Atlandı (signals/unknown):", p)
            continue

        print("İşleniyor:", p)
        try:
            df = pd.read_parquet(p)
            if len(df) < HORIZON_BARS + 3:
                print("  -> Yetersiz satır sayısı, atlanıyor.")
                continue
            out_df = engineer_features_df(df)
            out_name = os.path.basename(p).replace('.parquet','__features.parquet')
            out_path = os.path.join('features', out_name)
            out_df.to_parquet(out_path, index=False)
            print(f"  -> kaydedildi: {out_path} (satır: {len(out_df)})")
            processed_count += 1
        except Exception as e:
            print("  !! Hata:", str(e))

    print(f"\nTamamlandı. İşlenen dosya sayısı: {processed_count}")

if __name__ == "__main__":
    main()
