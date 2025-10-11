# fix_features.py
import pandas as pd
import joblib, glob, os

model_file = 'models/btc_offline_lgb_recomputed_optuna.joblib'
features_path_pattern = 'features/BTCUSDT__*__features_recomputed_auto.parquet'

model = joblib.load(model_file)
feature_order = model.feature_name_ if hasattr(model, 'feature_name_') else model.feature_name()

for ffile in glob.glob(features_path_pattern):
    df = pd.read_parquet(ffile)
    missing = set(feature_order) - set(df.columns)
    for m in missing:
        df[m] = 0.0  # eksik feature’ı 0 ile doldur
    df = df[feature_order]  # sıralamayı model ile aynı yap
    df.to_parquet(ffile, index=False)
    print(f"Fixed features: {ffile}, columns now: {len(df.columns)}")
