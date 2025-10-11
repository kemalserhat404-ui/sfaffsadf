import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    # Veri yükle
    print("Loading:", args.file)
    df = pd.read_parquet(args.file)

    # Features ve label
    X = df.drop(columns=['label','timestamp'])
    y = df['label']

    # Sample weight hesapla (sınıf dengesizliğini çözmek için)
    sample_weight = compute_sample_weight(class_weight='balanced', y=y)

    # Modeli oluştur
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        n_estimators=1000,
        max_depth=3,
        learning_rate=0.05,
        eval_metric='mlogloss',
        use_label_encoder=False
    )

    # Eğit
    print("Training model with balanced class weights...")
    model.fit(X, y, sample_weight=sample_weight)

    # Kaydet
    print("Saving model:", args.out)
    joblib.dump(model, args.out)
    print("Done.")

if __name__ == '__main__':
    main()
