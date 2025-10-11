import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# Mevcut XGB modelini yükle
xgb_model = joblib.load('models/btc_usdt_all_xgb.joblib')

# RandomForest modelini oluştur (örnek parametreler)
rf_model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
# Eğer RF modelini ayrı eğitmek istersen: rf_model.fit(X_train, y_train)

# Ensemble
ensemble = VotingClassifier(estimators=[('xgb', xgb_model), ('rf', rf_model)], voting='soft')
# Eğer fit gerekiyorsa: ensemble.fit(X_train, y_train)

joblib.dump(ensemble, 'models/ensemble_model.joblib')
print("Ensemble modeli kaydedildi.")
