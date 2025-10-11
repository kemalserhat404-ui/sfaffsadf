import joblib
import pandas as pd
from risk_engine import calculate_position_size
from paper_trade import client

# Feature yükle
df = pd.read_parquet('features/all_features.parquet')

# Ensemble modeli yükle
ensemble = joblib.load('models/ensemble_model.joblib')

# Tahmin
X = df.drop(columns=['timestamp', 'signal'])  # sadece feature sütunları
preds = ensemble.predict_proba(X)[:, 1]

# Risk ve pozisyon
account_balance = 1000  # örnek bakiye
risk_per_trade = 0.01
stop_loss_pips = 50

position_size = calculate_position_size(account_balance, risk_per_trade, stop_loss_pips)
print(f"Önerilen pozisyon boyutu: {position_size}")

# Paper-trade örnek emir
# client.order_market_buy(symbol='BTCUSDT', quantity=position_size)
