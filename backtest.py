import pandas as pd
import os
from data_fetcher import fetch_ohlcv
from strategies.ema_rsi_strategy import ema_rsi_strategy

# Ayarlar
pairs = ['BTC/USDT', 'ETH/USDT']  # İstediğin kadar varlık ekleyebilirsin
timeframe = '1h'
limit = 500

# Backtests klasörü oluştur (varsa hata vermez)
os.makedirs('backtests', exist_ok=True)

# Her varlık için backtest
for pair in pairs:
    # Veriyi çek
    df = fetch_ohlcv(pair, timeframe, limit=limit)

    results = []

    # Backtest döngüsü
    for i in range(len(df)):
        data_slice = df.iloc[:i+1]  # O ana kadar olan veri
        signal, confidence = ema_rsi_strategy(pair, timeframe, data_slice)
        results.append({
            'timestamp': df.iloc[i]['timestamp'],
            'signal': signal,
            'confidence': confidence
        })

    results_df = pd.DataFrame(results)

    # CSV kaydet
    results_df.to_csv(f'backtests/backtest_{pair.replace("/", "_")}_{timeframe}.csv', index=False)

    print(f"Backtest tamamlandı: {pair}. Dosya 'backtests/backtest_{pair.replace('/', '_')}_{timeframe}.csv' olarak kaydedildi.")
    print(results_df.tail())
