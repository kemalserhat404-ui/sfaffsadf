import pandas as pd
import numpy as np

def ema_rsi_strategy(pair, timeframe, historical_data=None):
    """
    Basit EMA + RSI stratejisi.
    historical_data: Backtest için kullanılan DataFrame, None ise gerçek zamanlı veri çekilecek
    """
    if historical_data is None:
        from data_fetcher import fetch_ohlcv
        df = fetch_ohlcv(pair, timeframe, limit=500)
    else:
        df = historical_data.copy()

    # EMA
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    rs = roll_up / roll_down
    df['rsi'] = 100 - (100 / (1 + rs))

    # Sinyal mantığı
    last = df.iloc[-1]
    if last['ema12'] > last['ema26'] and last['rsi'] < 70:
        signal = 'BUY'
        confidence = np.clip((70 - last['rsi']) / 70, 0, 1)
    elif last['ema12'] < last['ema26'] and last['rsi'] > 30:
        signal = 'SELL'
        confidence = np.clip((last['rsi'] - 30) / 70, 0, 1)
    else:
        signal = 'NoSignal'
        confidence = 0.5

    return signal, confidence
