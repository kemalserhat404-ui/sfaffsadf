import pandas as pd
import numpy as np

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def macd_calc(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1 * delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

def vwap(df, window=14):
    tp = (df['high'] + df['low'] + df['close']) / 3
    pv = tp * df['volume']
    vw = pv.rolling(window=window, min_periods=1).sum() / df['volume'].rolling(window=window, min_periods=1).sum()
    return vw

def obv(df):
    close = df['close']
    vol = df['volume']
    direction = np.sign(close.diff().fillna(0))
    obv_series = (direction * vol).fillna(0).cumsum()
    return obv_series

def compute_indicators(df):
    """
    Input: df with columns ['timestamp','open','high','low','close','volume']
    Returns: df copy with indicators added
    """
    df = df.copy().reset_index(drop=True)

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column: {col}")

    # Basit teknik göstergeler
    df['return1'] = df['close'].pct_change().fillna(0)
    df['ema12'] = ema(df['close'], 12)
    df['ema26'] = ema(df['close'], 26)

    macd, macd_signal, macd_hist = macd_calc(df['close'], 12, 26, 9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist

    df['rsi14'] = rsi(df['close'], period=14)
    df['atr14'] = atr(df, period=14)
    df['vwap14'] = vwap(df, window=14)
    df['obv'] = obv(df)

    if 'bid' in df.columns and 'ask' in df.columns:
        df['spread'] = (df['ask'] - df['bid']) / ((df['ask'] + df['bid']) / 2)
    else:
        df['spread'] = 0.0

    # Pandas 3.0 uyumlu doldurma
    df = df.ffill().bfill().fillna(0)
    return df
