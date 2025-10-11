import pandas as pd
import numpy as np

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Lag features
    for lag in [1,2,3,6,12]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'return_{lag}'] = df['close'].pct_change(lag)
    df['return1'] = df['close'].pct_change(1)
    
    # Rolling / volatility
    df['rolling_mean_14'] = df['close'].rolling(14).mean()
    df['rolling_std_14'] = df['close'].rolling(14).std()
    df['volatility'] = df['return1'].rolling(14).std()
    df['momentum14'] = df['close'] - df['close'].shift(14)
    df['price_range'] = df['high'] - df['low']
    
    # Volume features
    df['vol_mean_14'] = df['volume'].rolling(14).mean()
    df['vol_spike'] = df['volume'] / df['vol_mean_14'] - 1
    
    # EMA / MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['ema_ratio'] = df['ema12'] / df['ema26']
    
    # Price range ratio
    df['price_high_low'] = df['high'] - df['low']
    
    # ATR
    df['atr14_simple'] = df['price_range'].rolling(14).mean()
    
    df = df.fillna(0)
    return df
