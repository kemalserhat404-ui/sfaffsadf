# test_indicators.py
from data_fetcher import fetch_ohlcv
from utils.indicators import add_common_indicators
import pandas as pd

pair = 'BTC/USDT'
tf = '15m'
df = fetch_ohlcv(pair, tf, limit=200)
df2 = add_common_indicators(df)
pd.set_option('display.max_columns', 30)
print(df2.tail().to_string())
