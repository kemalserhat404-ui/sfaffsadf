import ccxt
import pandas as pd
import time
from datetime import datetime, timezone

def fetch_ohlcv(symbol="BTC/USDT", interval="1h", limit=500, max_retries=3, exchange=None):
    """
    Binance OHLCV çekme fonksiyonu (güçlendirilmiş)
    - Retry/backoff sistemi
    - UTC normalize edilmiş timestamp
    - Opsiyonel ccxt exchange objesi (yeniden oluşturmayı önler)
    """
    if exchange is None:
        exchange = ccxt.binance({
            "enableRateLimit": True,
            "rateLimit": 1200,
        })

    for attempt in range(max_retries):
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            return df
        except ccxt.NetworkError as e:
            print(f"[Retry {attempt+1}/{max_retries}] NetworkError: {e}")
            time.sleep(2 ** attempt)
        except ccxt.ExchangeError as e:
            print(f"ExchangeError: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break

    raise RuntimeError(f"Failed to fetch OHLCV for {symbol} after {max_retries} retries.")
