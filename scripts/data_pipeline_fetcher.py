# scripts/data_pipeline_fetcher.py
import ccxt, time, os
import pandas as pd
from datetime import datetime, timedelta

EXCHANGE = ccxt.binance({'enableRateLimit': True})

def ms(ts): return int(ts.timestamp() * 1000)

def fetch_ohlcv_pair(pair="BTC/USDT", timeframe="1m", since_ts=None, limit=1000, out_root="data/raw"):
    out_dir = os.path.join(out_root, pair.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    since_ms = since_ts
    chunk_idx = 0
    while True:
        try:
            chunk = EXCHANGE.fetch_ohlcv(pair, timeframe=timeframe, since=since_ms, limit=limit)
        except Exception as e:
            print("Fetch error:", e)
            time.sleep(2)
            continue
        if not chunk:
            break
        df = pd.DataFrame(chunk, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        iso = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = f"{pair.replace('/','_')}__{timeframe}__{iso}__chunk{chunk_idx}.csv"
        df.to_csv(os.path.join(out_dir, fname), index=False)
        print("Saved:", fname, "rows:", len(df))
        last_ts_ms = int(df['timestamp'].astype('int64').iloc[-1] // 10**6)
        since_ms = last_ts_ms + 1
        chunk_idx += 1
        # if chunk smaller than limit assume finished
        if len(chunk) < limit:
            break
        time.sleep(EXCHANGE.rateLimit / 1000.0)
    return

if __name__ == "__main__":
    # örnek: son 7 gün 1m:
    since = datetime.utcnow() - timedelta(days=7)
    fetch_ohlcv_pair("BTC/USDT", "1m", since_ts=ms(since))
