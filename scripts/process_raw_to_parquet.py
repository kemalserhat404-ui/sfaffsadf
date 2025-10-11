# scripts/process_raw_to_parquet.py
import pandas as pd, glob, os
from pathlib import Path

raw_root = "data/raw"
out_root = "data/processed"
os.makedirs(out_root, exist_ok=True)

pairs = [p for p in Path(raw_root).iterdir() if p.is_dir()]
for p in pairs:
    pairname = p.name  # e.g. BTC_USDT
    files = sorted(glob.glob(str(p/"*.csv")))
    if not files:
        continue
    # collect per timeframe by filename pattern __{tf}__
    frames_by_tf = {}
    for f in files:
        fname = os.path.basename(f)
        # name format: <PAIR>__{tf}__{iso}__chunkX.csv
        parts = fname.split("__")
        if len(parts) < 2:
            continue
        tf = parts[1]
        df = pd.read_csv(f, parse_dates=['timestamp'])
        df = df[['timestamp','open','high','low','close','volume']]
        df = df.drop_duplicates('timestamp').sort_values('timestamp')
        frames_by_tf.setdefault(tf, []).append(df)

    for tf, dfs in frames_by_tf.items():
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
        # set timezone-naive to UTC
        combined['timestamp'] = pd.to_datetime(combined['timestamp'], utc=True)
        combined = combined.set_index('timestamp')
        # ensure continuous index at that timeframe:
        freq_map = {"1m":"1min","5m":"5min","15m":"15min","1h":"1H","4h":"4H","1d":"1D"}
        freq = freq_map.get(tf, None)
        if freq:
            combined = combined.asfreq(freq).ffill().bfill()
        combined = combined.reset_index()
        out_file = os.path.join(out_root, f"{pairname}__{tf}.parquet")
        combined.to_parquet(out_file)
        print("Written:", out_file, "rows:", len(combined))
