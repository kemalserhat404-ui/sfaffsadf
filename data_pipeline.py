# data_pipeline.py
# Basit, güvenilir OHLCV toplayıcı (prototip). fetch_ohlcv fonksiyonunu kullanır.
import os
import time
from datetime import datetime
import pandas as pd

# projede zaten varsa kendi fetch_ohlcv kullan
try:
    from data_fetcher import fetch_ohlcv
except Exception as e:
    raise RuntimeError("data_fetcher.fetch_ohlcv bulunamadı. Dosya mevcut mu?") from e

# config'den pairs al (eğer yoksa default kullan)
try:
    from config import pairs, timeframe as default_timeframe, limit as default_limit
    # config.timeframe tek ise timeframes listine çevirebiliriz
except Exception:
    pairs = ['BTC/USDT', 'ETH/USDT']
    default_timeframe = '1h'
    default_limit = 500

# Hangi timeframe'leri toplayacağız — ihtiyaca göre düzenle
timeframes = ['1m', '5m', '15m', '1h']  # prototip: çeşitli TF'ler
limit = default_limit if isinstance(default_limit, int) else 500

OUT_DIR = 'data'  # ham veriler burada depolanacak
os.makedirs(OUT_DIR, exist_ok=True)

def safe_filename(pair):
    return pair.replace('/', '_')

def save_ohlcv_csv(pair, timeframe, df):
    pair_dir = os.path.join(OUT_DIR, safe_filename(pair))
    os.makedirs(pair_dir, exist_ok=True)
    fname = f"{safe_filename(pair)}__{timeframe}__{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    path = os.path.join(pair_dir, fname)
    df.to_csv(path, index=False)
    return path

def latest_csv_exists(pair, timeframe):
    pair_dir = os.path.join(OUT_DIR, safe_filename(pair))
    if not os.path.isdir(pair_dir):
        return False
    # varsa aynı timeframe için son dosya olup olmadığını kontrol et
    files = [f for f in os.listdir(pair_dir) if f.endswith('.csv') and f"__{timeframe}__" in f]
    return len(files) > 0

def run_collection(pairs_list, timeframes_list, limit_per_fetch=500, sleep_between_calls=0.6):
    print(f"Başlatıldı: {len(pairs_list)} pair x {len(timeframes_list)} timeframes. Limit={limit_per_fetch}")
    for pair in pairs_list:
        for tf in timeframes_list:
            try:
                print(f"[{datetime.utcnow().isoformat()}] {pair} - {tf} çekiliyor...")
                df = fetch_ohlcv(pair, tf, limit=limit_per_fetch)
                if df is None or df.empty:
                    print(f"  !! Veri yok: {pair} {tf}")
                    time.sleep(sleep_between_calls)
                    continue
                # normalize timestamp column varsa, yoksa dene
                if 'timestamp' not in df.columns:
                    # bazı fetch fonksiyonları 'datetime' veya index kullanabilir
                    if df.index.name is not None:
                        df = df.reset_index()
                        df.rename(columns={df.columns[0]:'timestamp'}, inplace=True)
                    else:
                        # fallback: create sequential timestamps (not ideal)
                        df['timestamp'] = pd.NaT
                # sakla
                path = save_ohlcv_csv(pair, tf, df)
                print(f"  Kaydedildi: {path} (satır: {len(df)})")
                # rate limit koruması
                time.sleep(sleep_between_calls)
            except Exception as e:
                print(f"  Hata {pair} {tf}: {e}")
                # hata alırsak kısa bekle
                time.sleep(2.0)

if __name__ == "__main__":
    run_collection(pairs, timeframes, limit_per_fetch=limit, sleep_between_calls=0.6)
    print("Tamamlandı. CSV'ler data/ içinde.")
