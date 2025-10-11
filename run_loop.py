# run_loop.py (model-backed run loop)
import time, os
from data_fetcher import fetch_ohlcv
from strategies.ema_rsi_strategy import ema_rsi_strategy
from config import pairs, timeframe, confidence_threshold
from save_signals import save_signals
from utils.indicators import add_common_indicators

# dynamic import of predictor
MODEL_TEMPLATE = "models/{pair}_{tf}_xgb.joblib"  # pair like BTC_USDT (no slash) and tf like 1h

def model_path_for(pair, tf):
    fn = MODEL_TEMPLATE.format(pair=pair.replace("/","_").upper(), tf=tf)
    return fn if os.path.exists(fn) else None

def main():
    print("Trading bot başlatılıyor...")
    print(f"Takip edilecek varlıklar: {pairs}")
    print(f"Zaman dilimi: {timeframe}")
    print(f"Güven eşiği: {confidence_threshold}")
    print("Run loop starting. Press Ctrl+C to stop.")

    # preload model availability
    model_map = {pair: model_path_for(pair, timeframe) for pair in pairs}
    for pair, mp in model_map.items():
        print(f"Model for {pair}: {mp or 'Yok (fallback EMA/RSI)'}")

    try:
        while True:
            for pair in pairs:
                # fetch last N bars (OHLCV)
                df = fetch_ohlcv(pair, timeframe, limit=500)  # df: timestamp, open, high, low, close, volume

                model_file = model_map.get(pair)
                if model_file:
                    try:
                        from models.predictor import predict_latest
                        pred, conf = predict_latest(model_file, df)
                        action = "BUY" if pred == 1 else "SELL"
                    except Exception as e:
                        print("Model prediction failed:", e)
                        action, conf = ema_rsi_strategy(pair, timeframe, df)
                else:
                    action, conf = ema_rsi_strategy(pair, timeframe, df)

                if conf >= confidence_threshold:
                    print(f"SINYAL -> {pair} {action} (Güven: {conf:.2f})")
                else:
                    print(f"NoSignal -> {pair} (Güven: {conf:.2f})")

                # kaydet
                try:
                    save_signals(pair, action, conf)
                except Exception as e:
                    print("save_signals hatası:", e)
            time.sleep(60)
    except KeyboardInterrupt:
        print("Run loop durduruldu (Ctrl+C).")

if __name__ == "__main__":
    main()
