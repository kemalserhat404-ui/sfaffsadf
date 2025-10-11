from strategies.simple_strategy import simple_strategy
from utils.logger import setup_logger

# Ayarlar
PAIRS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
CONFIDENCE_THRESHOLD = 0.8

print("Trading bot başlatılıyor...")
print(f"Takip edilecek varlıklar: {PAIRS}")
print(f"Zaman dilimi: {TIMEFRAME}")
print(f"Güven eşiği: {CONFIDENCE_THRESHOLD}")

for pair in PAIRS:
    logger = setup_logger(pair)
    action, confidence = simple_strategy(pair, TIMEFRAME)

    if confidence >= CONFIDENCE_THRESHOLD:
        print(f"{pair} için sinyal: {action} (Güven: {confidence:.2f})")
        logger.info(f"Sinyal: {action}, Güven: {confidence:.2f}")
    else:
        print(f"{pair} için güven düşük ({confidence:.2f}), sinyal yok.")
        logger.info(f"Güven düşük ({confidence:.2f}), sinyal yok.")
