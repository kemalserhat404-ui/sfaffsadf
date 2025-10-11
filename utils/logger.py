import logging
import os

def setup_logger(pair: str):
    # logs klasörü yoksa oluştur
    if not os.path.exists("logs"):
        os.makedirs("logs")

    logger = logging.getLogger(pair)
    logger.setLevel(logging.INFO)

    # Dosya handler
    fh = logging.FileHandler(f"logs/{pair.replace('/', '_')}.log")
    fh.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Logger’a ekle
    if not logger.handlers:
        logger.addHandler(fh)

    return logger

