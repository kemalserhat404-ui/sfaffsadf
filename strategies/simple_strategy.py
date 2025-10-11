def simple_strategy(pair: str, timeframe: str):
    """
    Basit örnek strateji:
    - 0.8 üzerinde confidence ise BUY
    - 0.8 altında ise sinyal yok
    """
    import random
    confidence = random.random()  # 0-1 arasında rastgele değer

    if confidence >= 0.8:
        action = "BUY"
    else:
        action = None

    return action, confidence
