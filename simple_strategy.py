def simple_strategy(pair: str, timeframe: str):
    import random
    confidence = random.random()  # 0-1 arasında rastgele değer
    if confidence > 0.8:
        return "BUY", confidence
    elif confidence < 0.2:
        return "SELL", confidence
    else:
        return "HOLD", confidence
