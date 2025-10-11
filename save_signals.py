import os

def save_signals(pair, action, confidence):
    """
    Gelen sinyalleri logs klasöründe ilgili varlık dosyasına kaydeder.
    """
    # logs klasörünün varlığını kontrol et, yoksa oluştur
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # dosya adı BTC_USDT.log gibi olacak, / işaretlerini _ ile değiştir
    filename = f"logs/{pair.replace('/', '_')}.log"
    
    with open(filename, "a") as f:
        if action:
            f.write(f"SINYAL -> {pair} {action} confidence={confidence:.2f}\n")
        else:
            f.write(f"NoSignal -> {pair} confidence={confidence:.2f}\n")
