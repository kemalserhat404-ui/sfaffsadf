from dotenv import load_dotenv
load_dotenv()

from adapters.binance_testnet import BinanceTestnetAdapter

adapter = BinanceTestnetAdapter()          # .env'den anahtarları alır
print("Symbol:", adapter.symbol)
print("BTC price:", adapter.get_ticker_price())
bal = adapter.get_balance()
print("Balance total (snapshot):")
# yazdırılabilir şekilde basit göster
print({k: v for k, v in bal.get("total", {}).items() if k in ("USDT","BTC")})
# küçük bir test order: 10 USDT ile BUY (testnet ise gerçek para kullanılmaz)
try:
    order = adapter.place_order("BUY", usd_amount=10)
    print("Order placed:", order.get("id"))
except Exception as e:
    print("Order error:", e)
adapter.log_fills("trades/log.csv")
print("Test completed.")
