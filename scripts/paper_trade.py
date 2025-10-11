from binance.client import Client
import os

api_key = os.getenv("TESTNET_API_KEY")
api_secret = os.getenv("TESTNET_API_SECRET")
client = Client(api_key, api_secret, testnet=True)

# Örnek market buy
order = client.order_market_buy(symbol='BTCUSDT', quantity=0.001)
print(order)
