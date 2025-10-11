# adapters/binance_testnet.py
"""
Basit, güvenli CCXT tabanlı Binance testnet adapter'i.
Kullanım:
  from dotenv import load_dotenv
  load_dotenv()
  from adapters.binance_testnet import BinanceTestnetAdapter
  a = BinanceTestnetAdapter()           # .env'den anahtarları alır
  a.get_ticker_price()                  # güncel price
  a.place_order("BUY", usd_amount=10)   # USD bazlı küçük market buy
  a.log_fills("trades/log.csv")
"""

from dotenv import load_dotenv
load_dotenv()

import os
import time
import csv
import pathlib
import logging
import json

import ccxt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceTestnetAdapter:
    def __init__(self, api_key=None, api_secret=None, symbol="BTC/USDT", testnet=True):
        api_key = api_key or os.getenv("BINANCE_API_KEY")
        api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            raise ValueError("BINANCE_API_KEY / BINANCE_API_SECRET environment variables required (check .env)")

        self.symbol = symbol
        # CCXT exchange nesnesi
        self.exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })
        # testnet sandbox modu
        if testnet:
            self.exchange.set_sandbox_mode(True)

        # load markets (ilk kullanımda)
        self.exchange.load_markets()
        self.fills = []  # local memory'de tutulan fill'ler (loglanacak)

    def get_ticker(self, symbol=None):
        symbol = symbol or self.symbol
        return self.exchange.fetch_ticker(symbol)

    def get_ticker_price(self, symbol=None):
        t = self.get_ticker(symbol)
        # ccxt fetch_ticker returns dict with 'last'
        return float(t.get("last") or t.get("close") or t.get("info", {}).get("price"))

    def get_balance(self):
        return self.exchange.fetch_balance()

    def _amount_precision(self, amount):
        """Round amount to market's allowed precision."""
        market = self.exchange.market(self.symbol)
        precision = market.get("precision", {}).get("amount")
        if precision is None:
            return amount
        # ccxt provides helper
        try:
            return float(self.exchange.amount_to_precision(self.symbol, amount))
        except Exception:
            # fallback
            return round(amount, precision)

    def _cost_precision(self, cost):
        market = self.exchange.market(self.symbol)
        precision = market.get("precision", {}).get("price")
        if precision is None:
            return cost
        try:
            return float(self.exchange.price_to_precision(self.symbol, cost))
        except Exception:
            return round(cost, precision)

    def place_order(self, side, usd_amount=None, amount=None, price=None, order_type="market", params=None):
        """
        side: "BUY" or "SELL"
        usd_amount: float (örnek: 10 => 10 USDT) -> amount hesaplanır
        amount: float (base asset miktarı, örn 0.001 BTC)
        price: limit emir için fiyat
        order_type: "market" veya "limit"
        """
        side = side.lower()
        if usd_amount is not None and amount is None:
            market_price = self.get_ticker_price()
            if market_price is None:
                raise RuntimeError("Market price unavailable to convert usd_amount -> amount")
            amount = usd_amount / market_price

        if amount is None:
            raise ValueError("Either `amount` (base asset) or `usd_amount` must be provided")

        # rounds
        amount = self._amount_precision(amount)
        if price is not None:
            price = self._cost_precision(price)

        try:
            order = self.exchange.create_order(symbol=self.symbol, type=order_type, side=side, amount=amount, price=price, params=params or {})
            # normalize minimal fields for logging
            recorded = {
                "id": order.get("id"),
                "timestamp": order.get("timestamp"),
                "datetime": order.get("datetime"),
                "symbol": order.get("symbol"),
                "type": order.get("type"),
                "side": order.get("side"),
                "price": order.get("price"),
                "amount": order.get("amount"),
                "filled": order.get("filled"),
                "remaining": order.get("remaining"),
                "status": order.get("status"),
                "info": order.get("info"),
            }
            self.fills.append(recorded)
            logger.info("Order placed: %s %s %s (amount=%s)", recorded["side"], recorded["symbol"], recorded["id"], recorded["amount"])
            return recorded
        except ccxt.BaseError as e:
            logger.exception("Order failed: %s", e)
            raise

    def fetch_open_orders(self):
        return self.exchange.fetch_open_orders(symbols=[self.symbol])

    def fetch_order(self, id):
        return self.exchange.fetch_order(id, symbol=self.symbol)

    def cancel_order(self, id):
        return self.exchange.cancel_order(id, symbol=self.symbol)

    def log_fills(self, path="trades/log.csv"):
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # collect union of keys
        keys = set()
        for f in self.fills:
            keys.update(f.keys())
        keys = list(keys)
        write_header = not p.exists()
        with p.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            if write_header:
                writer.writeheader()
            for f in self.fills:
                # serialize nested objects to json string for CSV
                row = {k: (json.dumps(f[k], default=str, ensure_ascii=False) if isinstance(f.get(k), (dict, list)) else f.get(k)) for k in keys}
                writer.writerow(row)
        logger.info("Logged %d fills to %s", len(self.fills), str(p))
        # clear in-memory fills after logging
        self.fills = []
