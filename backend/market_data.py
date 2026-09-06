"""Turning raw BloFin API responses into clean, typed data this bot can use.

Two kinds of things live here:
  - small Decimal-math helpers (decimal_from, quantize_down, decimal_to_str)
  - functions that fetch from / parse BloFin's REST and websocket payloads
    into a consistent candle/price shape, regardless of which endpoint or
    message format they came from.
"""

from decimal import ROUND_DOWN, Decimal
from typing import Any, Dict, List, Optional

from blofin.rest_market import MarketAPI

from config import CANDLE_LIMIT, INST_ID, BAR


def decimal_from(value: Any, fallback: str = "0") -> Decimal:
    if value is None or value == "":
        value = fallback
    return Decimal(str(value))


def quantize_down(value: Decimal, increment: Decimal) -> Decimal:
    if increment <= 0:
        return value
    units = (value / increment).to_integral_value(rounding=ROUND_DOWN)
    return units * increment


def decimal_to_str(value: Decimal) -> str:
    return format(value.normalize(), "f")


def candle_to_record(candle: Any) -> Optional[Dict[str, Any]]:
    """Normalize a candle from either the REST API (dict) or websocket
    (list) shape into {ts, open, high, low, close} with Decimal prices."""
    if isinstance(candle, dict):
        timestamp = candle.get("ts") or candle.get("time")
        open_price = candle.get("open")
        high = candle.get("high")
        low = candle.get("low")
        close = candle.get("close")
    elif isinstance(candle, list) and len(candle) >= 5:
        timestamp = candle[0]
        open_price = candle[1]
        high = candle[2]
        low = candle[3]
        close = candle[4]
    else:
        return None

    try:
        return {
            "ts": int(timestamp),
            "open": decimal_from(open_price),
            "high": decimal_from(high),
            "low": decimal_from(low),
            "close": decimal_from(close),
        }
    except Exception:
        return None


def ticker_to_price(ticker: Any) -> Optional[Decimal]:
    if not isinstance(ticker, dict):
        return None

    for key in ("last", "lastPrice", "markPrice", "price"):
        price = decimal_from(ticker.get(key))
        if price > 0:
            return price

    return None


def fetch_rest_candles(market_api: MarketAPI) -> List[Dict[str, Any]]:
    response = market_api.getCandlesticks(instId=INST_ID, bar=BAR, limit=CANDLE_LIMIT)
    return [
        record
        for candle in response.get("data", [])
        if (record := candle_to_record(candle)) is not None
    ]


def fetch_tick_size(market_api: MarketAPI) -> Decimal:
    response = market_api.getInstruments(instId=INST_ID)
    data = response.get("data")
    instrument = data[0] if isinstance(data, list) and data else {}
    return decimal_from(instrument.get("tickSize"), "0.1")


def fetch_rest_ticker_price(market_api: MarketAPI) -> Optional[Decimal]:
    response = market_api.getTickers(instId=INST_ID)
    data = response.get("data")
    ticker = data[0] if isinstance(data, list) and data else {}
    return ticker_to_price(ticker)
