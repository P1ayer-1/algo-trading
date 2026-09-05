import asyncio
import argparse
import builtins
import html
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

SDK_SRC = Path(__file__).resolve().parent / "blofin-sdk-python" / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

print = partial(builtins.print, flush=True)


def load_local_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


try:
    from blofin.client import DemoClient
    from blofin.exceptions import BlofinAPIException
    from blofin.rest_market import MarketAPI
    from blofin.rest_trading import TradingAPI
    from blofin.websocket_client import BlofinWsPrivateClient, BlofinWsPublicClient
except ModuleNotFoundError as exc:
    missing = exc.name or "a dependency"
    raise SystemExit(
        f"Missing Python package '{missing}'. Install SDK dependencies with: "
        "python -m pip install -r blofin-sdk-python\\requirements.txt"
    ) from exc

load_local_env(Path(__file__).resolve().parent / ".env")

INST_ID = os.getenv("BLOFIN_INST_ID", "BTC-USDT")
PRICE_OFFSET = Decimal(os.getenv("BLOFIN_PRICE_OFFSET", "2000"))
MARGIN_MODE = os.getenv("BLOFIN_MARGIN_MODE", "cross")
POSITION_SIDE = os.getenv("BLOFIN_POSITION_SIDE", "net")
ORDER_SIDE = "buy"
ORDER_TYPE = "limit"
WATCH_SECONDS = int(os.getenv("BLOFIN_WATCH_SECONDS", "300"))
REST_POLL_SECONDS = int(os.getenv("BLOFIN_REST_POLL_SECONDS", "10"))
SUPPORT_BAR = os.getenv("BLOFIN_SUPPORT_BAR", "15m")
SUPPORT_CANDLE_LIMIT = os.getenv("BLOFIN_SUPPORT_CANDLE_LIMIT", "500")
SUPPORT_LOOKBACK = int(os.getenv("BLOFIN_SUPPORT_LOOKBACK", "3"))
SUPPORT_CLUSTER_PCT = Decimal(os.getenv("BLOFIN_SUPPORT_CLUSTER_PCT", "0.001"))
SUPPORT_DEFAULT_BELOW = Decimal(os.getenv("BLOFIN_SUPPORT_DEFAULT_BELOW", "5000"))
SUPPORT_LEVEL_COUNT = int(os.getenv("BLOFIN_SUPPORT_LEVEL_COUNT", "8"))
RESISTANCE_DEFAULT_ABOVE = Decimal(os.getenv("BLOFIN_RESISTANCE_DEFAULT_ABOVE", "5000"))
RESISTANCE_LEVEL_COUNT = int(os.getenv("BLOFIN_RESISTANCE_LEVEL_COUNT", str(SUPPORT_LEVEL_COUNT)))
CHART_PATH = Path(os.getenv("BLOFIN_CHART_PATH", "support-resistance-chart.html"))

FINAL_STATES = {"filled", "canceled", "partially_canceled", "order_failed"}


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def first_data(response: Dict[str, Any]) -> Dict[str, Any]:
    data = response.get("data")
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    return {}


def decimal_from(value: Any, fallback: str = "0") -> Decimal:
    if value is None or value == "":
        value = fallback
    return Decimal(str(value))


def quantize_down(value: Decimal, increment: Decimal) -> Decimal:
    if increment <= 0:
        return value
    units = (value / increment).to_integral_value(rounding=ROUND_DOWN)
    return units * increment


def decimal_to_api(value: Decimal) -> str:
    normalized = value.normalize()
    return format(normalized, "f")


def extract_order_id(place_order_response: Dict[str, Any]) -> Optional[str]:
    data = place_order_response.get("data")
    if isinstance(data, list) and data:
        return data[0].get("orderId")
    if isinstance(data, dict):
        return data.get("orderId")
    return None


def find_order_by_ids(
    response: Dict[str, Any],
    *,
    order_id: Optional[str],
    client_order_id: str,
) -> Optional[Dict[str, Any]]:
    data = response.get("data")
    if not isinstance(data, list):
        return None

    for order in data:
        if order_id and order.get("orderId") == order_id:
            return order
        if order.get("clientOrderId") == client_order_id:
            return order
    return None


def summarize_order(order: Dict[str, Any]) -> str:
    fields = {
        "orderId": order.get("orderId"),
        "clientOrderId": order.get("clientOrderId"),
        "state": order.get("state"),
        "price": order.get("price"),
        "size": order.get("size"),
        "filledSize": order.get("filledSize"),
        "averagePrice": order.get("averagePrice"),
    }
    return ", ".join(f"{key}={value}" for key, value in fields.items() if value not in (None, ""))


def candle_to_record(candle: Any) -> Optional[Dict[str, Any]]:
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


def find_support_levels(
    candles: List[Dict[str, Any]],
    *,
    low_range: Decimal,
    high_range: Decimal,
    tick_size: Decimal,
) -> List[Dict[str, Any]]:
    if len(candles) < (SUPPORT_LOOKBACK * 2) + 1:
        return []

    sorted_candles = sorted(candles, key=lambda item: item["ts"])
    swing_lows: List[Dict[str, Any]] = []

    for index in range(SUPPORT_LOOKBACK, len(sorted_candles) - SUPPORT_LOOKBACK):
        candle = sorted_candles[index]
        low = candle["low"]
        if low < low_range or low > high_range:
            continue

        neighbors = (
            sorted_candles[index - SUPPORT_LOOKBACK:index]
            + sorted_candles[index + 1:index + SUPPORT_LOOKBACK + 1]
        )
        if all(low <= neighbor["low"] for neighbor in neighbors):
            swing_lows.append(candle)

    clusters: List[Dict[str, Any]] = []
    min_cluster_width = tick_size * Decimal("10") if tick_size > 0 else Decimal("0")

    for candle in sorted(swing_lows, key=lambda item: item["low"]):
        price = candle["low"]
        tolerance = max(price * SUPPORT_CLUSTER_PCT, min_cluster_width)
        target = None

        for cluster in clusters:
            if abs(price - cluster["level"]) <= tolerance:
                target = cluster
                break

        if target is None:
            clusters.append({
                "level": price,
                "touches": 1,
                "last_ts": candle["ts"],
                "prices": [price],
            })
            continue

        target["prices"].append(price)
        target["touches"] += 1
        target["last_ts"] = max(target["last_ts"], candle["ts"])
        target["level"] = sum(target["prices"], Decimal("0")) / Decimal(len(target["prices"]))

    levels = []
    for cluster in clusters:
        level = quantize_down(cluster["level"], tick_size)
        if low_range <= level <= high_range:
            levels.append({
                "level": level,
                "touches": cluster["touches"],
                "last_ts": cluster["last_ts"],
            })

    return sorted(levels, key=lambda item: (item["touches"], item["last_ts"]), reverse=True)


def find_resistance_levels(
    candles: List[Dict[str, Any]],
    *,
    low_range: Decimal,
    high_range: Decimal,
    tick_size: Decimal,
) -> List[Dict[str, Any]]:
    if len(candles) < (SUPPORT_LOOKBACK * 2) + 1:
        return []

    sorted_candles = sorted(candles, key=lambda item: item["ts"])
    swing_highs: List[Dict[str, Any]] = []

    for index in range(SUPPORT_LOOKBACK, len(sorted_candles) - SUPPORT_LOOKBACK):
        candle = sorted_candles[index]
        high = candle["high"]
        if high < low_range or high > high_range:
            continue

        neighbors = (
            sorted_candles[index - SUPPORT_LOOKBACK:index]
            + sorted_candles[index + 1:index + SUPPORT_LOOKBACK + 1]
        )
        if all(high >= neighbor["high"] for neighbor in neighbors):
            swing_highs.append(candle)

    clusters: List[Dict[str, Any]] = []
    min_cluster_width = tick_size * Decimal("10") if tick_size > 0 else Decimal("0")

    for candle in sorted(swing_highs, key=lambda item: item["high"]):
        price = candle["high"]
        tolerance = max(price * SUPPORT_CLUSTER_PCT, min_cluster_width)
        target = None

        for cluster in clusters:
            if abs(price - cluster["level"]) <= tolerance:
                target = cluster
                break

        if target is None:
            clusters.append({
                "level": price,
                "touches": 1,
                "last_ts": candle["ts"],
                "prices": [price],
            })
            continue

        target["prices"].append(price)
        target["touches"] += 1
        target["last_ts"] = max(target["last_ts"], candle["ts"])
        target["level"] = sum(target["prices"], Decimal("0")) / Decimal(len(target["prices"]))

    levels = []
    for cluster in clusters:
        level = quantize_down(cluster["level"], tick_size)
        if low_range <= level <= high_range:
            levels.append({
                "level": level,
                "touches": cluster["touches"],
                "last_ts": cluster["last_ts"],
            })

    return sorted(levels, key=lambda item: (item["touches"], item["last_ts"]), reverse=True)


def get_support_resistance_range(current_price: Decimal) -> tuple[Decimal, Decimal]:
    configured_low = os.getenv("BLOFIN_SUPPORT_LOW")
    configured_high = os.getenv("BLOFIN_RESISTANCE_HIGH") or os.getenv("BLOFIN_SUPPORT_HIGH")

    low_range = decimal_from(configured_low, str(current_price - SUPPORT_DEFAULT_BELOW))
    high_range = decimal_from(configured_high, str(current_price + RESISTANCE_DEFAULT_ABOVE))

    if low_range > high_range:
        low_range, high_range = high_range, low_range

    return low_range, high_range


def fetch_candles(market_api: MarketAPI) -> List[Dict[str, Any]]:
    candles_response = market_api.getCandlesticks(
        instId=INST_ID,
        bar=SUPPORT_BAR,
        limit=SUPPORT_CANDLE_LIMIT,
    )
    return [
        record
        for candle in candles_response.get("data", [])
        if (record := candle_to_record(candle)) is not None
    ]


def level_line(level: Dict[str, Any], current_price: Decimal) -> str:
    last_seen = datetime.fromtimestamp(level["last_ts"] / 1000, tz=timezone.utc)
    distance = abs(level["level"] - current_price)
    return (
        f"price={decimal_to_api(level['level'])} "
        f"touches={level['touches']} "
        f"distance={decimal_to_api(distance)} "
        f"lastSeenUtc={last_seen.isoformat(timespec='seconds')}"
    )


def generate_support_resistance_chart(
    candles: List[Dict[str, Any]],
    *,
    current_price: Decimal,
    support_levels: List[Dict[str, Any]],
    resistance_levels: List[Dict[str, Any]],
    chart_path: Path,
) -> Path:
    if not candles:
        raise RuntimeError("Cannot render chart without candles.")

    chart_path = chart_path if chart_path.is_absolute() else Path(__file__).resolve().parent / chart_path
    sorted_candles = sorted(candles, key=lambda item: item["ts"])
    width = 1180
    height = 680
    left = 76
    right = 28
    top = 42
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom

    prices = [item["low"] for item in sorted_candles] + [item["high"] for item in sorted_candles]
    prices += [level["level"] for level in support_levels + resistance_levels]
    prices.append(current_price)
    min_price = min(prices)
    max_price = max(prices)
    pad = max((max_price - min_price) * Decimal("0.08"), Decimal("1"))
    min_price -= pad
    max_price += pad

    def x_for(index: int) -> Decimal:
        if len(sorted_candles) == 1:
            return Decimal(left + plot_w / 2)
        return Decimal(left) + (Decimal(index) / Decimal(len(sorted_candles) - 1)) * Decimal(plot_w)

    def y_for(price: Decimal) -> Decimal:
        return Decimal(top) + ((max_price - price) / (max_price - min_price)) * Decimal(plot_h)

    close_points = " ".join(
        f"{float(x_for(index)):.2f},{float(y_for(candle['close'])):.2f}"
        for index, candle in enumerate(sorted_candles)
    )

    wick_step = max(1, len(sorted_candles) // 180)
    wicks = []
    for index, candle in enumerate(sorted_candles):
        if index % wick_step:
            continue
        x = float(x_for(index))
        y_high = float(y_for(candle["high"]))
        y_low = float(y_for(candle["low"]))
        wicks.append(f'<line x1="{x:.2f}" y1="{y_high:.2f}" x2="{x:.2f}" y2="{y_low:.2f}" class="wick" />')

    def render_level(level: Dict[str, Any], kind: str, rank: int) -> str:
        y = float(y_for(level["level"]))
        price = html.escape(decimal_to_api(level["level"]))
        touches = html.escape(str(level["touches"]))
        label_x = left + 8 if kind == "support" else width - right - 166
        label = "S" if kind == "support" else "R"
        return (
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" class="{kind}-line" />'
            f'<text x="{label_x}" y="{y - 6:.2f}" class="{kind}-label">{label}{rank} {price} ({touches}x)</text>'
        )

    support_svg = "\n".join(
        render_level(level, "support", index)
        for index, level in enumerate(support_levels[:SUPPORT_LEVEL_COUNT], start=1)
    )
    resistance_svg = "\n".join(
        render_level(level, "resistance", index)
        for index, level in enumerate(resistance_levels[:RESISTANCE_LEVEL_COUNT], start=1)
    )

    y_ticks = []
    for tick in range(6):
        ratio = Decimal(tick) / Decimal(5)
        price = max_price - ((max_price - min_price) * ratio)
        y = float(y_for(price))
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" class="grid" />'
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="axis-text">{html.escape(decimal_to_api(quantize_down(price, Decimal("0.1"))))}</text>'
        )

    x_ticks = []
    tick_count = 5
    for tick in range(tick_count):
        index = round(tick * (len(sorted_candles) - 1) / (tick_count - 1))
        candle = sorted_candles[index]
        x = float(x_for(index))
        stamp = datetime.fromtimestamp(candle["ts"] / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}" class="grid" />'
            f'<text x="{x:.2f}" y="{height - bottom + 26}" text-anchor="middle" class="axis-text">{stamp}</text>'
        )

    current_y = float(y_for(current_price))
    title = html.escape(f"{INST_ID} {SUPPORT_BAR} support and resistance")
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    document = f"""<div id="support-resistance-chart">
  <style>
    #support-resistance-chart {{
      color: var(--foreground);
      font-family: inherit;
      width: 100%;
    }}
    #support-resistance-chart h2 {{
      margin: 0 0 0.65rem;
      font-weight: 500;
    }}
    #support-resistance-chart .meta {{
      color: var(--muted-foreground);
      margin: 0 0 0.75rem;
    }}
    #support-resistance-chart svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    #support-resistance-chart .frame {{
      fill: none;
      stroke: var(--border);
    }}
    #support-resistance-chart .grid {{
      stroke: var(--border);
      stroke-width: 1;
      opacity: 0.7;
    }}
    #support-resistance-chart .wick {{
      stroke: var(--muted-foreground);
      stroke-width: 1;
      opacity: 0.45;
    }}
    #support-resistance-chart .close-line {{
      fill: none;
      stroke: var(--viz-series-1);
      stroke-width: 2.4;
    }}
    #support-resistance-chart .support-line {{
      stroke: var(--green);
      stroke-width: 1.7;
      opacity: 0.9;
    }}
    #support-resistance-chart .resistance-line {{
      stroke: var(--red);
      stroke-width: 1.7;
      opacity: 0.9;
    }}
    #support-resistance-chart .current-line {{
      stroke: var(--foreground);
      stroke-width: 1.4;
      stroke-dasharray: 6 5;
      opacity: 0.75;
    }}
    #support-resistance-chart text {{
      fill: var(--foreground);
      font-size: 12px;
    }}
    #support-resistance-chart .axis-text,
    #support-resistance-chart .caption {{
      fill: var(--muted-foreground);
    }}
    #support-resistance-chart .support-label {{
      fill: var(--green);
    }}
    #support-resistance-chart .resistance-label {{
      fill: var(--red);
    }}
  </style>
  <h2>{title}</h2>
  <p class="meta">Generated {html.escape(generated)} UTC. Current price {html.escape(decimal_to_api(current_price))}.</p>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
    <title>{title}</title>
    <desc>Close price line with candle ranges, support levels, resistance levels, and current price.</desc>
    <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="frame" />
    {"".join(y_ticks)}
    {"".join(x_ticks)}
    {"".join(wicks)}
    <polyline points="{close_points}" class="close-line" />
    {support_svg}
    {resistance_svg}
    <line x1="{left}" y1="{current_y:.2f}" x2="{width - right}" y2="{current_y:.2f}" class="current-line" />
    <text x="{width - right - 128}" y="{current_y - 8:.2f}" class="caption">current {html.escape(decimal_to_api(current_price))}</text>
    <text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle" class="axis-text">Time UTC</text>
    <text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20 {top + plot_h / 2})" text-anchor="middle" class="axis-text">Price USDT</text>
  </svg>
</div>
"""
    chart_path.write_text(document, encoding="utf-8")
    return chart_path


def analyze_support_resistance(
    market_api: MarketAPI,
    *,
    current_price: Decimal,
    tick_size: Decimal,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Path]:
    low_range, high_range = get_support_resistance_range(current_price)
    candles = fetch_candles(market_api)
    support_levels = find_support_levels(
        candles,
        low_range=low_range,
        high_range=current_price,
        tick_size=tick_size,
    )
    resistance_levels = find_resistance_levels(
        candles,
        low_range=current_price,
        high_range=high_range,
        tick_size=tick_size,
    )
    chart_path = generate_support_resistance_chart(
        candles,
        current_price=current_price,
        support_levels=support_levels,
        resistance_levels=resistance_levels,
        chart_path=CHART_PATH,
    )

    print(
        "Support/resistance scan:",
        f"bar={SUPPORT_BAR}",
        f"candles={len(candles)}",
        f"range={decimal_to_api(low_range)}-{decimal_to_api(high_range)}",
    )

    if not support_levels:
        print("No support levels found in that range.")
    for index, level in enumerate(support_levels[:SUPPORT_LEVEL_COUNT], start=1):
        print(f"Support {index}:", level_line(level, current_price))

    if not resistance_levels:
        print("No resistance levels found in that range.")
    for index, level in enumerate(resistance_levels[:RESISTANCE_LEVEL_COUNT], start=1):
        print(f"Resistance {index}:", level_line(level, current_price))

    print(f"Chart written to: {chart_path}")
    return candles, support_levels, resistance_levels, chart_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BloFin demo BTC support/resistance and order tracker.")
    parser.add_argument(
        "--no-order",
        action="store_true",
        help="Only track price, compute levels, and write the chart. Do not place an order.",
    )
    return parser.parse_args()


async def wait_for_btc_price(public_ws: BlofinWsPublicClient) -> Decimal:
    await public_ws.subscribeTickers(INST_ID)
    print(f"Tracking {INST_ID} price on demo websocket...")

    async for message in public_ws.listen():
        if message.get("arg", {}).get("channel") != "tickers":
            continue

        data = message.get("data")
        if not isinstance(data, list) or not data:
            continue

        ticker = data[0]
        last = decimal_from(ticker.get("last"))
        if last <= 0:
            continue

        print(f"Latest demo websocket price: {decimal_to_api(last)}")
        return last

    raise RuntimeError("Public websocket closed before a ticker price was received.")


async def watch_private_orders(
    private_ws: BlofinWsPrivateClient,
    *,
    order_id: Optional[str],
    client_order_id: str,
    status_queue: asyncio.Queue,
) -> None:
    async for message in private_ws.listen():
        if message.get("arg", {}).get("channel") != "orders":
            continue

        data = message.get("data")
        if not isinstance(data, list):
            continue

        for order in data:
            if order_id and order.get("orderId") == order_id:
                await status_queue.put(("websocket", order))
            elif order.get("clientOrderId") == client_order_id:
                await status_queue.put(("websocket", order))


async def poll_order_rest(
    trading_api: TradingAPI,
    *,
    order_id: Optional[str],
    client_order_id: str,
    status_queue: asyncio.Queue,
) -> None:
    while True:
        await asyncio.sleep(REST_POLL_SECONDS)
        try:
            pending = trading_api.getOrdersPending(instId=INST_ID, limit="100")
            pending_order = find_order_by_ids(
                pending,
                order_id=order_id,
                client_order_id=client_order_id,
            )
            if pending_order:
                await status_queue.put(("rest-pending", pending_order))
                continue

            history = trading_api.getOrdersHistory(instId=INST_ID, limit="100")
            history_order = find_order_by_ids(
                history,
                order_id=order_id,
                client_order_id=client_order_id,
            )
            if history_order:
                await status_queue.put(("rest-history", history_order))

        except BlofinAPIException as exc:
            await status_queue.put(("rest-error", {"state": "unknown", "error": str(exc)}))


async def main() -> None:
    args = parse_args()
    api_key = require_env("API_KEY")
    secret = require_env("SECRET")
    passphrase = require_env("PASSPHRASE")

    rest_client = DemoClient(apiKey=api_key, apiSecret=secret, passphrase=passphrase)
    market_api = MarketAPI(rest_client)
    trading_api = TradingAPI(rest_client)

    public_ws = BlofinWsPublicClient(isDemo=True)
    private_ws = BlofinWsPrivateClient(
        apiKey=api_key,
        secret=secret,
        passphrase=passphrase,
        isDemo=True,
    )

    private_watch_task: Optional[asyncio.Task] = None
    rest_poll_task: Optional[asyncio.Task] = None

    try:
        await public_ws.connect()
        current_price = await wait_for_btc_price(public_ws)

        instrument = first_data(market_api.getInstruments(instId=INST_ID))
        tick_size = decimal_from(instrument.get("tickSize"), "0.1")
        min_size = decimal_from(instrument.get("minSize"), "1")
        lot_size = decimal_from(instrument.get("lotSize"), "1")

        analyze_support_resistance(
            market_api,
            current_price=current_price,
            tick_size=tick_size,
        )

        if args.no_order:
            print("No-order mode enabled. Skipping demo order placement.")
            return

        raw_limit_price = current_price - PRICE_OFFSET
        if raw_limit_price <= 0:
            raise SystemExit(f"Calculated limit price is not positive: {raw_limit_price}")

        limit_price = quantize_down(raw_limit_price, tick_size)
        configured_size = os.getenv("BLOFIN_ORDER_SIZE")
        size = decimal_from(configured_size, str(min_size))
        size = quantize_down(size, lot_size)
        if size < min_size:
            size = min_size

        client_order_id = f"demoBtcBuy{int(time.time())}"[-32:]
        order_payload = {
            "instId": INST_ID,
            "marginMode": MARGIN_MODE,
            "positionSide": POSITION_SIDE,
            "side": ORDER_SIDE,
            "orderType": ORDER_TYPE,
            "size": decimal_to_api(size),
            "price": decimal_to_api(limit_price),
            "clientOrderId": client_order_id,
        }

        try:
            await private_ws.connect()
            await private_ws.subscribeOrders(INST_ID)
        except BlofinAPIException as exc:
            raise SystemExit(
                "Demo private websocket authentication failed before order placement. "
                "No order was placed. Check that API_KEY, SECRET, and PASSPHRASE are "
                f"demo-trading credentials. Details: {exc}"
            ) from exc

        print("Placing DEMO limit buy order via REST:")
        print(order_payload)
        place_response = trading_api.placeOrder(**order_payload)
        print(f"REST placeOrder response: {place_response}")

        order_id = extract_order_id(place_response)
        status_queue: asyncio.Queue = asyncio.Queue()
        private_watch_task = asyncio.create_task(
            watch_private_orders(
                private_ws,
                order_id=order_id,
                client_order_id=client_order_id,
                status_queue=status_queue,
            )
        )
        rest_poll_task = asyncio.create_task(
            poll_order_rest(
                trading_api,
                order_id=order_id,
                client_order_id=client_order_id,
                status_queue=status_queue,
            )
        )

        print(f"Tracking fill status for up to {WATCH_SECONDS} seconds...")
        deadline = time.monotonic() + WATCH_SECONDS
        last_status = None

        while time.monotonic() < deadline:
            timeout = min(5, max(0.1, deadline - time.monotonic()))
            try:
                source, order = await asyncio.wait_for(status_queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                continue

            if source == "rest-error":
                print(f"REST status check error: {order.get('error')}")
                continue

            summary = summarize_order(order)
            if summary != last_status:
                print(f"Order status from {source}: {summary}")
                last_status = summary

            if order.get("state") in FINAL_STATES:
                print("Order reached a final state.")
                return

        print("Watch window ended before a final fill/cancel state.")
        print("Latest status:", last_status or "No order update received yet.")

    finally:
        if private_watch_task:
            private_watch_task.cancel()
        if rest_poll_task:
            rest_poll_task.cancel()
        await private_ws.close()
        await public_ws.close()


if __name__ == "__main__":
    asyncio.run(main())
