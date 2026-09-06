"""The only 'strategy-adjacent' logic in this project right now.

This computes swing-based support/resistance levels from recent candles:
find local highs/lows that stand out from their neighbors ("swings"), then
group nearby swing prices into clusters (a level gets more weight the more
times price has touched near it).

This is purely descriptive — it draws lines on the chart. It does not decide
when to buy or sell. See the README roadmap for what still needs to be built
before this project does any actual trading.
"""

from typing import Any, Dict, List

from decimal import Decimal

from config import LOOKBACK, CLUSTER_PCT
from market_data import quantize_down


def cluster_swings(
    swings: List[Dict[str, Any]],
    *,
    price_key: str,
    low_range: Decimal,
    high_range: Decimal,
    tick_size: Decimal,
) -> List[Dict[str, Any]]:
    """Group swing points that are close in price into single levels."""
    clusters: List[Dict[str, Any]] = []
    min_cluster_width = tick_size * Decimal("10") if tick_size > 0 else Decimal("0")

    for candle in sorted(swings, key=lambda item: item[price_key]):
        price = candle[price_key]
        tolerance = max(price * CLUSTER_PCT, min_cluster_width)
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


def _find_swings(
    candles: List[Dict[str, Any]],
    *,
    price_key: str,
    is_better: Any,
    low_range: Decimal,
    high_range: Decimal,
) -> List[Dict[str, Any]]:
    """Shared scan for both supports (local lows) and resistances (local
    highs): a candle is a swing point if no neighbor within LOOKBACK bars on
    either side beats it."""
    candles = sorted(candles, key=lambda item: item["ts"])
    if len(candles) < (LOOKBACK * 2) + 1:
        return []

    swings = []
    for index in range(LOOKBACK, len(candles) - LOOKBACK):
        candle = candles[index]
        price = candle[price_key]
        if price < low_range or price > high_range:
            continue
        neighbors = candles[index - LOOKBACK:index] + candles[index + 1:index + LOOKBACK + 1]
        if all(is_better(price, neighbor[price_key]) for neighbor in neighbors):
            swings.append(candle)

    return swings


def find_support_levels(
    candles: List[Dict[str, Any]],
    *,
    low_range: Decimal,
    high_range: Decimal,
    tick_size: Decimal,
) -> List[Dict[str, Any]]:
    """Support = clusters of local lows (price <= every nearby low)."""
    swings = _find_swings(
        candles,
        price_key="low",
        is_better=lambda price, neighbor: price <= neighbor,
        low_range=low_range,
        high_range=high_range,
    )
    return cluster_swings(
        swings,
        price_key="low",
        low_range=low_range,
        high_range=high_range,
        tick_size=tick_size,
    )


def find_resistance_levels(
    candles: List[Dict[str, Any]],
    *,
    low_range: Decimal,
    high_range: Decimal,
    tick_size: Decimal,
) -> List[Dict[str, Any]]:
    """Resistance = clusters of local highs (price >= every nearby high)."""
    swings = _find_swings(
        candles,
        price_key="high",
        is_better=lambda price, neighbor: price >= neighbor,
        low_range=low_range,
        high_range=high_range,
    )
    return cluster_swings(
        swings,
        price_key="high",
        low_range=low_range,
        high_range=high_range,
        tick_size=tick_size,
    )
