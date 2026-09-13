"""Which BloFin instruments can the lead quoter (README 9ae) be run on? READ ONLY.

    python backend\\analysis\\lead_quote_universe.py                 # the passing list, by BloFin volume
    python backend\\analysis\\lead_quote_universe.py --all           # every USDT swap with its reasons
    python backend\\analysis\\lead_quote_universe.py --top 12 --min-blofin-usd 2e6

Why a screen and not the backtest
---------------------------------
The 9ae backtest ran on the six instruments the recorder had archived, and
it needed three days of BloFin books plus Binance aggTrades per instrument.
The paper quoter is the cheaper test: it reads production feeds and fills
itself from BloFin's tape, sending nothing. So the question for a new pair
is only whether it can pass the quoter's own gate and has a tape to fill
from, and that is three numbers per instrument, all public:

  tick, in bps of price    the gate in `plan.py` (MAX_TICK_BPS = 2.5; ADA at
                           4.8 lost every day in the study). Below ~1 bp the
                           other failure appears: BTC's tick is 0.01 bp and
                           its BloFin book tracks Binance inside the venue's
                           100 ms batch, so a 7 bp gap with an empty level
                           never showed up (0 posts in 5.6 h, 2026-09-13).
  spread, in ticks         a bid at `ask - tick` must sit above the bid, so
                           the spread has to be two ticks or more, often.
                           A one-tick spread posts nothing. One snapshot is
                           not enough: LINK read 1 tick, then 2, then 6 in
                           three reads a minute apart (2026-09-13), so the
                           screen samples the tickers `--samples` times and
                           gates on the median, and prints the share of
                           samples with room to post ("room").
  BloFin 24h volume, USD   paper fills come from BloFin prints at our price;
                           SUI's ~$2.7M/day gave 20-40 fills a day. Below
                           about $1M/day a run is mostly waiting.

Binance's 24h quote volume is shown because the leader has to lead: a
symbol Binance barely trades has no signal in its book ticker.

Every USDT-margined swap on BloFin has the same symbol on Binance USDT-M
futures with the dash removed (`1000BONK-USDT` -> `1000BONKUSDT`; checked
2026-09-13: 39 of 488 unmatched, all BTC-USD-style coin-margined or not on
Binance). Unmatched ones are listed as such, never guessed.

The demo host lists 87 of production's 488 swaps (2026-09-13), so a pair
can be paper-quoted from production feeds but not mirrored with `--confirm`
(FLOCK and USELESS started and exited before writing a log that day). The
`demo` column says which, and the launch lines are split accordingly.

The six instruments 9ae studied carry their backtest verdict as a note, so
a screen pass is never read as overriding a measured loss (LTC: zero fills;
ADA: negative every day).

This ranks candidates; it is not evidence. Evidence is a week of paper
fills from the Tokyo box, read back with `run_lead_quote.py --summary`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading.strategies.lead_quote.plan import MAX_TICK_BPS  # noqa: E402

BLOFIN = "https://openapi.blofin.com"
BLOFIN_DEMO = "https://demo-trading-openapi.blofin.com"   # lists 87 of production's 488 swaps (2026-09-13)
BINANCE = "https://fapi.binance.com"
MIN_TICK_BPS = 0.3        # below this the BloFin book never lags a whole edge behind (BTC, 2026-09-13)
MIN_SPREAD_TICKS = 2.0    # a bid at ask - tick must be above the bid
STUDIED = {               # README 9ae, edge 7 / stop 3, three archived days
    "SUI-USDT": "9ae: positive 3/3 days", "DOGE-USDT": "9ae: positive 3/3 days",
    "AVAX-USDT": "9ae: positive 3/3 days", "BTC-USDT": "9ae: positive 3/3 days",
    "LTC-USDT": "9ae: zero fills", "ADA-USDT": "9ae: negative every day",
}


def _fetch(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass
class Candidate:
    inst_id: str
    price: float
    tick: float
    tick_bps: float
    spread_bps: float
    spread_ticks: float
    blofin_usd: float
    binance_usd: Optional[float]
    binance_trades: Optional[int]
    room_share: float = float("nan")     # share of ticker samples with spread >= MIN_SPREAD_TICKS
    on_demo: bool = False                # listed on the demo host, so --confirm can mirror it
    note: str = ""
    reasons: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons


def spread_samples(snapshots: List[List[Dict[str, Any]]]) -> Dict[str, List[float]]:
    """Per instrument, the ask - bid of every snapshot that had both sides."""
    out: Dict[str, List[float]] = {}
    for snapshot in snapshots:
        for row in snapshot:
            try:
                bid, ask = float(row["bidPrice"]), float(row["askPrice"])
            except (KeyError, TypeError, ValueError):
                continue
            if bid > 0 and ask > 0:
                out.setdefault(str(row.get("instId")), []).append(ask - bid)
    return out


def screen(instruments: List[Dict[str, Any]], tickers: List[Dict[str, Any]],
           binance: List[Dict[str, Any]], *, min_blofin_usd: float, min_binance_usd: float,
           max_tick_bps: float = MAX_TICK_BPS, min_tick_bps: float = MIN_TICK_BPS,
           min_spread_ticks: float = MIN_SPREAD_TICKS,
           spreads: Optional[Dict[str, List[float]]] = None,
           demo_instruments: Optional[List[Dict[str, Any]]] = None) -> List[Candidate]:
    """Every USDT swap with its numbers and every reason it fails, sorted by BloFin volume.

    `spreads` (instrument -> sampled ask - bid values) overrides the single
    spread in `tickers`; the gate is on the median and `room_share` is the
    share of samples at or above `min_spread_ticks`.
    """
    rules = {row["instId"]: row for row in instruments}
    leader = {row["symbol"]: row for row in binance}
    demo = {row.get("instId") for row in (demo_instruments or [])}
    out: List[Candidate] = []
    for ticker in tickers:
        inst_id = str(ticker.get("instId") or "")
        if not inst_id.endswith("-USDT"):
            continue
        rule = rules.get(inst_id)
        if rule is None:
            continue
        try:
            price = float(ticker["last"])
            bid, ask = float(ticker["bidPrice"]), float(ticker["askPrice"])
            tick = float(rule["tickSize"])
            contract = float(rule.get("contractValue") or 1.0)
            vol = float(ticker.get("vol24h") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        if price <= 0 or tick <= 0:
            continue
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else price
        samples = (spreads or {}).get(inst_id) or ([ask - bid] if bid > 0 and ask > 0 else [])
        spread = statistics.median(samples) if samples else float("nan")
        room = (sum(1 for x in samples if x / tick >= min_spread_ticks) / len(samples)
                if samples else float("nan"))
        lead = leader.get(inst_id.replace("-", ""))
        cand = Candidate(
            inst_id=inst_id, price=price, tick=tick, tick_bps=tick / price * 1e4,
            spread_bps=spread / mid * 1e4, spread_ticks=spread / tick,
            blofin_usd=vol * contract * price,
            binance_usd=float(lead["quoteVolume"]) if lead else None,
            binance_trades=int(lead["count"]) if lead and "count" in lead else None,
            room_share=room, on_demo=inst_id in demo, note=STUDIED.get(inst_id, ""))
        if str(rule.get("state", "live")) != "live":
            cand.reasons.append("not live on BloFin")
        if cand.tick_bps > max_tick_bps:
            cand.reasons.append("tick {:.2f} bps > {:.1f}".format(cand.tick_bps, max_tick_bps))
        if cand.tick_bps < min_tick_bps:
            cand.reasons.append("tick {:.2f} bps < {:.1f}: book never lags a whole edge".format(
                cand.tick_bps, min_tick_bps))
        if not (cand.spread_ticks >= min_spread_ticks):
            cand.reasons.append("spread {:.1f} ticks < {:.0f}: nothing to post inside".format(
                cand.spread_ticks, min_spread_ticks))
        if cand.blofin_usd < min_blofin_usd:
            cand.reasons.append("BloFin ${:.1f}M/day < ${:.1f}M".format(
                cand.blofin_usd / 1e6, min_blofin_usd / 1e6))
        if lead is None:
            cand.reasons.append("no Binance USDT-M future named " + inst_id.replace("-", ""))
        elif cand.binance_usd is not None and cand.binance_usd < min_binance_usd:
            cand.reasons.append("Binance ${:.0f}M/day < ${:.0f}M".format(
                cand.binance_usd / 1e6, min_binance_usd / 1e6))
        out.append(cand)
    out.sort(key=lambda c: -c.blofin_usd)
    return out


def lines(cands: List[Candidate], *, show_all: bool, top: int) -> List[str]:
    out = ["{:<16} {:>10} {:>8} {:>9} {:>7} {:>5} {:>10} {:>10} {:>8} {:>5}  {}".format(
        "instrument", "price", "tick bps", "sprd bps", "ticks", "room", "BloFin $M", "Binance$M", "trades/d", "demo", "")]
    shown = 0
    for cand in cands:
        if not show_all and not cand.ok:
            continue
        if shown >= top:
            break
        shown += 1
        verdict = "OK" if cand.ok else "; ".join(cand.reasons)
        if cand.note:
            verdict += "  [" + cand.note + "]"
        out.append("{:<16} {:>10.5g} {:>8.2f} {:>9.2f} {:>7.1f} {:>4.0%} {:>10.2f} {:>10.0f} {:>8} {:>5}  {}".format(
            cand.inst_id, cand.price, cand.tick_bps, cand.spread_bps, cand.spread_ticks, cand.room_share,
            cand.blofin_usd / 1e6, (cand.binance_usd or 0) / 1e6,
            "{:,}".format(cand.binance_trades // 1000) + "k" if cand.binance_trades else "-",
            "yes" if cand.on_demo else "no", verdict))
    return out


def main(argv: Optional[List[str]] = None, fetch: Callable[[str], Any] = _fetch,
         sleep: Callable[[float], None] = time.sleep) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--all", action="store_true", help="show failing instruments and why")
    parser.add_argument("--min-blofin-usd", type=float, default=1e6, help="BloFin 24h volume floor, USD")
    parser.add_argument("--min-binance-usd", type=float, default=2e7, help="Binance 24h quote volume floor")
    parser.add_argument("--minutes", type=int, default=1440, help="for the printed command lines")
    parser.add_argument("--samples", type=int, default=10, help="ticker snapshots for the spread gate")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between snapshots")
    args = parser.parse_args(argv)

    instruments = fetch(BLOFIN + "/api/v1/market/instruments?instType=SWAP").get("data") or []
    demo_instruments = fetch(BLOFIN_DEMO + "/api/v1/market/instruments?instType=SWAP").get("data") or []
    snapshots = []
    for i in range(max(1, args.samples)):
        if i:
            sleep(args.interval)
        snapshots.append(fetch(BLOFIN + "/api/v1/market/tickers?instType=SWAP").get("data") or [])
    tickers = snapshots[-1]
    binance = fetch(BINANCE + "/fapi/v1/ticker/24hr")
    cands = screen(instruments, tickers, binance, min_blofin_usd=args.min_blofin_usd,
                   min_binance_usd=args.min_binance_usd, spreads=spread_samples(snapshots),
                   demo_instruments=demo_instruments)
    passing = [c for c in cands if c.ok]
    print("{} USDT swaps on BloFin, {} pass the 9ae gate at tick {:.1f}-{:.1f} bps, median spread >= {:.0f} ticks "
          "over {} samples, BloFin >= ${:.1f}M/day, Binance >= ${:.0f}M/day".format(
              len(cands), len(passing), MIN_TICK_BPS, MAX_TICK_BPS, MIN_SPREAD_TICKS, len(snapshots),
              args.min_blofin_usd / 1e6, args.min_binance_usd / 1e6))
    print()
    for line in lines(cands, show_all=args.all, top=args.top if not args.all else len(cands)):
        print(line)
    chosen = passing[:args.top]
    mirrored = [c.inst_id for c in chosen if c.on_demo]
    paper_only = [c.inst_id for c in chosen if not c.on_demo]
    if mirrored:
        print()
        print("one process each, on the Tokyo box; these are on the demo host, so --confirm mirrors them:")
        print("for i in {}; do nohup python backend/run_lead_quote.py --instruments $i --minutes {} --confirm "
              "> lq_$i.out 2>&1 & done".format(" ".join(mirrored), args.minutes))
    if paper_only:
        print()
        print("these are NOT on the demo host (--confirm would exit before writing a log); paper only:")
        print("for i in {}; do nohup python backend/run_lead_quote.py --instruments $i --minutes {} "
              "> lq_$i.out 2>&1 & done".format(" ".join(paper_only), args.minutes))
    print()
    print("volume is one 24h window and spread a few seconds of samples; the tick gate is the only hard one. "
          "A pair is evidence only after a week of --summary.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
