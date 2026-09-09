"""What are BloFin's OWN spreads? The venue-native version of the fee gate.

    python backend\\analysis\\blofin_spread_survey.py
    python backend\\analysis\\blofin_spread_survey.py --minutes 30 --interval 5
    python backend\\analysis\\blofin_spread_survey.py --min-volume-usd 5e6 --top 40

`spread_survey.py` answered "which instruments' spreads cover the maker fee?"
from Tardis archives of **binance-futures**. That was the venue available, not
the venue that matters: this account trades on BloFin, a far less arbitraged
book. Every execution-cost conclusion in this repo is therefore, strictly, a
conclusion about somebody else's exchange.

This closes that gap for the price of a poll. `getTickers()` returns best bid,
best ask and 24h volume for every instrument in one unauthenticated REST call
- 487 of them at the time of writing - so a whole-venue survey needs no
download, no API key and no archive.

The gate is imported from `passive_sim.clears_fee_gate` rather than restated,
so this can never disagree with the simulator about what passing means.

What it does NOT tell you
-------------------------
**A wide spread is not free money.** There are two reasons a spread is wide
and they have opposite implications, so the report separates them:

1. *Competition-limited.* Makers widen when the flow is more informed, so
   spread and adverse selection move together. Measured on Binance, adverse
   selection ran 0.29-0.65 bps per leg across a 400x range of spreads, with no
   trend - roughly half a basis point regardless. That is why the report shows
   a second, stricter bar of `2 x (0.5 + maker fee)` beside the gate.

2. *Tick-limited.* When an instrument sits at its minimum tick essentially
   always, makers *cannot* compete the spread away, so they queue behind it
   instead. On Binance that meant 173,000 contracts resting at ADA's best bid
   against 3 at BTC's. A tick-bound spread is a queue to get to the front of,
   not a payment for risk - and queue position is precisely what `passive_sim`
   refuses to model and brackets instead.

So the report prints `tick bps`, the spread floor implied by `tickSize/mid`,
beside the measured spread. Where the two are equal the instrument is pinned
at its tick and reason 2 applies. Where the spread sits well above its floor,
makers are setting the width and reason 1 applies.

**Nor does it say anything about fills.** 24h volume is shown to keep you off
instruments whose wide spread is really an absence of anyone trading, but
volume is not queue position and neither is top-of-book size.

Sampling
--------
Tick-bound spreads are pinned - on Binance, p25 and p75 sat within 0.01 bps of
the median - so a short poll gives a stable median for exactly the instruments
most likely to clear the gate. The default 10 minutes at 5s is 120 samples.
Widen `--minutes` before believing anything about an instrument whose p25 and
p75 are far apart, because that is the one whose median still moves.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.passive_sim import (  # noqa: E402
    MAKER_BPS,
    ROUND_TRIP_MAKER_BPS,
    clears_fee_gate,
)

# Measured adverse selection per passive leg, from the step 9a simulations:
# 0.29-0.65 bps across a 400x spread range, with no trend against spread.
# Treated as a constant because that is what the measurement showed.
ADVERSE_SELECTION_BPS_PER_LEG = 0.5

# The bar including it. `clears_fee_gate` deliberately excludes adverse
# selection - it answers "does the arithmetic work at all". An instrument that
# passes the gate and fails this is not tradeable, only worth simulating.
EMPIRICAL_GATE_BPS = 2 * (ADVERSE_SELECTION_BPS_PER_LEG + float(MAKER_BPS))


@dataclass
class Instrument:
    """Static contract facts, from getInstruments()."""

    inst_id: str
    tick_size: float
    contract_type: str
    quote: str
    state: str


@dataclass
class SpreadSamples:
    """Accumulated top-of-book observations for one instrument."""

    inst_id: str
    spreads_bps: List[float] = field(default_factory=list)
    mid: float = float("nan")
    volume_usd_24h: float = float("nan")
    tick_bps: float = float("nan")
    contract_type: str = ""

    def add(self, bid: float, ask: float) -> None:
        # A crossed or locked book is not a spread measurement - it is a stale
        # cache on their side or a genuinely crossed quote. Averaging it in
        # would understate the width, which is the direction that flatters.
        if not (bid > 0 and ask > 0 and ask > bid):
            return
        mid = (bid + ask) / 2.0
        self.spreads_bps.append((ask - bid) / mid * 10_000.0)
        self.mid = mid

    @property
    def samples(self) -> int:
        return len(self.spreads_bps)

    @property
    def median_bps(self) -> float:
        return statistics.median(self.spreads_bps) if self.spreads_bps else float("nan")

    @property
    def p25_bps(self) -> float:
        return self._quantile(0.25)

    @property
    def p75_bps(self) -> float:
        return self._quantile(0.75)

    def _quantile(self, q: float) -> float:
        if not self.spreads_bps:
            return float("nan")
        ordered = sorted(self.spreads_bps)
        index = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1) + 0.5)))
        return ordered[index]

    @property
    def above_gate(self) -> float:
        """Fraction of samples clearing the arithmetic gate.

        An instrument at 0.9 bps on median but above the gate a third of the
        time is a *selective* quoting opportunity; one pinned above it all day
        is a different and better one. The median alone cannot separate them.
        """
        if not self.spreads_bps:
            return float("nan")
        clearing = sum(1 for value in self.spreads_bps if clears_fee_gate(value))
        return clearing / len(self.spreads_bps)

    @property
    def passes(self) -> bool:
        return self.samples > 0 and clears_fee_gate(self.median_bps)

    @property
    def passes_empirical(self) -> bool:
        return self.samples > 0 and self.median_bps >= EMPIRICAL_GATE_BPS

    @property
    def tick_bound(self) -> bool:
        """Is the measured spread sitting on its tick floor?

        Within 20% of it. Exact equality is too strict: the floor is computed
        from a mid that moves between samples, so a genuinely pinned
        instrument still lands slightly either side of the line.
        """
        if not (self.tick_bps > 0) or not self.spreads_bps:
            return False
        return self.median_bps <= self.tick_bps * 1.2


def fetch_instruments(api) -> Dict[str, Instrument]:
    """Static contract facts, keyed by instId. One call for the whole venue."""
    payload = api.getInstruments()
    instruments: Dict[str, Instrument] = {}
    for row in payload.get("data", []) or []:
        inst_id = row.get("instId", "")
        if not inst_id:
            continue
        try:
            tick = float(row.get("tickSize", "0") or 0)
        except (TypeError, ValueError):
            tick = 0.0
        instruments[inst_id] = Instrument(
            inst_id=inst_id,
            tick_size=tick,
            contract_type=row.get("contractType", ""),
            quote=row.get("quoteCurrency", ""),
            state=row.get("state", ""),
        )
    return instruments


def poll_once(api, samples: Dict[str, SpreadSamples],
              instruments: Dict[str, Instrument]) -> int:
    """Fold one getTickers() call into the accumulator. Returns rows seen."""
    payload = api.getTickers()
    rows = payload.get("data", []) or []
    for row in rows:
        inst_id = row.get("instId", "")
        if not inst_id:
            continue
        try:
            bid = float(row.get("bidPrice", "0") or 0)
            ask = float(row.get("askPrice", "0") or 0)
            volume = float(row.get("volCurrency24h", "0") or 0)
        except (TypeError, ValueError):
            continue

        entry = samples.get(inst_id)
        if entry is None:
            meta = instruments.get(inst_id)
            entry = SpreadSamples(
                inst_id=inst_id,
                contract_type=meta.contract_type if meta else "",
            )
            samples[inst_id] = entry
        entry.add(bid, ask)

        if entry.mid > 0:
            meta = instruments.get(inst_id)
            if meta is not None and meta.tick_size > 0:
                # The narrowest spread this instrument can legally quote.
                entry.tick_bps = meta.tick_size / entry.mid * 10_000.0
            # volCurrency24h is denominated in the base currency; price it in
            # USD so instruments of different unit sizes are comparable at all.
            if volume > 0:
                entry.volume_usd_24h = volume * entry.mid
    return len(rows)


def collect(api, minutes: float, interval: float,
            instruments: Dict[str, Instrument],
            sleep=time.sleep) -> Dict[str, SpreadSamples]:
    """Poll the whole venue until the window closes."""
    samples: Dict[str, SpreadSamples] = {}
    deadline = time.time() + minutes * 60.0
    polls = 0
    while True:
        try:
            count = poll_once(api, samples, instruments)
            polls += 1
            print(f"  poll {polls:>4}  {count} instruments  "
                  f"{max(0.0, deadline - time.time()):.0f}s left", flush=True)
        except Exception as exc:
            # One failed poll is not a failed survey. Losing the whole window
            # to a transient HTTP error would be the same mistake that ended
            # the overnight recorder run.
            print(f"  poll failed: {exc}", flush=True)
        if time.time() >= deadline:
            break
        sleep(interval)
    return samples


def eligible(entry: SpreadSamples, instruments: Dict[str, Instrument], *,
             quote: Optional[str], linear_only: bool,
             min_volume_usd: float) -> bool:
    meta = instruments.get(entry.inst_id)
    if meta is not None:
        if meta.state and meta.state != "live":
            return False
        if quote and meta.quote != quote:
            return False
        # Inverse contracts are quoted in the base currency and their PnL is
        # non-linear in price. The passive arithmetic in this repo assumes a
        # linear contract throughout, so including them would compare numbers
        # that are not the same number.
        if linear_only and meta.contract_type and meta.contract_type != "linear":
            return False
    if entry.samples == 0:
        return False
    volume = entry.volume_usd_24h
    if min_volume_usd > 0 and not (volume == volume and volume >= min_volume_usd):
        return False
    return True


def report(rows: Sequence[SpreadSamples], *, minutes: float, interval: float,
           min_volume_usd: float, top: int) -> None:
    print("\n" + "=" * 86)
    print(f"BLOFIN SPREAD SURVEY  ({minutes:g} min at {interval:g}s, "
          f"{len(rows)} instruments)")
    print("=" * 86)
    print(f"  gate: a passive round trip captures the whole spread and pays "
          f"{ROUND_TRIP_MAKER_BPS:.2f} bps")
    print(f"        ({MAKER_BPS:.2f} per leg), so an instrument needs a median "
          f"spread at or above that.")
    print(f"  and:  measured adverse selection adds "
          f"{ADVERSE_SELECTION_BPS_PER_LEG:.1f} bps per leg, so the empirical "
          f"bar is {EMPIRICAL_GATE_BPS:.2f} bps.")
    if min_volume_usd > 0:
        print(f"  filtered to 24h volume >= ${min_volume_usd:,.0f}.")

    if not rows:
        print("\n  Nothing measured. Loosen --min-volume-usd or --quote.")
        return

    ranked = sorted(rows, key=lambda item: -item.median_bps)
    shown = ranked[:top]

    print(f"\n  {'instrument':<18}{'spread':>9}{'p25':>8}{'p75':>8}"
          f"{'tick bps':>10}{'above gate':>12}{'24h vol':>13}   verdict")
    print("  " + "-" * 82)
    for entry in shown:
        volume = (f"${entry.volume_usd_24h/1e6:,.1f}M"
                  if entry.volume_usd_24h == entry.volume_usd_24h else "-")
        if entry.passes_empirical:
            verdict = "CLEARS BOTH"
        elif entry.passes:
            verdict = "clears fee only"
        else:
            verdict = ""
        pinned = "*" if entry.tick_bound else " "
        print(f"  {entry.inst_id:<18}{entry.median_bps:>9.3f}"
              f"{entry.p25_bps:>8.3f}{entry.p75_bps:>8.3f}"
              f"{entry.tick_bps:>9.3f}{pinned}{entry.above_gate:>11.1%}"
              f"{volume:>13}   {verdict}")
    if len(ranked) > len(shown):
        print(f"  ... {len(ranked) - len(shown)} more below "
              f"{shown[-1].median_bps:.3f} bps")
    print("\n  * spread is sitting on its tick floor: makers cannot compete it "
          "away, so they\n    queue behind it instead. That width is a queue, "
          "not a payment for risk.")

    passing = [row for row in ranked if row.passes]
    both = [row for row in ranked if row.passes_empirical]
    selective = [row for row in ranked
                 if not row.passes and row.above_gate >= 0.10]

    print("\n" + "=" * 86)
    print("VERDICT")
    print("=" * 86)
    widest = ranked[0]
    if both:
        names = ", ".join(row.inst_id for row in both[:8])
        print(f"  {len(both)} instrument(s) clear the fee gate AND the "
              f"{EMPIRICAL_GATE_BPS:.2f} bps empirical bar:\n    {names}")
        print("\n  This is the first BloFin-native shortlist in the project, "
              "and it is a\n  shortlist, not a result. Two things still stand "
              "between it and an edge:")
        print("    - Queue position. A starred instrument is pinned at its "
              "tick, which means\n      the spread is wide because the queue "
              "is long. Step 9 measured markout at\n      -0.400 bps at the "
              "instant of a back-of-queue fill: the fill arrives\n      "
              "exactly when the level is being swept.")
        print("    - Adverse selection here is ASSUMED at "
              f"{ADVERSE_SELECTION_BPS_PER_LEG:.1f} bps/leg from Binance "
              "measurements.\n      It has never been measured on BloFin, and "
              "a less arbitraged venue is not\n      obviously the same.")
        print("\n  Both need a book/trade recording on the instrument itself. "
              "The recorder\n  already does that - point BLOFIN_INST_ID at the "
              "best candidate and run it.")
    elif passing:
        names = ", ".join(row.inst_id for row in passing[:8])
        print(f"  {len(passing)} instrument(s) clear the {ROUND_TRIP_MAKER_BPS:.2f} "
              f"bps fee gate but none clear the\n  {EMPIRICAL_GATE_BPS:.2f} bps "
              f"bar once adverse selection is included:\n    {names}")
        print("\n  That is the same shape of answer step 9 got on Binance: the "
              "arithmetic\n  works and the microstructure does not. Worth "
              "simulating only if you think\n  BloFin's adverse selection is "
              "materially below Binance's - which is\n  plausible on a less "
              "arbitraged venue, and is a measurement, not a hope.")
    elif selective:
        names = ", ".join(f"{row.inst_id} ({row.above_gate:.0%})"
                          for row in selective[:8])
        print("  Nothing clears the gate on median spread, but some clear it "
              "part of the\n  time:\n    " + names)
        print("\n  That is a *selective* quoting question: quote only while the "
              "spread is wide.\n  A real strategy and a harder one, because the "
              "spread is widest exactly when\n  the flow is worst.")
    else:
        print("  NOTHING CLEARS THE FEE GATE ON THIS VENUE")
        print(f"  The widest instrument surveyed is {widest.inst_id} at "
              f"{widest.median_bps:.3f} bps,\n  which is "
              f"{ROUND_TRIP_MAKER_BPS / max(widest.median_bps, 1e-9):.1f}x "
              "short of the fee.")
        print("\n  If BloFin is no wider than Binance, the passive branch is "
              "closed on both\n  venues at this fee tier, and the honest next "
              "branch is funding carry -\n  which needs no spread and no "
              "directional forecast.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--minutes", type=float, default=10.0,
                        help="How long to poll (default 10).")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Seconds between polls (default 5).")
    parser.add_argument("--quote", default="USDT",
                        help="Quote currency to survey, or '' for all.")
    parser.add_argument("--all-contract-types", action="store_true",
                        help="Include inverse contracts (excluded by default: "
                             "their PnL is not linear in price, so the "
                             "passive arithmetic here does not apply).")
    parser.add_argument("--min-volume-usd", type=float, default=1e6,
                        help="Drop instruments below this 24h volume "
                             "(default 1e6). A wide spread on an instrument "
                             "nobody trades is not an opportunity.")
    parser.add_argument("--top", type=int, default=25,
                        help="Rows to print (default 25).")
    args = parser.parse_args(argv)

    from blofin.client import Client
    from blofin.rest_market import MarketAPI

    api = MarketAPI(Client())

    print("Fetching instrument definitions...", flush=True)
    instruments = fetch_instruments(api)
    print(f"  {len(instruments)} instruments on the venue.")

    print(f"\nPolling top of book for {args.minutes:g} minutes "
          f"every {args.interval:g}s:", flush=True)
    samples = collect(api, args.minutes, args.interval, instruments)

    rows = [entry for entry in samples.values()
            if eligible(entry, instruments,
                        quote=args.quote or None,
                        linear_only=not args.all_contract_types,
                        min_volume_usd=args.min_volume_usd)]
    report(rows, minutes=args.minutes, interval=args.interval,
           min_volume_usd=args.min_volume_usd, top=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
