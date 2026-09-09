"""Is adverse selection on BloFin the same as on Binance? Measure both, paired.

    python backend\\analysis\\venue_compare.py
    python backend\\analysis\\venue_compare.py --instruments ADA-USDT,DOGE-USDT
    python backend\\analysis\\venue_compare.py --hours 2 --horizon 60

The passive branch rests on one number that has never been measured on the
venue this account trades on. `passive_sim.py` found adverse selection of
0.29-0.65 bps per leg across a 400x range of spreads, with no trend against
spread - a genuinely useful regularity, and a **Binance** one. Every gate in
this repo has since assumed 0.5 bps/leg applies to BloFin too.

It might not. BloFin's spreads run 1.8-2.7x wider on matched instruments
(`blofin_spread_survey.py`), which says there is less maker competition there.
Less competition is consistent with adverse selection being *lower* (fewer
informed makers picking you off) or *higher* (the flow that does arrive is
disproportionately informed). Both stories are plausible, which is exactly why
it needs measuring rather than arguing about.

So this runs the identical `passive_sim` pipeline over both venues' data for
the same instruments and reports the paired difference. Same `collect`, same
`observable`, same `simulate`, same `markout_table`, same
`adverse_selection` - imported, not reimplemented, so the two sides cannot
disagree about what is being measured.

The confound, stated up front
-----------------------------
**The two sides are not the same days.** BloFin data is whatever the recorder
has captured; the Binance side comes from the Tardis free sample, which is the
1st of a month and nothing else. Adverse selection varies with regime, so a
difference between venues and a difference between dates arrive here wearing
the same clothes.

BTC-USDT is the control that makes the rest readable. It is on both venues,
tick-bound on both, and its spread was measured identical to three decimal
places. If BTC's adverse selection also matches across the two sides, the date
gap is not doing much and the differences on other instruments are more likely
to be about the venue. If BTC's differs materially, this whole comparison is
date-confounded and says so in the verdict.

That is weaker than a same-day comparison and stronger than an assumption.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.layout import instrument_dirs  # noqa: E402
from analysis.passive_sim import (  # noqa: E402
    MODELS,
    adverse_selection,
    collect,
    markout_table,
    observable,
    raw_archive_events,
    simulate,
    tardis_events,
    touch_cancel_share,
)
from analysis.tardis_import import FREE_SAMPLE_DAY  # noqa: E402


def latest_free_sample_day(today: Optional["date"] = None) -> str:
    """The most recent day Tardis gives away, as YYYY-MM-DD.

    `FREE_SAMPLE_DAY` is a day-of-MONTH (1), not a date - the free sample is
    the 1st of any month and nothing else. So the default here is the most
    recent 1st strictly in the past, which is also the day every cached file
    from the step 9a survey happens to be.
    """
    import datetime as dt

    today = today or dt.date.today()
    first = today.replace(day=FREE_SAMPLE_DAY)
    if first >= today:
        first = (first - dt.timedelta(days=1)).replace(day=FREE_SAMPLE_DAY)
    return first.isoformat()

# The instruments with measurements on both sides. Deliberately the five the
# step 9a survey simulated, because a paired comparison needs a pair.
DEFAULT_INSTRUMENTS = ("BTC-USDT", "ADA-USDT", "DOGE-USDT", "LTC-USDT",
                       "AVAX-USDT")

# BTC is tick-bound on both venues and its spread matched to three decimals,
# so it carries no venue difference of its own. Whatever it shows here is the
# date gap plus noise.
CONTROL = "BTC-USDT"

# A difference this size or below is not worth reading as a venue effect: it
# is inside the spread of the per-leg estimates the original survey produced
# (0.29-0.65) on one venue alone.
NEGLIGIBLE_BPS = 0.2

# Below this much recorded BloFin data, say so before anything else. Adverse
# selection varies with regime, and a run of under a shift is one regime -
# the Binance side is a whole sampled day, so a short recording is not even
# comparable in span. This does not block the comparison, it labels it: the
# numbers are still worth seeing while the recorder fills in behind them.
MIN_HOURS = 6.0


def binance_symbol(inst_id: str) -> str:
    """BloFin `ADA-USDT` -> Binance `ADAUSDT`. Also `1000BONK-USDT`."""
    return inst_id.replace("-", "")


@dataclass
class Measurement:
    """One venue's answer for one instrument, or why there isn't one."""

    venue: str
    symbol: str
    unavailable: Optional[str] = None

    median_spread_bps: float = float("nan")
    adverse_bps: float = float("nan")
    adverse_stderr: float = float("nan")
    markout_at_fill: float = float("nan")
    fills: int = 0
    span_hours: float = float("nan")

    @property
    def usable(self) -> bool:
        return self.unavailable is None and self.fills > 0


def measure(events: Iterable[dict], *, venue: str, symbol: str,
            quote_interval_ms: int, timeout_ms: int, horizon: float,
            model: str, signals: Sequence[str]) -> Measurement:
    """Run the passive_sim pipeline over one venue's events."""
    horizons = (0.0, horizon)
    market, quotes = collect(events, quote_interval_ms=quote_interval_ms,
                             signals=signals)
    if len(market.book_ts) == 0 or len(quotes) == 0:
        return Measurement(venue, symbol, unavailable="no usable book events")

    quotes = observable(quotes, market, timeout_ms=timeout_ms,
                        max_horizon_ms=int(horizon * 1000))
    if len(quotes) == 0:
        # Every quote needs its fill window AND its markout window inside the
        # data. A recording shorter than timeout + horizon has none.
        return Measurement(
            venue, symbol,
            unavailable=f"recording too short for a {horizon:g}s markout")

    cancel_share = {side: touch_cancel_share(market, side) for side in ("bid", "ask")}
    fills = simulate(market, quotes, timeout_ms=timeout_ms,
                     cancel_share=cancel_share)
    table = markout_table(market, quotes, fills, horizons_s=horizons)
    decay, stderr = adverse_selection(table, horizons, model, horizon)
    mean, _, count = table[model]

    return Measurement(
        venue=venue,
        symbol=symbol,
        median_spread_bps=float(np.median(quotes.spread_bps)),
        adverse_bps=decay,
        adverse_stderr=stderr,
        markout_at_fill=float(mean[0]) if count else float("nan"),
        fills=count,
        span_hours=market.span_seconds / 3600.0,
    )


def difference_in_differences(
    instrument: Tuple[str, Measurement, Measurement],
    control: Tuple[str, Measurement, Measurement],
) -> Tuple[float, float]:
    """The instrument's venue difference NET of the control's.

    The control measures what this comparison reports when there is nothing to
    report: BTC-USDT is tick-bound with an identical spread on both venues, so
    any difference it shows is the date gap, the regime, and whatever else
    separates two samples that are not the same day.

    Subtracting it is the standard use of a control, and it is strictly better
    than the threshold test this replaced. That test compared the control's
    POINT estimate against a fixed bar and ignored its interval entirely -
    which, in a tool whose whole discipline is refusing to read a number
    without one, was the inconsistent step. At 42 minutes the control sat at
    -2.901 [-4.21, -1.59] and vetoed everything; at 16 hours it sits at
    -0.351 [-0.95, +0.25], indistinguishable from zero, and a threshold on
    0.351 would still have vetoed everything.

    Errors add in quadrature: the control and the instrument are measured on
    different instruments and different quotes, so they are independent.
    """
    instrument_difference, instrument_stderr = paired_difference(
        instrument[1], instrument[2])
    control_difference, control_stderr = paired_difference(control[1], control[2])
    return (instrument_difference - control_difference,
            float(np.hypot(instrument_stderr, control_stderr)))


def paired_difference(blofin: Measurement,
                      binance: Measurement) -> Tuple[float, float]:
    """(BloFin adverse selection minus Binance's, standard error).

    The two samples are independent - different venues, different days,
    different quotes - so the errors add in quadrature. Unlike the within-
    instrument decay this is a genuinely unpaired difference, and the interval
    is honest rather than conservative.
    """
    difference = blofin.adverse_bps - binance.adverse_bps
    stderr = float(np.hypot(blofin.adverse_stderr, binance.adverse_stderr))
    return difference, stderr


def report(rows: List[Tuple[str, Measurement, Measurement]], *,
           horizon: float, model: str, blofin_date: Optional[str],
           binance_date: str) -> None:
    print("\n" + "=" * 88)
    print(f"ADVERSE SELECTION BY VENUE  ({model} queue, {horizon:g}s markout)")
    print("=" * 88)
    print(f"  BloFin   {blofin_date or 'all recorded dates'}   (live archive)")
    print(f"  Binance  {binance_date}   (Tardis free sample)")
    print("  Different days. BTC-USDT is the control: it is tick-bound on both "
          "venues with\n  an identical spread, so what it shows here is the "
          "date gap, not the venue.\n")

    usable = [(name, a, b) for name, a, b in rows if a.usable and b.usable]
    missing = [(name, a, b) for name, a, b in rows if not (a.usable and b.usable)]

    if usable:
        print(f"  {'instrument':<14}{'spread B/F':>12}{'spread BN':>11}"
              f"{'adv B/F':>11}{'adv BN':>11}{'difference':>22}")
        print("  " + "-" * 82)
        for name, blofin, binance in usable:
            difference, stderr = paired_difference(blofin, binance)
            low, high = difference - 1.96 * stderr, difference + 1.96 * stderr
            marker = " *" if name == CONTROL else "  "
            print(f"  {name:<14}{blofin.median_spread_bps:>12.3f}"
                  f"{binance.median_spread_bps:>11.3f}"
                  f"{blofin.adverse_bps:>11.3f}{binance.adverse_bps:>11.3f}"
                  f"{difference:>+11.3f} [{low:+.2f},{high:+.2f}]{marker}")
        print("\n  * the control. Positive difference = BloFin selects against "
              "a passive quote\n    harder than Binance does.")
        print(f"\n  {'instrument':<14}{'fills B/F':>11}{'fills BN':>11}"
              f"{'hours B/F':>12}{'hours BN':>11}")
        print("  " + "-" * 59)
        for name, blofin, binance in usable:
            print(f"  {name:<14}{blofin.fills:>11,}{binance.fills:>11,}"
                  f"{blofin.span_hours:>12.1f}{binance.span_hours:>11.1f}")

    if missing:
        print("\n  not compared:")
        for name, blofin, binance in missing:
            reason = blofin.unavailable or binance.unavailable or "no fills"
            side = "BloFin" if not blofin.usable else "Binance"
            print(f"    {name:<14} {side}: {reason}")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)

    if not usable:
        print("  Nothing could be compared. The BloFin side needs a recording "
              "at least\n  `timeout + horizon` long on an instrument the "
              "Binance sample also covers.")
        return

    control = next((row for row in usable if row[0] == CONTROL), None)
    if control is None:
        print(f"  NO CONTROL. {CONTROL} was not compared, so there is nothing "
              "separating a\n  venue difference from a date difference. Treat "
              "everything above as suggestive\n  and re-run once the control "
              "is available.")
        return

    thinnest = min(row[1].span_hours for row in usable)
    if thinnest < MIN_HOURS:
        print(f"  THE BLOFIN SIDE IS {thinnest:.1f} HOURS. Everything above is "
              f"provisional.")
        print(f"  Adverse selection varies with regime, and {thinnest:.1f} "
              "hours is one regime. The\n  Binance side is a full sampled day, "
              "so the two are not comparable in span\n  either. Re-run at "
              f"{MIN_HOURS:.0f}+ hours before reading any of it as a venue "
              "difference.\n")

    control_difference, control_stderr = paired_difference(control[1], control[2])
    print(f"  Control ({CONTROL}): {control_difference:+.3f} bps "
          f"+-{1.96 * control_stderr:.3f}")

    control_significant = abs(control_difference) > 1.96 * control_stderr
    if control_significant:
        print("  The control's interval EXCLUDES zero, so there is a real "
              "systematic bias\n  between these two samples. Subtracting it is "
              "the only way to read the rest.")
    else:
        print("  The control's interval includes zero: no measurable bias "
              "between the samples.")

    others = [row for row in usable if row[0] != CONTROL]
    if not others:
        print("\n  Control only. Nothing to compare it against yet.")
        return

    print("\n  Each instrument NET of the control (difference-in-differences), "
          "which is what\n  isolates the venue from the dates:\n")
    print(f"  {'instrument':<14}{'raw diff':>10}{'net of control':>28}")
    print("  " + "-" * 54)

    significant = []
    for row in others:
        name = row[0]
        raw, _ = paired_difference(row[1], row[2])
        effect, stderr = difference_in_differences(row, control)
        low, high = effect - 1.96 * stderr, effect + 1.96 * stderr
        clears = abs(effect) > 1.96 * stderr and abs(effect) > NEGLIGIBLE_BPS
        print(f"  {name:<14}{raw:>+10.3f}{effect:>+16.3f} "
              f"[{low:+.2f},{high:+.2f}]{'  *' if clears else ''}")
        if clears:
            significant.append((name, effect))
    print("\n  * clears its own interval and the 0.2 bps floor below which a "
          "difference is\n    smaller than the spread of this quantity on one "
          "venue alone.\n")

    if not significant:
        print("  NO INSTRUMENT SHOWS A VENUE EFFECT THAT CLEARS ITS OWN INTERVAL.")
        print("  On this evidence the 0.5 bps/leg assumption carried over from "
              "Binance is not\n  contradicted, which is the most useful thing "
              "a null result here can say -\n  every gate in the repo depends "
              "on it.")
        return

    worse = [name for name, difference in significant if difference > 0]
    better = [name for name, difference in significant if difference < 0]
    if worse:
        print(f"  BloFin selects HARDER on: {', '.join(worse)}")
        print("  The wider spread is being paid for. Re-run the gate with the "
              "measured number\n  rather than 0.5 bps/leg before treating any "
              "of these as tradeable.")
    if better:
        print(f"  BloFin selects LESS on: {', '.join(better)}")
        print("  Wider spread AND cheaper adverse selection is the combination "
              "the passive\n  branch has been looking for. Confirm it on more "
              "days before believing it -\n  this is one sample against one "
              "sample.")


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instruments", default=",".join(DEFAULT_INSTRUMENTS),
                        help="BloFin instrument ids, comma separated.")
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data")
    parser.add_argument("--blofin-date", default=None,
                        help="YYYY-MM-DD; omit for every recorded date.")
    parser.add_argument("--binance-date", default=latest_free_sample_day(),
                        help="Tardis sample day. The free sample is the 1st "
                             "of a month; anything else needs TARDIS_API_KEY.")
    parser.add_argument("--exchange", default="binance-futures")
    parser.add_argument("--depth", type=int, default=5, choices=(5, 25))
    parser.add_argument("--hours", type=float, default=2.0,
                        help="Limit the Binance parse. The BloFin side uses "
                             "whatever has been recorded.")
    parser.add_argument("--horizon", type=float, default=60.0,
                        help="Markout horizon the decay is measured over.")
    parser.add_argument("--model", default="pessimistic", choices=MODELS,
                        help="Queue assumption. Adverse selection is largest "
                             "at the back of the queue, which is where a fill "
                             "arrives as the level is swept.")
    parser.add_argument("--quote-interval", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--api-key", default=os.environ.get("TARDIS_API_KEY"))
    args = parser.parse_args(argv)

    instruments = [part.strip() for part in args.instruments.split(",")
                   if part.strip()]
    cache = args.cache or repo_root / "data" / "tardis" / "raw"
    timeout_ms = int(args.timeout * 1000)
    signals: Tuple[str, ...] = ("obi_1",)
    available = instrument_dirs(args.data_dir)

    rows: List[Tuple[str, Measurement, Measurement]] = []
    for index, inst_id in enumerate(instruments, start=1):
        symbol = binance_symbol(inst_id)
        print(f"\n[{index}/{len(instruments)}] {inst_id}")

        raw_dir = args.data_dir / inst_id / "raw"
        if inst_id not in available or not raw_dir.is_dir():
            blofin = Measurement("blofin", inst_id,
                                 unavailable=f"nothing recorded at {raw_dir}")
        else:
            print(f"  BloFin   {raw_dir}")
            try:
                blofin = measure(
                    raw_archive_events(raw_dir, args.blofin_date),
                    venue="blofin", symbol=inst_id,
                    quote_interval_ms=args.quote_interval,
                    timeout_ms=timeout_ms, horizon=args.horizon,
                    model=args.model, signals=signals)
            except SystemExit as exc:
                blofin = Measurement("blofin", inst_id, unavailable=str(exc))

        print(f"  Binance  {symbol} {args.binance_date}")
        try:
            binance = measure(
                tardis_events(cache, args.exchange, symbol, args.binance_date,
                              args.depth, args.hours, args.api_key),
                venue="binance", symbol=symbol,
                quote_interval_ms=args.quote_interval,
                timeout_ms=timeout_ms, horizon=args.horizon,
                model=args.model, signals=signals)
        except (SystemExit, Exception) as exc:  # noqa: BLE001 - reported, not raised
            binance = Measurement("binance", symbol, unavailable=str(exc))

        rows.append((inst_id, blofin, binance))

    report(rows, horizon=args.horizon, model=args.model,
           blofin_date=args.blofin_date, binance_date=args.binance_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
