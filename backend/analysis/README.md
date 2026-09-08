# backend/analysis — evaluation and data tooling

Everything here reads recorded data. Nothing here touches the exchange.

Your interpreter:

```
D:\Micromamba\micromambaenv\envs\crypto_bot\python.exe
```

Shortened to `python` below. Run from the repo root.

Only `numpy` is required. `pyarrow`/`duckdb` are optional and only for
`compact.py`.

---

## check_features.py — does any of this predict anything?

```
python backend\analysis\check_features.py --horizon 900
```

`--cost-bps` now defaults to `config.ROUND_TRIP_COST_BPS` (10bps = taker on
both sides at VIP 1), so it no longer has to be passed by hand. Pass
`--cost-bps 1.2` to see the maker-only case — but only once a passive
execution engine exists and its fill rate has been measured.

**This is a gate, not a formality.** Run it before writing a line of model
code. If the features carry no out-of-sample information, no amount of
LightGBM will manufacture an edge, and finding out here costs an afternoon
instead of a month.

It reports, in order:

1. **Data summary** — span, sampling rate, label balance, and the *effective*
   sample size.
2. **Information coefficients** — Pearson and Spearman per feature, with
   t-statistics on the effective N.
3. **Leakage checks** — loud, because a leak looks exactly like success.
4. **Out-of-sample test** — time-ordered, purged split; logistic baseline.
5. **Economic test** — the only one that decides anything.

### Three things it does that most home-made checks don't

**Effective sample size.** At 250ms sampling with a 5s horizon, adjacent rows
share 4.75s of their forward window. Twenty consecutive rows are roughly *one*
independent observation. A t-statistic computed on 350,000 rows when the real
N is 17,500 overstates significance by ~4.5x. This is the single most common
way to convince yourself noise is signal, and the script corrects for it
everywhere.

**Purged split.** Never shuffled, and it drops a gap of one horizon between
train and test. Without the gap the last training rows carry labels reaching
forward into the test period, leaking the answer.

**The economic test.** Accuracy is not profitability. A 55%-accurate model
predicting 1bps moves loses money against 6bps of costs, every time. The
script reports mean forward return by predicted-probability decile and
compares the top decile against `--cost-bps`. Set that from your *measured*
round-trip cost.

### Reading the verdict

| Verdict | Meaning |
|---|---|
| `NO SIGNAL` | AUC ≈ 0.5. Don't build a model. Collect more data, try a shorter horizon, or add features. |
| `STATISTICAL SIGNAL, BUT NOT TRADEABLE` | Direction is predictable but moves are smaller than costs. **The most common real outcome** — a genuine result, not a failure. Next step is cheaper execution (maker-only), not a bigger model. |
| `PROMISING` | Worth building properly. Confirm on a *different day* before believing it. |

Calibration: single-feature |IC| of **0.01–0.06** is a realistic, usable
signal at these horizons. |IC| > 0.20 is nearly always a bug, and the script
says so.

The script is tested against data with a deliberately planted edge, against a
realistic weak edge, and against pure noise — a check that can only ever say
"promising" is worse than no check.

---

## bars_import.py — the actual training set

```
python backend\analysis\bars_import.py
python backend\analysis\bars_import.py --start 2023-01-01 --end 2026-09-06
python backend\analysis\check_features.py --data-dir data\bars\BTCUSDT-... --horizon 900
```

The two tick importers below reconstruct an order book, because the features
they feed operate on a horizon of seconds. This one does not, because the
horizons this project has to trade are minutes, and at those horizons the tick
data is both unnecessary and the expensive part: tick-level L2 history is a
$1k/month product, and the one free source of it (Binance `bookTicker`) stopped
being published on 2024-03-30.

Everything else on `data.binance.vision` is still current, still free, and
still needs no API key:

| dataset | files | earliest | per day | what it carries |
|---|---|---|---|---|
| `klines/1m` | 2,442 | 2019-12-31 | 0.06 MB | OHLCV, trade count, taker buy volume |
| `metrics` | 2,197 | 2020-09-01 | 0.01 MB | open interest, long/short ratios, taker ratio |
| `premiumIndexKlines` | 2,443 | 2019-12-24 | small | premium / basis, i.e. funding |
| `bookDepth` | 1,342 | 2023-01-01 | 0.49 MB | depth in bands around mid (opt-in) |

Defaults to the last 365 days at one row every five minutes, writing a feature
CSV in exactly the recorder's format — so `check_features.py`, `stats.py` and
`compact.py` all work on it unchanged. Monthly archives are used wherever a
whole calendar month is covered, which turns 2,442 requests into 80.

**What it is worth, in the only unit that matters.** Six hours of live
recording gives 16 independent observations at a 900s horizon. Forty days from
this importer gives 3,838. The full archive gives roughly 234,000.

### The lag that is not in Binance's documentation

A `metrics` row timestamped T describes the window **[T, T+5min)** — not the
window ending at T. Measured over 5,758 paired samples
(2026-07-01..2026-07-20), `sum_taker_long_short_vol_ratio` has a Spearman of
**+0.44** against the price move over the *next* five minutes and +0.25 against
the *previous* five.

An ordinary as-of join at T therefore hands the model five minutes of the
future. The first run of this importer did exactly that, and it did not look
like a bug — it looked like a discovery: `check_features.py` reported the
feature at IC +0.25 and flagged it `SUSPICIOUS`, which is the same thing a real
edge would look like to someone who wanted one.

Every metrics lookup is now taken as-of `anchor - 5min`, and the lag is applied
to the whole dataset rather than the one column it was proved on, because all
eight share a `create_time` and the others cannot be verified independently.
`test_bars_import.py` pins this.

### bookDepth is opt-in, and should stay that way until someone checks it

Its rows are cumulative depth in percentage bands either side of mid (±0.2, ±1,
±2, ±3, ±4, ±5), roughly every 30 seconds. On the day sampled while writing
this, the implied average price of the **positive**-percentage side
(`notional / depth`) came out *below* the contemporaneous mid — which cannot be
true of resting asks above the mid.

So `--with-depth` prints that diagnostic on every run, and the features are
named neutrally (`depth_imb_*` is the negative-percentage side minus the
positive one) rather than `bid`/`ask`. Read the diagnostic before believing
anything they tell you.

### What it does not tell you

The same caveat as the tick importers: this is Binance, not BloFin. Different
venue, different participants, different fee schedule. Binance BTCUSDT perp is
among the most heavily arbitraged instruments in existence, so an edge visible
there is quite likely present on a less efficient venue — while an absence
there is weak evidence either way. Confirm on BloFin data before trading.

---

## Two importers — test the pipeline today, without waiting

Both download free historical data and run it through the **same** OrderBook /
TradeTape / FeatureEngine / FeatureRecorder as the live bot, producing a
feature CSV in the identical format. They share `importer_core.py`, so neither
can drift from the other or from the live path.

**Reach for `tardis_import.py` first.** It has real depth and current data.

| | `tardis_import.py` | `binance_import.py` |
|---|---|---|
| Source | Tardis.dev free samples | data.binance.vision |
| Venue | binance-futures (or bybit, bitget, okex-swap...) | Binance USDT-M |
| Dates | **1st of any month**, to the present | **2023-05-16 … 2024-03-30 only** |
| Book | 25 levels per side | top of book only |
| Book rate | ~27/sec | ~470/sec |
| Trades | ~3.6M/day | ~1.5M/day |
| Full day | ~15–20 min | ~1 hour |
| Account | none | none |

### Why there are two

Binance **stopped publishing `bookTicker`** after 2024-03-30 — daily and
monthly both. `aggTrades`, `klines` and `bookDepth` are still current, but
nothing on `data.binance.vision` carries top of book after that date, and spot
never had it at all. So `binance_import.py` can only ever run on data that is
now years stale. It is kept because it is the only free source with
touch-resolution book updates, which matters for OFI.

Tardis gives the 1st of every month away free, for every exchange and every
data type, with no account and no API key. Every other day returns 401 unless
you set `--api-key` / `$TARDIS_API_KEY`. Both importers check the date *before*
downloading, so a wrong one costs a second rather than a 90 MB transfer.

---

### tardis_import.py — recent data, with real depth

```
python backend\analysis\tardis_import.py --date 2026-09-01 --hours 2
python backend\analysis\check_features.py ^
    --data-dir data\tardis\binance-futures-BTCUSDT-2026-09-01 ^
    --horizon 900
```

Two datasets per day: `book_snapshot_25` (25 levels per side, every book
change) and `trades` (every trade, with the aggressor side).

**This fixes the degraded columns.** `obi_5`, `obi_20`, `bid_depth_20` and
`ask_depth_20` are all fully valid here — under `binance_import.py` `obi_5`
and `obi_20` are literally copies of `obi_1`. Only `funding_rate` is still
always 0; it is in neither dataset.

**The trade-off is time resolution at the touch.** Binance's archived
bookTicker fired on every change to the best bid/ask (~470/sec). Tardis
snapshots come from the depth stream at ~27/sec (median gap 26ms). OFI is
defined purely on the touch, so it sees less churn here. At 250ms recorder
sampling that is still ~7 book updates per row, which is workable — but if OFI
specifically is what you are testing, run both and compare.

`--exchange` takes any Tardis venue id, so `--exchange bybit` or
`--exchange bitget-futures` works with no code change. **BloFin is not on
Tardis**, so this is still a proxy, not the real thing.

#### The sign convention

Tardis normalises `side` to the **aggressor** (the liquidity taker) — the same
convention BloFin uses and the same thing `TradeTape` expects, so this importer
passes it through unchanged. `binance_import.py` must do the opposite, because
Binance reports `is_buyer_maker`, which is the inverse. There are tests pinning
both directions.

---

### binance_import.py — touch-resolution, but only up to 2024-03-30

```
python backend\analysis\binance_import.py --date 2024-03-01 --hours 2
python backend\analysis\check_features.py ^
    --data-dir data\binance\BTCUSDT-2024-03-01 --horizon 900
```

Two datasets per day: `bookTicker` (every change to the best bid/ask, with
sizes) and `aggTrades` (every trade). A full BTCUSDT day is 20–40M updates and
takes roughly an hour; `--hours 2` gives ~30k rows in about five minutes.

#### Degraded columns — bookTicker is top-of-book only

| Column | Status |
|---|---|
| `obi_1`, `ofi_*`, `tfi_*`, `microprice`, `spread_bps`, `ret_*`, `rv_*` | fully valid |
| `obi_5`, `obi_20` | **identical to `obi_1`** — no depth in this feed |
| `bid_depth_20`, `ask_depth_20` | best-level size only |
| `funding_rate` | always 0 — not in this dataset |

Use `tardis_import.py` if you need those columns.

---

### One directory per import — and why

Each run writes to its own directory by default:

```
data\tardis\binance-futures-BTCUSDT-2026-09-01
data\binance\BTCUSDT-2024-03-01
```

This is not tidiness. `FeatureRecorder` names its file `features-<today>.csv`
by **wall-clock** date and opens it in **append** mode. That is correct for
live recording, where wall clock and market time are the same thing, and wrong
for importing, where they are not — two imports run on the same afternoon
would land in one file, ordered by when you ran them rather than by market
time. `check_features.py` would then do a time-ordered, purged split on rows
that are not in time order and report a number that means nothing, without
complaining. Both importers refuse to start if the output directory already
holds a feature CSV.

### What these do and don't tell you

Answers: does the pipeline work end to end on real data? Do OBI / OFI /
trade-flow carry predictive information at all? How large is it, and does it
survive costs?

Does **not** answer: whether an edge exists *on BloFin*. Different venue, fees,
tick size, and participants.

The direction of the bias is worth knowing. Binance BTCUSDT perp is among the
most liquid and most heavily arbitraged instruments anywhere, so edges there
are competed down hard. A smaller venue like BloFin is generally *less*
efficient — so a signal visible on Binance is quite likely present on BloFin
too, while a signal absent on Binance is weak evidence either way.

### If a format ever changes

Both parsers read column names from the header, hard-validate the first 50
rows (bid < ask, levels correctly sorted, plausible prices, plausible epoch),
and auto-detect millisecond vs microsecond timestamps. A layout change raises
`SchemaError` with a clear message rather than producing a plausible-looking,
wrong dataset.

---

## cross_sectional_import.py — which coin outperforms, not where BTC goes

```
python backend\analysis\cross_sectional_import.py
python backend\analysis\check_features.py --data-dir data\cross\... --horizon 900
python backend\analysis\train_model.py    --data-dir data\cross\... --horizon 900
```

`bars_import.py` asks "will BTC be higher in 15 minutes?" A year of data and
two model classes said no, with intervals tight enough to believe it. That is
the expected answer — BTC-USDT perp is among the most arbitraged instruments in
existence.

This asks a different question. The label is each symbol's forward return
**minus the cross-sectional mean**, so the market factor is subtracted out
rather than predicted, and what remains is dispersion across the ten most
liquid USDT-M perpetuals. Features are the same ones converted to
cross-sectional ranks in `[-1, +1]`: a raw `ret_60m` of +40bps says little,
but being the strongest of ten majors over the last hour is a statement about
relative positioning that is scale-free and comparable across regimes.

Ranks rather than z-scores, deliberately — a z-score is dominated by whichever
coin had an outlier that minute, and this data has an outlier most minutes.

### Two assumptions it breaks downstream, and how they are fixed

Rows become one per (timestamp, symbol). Both fixes live in
`check_features.panel_geometry()` and are no-ops for single-asset files.

**The purge gap is measured in rows.** With ten symbols per timestamp, a gap
computed from `span / (rows - 1)` is ten times too short *in time*, so training
rows end up inside the test period's forward window. The gap is now
`horizon / interval x rows_per_timestamp`, with the interval taken from
distinct timestamps.

**Bootstrap resampling must draw whole timestamps.** Ten symbols observed at
one instant are ten correlated measurements of one moment. Resampling them as
independent rows reports an interval roughly `sqrt(10)` too narrow.
`train_model.py` switches to a cluster bootstrap automatically when it sees
duplicate timestamps.

### The cost of not having to know where the market is going

A cross-sectional position is two legs, so it pays two round trips: 2.4bps
maker or 20bps taker at VIP 1. `--threshold-bps` defaults to the latter. The
edge has to clear double what a directional trade needed.

Note also that `--sample-minutes` defaults to 15 rather than 5: ten symbols
multiply the row count, and every tool downstream loads the file into memory.

---

## train_model.py — LightGBM, and whether it beats nothing

```
python backend\analysis\train_model.py --data-dir data\bars\BTCUSDT-... --horizon 900
```

Roadmap step 7. `check_features.py` fits a logistic regression, which is the
right baseline and the wrong ceiling — it cannot represent an interaction, and
these features are plausibly conditional on one another. This fits LightGBM on
the same matrix and asks whether the non-linearity is actually there.

Requires `lightgbm` and `scipy`; everything else here runs on numpy alone.

### Four things it does that a naive `lgb.train()` does not

**A three-way split, purged twice.** Train / validation / test, with a
horizon-sized gap either side of validation. Early stopping *reads* the
validation block, so validation is not out-of-sample in any useful sense. Only
the final block is untouched, and only its numbers are reported.

**The test set is decimated to non-overlapping rows.** At a 900s horizon
sampled every 300s, three consecutive rows share most of their forward window.
20,960 test rows are really 6,987 observations, and a mean over the full set
implies three times the confidence the data supports.

**A shuffled-label control, run five times.** The identical pipeline retrained
on shuffled labels is the noise floor of this procedure on this data. It is run
five times rather than once because one control is one draw from a
distribution: on this repo's own data, four control seeds gave +0.34, +0.32,
−0.15 and +0.93 bps, and in an earlier single-control run the control *beat*
the model. A single control had made the same model look like a result.

**A paired bootstrap.** The verdict rests on
`top_decile(model) − top_decile(control)` resampled on the *same* rows. Both
are scored on one test set, so most of each interval is the same shared
uncertainty — which fortnight the test block landed on, which few large moves
fell in the top decile. Pairing cancels it. Comparing one model's point
estimate against the other's interval answers a harsher question and would
reject a real difference whenever the test set is small, which is exactly when
it matters.

### Reading the verdict

| Verdict | Meaning |
|---|---|
| `NOT DISTINGUISHABLE FROM ZERO` | The top-decile interval includes zero. No edge to cost, let alone trade. |
| `NOT SEPARABLE FROM SHUFFLED LABELS` | There is a number, but the same pipeline produces numbers that size from data with no signal in it. |
| `REAL BUT NOT TRADEABLE` | Clears zero and clears the control, but not the cheapest round trip. Cheaper execution or a longer horizon. |
| `PROMISING` | Clears all three. Confirm on a different date range, then paper-trade. Do not size it from the backtest. |

The seed spread printed under the table is worth as much as the verdict: a
model whose top decile swings from +1.06 to +1.66 across seeds is reporting
seed noise in its third digit, and any decision that depends on that digit is
not supported.

---

## passive_sim.py — what a passive quote actually earns

```
python backend\analysis\passive_sim.py --date 2026-09-01 --hours 2
python backend\analysis\passive_sim.py --source raw --date 2026-09-07
python backend\analysis\passive_sim.py --date 2026-09-01 --hours 6 ^
    --signals obi_1,tfi_5s --decision-horizon 5
```

Roadmap step 9, and the first tool here that measures *execution* rather than
prediction. Every result in this repo has died on cost rather than on
forecasting, and the one effect that survived — short-horizon cross-sectional
reversal — is the return to providing liquidity to whatever just moved. Both
point at the same open question: what does a resting order earn, once the
people who fill it are done selecting against you?

It places a hypothetical passive quote on both sides of the touch every
`--quote-interval` ms, leaves it there until it fills or `--timeout` elapses,
and reports:

* **fill rate** — how often a quote trades at all,
* **the markout curve** — where the mid sits 0s / 1s / 5s / … after the fill,
  signed so positive is money. At 0s it is the half spread you captured; its
  decay after that is adverse selection, priced in basis points.

Unconditional first, then conditional on a signal. That order is deliberate: a
conditional number with nothing to compare it against is not a number.

### The queue problem, and why this brackets instead of modelling

You cannot see how many orders sit ahead of yours at a price, and you cannot
see how many of them cancel. Neither is in any market data feed at any price.
So the simulator does not model queue position — it runs the identical markout
machinery under two rules that are wrong in known, opposite directions:

| | rule |
|---|---|
| **Pessimistic** | You join behind the entire visible size `Q` and fill only once cumulative same-side aggressor volume at that price exceeds `Q`. Nobody ahead of you cancels. |
| **Optimistic** | You fill the moment any trade occurs at your price. |

Truth is between them, and closer to the optimistic end than the pessimistic
bound suggests, because real queues shrink by cancellation as well as by
trading and the pessimistic rule counts only the trading.

**Plan with the pessimistic number and treat the gap as your uncertainty.**
Every figure in the report is printed as three columns for exactly that
reason.

Both rules are the same function with a different amount of volume to clear —
optimistic is the pessimistic rule with `Q = 0`. There is no second code path
that could drift from the first, and a property test asserts the bracket never
inverts on random tapes.

### Locating the truth inside the bracket, without a queue model

*How far* toward the optimistic end is not a matter of opinion. Queue position
is unobservable; **cancellation is not**. Between two book updates at an
unchanged touch price, the size fell by more than the volume that printed
there, and the excess was cancelled. `touch_cancel_share` measures exactly
that.

On the Tardis 2026-09-01 BTCUSDT sample it is **92.2% on the bid and 90.8% on
the ask** — nine tenths of the queue ahead of you evaporates rather than
trading. That is the number that says the pessimistic bound is very
pessimistic indeed.

A third column, `cancel-adj`, is the pessimistic rule with `Q` scaled by
`1 - c`. It is an interpolation driven by that measurement, not a third model,
and it sits inside the bracket by construction. The measurement is biased
*down* — intervals where someone adds size hide removals inside a smaller net
change — which makes `cancel-adj` conservative rather than optimistic. That is
the safe direction to be wrong in.

### Read the arithmetic gate before anything else

The report prints the median half spread next to the maker round trip before
any simulation result, because on a tight instrument that one comparison
settles the question:

```
  median half spread               0.006 bps  <- the most a passive fill can capture
  round trip, both legs passive    1.200 bps  (0.60 per leg)
  a flawless fill, marked out instantly, earns -1.194 bps
```

Binance BTCUSDT perp trades one tick wide almost always. A tick is $0.10 on a
~$110k instrument, so the half spread is **0.006 bps against a 1.2 bps maker
round trip — roughly 200x too small**. A perfect fill, at the front of the
queue, marked out instantly, against a counterparty who knows nothing, still
loses. No signal, queue assumption or horizon repairs that, and the verdict
says so rather than burying it under a table.

That gate is worth knowing before spending a week on a fill model.

### What it found

Six hours of BTCUSDT on 2026-09-01, quoting every second with a 60s timeout
(20,665 quote pairs, 590k book updates, 592k trades):

| | optimistic | cancel-adj | pessimistic |
|---|---|---|---|
| fill rate, bid | 96.7% | 84.6% | 63.2% |
| fill rate, ask | 97.4% | 86.1% | 67.7% |
| median wait, bid | 0.4s | 2.9s | 10.4s |
| markout @ 0s | +0.019 | +0.005 | **−0.400** |
| markout @ 1s | −0.079 | −0.416 | −1.161 |
| markout @ 60s | −0.110 ±0.155 | −0.607 ±0.157 | −1.489 ±0.159 |
| **net of fees @ 60s** | **−1.310** | **−1.807** | **−2.689** |

Three things in that table are worth more than the verdict.

**The bracket is wide, and wide in the way that matters.** Fill rates run
63%–97%, and the 60s markout runs −1.49 to −0.11 bps. Anyone quoting a single
fill-rate number for this instrument is quoting an assumption, not a
measurement — which is the whole reason for the two bounds.

**Markout is already −0.400 bps at the instant of the pessimistic fill.** The
half spread is 0.006. So by the time enough volume had printed to clear a
whole queue, the mid had moved through the quote by sixty times what providing
liquidity paid. Waiting at the back of the queue does not buy a fill at a good
price — it buys a fill precisely when the level is being swept. That single
number is the clearest statement in this repo of what queue position is worth.

**Adverse selection saturates in about a second.** Optimistic markout goes
−0.079 at 1s, −0.109 at 10s, −0.110 at 60s: essentially all the damage is
immediate and the rest is drift. The standard errors say the same thing from
the other side — ±0.155 bps at 60s and ±0.805 at 300s, against point estimates
an order of magnitude smaller. **Nothing past ~10s in this table is measured**,
and the errors are computed on the *effective* sample size for the reason
`check_features.py` uses it: consecutive fills share nearly all of their
markout window. Prefer `--decision-horizon 5`.

### The conditional section picks out of sample, on purpose

Choosing the best signal bucket on the rows you then report is the oldest way
to manufacture a backtest — with five buckets and a noisy metric, one of them
looks good whether or not anything is there. So the bucket edges are fitted on
the first `--train-fraction` of the session, the best bucket is chosen there,
and the number reported is what that choice earned *afterwards*, printed
beside the unconditional figure over the same rows.

It survived, and it is far too small to matter:

| signal | picked | in-sample | out-of-sample | unconditional | fill rate |
|---|---|---|---|---|---|
| `obi_1` | bucket 3 | −1.362 | **−1.207** | −1.521 | 67.8% vs 63.8% |
| `ofi_1s` | bucket 1 | −1.448 | −1.397 | −1.521 | 64.6% vs 63.8% |
| `ret_5s` | bucket 1 | −1.421 | −1.430 | −1.521 | 64.3% vs 63.8% |
| `tfi_5s` | bucket 2 | −1.332 | −1.467 | −1.521 | 63.8% vs 63.8% |

All four beat unconditional quoting out of sample, by 0.05 to 0.31 bps, and
`obi_1` did it while *raising* the fill rate — so it is not the usual failure
mode where a signal wins by quoting into states nobody trades against. The
best of them, quoting only when the book is balanced, is a genuine effect of
about a third of a basis point.

Against a 1.2 bps fee and a 1.5 bps markout deficit, a third of a basis point
is a rounding error. The honest summary is that conditioning works and does
not remotely close the gap.

### Reading the verdict

| Verdict | Meaning |
|---|---|
| `THE SPREAD NEVER COVERED THE FEE` | Arithmetic, not execution. The half spread is below the maker round trip, so a flawless fill loses. Look at a wider-spread symbol or venue, or a tier with a maker rebate. |
| `ADVERSE SELECTION EXCEEDS THE SPREAD` | The spread would have covered the fee, but whoever fills you knows where price is going. No queue assumption rescues it. |
| `DEPENDS ENTIRELY ON QUEUE POSITION` | The bracket straddles zero. This is the case where queue position *is* the strategy — a late passive fill is a different trade from an early one. |
| `PASSIVE ENTRY PAYS, EVEN AT THE BACK OF THE QUEUE` | Clears cost under the pessimistic rule. Confirm on another date, size from the pessimistic number, treat the rest as headroom. |

### Options worth knowing

| flag | default | note |
|---|---|---|
| `--source` | `tardis` | `raw` reads `data/raw/` — the real venue, through the same reader `replay.py` uses. |
| `--quote-interval` | `1000` ms | Quotes overlap whenever this is below `--timeout`. The effective-sample-size correction accounts for it; the raw fill counts do not. |
| `--timeout` | `60` s | How long a quote rests. Longer raises the fill rate and worsens the markout — both bounds move together. |
| `--decision-horizon` | `60` s | Horizon the verdict is taken at. On this data anything past ~10s is inside the noise. |
| `--conditional-model` | `pessimistic` | Queue rule the conditional table uses — the one you should plan with. |
| `--hours` | all | Both sources parse into memory, as the importers do. Start with 2; six hours is ~10 minutes and ~2 GB. |

### What it does not model

Your own order changes the book, and nothing here accounts for that: the quote
is assumed small enough not to matter to the queue and large enough to matter
to you. It rests at a fixed price rather than being requoted as the touch
moves, which is the conservative choice — a managed quote fills more often and
is selected against harder.

And with `--source tardis` this is Binance, not BloFin. Fill rates are a
property of one venue's queue and do not transfer. Binance BTCUSDT perp is
also close to the worst case for this question: the most arbitraged perpetual
in existence, quoted one tick wide. A less efficient venue, or a symbol whose
spread is several ticks, is where the arithmetic gate above could plausibly
come out the other way — and testing that costs one `--symbol` flag.

The BloFin answer needs `--source raw` and enough recorded hours to be worth
reading, which is one more argument for leaving the recorder running.

---

## replay.py — rebuild features from the raw archive

```
python backend\analysis\replay.py --date 2026-09-06
python backend\analysis\replay.py --sample-ms 100 --horizons 0.5,2,10 --out data\replayed
```

Regenerates a feature CSV from `data/raw/`, using the **same** `OrderBook`,
`TradeTape` and `FeatureEngine` as the live path — so replayed features are
identical to what the bot computes live (there's a test pinning this).

This is what makes the raw archive worth keeping: add a feature or change a
horizon, replay everything you've ever recorded, and get a new dataset in
minutes instead of waiting days to collect one.

---

## compact.py — Parquet conversion and storage report

```
python backend\analysis\compact.py --report        # sizes only
python backend\analysis\compact.py                # convert to Parquet
```

**You probably don't need this yet.** CSV plus numpy handles a few million
rows fine. Run it when loading starts to annoy you — past ~10M rows.

Then query with DuckDB, no server required:

```python
import duckdb
duckdb.sql("SELECT AVG(fwd_ret_bps_5s) FROM 'data/parquet/*.parquet' "
           "WHERE obi_1 > 0.5").show()
```

---

## Storage: measured, not guessed

| What | Per day | Per month | Per year |
|---|---|---|---|
| Raw archive (gzipped) | ~100 MB | ~3 GB | ~37 GB |
| Feature CSV @ 1s (default) | ~40 MB | ~1.2 GB | ~15 GB |
| **Combined** | **~140 MB** | **~4.2 GB** | **~52 GB** |

Feature CSV scales linearly with sample rate: 100ms ≈ 400 MB/day, 1s ≈ 40
MB/day. Measured at 461 bytes/row over 40,000 real rows.

### Why files and not a database

The workload is append-only with a single writer, and reads are full
sequential scans for training. There are no updates, no deletes, no
concurrent writers, and no transactions. A relational database would add
operational overhead and ACID guarantees you don't need, while being *slower*
at the full-table scans that are the actual access pattern.

If you want SQL, DuckDB over Parquet gives you it with nothing to install,
run, back up, or keep alive. A real server database (ClickHouse is the right
one for this shape of data) only starts to pay off across many instruments and
billions of rows — worth revisiting then, not now.

### The one rule about deleting things

**The raw archive is irreplaceable. The feature CSVs are not.**

Feature CSVs can be regenerated from raw at any time with `replay.py`, so if
you need space, delete those. Raw events cannot be recovered once gone — that
market moment is over.
