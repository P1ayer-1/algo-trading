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
python backend\analysis\check_features.py --horizon 5 --cost-bps 6
```

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

## binance_import.py — test the pipeline today, without waiting

```
python backend\analysis\binance_import.py --date 2026-09-01 --hours 2
python backend\analysis\check_features.py --data-dir data\binance --horizon 5 --cost-bps 6
```

Downloads free historical USDⓈ-M futures data from
`https://data.binance.vision` (no API key, no account) and runs it through the
**same** OrderBook / TradeTape / FeatureEngine / FeatureRecorder as the live
bot, producing a feature CSV in the identical format.

Two datasets per day:
- `bookTicker` — every change to the best bid/ask, with sizes
- `aggTrades` — every trade, with the aggressor side

**Runtime:** ~6k events/sec, so a full BTCUSDT day (20–40M updates) takes
roughly an hour. `--hours 2` gives ~30k rows in about five minutes — start
there.

### What it does and doesn't tell you

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

### Degraded columns — bookTicker is top-of-book only

| Column | Status |
|---|---|
| `obi_1`, `ofi_*`, `tfi_*`, `microprice`, `spread_bps`, `ret_*`, `rv_*` | fully valid |
| `obi_5`, `obi_20` | **identical to `obi_1`** — no depth in this feed |
| `bid_depth_20`, `ask_depth_20` | best-level size only |
| `funding_rate` | always 0 — not in this dataset |

OFI is unaffected, which matters: it's defined purely on the touch and has the
strongest theoretical basis of the features here. The importer prints this
warning on every run.

### If the format ever changes

The parser reads column names from the header when present, falls back to
documented positions when absent, auto-detects millisecond vs microsecond
timestamps, and hard-validates the first 50 rows (bid < ask, plausible prices,
plausible epoch). A layout change raises `SchemaError` with a clear message
rather than producing a plausible-looking, wrong dataset.

One convention worth knowing, since getting it backwards inverts every flow
signal: Binance reports `is_buyer_maker`, which is the **inverse** of BloFin's
`side`. `is_buyer_maker=true` means the buyer was passive, so the *aggressor
was a seller*. The importer inverts it; there are tests pinning both
directions.

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
| Feature CSV @ 250ms | ~160 MB | ~4.8 GB | ~58 GB |
| **Combined** | **~260 MB** | **~8 GB** | **~95 GB** |

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
