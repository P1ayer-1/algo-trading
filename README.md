# Crypto Bot

Intended goal: a high-frequency, high-leverage crypto trading bot that uses
technical indicators to decide entries/exits and keeps liquidation levels at a
comfortable distance to manage risk.

## Current state (be honest with yourself here)

Right now this repo does **not** trade anything. It cannot: there is no code
path that sends an order to an exchange. What exists is:

**A live chart viewer for BloFin**
- BTC-USDT candles and the live ticker from BloFin (REST + public websocket).
- Simple swing-based support/resistance levels from recent candles.
- A local webpage with a live candlestick chart and a support/resistance panel.

**A market-microstructure data stack** (`backend/trading/`)
- A maintained L2 order book from BloFin's `books` channel, with sequence-gap
  detection and automatic resync.
- A rolling trade tape using the exchange's aggressor side.
- A feature engine: order-book imbalance, order flow imbalance (Cont et al.),
  microprice, spread, multi-horizon returns, realised volatility, funding.
- A recorder that writes those features to CSV with **forward-looking labels**,
  so there is something to train a model on later.
- A raw event archive (`data/raw/`) keeping every websocket message, so any
  feature invented later can be recomputed over all past history.
- A standalone, tested risk engine: liquidation-price math, volatility- and
  Kelly-based position sizing, hard limits, and a kill switch.

**Analysis tooling** (`backend/analysis/`)
- `check_features.py` — the gate: information coefficients, leakage checks, a
  purged out-of-sample split, and a cost-adjusted economic test.
- `replay.py` — rebuild features from the raw archive with different
  parameters, without re-collecting data.
- `compact.py` — Parquet conversion and storage reporting.

**Still missing**: the prediction model, regime detection, the signal
combiner, the execution engine, position persistence, and backtesting. The
risk engine is written and tested but is currently guarding a door that does
not open yet.

A note on ordering, because it is the whole reason the project is shaped this
way: the ML layers cannot be built before the recorder has run. There is no
historical order-book dataset sitting anywhere — it has to be captured live,
first. Recording is therefore step one, not step four.

## Layout

```
.
├── backend/
│   ├── live-chart.py        # entrypoint: run this. Wires the pieces below together.
│   ├── config.py             # all settings: env vars, paths, ports
│   ├── market_data.py        # fetch/parse BloFin candles & prices
│   ├── support_resistance.py # swing-detection + clustering (the only "analysis" so far)
│   ├── state.py               # shared LiveChartState + websocket broadcast
│   ├── server.py              # HTTP/websocket servers + background refresh loops
│   ├── trading/               # the trading stack (see trading/README.md)
│   │   ├── orderbook.py       # L2 book w/ sequence-gap detection
│   │   ├── tape.py            # rolling trade tape
│   │   ├── features.py        # OBI / OFI / microprice / volatility
│   │   ├── recorder.py        # labelled feature rows -> CSV
│   │   ├── ingest.py          # live book/trade websocket loop
│   │   ├── rawlog.py          # raw event archive (gzipped JSONL)
│   │   └── risk.py            # liquidation math, sizing, hard limits
│   ├── analysis/              # offline tooling (see analysis/README.md)
│   │   ├── bars_import.py     # free Binance bar/OI/funding history -> dataset
│   │   ├── cross_sectional_import.py  # the same, as a multi-symbol panel
│   │   ├── check_features.py  # do the features predict anything?
│   │   ├── train_model.py     # LightGBM + shuffled-label control + paired test
│   │   ├── passive_sim.py     # markout curves + bracketed passive fill rates
│   │   ├── spread_survey.py   # which instruments' spreads cover the maker fee
│   │   ├── replay.py          # rebuild features from raw events
│   │   ├── compact.py         # CSV -> Parquet, storage report
│   │   └── stats.py           # IC, AUC, logistic regression, purged split
│   └── tests/                 # pytest suite (308 tests)
├── data/                      # recorded data (gitignored)
│   ├── features-*.csv         #   labelled features — regenerable
│   └── raw/                   #   raw events — IRREPLACEABLE
├── frontend/
│   ├── live-chart.html     # thin page shell
│   ├── styles.css          # all page styling
│   ├── app.js              # chart rendering + websocket client logic
│   ├── package.json        # npm dep: lightweight-charts
│   └── node_modules/       # installed by `npm install` in frontend/ (gitignored)
├── blofin-sdk-python/      # vendored BloFin API SDK (its own git repo/history)
├── .env                    # BloFin API credentials (gitignored, never commit this)
└── .gitignore
```

`backend/live-chart.py` is intentionally thin — it just parses args and calls
into the other modules. If you're reading this codebase for the first time,
read the backend files in the order listed above (config → market_data →
support_resistance → state → server → live-chart.py).

## Setup

This project uses the micromamba environment at:

```
D:\Micromamba\micromambaenv\envs\crypto_bot\python.exe
```

Written as `python` throughout this README. If it isn't on your PATH, use the
full path, or activate the env first with
`micromamba activate crypto_bot`.

1. Python deps (from the repo root):
   ```
   python -m pip install -r blofin-sdk-python\requirements.txt
   python -m pip install numpy                  # for backend\analysis
   python -m pip install lightgbm scipy         # for backend\analysis\train_model.py
   ```
   `lightgbm` and `scipy` are only needed to train a model; everything else in
   `backend\analysis` runs on numpy alone. `pyarrow` and `duckdb` are optional
   too — only for `backend\analysis\compact.py`, and not until the data gets
   large.
2. Frontend deps:
   ```
   cd frontend
   npm install
   ```
3. `.env` at the repo root holds BloFin API credentials (`API_KEY`, `SECRET`,
   `PASSPHRASE`). Gitignored — never commit real keys.

   **Not needed for data collection.** The book, trade and funding channels
   are public, and the REST market endpoints accept no credentials. Keys only
   matter once there is an order path, which there isn't yet.

## Running the live chart

From the repo root:
```
python backend\live-chart.py
```

This starts a local HTTP server (default `http://127.0.0.1:8765/live-chart.html`)
and a websocket feed (default `ws://127.0.0.1:8766`) that pushes candle/price
updates to the page.

Useful environment variables (set in `.env` or the shell):

| Variable | Purpose | Default |
|---|---|---|
| `BLOFIN_INST_ID` | Instrument to track | `BTC-USDT` |
| `BLOFIN_SUPPORT_BAR` | Candle timeframe | `15m` |
| `BLOFIN_USE_DEMO` | Use BloFin demo endpoints | `false` |
| `BLOFIN_CHART_HTTP_PORT` / `BLOFIN_CHART_WS_PORT` | Local server ports | `8765` / `8766` |
| `BLOFIN_MICRO_ENABLED` | Run the order book / feature feed | `true` |
| `BLOFIN_BOOK_DEPTH` | `books` (200 levels) or `books5` | `books` |
| `BLOFIN_FEED_STALL_TIMEOUT_S` | Silence before the feed is reconnected | `30` |
| `BLOFIN_RECORD_FEATURES` | Write labelled features to `data/` | `true` |
| `BLOFIN_FEATURE_SAMPLE_MS` | How often to persist a row | `1000` |
| `BLOFIN_LABEL_HORIZONS` | Forward label horizons, seconds | `300,900,1800` |
| `BLOFIN_LABEL_THRESHOLD_BPS` | Move size counted as a signal | `10` |
| `BLOFIN_VIP_TIER` | BloFin fee tier (0, 1, 2, 5) | `0` |
| `BLOFIN_MAKER_FEE_RATE` / `BLOFIN_TAKER_FEE_RATE` | Override the tier's rates, per side | from tier |
| `BLOFIN_ROUND_TRIP_COST_BPS` | Cost the edge gate must clear | `10` |
| `BLOFIN_MAX_LEVERAGE` | Risk engine leverage cap | `5` |
| `BLOFIN_MIN_LIQ_BUFFER_PCT` | Required distance to liquidation | `0.15` |
| `BLOFIN_RECORD_RAW` | Archive raw events to `data/raw/` | `true` |

See `backend/config.py` for the full list — every setting lives there.

Run with `--no-microstructure` to start only the chart.

## Collecting data

Just run the bot and leave it. Recording is on by default:

```
python backend\live-chart.py
```

Startup confirms both writers, and the chart page's Microstructure panel shows
a live row count. **Nothing is written for the first 30 minutes** — the
longest label horizon is 1800s, and a row can't be written until its forward
window has actually elapsed. That's the lookahead guard, not a hang. The raw
archive starts writing immediately, so nothing is lost in the meantime.

Why 30 minutes and not 30 seconds: measured on BTC-USDT, the standard
deviation of the forward move is 0.97bps at 5s and 2.41bps at 30s, against a
round-trip cost of 1.2bps (maker both sides) to 10bps (taker both sides). At a
30-second horizon a perfect oracle capturing a full standard deviation still
loses money at taker fees, so second-scale horizons were unprofitable by
arithmetic before any model was involved. Price scales as a near-perfect random
walk here (measured exponent 0.495), so the fix is time: sigma grows as
sqrt(T), reaching ~7.6bps at 5 minutes and ~18.7bps at 30 minutes. The full
derivation is in `backend/config.py` next to the constants.

Storage, measured: **~140 MB/day** total (~100 MB raw + ~40 MB features at the
1s sample interval), about 4 GB/month. See `backend/analysis/README.md` for the
full table and the reasoning about databases.

Collect across **varied conditions** — a model trained only on quiet books
learns nothing about the regimes that actually hurt you.

### If it stops overnight

It shouldn't any more. On 2026-09-08 a run died at 03:02 UTC, seven hours in,
because one dropped HTTP keep-alive on the *chart's* candle endpoint raised
`RemoteDisconnected` in the one loop that had no error handler — and
`run_servers` was gathering the raw coroutines, so `asyncio.gather` propagated
it and ended the process, recorder and raw archive included.

Both halves are fixed (`backend/server.py`): every loop handles its own
errors, and `supervise` restarts any that still falls over, so no loop can end
the run. The chart going stale is now a cosmetic failure, which is what it
always should have been. Console lines carry a UTC timestamp and are flushed,
so a failure that happens while you are asleep leaves a usable record instead
of sitting in a block buffer that dies with the process.

**Nothing was lost in that incident** beyond the hours that never happened —
the `finally` block flushed cleanly, and the 7 hours already on disk are
intact. Worth knowing the shape of the loss, though: the raw archive is the
irreplaceable half, and a crash is only ever as expensive as the time before
someone notices.

The other half of that lesson — a *stalled* feed, TCP still open and no
messages arriving — is now handled too. It was the more dangerous of the two,
because it raises nothing: `supervise` restarts loops that fail, and a loop
waiting on a silent socket has not failed. The SDK cannot catch it either;
its receive loop swallows its own read timeout with `continue`, and
`listen()` then blocks forever on an empty queue.

So the feed now measures silence directly (`backend/trading/ingest.py`,
`_stream`): each message is awaited with a deadline, and
`BLOFIN_FEED_STALL_TIMEOUT_S` seconds without one tears the connection down
and reconnects, which is the only honest response to a feed we can no longer
account for. The deadline is per message, not per stream, so a slow feed is
never mistaken for a dead one. `feed.status()` reports `silenceSeconds` and a
`stalls` count, so the condition is visible on the chart's Microstructure
panel before the watchdog acts on it.

## Checking whether it predicts anything

After a few days:

```
python backend\analysis\check_features.py --horizon 900
```

This is the gate before any model work. See `backend/analysis/README.md` for
how to read the output — particularly why "statistically significant" and
"tradeable" are different questions, and why the effective sample size is far
smaller than the row count.

## Tests

```
cd backend
python -m pytest
```

308 tests covering the order book's gap handling, the OFI recursion, the
recorder's lookahead guard, the liquidation math (against hand-computed
values), the unrealized-drawdown breakers, the reduce-only close path, the
raw-archive round trip, the passive simulator's aggressor convention and
queue bracket, the maker-fee gate, the loop supervisor that keeps an
overnight run alive, the stall watchdog that reconnects a silent feed, the
loader that refuses to concatenate CSVs from two label generations, and the
evaluation statistics. They need no network,
credentials, or SDK.

## Roadmap toward the actual bot

Done:

1. ~~**Risk/liquidation module**~~ — `backend/trading/risk.py`. Standalone and
   tested, as planned. **Its MMR assumption still needs validating against
   BloFin's real tier table** — see `backend/trading/README.md`.
2. ~~**Microstructure feature layer**~~ — `backend/trading/features.py`.
   Replaced the original "indicator layer" idea: on a 15m candle, RSI and MACD
   say nothing useful about the next few seconds. Order-book and trade-flow
   imbalance are the signals that operate on an HFT horizon.
3. ~~**Data recording**~~ — `backend/trading/recorder.py` plus
   `backend/trading/rawlog.py` for the raw archive.
4. ~~**The predictiveness check**~~ — `backend/analysis/check_features.py`,
   with `replay.py` to regenerate features from raw whenever the feature set
   changes.

Next, in order:

5. **Collect data.** Leave the bot running. Nothing below can start until
   there are at least a few days of recorded features across different market
   conditions.
6. **Run the check.** `check_features.py`. If it says `NO SIGNAL`, stop and
   fix that — more model complexity cannot create information that isn't
   there. If it says `NOT TRADEABLE`, the problem is execution cost, not the
   model.
7. ~~**Prediction model**~~ — `backend\analysis\train_model.py`. LightGBM on
   a three-way purged time split, a shuffled-label control, and a paired
   bootstrap against both the control and the linear baseline.

   **Run, and the answer was no.** On 365 days of bar features, at every
   horizon tried:

   | horizon | LightGBM top decile | vs shuffled control | vs linear baseline |
   |---|---|---|---|
   | 300s | +0.89 bps | +0.37 `[-0.20, +0.88]` | −0.19 `[-0.65, +0.30]` |
   | 900s | +1.35 bps | +0.60 `[-0.98, +2.58]` | +0.27 `[-0.95, +1.68]` |
   | 1800s | +1.81 bps | +2.23 `[-1.15, +5.72]` | +0.20 `[-2.46, +3.24]` |

   Every paired interval includes zero. The non-linear model is not separable
   from the same pipeline trained on shuffled labels, and it does not beat the
   logistic regression. LightGBM does rank *direction* slightly better (test
   AUC 0.533 vs 0.528 at 300s) without ranking *returns* better, which says it
   finds direction on small moves — the ones costs eat first.

   The constraint is the feature set, not the model class. Adding capacity to
   a model that already cannot separate itself from noise is the one move
   guaranteed not to help.
8. **Cross-sectional panel** — `backend\analysis\cross_sectional_import.py`.
   Ten majors, 364 days, 349,420 rows: predict which coin outperforms rather
   than where BTC goes, so the market factor is subtracted from the label
   instead of forecast.

   **The clearest signal this project has found, and still not tradeable.**
   Ten features significant at |IC| 0.02-0.04, and *every* momentum sign
   negative — short-horizon cross-sectional reversal, the coin that led the
   last 15-240 minutes lagging the next 15. Sign consistency across ten
   independent lookbacks is what a real effect looks like; noise gives random
   signs. On a 70/30 purged split the linear decile ladder is monotonic at
   **+0.957**.

   But the size is wrong by an order of magnitude. Top decile +0.49bps, bottom
   −0.41bps, so a long/short spread of ~0.9bps against **2.4bps** for two
   maker legs. And under the stricter test — 60/20/20 with a shuffled-label
   control — neither model separates from its control at any horizon:

   | horizon | LightGBM | shuffled control | paired difference |
   |---|---|---|---|
   | 300s | +0.79 bps | +0.57 bps | +0.22 `[-0.05, +0.50]` |
   | 900s | +0.36 bps | +0.21 bps | +0.15 `[-0.53, +0.83]` |
   | 1800s | +0.50 bps | −0.04 bps | +0.54 `[-0.53, +1.54]` |

   Worth noting the control clears zero at 300s. That is the top-decile metric
   itself being biased upward for any score correlated with cross-sectional
   volatility — high-beta names have fatter right tails, so selecting on
   anything vol-adjacent lifts the mean. Catching that is precisely why the
   control exists.

   LightGBM does beat the linear baseline here in a way it never did on BTC
   alone (decile monotonicity +0.960 vs +0.697 at 900s), so the non-linearity
   is real in the cross-section. It is just nowhere near two round trips.
9. ~~**Execution — what a passive quote actually earns**~~ —
   `backend\analysis\passive_sim.py`. Two readings of step 8 pointed the same
   way. The effect found is *reversal*, which is the return to providing
   liquidity to whatever just moved, so the edge lives on the passive side of
   the book; and every result so far had died on cost rather than prediction.
   Measuring the real passive fill rate and its adverse selection was worth
   more than any further feature work.

   **Run, and it fails on arithmetic before it gets to adverse selection.**
   Six hours of Binance BTCUSDT (Tardis, 2026-09-01), quoting at the touch
   every second with a 60s timeout. Queue position is unobservable, so it is
   bracketed rather than modelled — pessimistic joins behind the whole visible
   size `Q` and fills only once same-side aggressor volume exceeds it,
   optimistic fills on any trade at the price:

   | | optimistic | cancel-adj | pessimistic |
   |---|---|---|---|
   | fill rate (bid) | 96.7% | 84.6% | 63.2% |
   | median wait | 0.4s | 2.9s | 10.4s |
   | markout @ 0s | +0.019 | +0.005 | **−0.400** |
   | markout @ 60s | −0.110 `±0.155` | −0.607 `±0.157` | −1.489 `±0.159` |
   | net of maker fees | −1.310 | −1.807 | **−2.689** |

   The median half spread is **0.006 bps against a 1.2 bps maker round trip**.
   Binance BTCUSDT perp is one tick wide almost always, and one tick is $0.10
   on a $110k instrument — so a flawless fill at the front of the queue,
   marked out instantly, against a counterparty who knows nothing, loses 1.19
   bps before adverse selection is even involved. That is not a queue problem
   or a signal problem. Conditioning on `obi_1` survives an out-of-sample
   split and helps by **+0.31 bps** while *raising* the fill rate, which is a
   real effect and a rounding error against the gap.

   Two things the bracket bought that a point estimate would not have. The
   fill-rate range is 63%–97%, so any single fill-rate assumption is an
   assumption; and markout is already **−0.400 bps at the instant of the
   pessimistic fill**, sixty times the half spread, which says a back-of-queue
   fill arrives precisely when the level is being swept. Queue position is not
   a detail here, it is the trade. Measured cancellation share at the touch is
   **92%**, so the truth sits far closer to the optimistic bound — and the
   optimistic bound also loses.

   Where this leaves the passive idea: not dead, but not on this instrument.
   The gate to clear is a spread wider than the maker round trip, which BTC
   perp on the most arbitraged venue in existence will never offer. Steps 9a
   and 9b below went looking for one that does.

9a. ~~**Which instruments' spreads cover the fee?**~~ —
   `backend\analysis\spread_survey.py`. The gate is one line — a passive
   round trip captures the whole spread and pays two maker fees, so it needs
   `spread >= COST_MAKER_MAKER_BPS` — and it is cheap enough to run across a
   universe. Ten majors on binance-futures, 2026-09-01:

   | | BTC | ETH | BNB | XRP | LINK | SOL | DOGE | AVAX | LTC | ADA |
   |---|---|---|---|---|---|---|---|---|---|---|
   | median spread (bps) | 0.013 | 0.041 | 0.144 | 0.725 | 0.881 | 0.970 | 1.207 | 1.378 | 2.053 | **5.019** |

   **A 400x range on one venue on one day.** Everything this project concluded
   about execution cost from BTCUSDT was a conclusion about the most
   arbitraged perpetual in existence. The spreads are also *pinned* — p25 and
   p75 within 0.01 bps of the median — because these instruments sit at their
   minimum tick essentially always, so the spread in bps is just `tick/price`.

   That has a sting in it. When the tick binds, makers cannot compete the
   spread away, so they queue behind it instead: 173,000 contracts resting at
   ADA's best bid against 3 at BTC's. A wide spread here is not payment for
   adverse selection, it is a queue to get to the front of — which is exactly
   what step 9 refuses to model and brackets instead.

   Simulating the four that cleared gave the first positive number in this
   project's history, and one clean empirical regularity:

   | symbol | spread | optimistic markout | adverse selection | pessimistic |
   |---|---|---|---|---|
   | ADAUSDT | 5.019 | **+1.967** | 0.543 | −4.338 |
   | LTCUSDT | 2.053 | +0.378 | 0.649 | −2.525 |
   | AVAXUSDT | 1.378 | +0.179 | 0.510 | −2.111 |
   | DOGEUSDT | 1.207 | +0.316 | 0.288 | −1.761 |
   | BTCUSDT | 0.013 | −0.107 | 0.113 | −1.321 |

   **Adverse selection is about half a basis point and does not scale with the
   spread** — 0.29 to 0.65 bps across a 400x spread range, no trend. So the
   whole question reduces to arithmetic you can do before downloading
   anything: a passive round trip needs `spread > 2 x (0.5 + maker fee per
   leg)`.

9b. ~~**The fee schedule, because the gate is made of it**~~ — `backend\config.py`.
   Every number above depends on a constant this repo had been assuming. Read
   from BloFin's published schedule on 2026-09-07 (their site 403s automated
   fetches, so this came from search results quoting it — **confirm against
   your own account**):

   | tier | maker | taker | qualification |
   |---|---|---|---|
   | VIP 0 | 0.0200% | 0.0600% | default |
   | VIP 1 | 0.0060% | 0.0500% | 50k USDT held, or 10M 30d futures, or 1M 30d spot |
   | VIP 2 | 0.0040% | 0.0450% | 2M USDT 30d spot |
   | VIP 5 | 0.0000% | 0.0350% | — |

   Two findings, and the second is the one that matters.

   **There is no maker rebate at any tier.** The floor is 0% at VIP 5. No
   BloFin schedule ever pays you to provide liquidity; the best case is that
   providing it becomes free. That closes off the "a rebate would invert the
   economics" idea entirely.

   **The repo had been assuming VIP 1 rates on an account that has never
   traded.** A fresh account is VIP 0, where the maker fee is 2.0 bps per leg
   rather than 0.6 — a 4.0 bps round trip, not 1.2. The default is now VIP 0
   (`BLOFIN_VIP_TIER`), because the entire history of this project is results
   that died once their cost assumption was made honest.

   The tier is now the single biggest lever in the project, and the threshold
   sits exactly between two verdicts:

   | | gate | best instrument surveyed | front-of-queue net |
   |---|---|---|---|
   | VIP 0 | 5.0 bps | ADAUSDT at 5.019 | **−2.03** |
   | VIP 1 | 2.2 bps | ADAUSDT at 5.019 | **+0.77** |

   VIP 1's cheapest route is **holding 50,000 USDT on the exchange** — an
   asset threshold, not a volume one, so it is reachable without trading a
   contract. Whether that is an acceptable thing to do is a decision, not a
   measurement, and it is now the decision the passive branch waits on.

10. **Regime detection** — replace the percentile-based `vol_regime`
   placeholder with a fitted model.
11. **Execution engine** — adaptive limit orders, wired to the risk engine's
   `check_order()`. This is where the existing kill switch finally guards
   something real. Demo account only.
12. **Backtesting / paper trading** — with realistic fees, queue position and
   slippage. Step 9 already built the queue-position half of this: reuse
   `passive_sim.py`'s bracket rather than picking a single fill assumption,
   and report both bounds. Expect the paper results to be considerably
   worse than the backtest; that gap is the honest measure of the model.

Given the stated leverage profile: this stays on the BloFin demo environment
until steps 5-11 are done and the liquidation math has been checked against the
exchange's own numbers. Leverage magnifies model *and* execution errors, and
the risk engine's default limits (5x, 15% liquidation buffer) are deliberately
far more conservative than the project's stated ambition.
