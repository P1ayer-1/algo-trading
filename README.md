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
- A raw event archive (`data/<INST-ID>/raw/`) keeping every websocket message,
  so any
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
│   ├── live-chart.py        # entrypoint: chart + one instrument's recorder
│   ├── record.py             # entrypoint: N instruments, headless, no ports
│   ├── record_oi.py          # entrypoint: OI or mark-price, alongside a live run
│   ├── record_hyperliquid.py # entrypoint: Hyperliquid market data + liquidation map
│   ├── plan_carry.py         # entrypoint: what a carry WOULD do. Sends nothing.
│   ├── run_carry.py          # entrypoint: opens it. Dry unless --confirm.
│   ├── monitor_carry.py      # entrypoint: scores an open carry. Sends nothing.
│   ├── close_carry.py        # entrypoint: takes it off. Dry unless --confirm.
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
│   │   ├── openinterest.py    # OI poller — the input for others' liquidation levels
│   │   ├── markprice.py       # mark + index price — the basis, REST-only
│   │   ├── hyperliquid.py     # second venue: feed, account reads, rate budget
│   │   ├── liquidation_map.py # others' liquidation levels, read off public positions
│   │   ├── rawlog.py          # raw event archive (gzipped JSONL)
│   │   ├── margin_tiers.py    # MMR for a size, from the account's own host
│   │   ├── risk.py            # liquidation math, sizing, hard limits
│   │   └── strategies/        # one directory per strategy, same three verbs
│   │       ├── __init__.py    # the plan/execute/monitor contract, as Protocols
│   │       ├── carry/
│   │       │   ├── plan.py     # sizes both legs; SENDS NOTHING
│   │       │   ├── execute.py  # puts it on (perp first), takes it off (spot first)
│   │       │   ├── monitor.py  # scores it against the plan; READ ONLY
│   │       │   └── broker.py   # the REST calls the other three are written against
│   │       └── range_trade/
│   │           └── levels.py   # the range + brackets; no other verbs (step 9k: no edge)
│   ├── analysis/              # offline tooling (see analysis/README.md)
│   │   ├── bars_import.py     # free Binance bar/OI/funding history -> dataset
│   │   ├── cross_sectional_import.py  # the same, as a multi-symbol panel
│   │   ├── check_features.py  # do the features predict anything?
│   │   ├── train_model.py     # LightGBM + shuffled-label control + paired test
│   │   ├── passive_sim.py     # markout curves + bracketed passive fill rates
│   │   ├── spread_survey.py   # which instruments' spreads cover the maker fee
│   │   ├── venue_compare.py   # adverse selection: BloFin vs Binance, paired
│   │   ├── blofin_spot.py     # the spot endpoints the SDK omits
│   │   ├── funding_carry.py   # long spot + short perp: does funding pay?
│   │   ├── carry_backtest.py  # the same position, run from every entry
│   │   ├── range_backtest.py  # fade the range: bracketed 1m fills vs a shuffled-day control
│   │   ├── range_information.py  # what a forecast range is worth: centre vs width
│   │   ├── range_harness.py   # the bar every range model is scored on
│   │   ├── range_gated.py     # trade the fade only when a model says the range holds
│   │   ├── range_multiscale.py # ranges at 4h/1d/3d/7d, each side scored separately
│   │   ├── attention_model.py # attention across timeframes, numpy, gradient-checked
│   │   ├── reversion_scale.py # is the reversion real, or is it the spread?
│   │   ├── fetch_klines.py    # bulk 1m klines, with the listing dates made visible
│   │   ├── panel_daily.py     # 1m archive -> a daily panel, Binance funding joined
│   │   ├── panel_blofin.py    # the same schema from BloFin's own endpoints
│   │   ├── panel_hyperliquid.py # and from Hyperliquid, which funds hourly
│   │   ├── factor_panel.py    # cross-sectional factors scored as money, not IC
│   │   ├── funding_dispersion.py # the same coin's funding on two venues
│   │   ├── panel_venue.py     # any venue behind one adapter: bybit, mexc, kraken...
│   │   ├── validate_liquidation.py  # our liq math vs the exchange's own
│   │   ├── blofin_spread_survey.py  # the same, live, on BloFin itself
│   │   ├── layout.py          # where recorded data lives; one owner
│   │   ├── replay.py          # rebuild features from raw events
│   │   ├── compact.py         # CSV -> Parquet, storage report
│   │   └── stats.py           # IC, AUC, logistic regression, purged split
│   └── tests/                 # pytest suite (972 tests)
├── data/                      # recorded data (gitignored)
│   ├── <INST-ID>/             #   ONE DIRECTORY PER INSTRUMENT (BloFin)
│   │   ├── features-*.csv     #     labelled features — regenerable
│   │   ├── raw/               #     raw events — IRREPLACEABLE
│   │   └── carry/             #     an open carry's frozen baseline + snapshots
│   └── hyperliquid/           #   a different VENUE, invisible to BloFin tools
│       ├── <COIN>/raw/        #     l2Book, trades, activeAssetCtx — IRREPLACEABLE
│       ├── <COIN>/liquidation-levels-*.jsonl  # the map — derived
│       └── _accounts/         #     clearinghouseState per account — IRREPLACEABLE
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

### What is actually recorded, and what cannot be recovered later

Two layers, and the difference between them decides what is urgent:

**The raw archive** (`data/<INST-ID>/raw/<day>/<channel>-<HH>.jsonl.gz`) holds
every message unmodified, wrapped with a local receive time and a per-process
counter: `{"t":…,"n":…,"m":{…}}`. Channels: `books`, `trades`, `funding-rate`
over websocket, plus `open-interest` and `mark-price` polled over REST. About
**1.35 GB/day across 15 instruments**.

A process never appends to a file it did not create. A restart inside an hour
writes `<channel>-<HH>.r001.jsonl.gz` beside the first run's file; it sorts
after it and matches the same `<channel>-*.jsonl.gz` glob. Appending is how it
used to work, and after a hard kill it cost the whole hour: the killed run's
gzip member has no trailer, the reader decoded through it into the restart's
member and gave up, and in the 2026-09-11 reproduction it returned **0 of 10**
records - the ones written before the crash included. The reader now recovers
every member of files written that way (`backend/trading/rawlog.py`), and
`python backend\analysis\audit_raw.py` reports which archived hours it
changes.

**The feature CSV** is a derived, deliberately lossy 1-second sample — 31
features plus `fwd_ret_bps_{300,900,1800}s` and their labels. `replay.py`
rebuilds it from raw with different parameters, so nothing in it is a
commitment.

The horizons are **seconds**: 5, 15 and 30 minutes. They used to be 1/5/30
*seconds*, and that was the reason the bot could not work — measured on
BTC-USDT, σ of the forward move is 0.45 bps at 1s and 2.41 bps at 30s against
a 1.2–10 bps round trip, so at 30 seconds an oracle with perfect knowledge of
the sign still loses money at taker fees. See `config.py` for the full
calculation. Consequence: **nothing lands on disk for the first 30 minutes**
after a restart, because a row cannot be written until its forward window
closes.

Anything derived can be recomputed from raw; anything never subscribed is
gone for good. That asymmetry is why `mark-price` was added (2026-09-10):
BloFin's public websocket refuses `mark-price`, `index-price`,
`index-tickers`, `open-interest`, `price-limit` and `liquidation-orders` with
`60012 Invalid request`, and its `tickers` channel carries only top-of-book,
last trade and 24h stats — all of which `books` and `trades` already provide.
`GET /api/v1/market/mark-price` returns mark **and** index together for all
486 instruments in one request, and has no history endpoint. Mark minus the
traded book is the basis: the `conv` cost `funding_carry.py` already prices,
and the price a carry's liquidation is actually struck against.

```
python backend\record_oi.py --channel mark-price --match-running
```

Safe to start against a recorder that is already days in — different channel,
different file, and the exclusive lock is named after the channel, so it
coexists with an open-interest poller and still refuses a second mark-price
one.

### Strategies: three verbs, and why they are separate files

Every strategy under `backend/trading/strategies/` is a directory with the
same three modules, and the split is a safety property rather than filing:

| | contract |
|---|---|
| `plan.py` | computes orders and **cannot send them**. No code path to `placeOrder` — checkable with a grep, and checked that way. Returns refusals *plural*: every failing gate, not the first. |
| `execute.py` | **dry unless a caller says otherwise**, same result shape either way, so rehearsal and live differ by one flag and nothing else. Owns leg ordering and unwinding. |
| `monitor.py` | no `--confirm`, **no path to `placeOrder` at all**. The one you can run at 3am without reading the source first. Verifies against the *exchange*, not against the plan. |

Every sizing and margin question gets answered and reviewed while the answer
is still only text; the code that can move money is a separate thing you ask
for by name.

There is deliberately **no `Strategy` base class**. There is one strategy, and
an interface extracted from a single example encodes that example's accidents
— carry has two legs, funding as its entire return source, a hold measured in
weeks, and no prediction anywhere in it, none of which a directional strategy
shares. So the lifecycle is stated as Protocols in
`strategies/__init__.py`, satisfied structurally and inherited from never:
`CarryPlan`, `ExecutionResult` and `MonitorReport` conform without importing
them. `tests/test_strategy_contract.py` asserts that, so the shape stays
load-bearing instead of becoming a stale comment. When a second strategy
arrives and both genuinely want the same behaviour, that is when to extract
it.

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
| `BLOFIN_VIP_TIER` | BloFin fee tier (0, 1, 2, 3, 5) | `0` |
| `BLOFIN_MAKER_FEE_RATE` / `BLOFIN_TAKER_FEE_RATE` | Override the tier's rates, per side | from tier |
| `BLOFIN_ROUND_TRIP_COST_BPS` | Cost the edge gate must clear | `10` |
| `BLOFIN_MAX_LEVERAGE` | Risk engine leverage cap | `5` |
| `BLOFIN_MIN_LIQ_BUFFER_PCT` | Required distance to liquidation | `0.15` |
| `BLOFIN_RECORD_RAW` | Archive raw events to `data/<INST-ID>/raw/` | `true` |

See `backend/config.py` for the full list — every setting lives there.

Run with `--no-microstructure` to start only the chart.

## Collecting data

For one instrument with a chart to watch, run the bot and leave it — recording
is on by default:

```
python backend\live-chart.py
```

For a **dataset**, which is usually what you want, record several instruments
at once with no chart, no HTTP server and no ports:

```
python backend\record.py --instruments BTC-USDT,ADA-USDT,PUMP-USDT
python backend\record.py --list-cost --instruments BTC-USDT,ADA-USDT   # disk only
```

`record.py` also polls open interest for the whole set — one HTTP request per
poll, one row per instrument per minute. To add it to a recorder that is
*already running*, without restarting it and losing continuity:

```
python backend\record_oi.py --match-running
```

That is a separate process writing a separate channel
(`raw/<day>/open-interest-<HH>.jsonl.gz`), so it cannot touch the `books-` and
`trades-` files a live recorder holds open. BloFin serves OI as a snapshot with
no history endpoint, which makes it capture-or-lose like the book — and it is
the only public input to estimating where *other* traders get liquidated. See
`trading/README.md` for what is and is not built on top of it.

### Hyperliquid, and other traders' liquidation levels

```
python backend\record_hyperliquid.py --check     # validate coins live, print the budget
python backend\record_hyperliquid.py             # BTC, ETH, SOL, HYPE by default
```

A second venue in a second process, sharing nothing with `record.py`.
Hyperliquid's ledger is public: any account's positions come back with the
exchange's own `liquidationPx`, and every trade names both accounts. So where
other traders get liquidated is **read, not reconstructed** from open
interest — which on BloFin is the only option. Each minute it writes a map per
coin to `data/hyperliquid/<COIN>/liquidation-levels-<day>.jsonl` and prints
a line:

```
BTC    mark 76,996  accounts 295  coverage L 2.9% S 3.5%  | 1%: $93k down / $0 up  2%: $776k down / $231k up ...
```

`2%: $776k down` is tracked longs whose own liquidation price lies within a 2%
fall. **Read every figure through the coverage beside it.** See
`trading/README.md` for what the map does and does not claim.

First live run, 2026-09-11, 200 seconds, four coins, default budget:

| | |
|---|---|
| account reads | 450/min — exactly the budget; 0 failed, 0 rate-limited |
| accounts discovered | 669 in 3 minutes, queue still growing: discovery outruns reads, by design |
| coverage after 3 minutes | 2–10% of open interest per side |
| tracked long size with no liquidation price | 28–90%; shorts 0%, structurally |
| reads that differed from the previous one | 92% — cross liquidation prices drift |

Storage is dominated by the account archive, and it is not small:

| | disk per day |
|---|---|
| `_accounts/` — `clearinghouseState` | **~580 MB** |
| per coin — book, trades, asset context | 16–29 MB |
| four coins, total | **~680 MB** |

Account records compress only ~5x and reach 71 KB for an account holding
dozens of positions (p90 15 KB), because whole accounts are kept: a cross
liquidation price depends on everything else the account holds. That cost
scales with `--weight-per-minute`, not with the number of coins. Three minutes
extrapolated, so an order of magnitude until a full day has run.

After a crash: `.recorder.lock` is left behind and taken over once its pid is
gone, and the address book is saved every five minutes and on a clean exit, so
a crash costs at most five minutes of discovered accounts.

Each feed opens its own websocket and writes to its own `data/<INST-ID>/`, so
one instrument desyncing, stalling or reconnecting cannot touch another's
data. Every feed runs under the same `supervise` wrapper as the chart's loops,
so no single failure can end the run. Measured cost is ~140 MB/day for
BTC-USDT; thinner instruments are a small fraction of that, so `--list-cost`
is an upper bound rather than an estimate.

### One directory per instrument, and why it matters

`data/<INST-ID>/` is not tidiness. Rows for two symbols are **structurally
identical** — same columns, same order, same dtypes — and differ only in which
instrument they describe. Written to a shared path they concatenate into a
matrix that no schema check can object to, because nothing about the schema is
wrong, and the result is a verdict about no instrument in particular. Layout
is the only thing that separates them.

So `check_features.py` refuses to cross a directory boundary. Point it at one:

```
python backend\analysis\check_features.py --horizon 900 --data-dir data\BTC-USDT
```

Given `--data-dir data` with several instruments beneath it, it names them and
stops rather than averaging them.

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

A *hard* kill turned out to cost more than that. Until 2026-09-11 a restart
inside an hour appended to that hour's archive file, and the reader, meeting
the killed run's trailerless gzip member, gave up there - losing the
restart's records and the last few before the tear. `backend\analysis\audit_raw.py`
over all 3,555 archive files found it had happened once: BTC-USDT, 2026-09-09
01:00 UTC, a run that died after 01:30:59 and restarted at 01:33:38. The
reader returned 6,401 records for that hour's books, trades and funding
against 11,790 on disk. Both halves are fixed in `backend/trading/rawlog.py` -
files are created exclusively, and the reader recovers every member - and one
more file, an open-interest hour on 2026-09-11 where two live writers
interleaved, is partly recovered (see `backend/analysis/README.md`).

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

972 tests covering the order book's gap handling, the OFI recursion, the
recorder's lookahead guard, the liquidation math (against hand-computed
values), the unrealized-drawdown breakers, the reduce-only close path, the
raw-archive round trip, the passive simulator's aggressor convention and
queue bracket, the maker-fee gate, the loop supervisor that keeps an
overnight run alive, the stall watchdog that reconnects a silent feed, the
loader that refuses to concatenate CSVs from two label generations or two
instruments, the multi-instrument recorder's isolation, and the evaluation
statistics. They need no network,
credentials, or SDK.

## Roadmap toward the actual bot

Done:

1. ~~**Risk/liquidation module**~~ — `backend/trading/risk.py`. Standalone and
   tested, as planned. ~~Its MMR assumption still needs validating against
   BloFin's real tier table.~~ **Validated 2026-09-09** against a real demo
   position via `backendnalysisalidate_liquidation.py`: 0.0568% relative
   error, and the MMR actually applied read out as exactly 0.500%. The 5.68
   bps residual is BloFin's liquidation fee rate, which the clean derivation
   omits and `fee_buffer_bps` should be set to ~6 to absorb. See
   `backend/trading/README.md`.
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

   **First real run, 2026-09-10, and it found a bug in the check before it
   found anything about the market.** On 1.55 days of recorded microstructure
   features the economic test compared each decile's mean forward return to
   *zero* rather than to the window's own drift. Every strategy pointing the
   same way as a trending sample therefore scored:

   | | move over window | raw verdict |
   |---|---|---|
   | SUI-USDT | −7.7% | PROMISING, net **short** edge +5.67 bps |
   | IOST-USDT | +26.0% | PROMISING, net **long** edge +8.02 bps |

   Same features, same code, opposite trade. Nine of SUI's ten deciles had
   negative mean returns because the window's drift was −8.30 bps; shorting
   anything scored. The drift *was* the edge.

   Two fixes, both in `check_features.py`:

   - **Everything is now measured as EXCESS over the sample's drift.** You
     cannot trade a drift you must know the sign of in advance — if you knew
     it, the model would be unnecessary. Note `decile monotonicity` was never
     affected: shifting every decile by a constant cannot change its
     correlation with rank, which is exactly why it stayed trustworthy while
     the edge numbers did not.
   - **`--across-instruments`** runs the whole panel and asks whether the edge
     keeps its SIGN. One instrument cannot tell an edge from a week; the
     features are normalised quantities whose relationship to forward returns
     should not care which symbol produced them. This is the test that made
     step 8's result credible, and the single-instrument check never had it.

   ```
   python backend\analysis\check_features.py --across-instruments --cost-bps 12
   ```

   **Then the actual answer, across 15 instruments: no.**

   ```
   top decile beats drift   7/15
   clears costs on excess   3/15
   would have passed on RAW returns but not on excess: 4/15
   sign agreement           8/15  (p = 0.500 if the sign were a coin flip)
   ```

   Mean AUC across the panel is **0.5036**. SUI's 0.586 is the top draw of a
   noise distribution centred on a coin flip, not a signal — and four of
   fifteen instruments were being handed false positives by the drift bug.
   1.55 days of one regime (everything down but IOST) cannot overturn step 7's
   365-day null result, and now it does not pretend to.
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

   | VIP Tier | Futures Maker Fee | Futures Taker Fee | Spot Maker Fee | Spot Taker Fee | Requirement                                       |
   | -------- | ----------------: | ----------------: | -------------: | -------------: | ------------------------------------------------- |
   | VIP 0    |           0.0200% |           0.0600% |        0.1000% |        0.1000% | default                                           |
   | VIP 1    |           0.0060% |           0.0500% |        0.0350% |        0.0600% | 50k USDT held, or 10M 30d futures, or 1M 30d spot |
   | VIP 2    |           0.0040% |           0.0450% |        0.0200% |        0.0500% | 2M USDT 30d spot                                  |
   | VIP 3    |           0.0020% |           0.0425% |        0.0150% |        0.0450% | —                                                 |
   | VIP 4    |           0.0010% |           0.0400% |        0.0125% |        0.0375% | —                                                 |
   | VIP 5    |           0.0000% |           0.0350% |        0.0100% |        0.0325% | —                                                 |

   Two findings, and the second is the one that matters.

   **There is no maker rebate at any tier.** The floor is 0% at VIP 5. No
   BloFin schedule ever pays you to provide liquidity; the best case is that
   providing it becomes free. That closes off the "a rebate would invert the
   economics" idea entirely.

   **The repo had been assuming VIP 1 rates on an account that has never
   traded.** A fresh account is VIP 0, where the maker fee is 2.0 bps per leg
   rather than 0.6 — a 4.0 bps round trip, not 1.2. The repo *default* is
   therefore VIP 0, because the entire history of this project is results that
   died once their cost assumption was made honest.

   The tier is the single biggest lever in the project, and the threshold sits
   exactly between two verdicts:

   | | gate | best instrument surveyed | front-of-queue net |
   |---|---|---|---|
   | VIP 0 | 5.0 bps | ADAUSDT at 5.019 | **−2.03** |
   | VIP 1 | 2.2 bps | ADAUSDT at 5.019 | **+0.77** |

   **This account is VIP 1** (confirmed 2026-09-09), so `BLOFIN_VIP_TIER=1` is
   set in `.env` and the working numbers are the second row. The repo default
   stays VIP 0 for anyone cloning it. Note what is and is not confirmed: the
   *tier* comes from the account, the *rates* for that tier still come from
   the search-result reading above, and at +0.77 bps a 0.2 bps error in the
   maker fee is a quarter of the result. Worth reading off a real fill.

   ### The tier has stopped being the biggest lever, and that is the finding

   Account-reported on 2026-09-09: VIP 3 is 0.0020% / 0.0425%, and the ladder
   **ends at VIP 5** — which independently corroborates the original reading
   of the top tier. VIP 4 is still unconfirmed and stays absent rather than
   interpolated. With tier 3 filled in, the whole ladder is visible at once,
   and it is front-loaded to the point of being lopsided:

   | step | gain, bps of round trip |
   |---|---|
   | VIP 0 → VIP 1 | **+2.80** |
   | VIP 1 → VIP 2 | +0.40 |
   | VIP 2 → VIP 3 | +0.40 |
   | VIP 3 → VIP 5 | +0.40 |

   **The first step is 70% of the entire ladder, and this account has already
   taken it.** Everything remaining between here and the top of BloFin's
   schedule is worth 1.20 bps.

   Net passive round trip at the front of the queue, from step 9a's measured
   markouts:

   | symbol | VIP 0 | VIP 1 | VIP 2 | VIP 3 | VIP 5 | queue bracket |
   |---|---|---|---|---|---|---|
   | ADAUSDT | −2.03 | +0.77 | +1.17 | +1.57 | +1.97 | **6.30** |
   | LTCUSDT | −3.62 | −0.82 | −0.42 | −0.02 | +0.38 | 2.90 |
   | DOGEUSDT | −3.68 | −0.88 | −0.48 | −0.08 | +0.32 | 2.08 |
   | AVAXUSDT | −3.82 | −1.02 | −0.62 | −0.22 | +0.18 | 2.29 |
   | BTCUSDT | −4.11 | −1.31 | −0.91 | −0.51 | −0.11 | 1.21 |

   Read the last column against the rest of the table. **ADA's uncertainty
   about queue position spans 6.30 bps. The entire fee ladder, VIP 0 to VIP 5,
   spans 4.00.** So the single largest unknown in the passive branch is no
   longer the fee schedule — it is where in the queue an order actually sits,
   and no tier upgrade touches it.

   There is a floor underneath all of this that no tier can lift. The maker
   fee bottoms out at zero, but adverse selection was measured at ~0.5 bps per
   leg and does not scale with spread, so a passive round trip needs about
   **1.0 bps of spread at any tier**, forever. VIP 5 does not make quoting
   free; it makes quoting cost exactly adverse selection.

   One more thing belongs in the decision, since VIP 3 is volume-qualified
   rather than asset-qualified: 30-day futures volume is earned by trading at
   your *current* tier's fees. If passive quoting is unprofitable at VIP 1,
   the route to VIP 3 is paid for in losses, and +0.80 bps is what is being
   bought. That is a real trade to evaluate, not an obvious yes.

   Re-run of the step 9a survey at the VIP 1 gate, from the same measurements:

   | symbol | spread | clears 1.2 gate | clears 2.2 w/ adv. sel. | net (optimistic) | net (pessimistic) |
   |---|---|---|---|---|---|
   | ADAUSDT | 5.019 | yes | yes | **+0.77** | −5.54 |
   | LTCUSDT | 2.053 | yes | no | −0.82 | −3.73 |
   | AVAXUSDT | 1.378 | yes | no | −1.02 | −3.31 |
   | DOGEUSDT | 1.207 | yes | no | −0.88 | −2.96 |
   | BTCUSDT | 0.013 | no | no | −1.31 | −2.52 |

   Three things this does not settle. The arithmetic gate in
   `passive_sim.clears_fee_gate` is `spread >= 1.2` and now admits four of ten
   majors rather than one — but it deliberately excludes adverse selection,
   and on the empirical `2 x (0.5 + fee)` bar only ADA survives. ADA's own
   bracket runs **+0.77 to −5.54**, so the sign of the answer is still decided
   by queue position, which step 9 refuses to model for good reason. And every
   number in that table is *Binance*, measured through Tardis; this account
   trades on **BloFin**, whose spreads have never been measured here at all.

   That last one is now the cheap missing measurement rather than a deposit
   decision, and step 9c is it.

9c. ~~**What are BloFin's own spreads?**~~ — `backendnalysislofin_spread_survey.py`.
   Everything the passive branch believes about execution cost was measured on
   a venue this account does not trade on. BloFin's `getTickers()` returns best
   bid/ask for every instrument, unauthenticated, in one REST call, so the
   venue-native version of step 9a costs a poll rather than a download.

   **Run, 20 minutes at 5s on 2026-09-09, and BloFin is systematically wider.**
   57 instruments cleared a $1M/24h volume filter; **37 clear the 2.20 bps
   VIP 1 bar**, against one of ten on Binance. On the four majors both surveys
   cover:

   | symbol | Binance (2026-09-01) | BloFin (2026-09-09) | ratio |
   |---|---|---|---|
   | ADA | 5.019 | 9.091 | 1.8x |
   | LTC | 2.053 | 5.530 | 2.7x |
   | AVAX | 1.378 | 3.749 | 2.7x |
   | DOGE | 1.207 | 3.316 | 2.7x |
   | **BTC** | **0.013** | **0.013** | **1.0x** |

   BTC is the row that makes the rest of the table mean something. It is
   identical to three decimal places, because BTC is pinned at its minimum
   tick on both venues and a pinned spread is arithmetic, not competition:
   `tick/price`, and both are $0.1 on a ~$78.7k instrument. So BloFin is
   **wider exactly where makers have room to choose the width, and identical
   where they do not**, which is the signature of less maker competition
   rather than a measurement artifact or a units error.

   The shape of the spread differs too, and this cuts against the opportunity.
   Binance's wide names were *pinned* — p25 and p75 within 0.01 bps of the
   median. BloFin's are not: ADA runs p25 4.551 / median 9.091 / p75 13.609,
   oscillating between sitting on its tick floor and three times wider. Only
   one instrument in the top 30 (NEAR) was tick-bound at all. A spread that
   wide and that variable on $2.4M of daily volume is at least as consistent
   with *nobody quoting* as with *money available for quoting*, and those two
   have very different consequences.

   **What this does not establish.** The headroom is bigger; the net is
   unknown. Converting spread into a net number needs the markout simulation,
   and that needs book and trade data for the instrument — `passive_sim.py`
   measured Binance ADA's optimistic markout at +1.967 against a 5.019 spread,
   so a quote at the touch captured well under half its width. Extrapolating
   that ratio to BloFin would be inventing the result. Two things must be
   measured on BloFin itself before any of this is tradeable:

   - **Adverse selection.** Assumed at 0.5 bps/leg from Binance throughout.
     Never measured here, and a less arbitraged venue is not obviously the
     same in either direction.
   - **Queue position.** Still the largest single unknown in the branch — a
     6.30 bps bracket on ADA against a 4.00 bps total fee ladder.

   Both need exactly what `recorder.py` already produces, pointed at one of
   these instruments instead of BTC-USDT. That is step 9d, and
   `backendnalysisenue_compare.py` is the tool that reads the answer out:
   it runs the identical `passive_sim` pipeline over both venues for the same
   instruments and reports the paired difference in adverse selection, with
   BTC-USDT as a control for the fact that the two sides are different days.

   **First run, 42 minutes in, and the control did its job:** BTC showed
   −2.901 bps `[-4.21, -1.59]` on an instrument that is tick-bound and
   identically priced on both venues and therefore cannot carry a venue
   effect. So the tool refused to draw a venue conclusion and said the
   comparison is date-confounded, which is correct. Re-run once there are
   6+ hours per instrument.

   The dates are eight days apart, which is the honest caveat on the ratio
   column. A consistent 2.7x across three independent instruments is larger
   than typical day-to-day variation, but it is not the same as a same-day
   comparison.

9d. **Record the shortlist.** ~~Fix the filename first~~ — `recorder.py` used
   to write `features-<date>.csv` with no instrument in the name and no
   instrument column, so two symbols would have merged into one matrix that
   nothing downstream could object to. Both writers are now scoped to
   `data/<INST-ID>/`, `check_features.py` refuses to cross that boundary, and
   `backend
ecord.py` runs N instruments headless in one process.

   **Running since 2026-09-09, 15 instruments**, chosen so the data answers
   two different questions:

   | group | instruments | what it buys |
   |---|---|---|
   | paired with Binance | BTC, ADA, DOGE, LTC, AVAX | `passive_sim.py` has already measured adverse selection on these five *on Binance*. The same five on BloFin make the venue comparison paired rather than anecdotal — and adverse selection at 0.5 bps/leg is the assumption the whole branch rests on. |
   | wide with real volume | PUMP, PEPE, 1000BONK, WLD, SUI, INJ, IOST | the actual passive candidates: 3.7-9.8 bps against a 2.20 bps bar, on $2-8M/day |
   | wide and thin | ATOM, CNPY, FARTCOIN | 16-18 bps on ~$1.2M/day. The widest spreads on the venue sit on the *lowest* volume, which is the signature of nobody quoting rather than money waiting. Recording them is how that gets settled instead of assumed. |

   BTC-USDT is in the first group twice over: it is the tick-bound control
   that made the venue comparison credible, and it carries the existing
   prediction-branch history, which now lives in `data/BTC-USDT/`.

9e. **Funding carry** — `backendnalysisunding_carry.py`. Every branch so
   far died on execution cost against a forecast that was too small. Carry
   needs no forecast: hold spot, short the perp against it, collect funding
   while delta-neutral.

   The vendored SDK is futures-only and `getInstruments()` returns 487 SWAPs,
   which made this look impossible. It is not — BloFin's spot endpoints are
   simply unwrapped, and answer fine when given `instType=SPOT`:
   **242 spot pairs, 175 of them with a matching linear perp.**

   One asymmetry shapes everything: there is no borrow, so **the spot leg can
   only be long**, and only POSITIVE funding is harvestable. That rules out
   BTC-USDT immediately — its median funding is below zero.

   **Run, and for the first time in this project something clears its cost by
   a wide margin.** 42 of 175 pairs clear a four-leg round trip over 30 days
   while crossing every leg:

   | | fund/day | pos% | spot spr | perp spr | cross RT | b/e days | net 30d | net after convergence |
   |---|---|---|---|---|---|---|---|---|
   | XRP-USDC | 8.40 | 100% | 3.52 | 1.41 | 24.93 | 3.0 | +227.1 | +218.3 |
   | SUI-USDT | 7.29 | 99% | 6.15 | 6.15 | 32.30 | 4.4 | +186.4 | +179.0 |
   | XRP-USDT | 6.93 | 100% | 4.93 | 0.70 | 25.64 | 3.7 | +182.3 | +175.2 |
   | LINK-USDT | 6.60 | 92% | 4.01 | 0.80 | 24.81 | 3.8 | +173.2 | +167.6 |
   | BTC-USDT | −0.60 | 42% | 0.23 | 0.01 | 20.24 | never | −38.2 | −41.2 |

   The constraint is the **spot spread**, not the funding rate: the perp side
   is often under 2 bps while the spot side runs 4-50, and a four-leg round
   trip is days of carry before the position is flat. That is why the gate is
   a break-even in days rather than a rate.

   Reasons this is not yet a result, in order of how much they could move it:

   - ~~**The spot fee schedule has never been read off the account.**~~ It
     still has not been, but a sensitivity sweep settled whether that matters
     and the answer is no. Pushing the spot taker from 0.6 to 30 bps — five
     times the futures taker — moves the count of clearing pairs from 38 to
     29 and the best net over 30 days from +219 to +160 bps:

     | spot taker bps | pairs clearing | best net 30d | best b/e days |
     |---|---|---|---|
     | 0.6 | 38 | +219.0 | 2.3 |
     | 5.0 | 37 | +210.2 | 3.4 |
     | 10.0 | 35 | +200.2 | 4.6 |
     | 20.0 | 31 | +180.2 | 6.9 |
     | 30.0 | 29 | +160.2 | 9.3 |

     A four-leg round trip is tens of bps and a 30-day hold collects hundreds,
     so fees are simply not the binding term over that horizon. Worth
     confirming before trading; not worth waiting for.
   - **33 days of funding history is one regime, and this is now the binding
     unknown.** The whole result is "funding stays roughly where it has been
     for a month". Positive-share and worst cumulative drawdown are reported
     per instrument for exactly this reason: RAY-USDT pays 76% of periods and
     has given back 27 bps in a run. Nothing here forecasts funding, and a
     regime where these rates compress or invert removes the entire return
     while leaving all four legs of cost in place.
   - **Size is not modelled.** Every number comes off the top of book, and the
     spot side of this venue trades $1-8M/day. A carry you can only put on in
     small size is a different proposition from the basis-point figures here.
   - **The basis is a real exposure.** The perp trades *below* spot on every
     instrument measured, so the hedge buys the expensive leg and shorts the
     cheap one; `net-conv` is the result if that gap closes completely.
   - **Returns are on notional, not capital**, and liquidation is not modelled
     at all — a delta-neutral position still has a leveraged leg that can be
     liquidated.

9f. ~~**Backtest the carry**~~ — `backendnalysis\carry_backtest.py`. The
   screen in 9e answers "does the median funding rate cover the round trip?",
   which is not a P&L: it assumes you enter today at today's spread and that
   funding behaves like its median for a month. This runs the position from
   **every** entry in 200 days, collecting the funding that actually printed
   and applying the basis move that actually happened.

   | instrument | median | p5 | worst | best | profit% | fund | basis | cost | eff N |
   |---|---|---|---|---|---|---|---|---|---|
   | CRV-USDT | +167.5 | +117.3 | +82.6 | +239.2 | 100% | +201.3 | +0.0 | 38.6 | 5 |
   | XRP-USDT | +121.4 | +79.4 | +66.1 | +180.9 | 100% | +153.9 | −0.2 | 32.5 | 5 |
   | DOGE-USDT | +149.3 | +76.9 | +63.9 | +190.5 | 100% | +180.4 | −0.0 | 29.9 | 5 |
   | SUI-USDT | +151.5 | +64.6 | +48.5 | +190.7 | 100% | +179.5 | −0.1 | 28.2 | 5 |
   | LTC-USDT | +93.7 | +30.6 | +14.6 | +163.3 | 100% | +135.0 | −0.1 | 40.4 | 5 |
   | **DOT-USDT** | **+92.3** | **−157.9** | **−195.3** | +201.3 | 70% | +146.3 | −0.2 | 49.6 | 5 |
   | BTC-USDT | −46.0 | −99.5 | −110.2 | +46.1 | 32% | −22.6 | −0.1 | 23.7 | 5 |

   **Five instruments were profitable from every entry point in the history**,
   which is the strongest statement this data can make. And DOT-USDT is why
   the backtest was worth building rather than trusting the screen: **+92 bps
   on median and −158 at the 5th percentile.** The screen would have sold you
   that one.

   Three things the table is built to stop you misreading:

   - **`eff N` is 5, not 510.** Adjacent windows share 89 of their 90 periods,
     so 200 days contains about five independent 30-day holds. "Profitable in
     100% of 510 windows" is one window counted 500 times, and the report says
     so beside every distribution.
   - **Ranked by p5, not median.** A carry is a position held through whatever
     arrives, so the bad entries are the question.
   - **The basis is variance, not a cost.** Its median 30-day move is ~0 —
     SUI's gap ranges −17.7 to +17.3 bps and mean-reverts — but the tail runs
     to ±56 bps on DOT, and that tail is already inside the p5 and worst
     columns rather than being bolted on as a worst case.

   Spreads remain the one assumption: there is no historical top of book for
   these markets, so entry and exit cost is today's spread held constant. It
   is the smaller assumption, since the four-leg cost is tens of bps against
   funding swings of hundreds.

9g. ~~**Plan the carry**~~ — `backend	rading\carry.py` and
   `backend\plan_carry.py`. The analysis says which instruments to carry and
   what they earned; neither says what to *do*. This computes the orders,
   the capital, the liquidation price and the expected return, and prints
   them. **It sends nothing** — there is no code path from it to
   `placeOrder`, which is checkable with a grep and is checked that way.

   ```
   python backend\plan_carry.py --instrument SUI-USDT --notional 2000 --leverage 3
   ```

   | | |
   |---|---|
   | BUY spot | 2534 SUI @ ~0.7896 ($2,000.85) |
   | SELL perp | 2534 contracts @ ~0.789, isolated 3x |
   | capital | $2,000.85 spot + $666.57 margin |
   | liquidation | 1.0461, **32.6% away** |
   | round trip | 32.14 bps, break-even 4.9 days |
   | expected 30d | +164.1 bps = $32.81 |

   Three things the planner exists to get right, none of which the analysis
   layer touches:

   - **Delta neutrality has to survive rounding.** The spot leg trades in base
     units on one lot size and the perp leg in contracts on another. Rounding
     them independently leaves the position quietly directional, so the perp
     leg is sized first in whole lots, spot is derived from it, and any
     residual the spot lot forces is reported in base and dollars.
   - **The short leg is liquidatable on its own.** Delta-neutral is not
     risk-neutral — the legs margin separately, and a rally that leaves the
     pair flat can still take the short out, leaving an unhedged long. The
     liquidation price uses the MMR and fee buffer measured in step 1's
     validation. Isolated margin, deliberately: cross would back the short
     with the spot leg's cash, which reads as a wider buffer and is really a
     bigger blast radius.
   - **Refusals are plural.** Every failing gate is listed, not the first.
     `--leverage 10` returns both "liquidation only 9.4% away, needs 15.0%"
     and "leverage 10 exceeds the 5 limit".

   What it deliberately does not decide: which leg to send first. Whichever
   fills leaves you directional until the other does, and that belongs to the
   executor that sends them.

9h. ~~**Open the carry**~~ — `backend\trading\carry_executor.py` and
   `backend\run_carry.py`. Dry unless `--confirm`, demo unless `--production`,
   and `--production --confirm` together are refused outright. Perp leg first
   so a failed spot leg leaves a short that closes instantly with a
   `reduce_only` order rather than a long that has to be sold; a spot leg that
   fails unwinds the perp instead of retrying. The hedge size is checked
   against the BALANCE, not the order response, because `targetCurrency`
   decides whether a market order's size means base or quote and the wrong one
   fills cleanly at the wrong size.

   **Run live on demo 2026-09-09, and the unwind path fired for real.** The
   first attempt's spot leg failed, the perp leg was closed automatically, and
   the second attempt filled both. Cost of the aborted open: $0.265, which is
   7.8% of the trade's whole 30-day expected profit — the argument for getting
   leg sequencing right is not theoretical.

9i. ~~**Watch it**~~ — `backend\trading\strategies\carry\monitor.py` and
   `backend\monitor_carry.py`. Everything above finishes in seconds and
   produces a position that takes a month. Nothing read it back, so
   "+170.4 bps over 30 days" was a forecast with no scoreboard.

   ```
   python backend\monitor_carry.py --instrument SUI-USDT
   ```

   - **Funding is derived, because it cannot be read.** This API version has
     no account-bills endpoint — `/api/v1/account/bills` and `-archive` both
     answer "not supported", and `/api/v1/asset/bills` covers transfers.
     `realizedPnl` on the position accumulates fees, closed-trade PnL and
     funding together, and everything but funding is observable from the
     fills, so `funding = realizedPnl + fees - fillPnl`. Checked against the
     position before any settlement had passed: `realizedPnl` was
     -0.11975592 and the entry fee was 0.11975592, so it returns exactly zero
     when zero is the true answer.
   - **A derived number gets a control.** The same funding is estimated
     independently from the public funding-rate history and the two are
     compared. When they disagree the report says so rather than preferring
     the one it computed itself.
   - **The baseline is reconstructed, not remembered.** The first carry was
     opened by a run that persisted nothing. The position carries its entry
     and creation time, the fills carry their fees, and the executor tagged
     both legs `carry<hex>p` / `carry<hex>s` — which is the only exact join
     between a futures position and the spot balance hedging it. Frozen to
     `data/<INST>/carry/baseline.json` on first run and never rewritten, so
     the forecast stops moving. Snapshots append beside it.
   - **The scoreboard is not the plan's.** The plan priced a 25.82 bps round
     trip off VIP 1; the demo account was charged VIP 0 — 6.0 bps perp and
     10.0 bps spot, measured from the fills. The real round trip is 32.0 bps
     and break-even is 4.9 days, not 3.9. Entry spreads are not recoverable
     after the fact, so this still understates it, and says so.
   - **It re-validates the liquidation model on every reading**, not once at
     open. Currently 0.04 bps from BloFin's own number.

   What ends a hold early gets its own alerts: a leg that vanished (either
   direction is named for what it leaves you holding), liquidation inside
   `min_open_liquidation_buffer_pct`, delta drift, funding that has gone
   negative, and funding far enough under forecast that break-even has moved
   past the hold.

   The delta threshold is deliberately NOT the planner's 0.001. That is a
   tolerance on lot *rounding*, measured before fees, and the commonest way a
   carry's delta actually breaks is a spot leg short by exactly its fee —
   0.1% of base, landing at 0.0997% of notional. A threshold set at 0.001
   sits a rounding error away from the one failure it most needs to catch,
   and misses it.

9j. ~~**Close it**~~ — `backend\close_carry.py`, and `CarryExecutor.close()`.
   The 32 bps round trip priced two exit legs no code could send.

   ```
   python backend\close_carry.py --instrument SUI-USDT
   python backend\close_carry.py --instrument SUI-USDT --confirm
   ```

   - **Spot leg first — the opposite of the open, for a stronger reason.**
     The open sends the perp first so a failed spot leg leaves a short that
     closes instantly. The close orders the legs so that *a first-leg failure
     is a no-op rather than a position*. The spot sell is the leg that can
     actually be refused: no `reduce_only` protection, needs a real balance,
     and the fee convention on a sell is still unmeasured (the BUY was charged
     in base currency, which is why the hedge came up 0.254 SUI short). If it
     bounces, the carry is still fully hedged and retrying costs nothing. What
     follows is a `reduce_only` perp close — deepest book, always approved,
     cannot overshoot into a long.
   - **Never unwound.** `open()` refuses to leave half a carry on. The close
     must NOT inherit that: undoing a close means re-opening the position
     somebody just decided to exit. A failed perp leg retries and then
     screams; re-buying spot to re-hedge is deliberately not done, because an
     executor that opens risk during a close is a surprise, and if the venue
     is rejecting orders the re-hedge is as likely to fail as the retry.
   - **Sized from the exchange, never from the plan** — which is weeks old by
     then, with fees out of the spot balance and funding through the margin.
     The useful consequence is that it is *idempotent*: interrupted after one
     leg, run it again and it finishes from whatever is left.
   - **Funding is captured immediately before the orders go.** It is derived
     from `realizedPnl`, which ceases to exist with the position, and there is
     no bills endpoint to recover it from. A 30-day test that cannot say what
     it earned has not concluded, it has only stopped. The exit fees are read
     after, because they do not exist until then, and the two halves are
     stitched into `data/<INST>/carry/closed.json`.

   A separate entrypoint rather than `run_carry.py --close`, because with a
   flag, the open command minus the flag **opens a second carry** — the most
   expensive typo available here. One verb per entrypoint deletes that.

9k. ~~**Fade the range**~~ — `backend\trading\strategies\range_trade\levels.py`
   and `backend\analysis\range_backtest.py`. Buy near the bottom of the last N
   hours' range, sell near the top, stop beyond both edges, take profit at the
   middle, with leverage. Proposed 2026-09-11 for a market that had been
   "trading sideways for three weeks".

   ```
   python backend\analysis\range_backtest.py
   ```

   The naive backtest of this lies in two ways, and the tool is built around
   both:

   - **Win rate is geometry.** A bracket with its stop S away and target T
     away wins S/(S+T) of the time on a driftless random walk, and still loses
     its fees. Out of sample the chosen configuration won **64%** of its
     trades, averaging +90.9 bps against losses of −185.4, and lost money.
   - **OHLC does not say whether the stop or the target printed first**, so
     fills are bracketed (1 bp trade-through and stop-first, against touch and
     target-first). A **shuffled-day control** — each UTC day's 1m bars
     permuted, which keeps the day's net move and volatility and destroys only
     the order of moves inside it — says what a random walk with the same days
     would have made. `real − control` is the mean-reversion edge, which is
     the only thing a fade can be harvesting.

   **Run on 365 days of Binance 1m klines for ten majors at BloFin VIP 1
   fees, and the answer is no.** 24 configurations (lookback 4/24/72h × entry
   0.1/0.25 × stop 0.25/0.5 × trend filter off/≤0.5), and **not one had a
   positive in-sample median across symbols.** The least bad (24h, entry
   0.25, stop 0.25), scored on the last 110 days without re-fitting:

   | | bps per trade | 95%, whole-day blocks |
   |---|---|---|
   | real, pessimistic | **−9.0** | [−21.4, +2.9] |
   | real, optimistic | −6.0 | [−18.9, +6.2] |
   | control, pessimistic | −0.8 | [−10.8, +9.0] |
   | real − control | −8.2 | [−19.1, +2.7] |

   2,966 trades, 9 of 10 symbols negative. The comparison with the control
   leans the wrong way: inside a day, moves at these scales tended to
   *continue* rather than reverse. That is not significant, but there is no
   hint of the reversion the strategy is a bet on. Ranking the grid in sample
   did not predict its ranking out of sample (Spearman **+0.07**).

   **Was it sideways?** The last 21 days against every 21-day window in the
   year, by trend ratio (net move / range width): BTC at the 45th percentile,
   ETH the 52nd — an ordinary three weeks for both. DOGE, ADA and AVAX were
   genuinely range-bound (10th–14th). Over exactly those 21 days the fade made
   −0.8 (pessimistic) / +5.2 (optimistic) bps per trade with an interval of
   about ±40, and the control made +3.5 on the same days. 21 days of one
   regime cannot tell this strategy from noise.

   **Leverage.** At 5x full allocation every symbol lost capital out of
   sample: ending equity 0.03–0.47x, max drawdown 83–99%. DOGE, the one symbol
   ahead at 1x (1.018x), finished at 0.47x — a +1.8 bps mean does not survive
   five times the variance. Liquidation never fired, since the stops sit well
   inside a 5x liquidation price. It did not need to.

   One thread worth pulling, and no more than that: the only positive
   out-of-sample medians in the grid all belong to the four **72h
   trend-filtered** configurations (+2.7 to +15.1), and all four were negative
   in sample (−7.0 to −12.6). That is either noise getting a second draw or a
   sign the regime filter needs a window of weeks rather than the lookback's.
   This data cannot say which, and a filter measured over weeks is the next
   test. Also not covered: stops triggered on BloFin's mark price rather than
   last trade (which would skip some wicks), and funding.

   No planner or executor was built. The plan/execute/monitor split exists so
   that order code is asked for after the evidence, and the evidence is
   negative.

9l. ~~**Price the forecast before building it**~~ —
   `backend\analysis\range_information.py`, `backend\analysis\range_harness.py`
   and `backend\analysis\fetch_klines.py`. Step 9k's fade lost money, and the
   obvious rescue is to forecast the range with a model. This prices that
   forecast before anyone builds one.

   ```
   python backend\analysis\range_information.py
   python backend\analysis\range_harness.py --target centre --model ridge
   ```

   The fade is fed a range interpolated in log space between the trailing
   range (no information) and the true future 24h high/low (perfect
   information), with the centre and the width dialled in separately. Five
   majors, a year of 1m bars, pessimistic fills, VIP 1:

   | centre known | width known | bps per trade |
   |---|---|---|
   | 0% | 0% | −7.1 |
   | 0% | **100%** | **−15.9** |
   | 5% | 0% | −9.2 |
   | 10% | 0% | −7.0 |
   | 15% | 0% | −2.9 |
   | 20% | 0% | +2.5 |
   | 25% | 0% | +11.2 |
   | 50% | 0% | +68.7 |
   | 100% | 0% | +102.1 |

   **Perfect knowledge of the next day's range WIDTH is worth less than
   nothing — it makes the strategy worse.** All of the value is in the CENTRE,
   and break-even sits at a centre IC of **0.177**. The curve is not linear
   near zero — it *dips* before it climbs, so a little centre skill is worse
   than none, and interpolating the threshold from the coarse 0/25% pair gave
   0.097 against a measured crossing of 0.177. Nearly double, in the direction
   that flatters a model, which is why the sweep now steps through 0.05–0.20.
   A range forecast that
   is symmetric around the current price — which is what a volatility model, a
   quantile regression on the high/low, or a Monte Carlo path simulation
   produces — has a centre IC of zero by construction. It forecasts the
   worthless half. The predictable half and the profitable half are different
   halves.

   `range_harness.py` is the scoreboard that follows from that, so model
   classes get measured instead of argued about: purged time split, effective
   N (a year is ~363 independent 24h windows per symbol, not 8,712 rows), a
   shuffled-label control over several seeds, sign agreement across symbols,
   and an IC→bps conversion so the verdict is about money rather than
   significance. A model is structural — `fit`/`predict`, no base class — and
   `--dump` / `--predictions` scores one that cannot run in this process
   (PyTorch, an LLM pipeline, an RL policy), with timestamp alignment checked
   rather than trusted: a prediction file one row out of step scores like a
   signal.

   First entries on the board — ridge, ten majors, 110 out-of-sample days:

   | target | pooled IC | control ceiling | reading |
   |---|---|---|---|
   | centre | +0.064 (t 2.12) | **+0.062** | does not separate from its own control; worth −8.6 bps against a 0.177 break-even, so it needs 2.7x this skill |
   | width | +0.600 (t 24.7) | +0.289 | real, and measured above as worth nothing |
   | contained | +0.242 (t 8.19) | +0.134 | the untested idea: it gates WHEN to fade, not where |

   Two things to read carefully there. The centre model does not separate from
   the same pipeline trained on shuffled targets, which is step 7 repeating on
   a new target. And the shuffled control itself reaches |IC| 0.13–0.29,
   because pooling symbols lets a prediction correlated with a symbol's
   average level score without any timing skill at all — so per-symbol
   demeaning belongs in the next version of the harness, and every pooled IC
   here should be read against its control rather than against zero.

   `fetch_klines.py` is the other half of the answer, since 363 windows cannot
   resolve an IC below ~0.06 at two standard errors and no architecture fixes
   that: 100 crypto perpetuals, five years of 1m bars, ~10 GB, ~90 requests
   per symbol-year. Two things it refuses, both found by running it. Tokenised
   equities and commodities — Binance lists gold, crude, silver and single
   stocks as `TRADIFI_PERPETUAL`, and seven were in the top hundred by volume;
   they trade in sessions, so their overnight gaps would read as market
   structure. And symbols whose names are not ASCII — four CJK meme tickers in
   the top hundred, unfetchable because the archive path is the symbol
   verbatim, one of which sat sixth by volume and ended the first run.

   **Then the sample was expanded 30x, and the centre signal disappeared.**
   35 symbols with a full five years, 454,305 rows, **18,922 independent
   windows** against 1,080:

   | | ten majors, one year | 35 symbols, five years |
   |---|---|---|
   | **centre** IC | +0.064 (t 2.12) | **+0.011** (t 1.55) |
   | its control ceiling | +0.062 | +0.039 |
   | symbols positive | 8/10 | 21/35 |
   | **contained** IC | +0.242 (t 8.19) | **+0.267** (t 38.0) |
   | its control ceiling | +0.134 | +0.098 |
   | symbols positive | 10/10 | **35/35** |

   **The two targets moved in opposite directions, and that contrast is the
   result.** More data made the centre skill *shrink toward zero* — what a
   small-sample draw does, and the opposite of what a real effect does. The
   ten-major +0.064 was noise that happened to point one way, and it never
   separated from its control even then. `contained` did the reverse: it held,
   strengthened slightly, and came back positive on **every one of 35
   symbols**, far clear of its control.

   That settles the question the expansion was built to answer: **the sample
   was not the binding constraint.** The data now resolves an IC of ~0.015 at
   two standard errors, break-even is 0.177, and the measurement is 0.011 —
   an order of magnitude short, with the error bars to say so. A bigger model
   over these features is not searching for a signal too faint to fit; it is
   searching where this says there is nothing, and the harness makes that
   claim cheap to overturn rather than merely asserted.

   What this does not settle: `contained` has never been converted into money,
   which needs the gated backtest rather than an IC; the features are bar-level
   only, so nothing here speaks to order flow, the liquidation map or news; and
   no larger model has been run against the bar yet. Making that cheap is what
   the harness is for.

9m. ~~**Gate the fade on the one signal that survived**~~ —
   `backend\analysis\range_gated.py`. `contained` was the only target left
   standing: IC +0.267 on 35 symbols over five years, positive on every one of
   them. An IC is not money, so this converts it — the fade run twice over the
   same out-of-sample bars, once ungated and once with entries allowed only
   where the model says the range holds.

   ```
   python backend\analysis\range_gated.py --keep 0.5
   ```

   **Gating on it makes the fade worse.** Ten majors over a year: −24.2 bps
   per trade gated against −9.2 ungated, all ten symbols degraded, and worse
   than a control gate that trades as little at shuffled times (−11.8). On 35
   symbols over five years, 49,056 trades:

   | | bps per trade | 95%, whole-day blocks | trades |
   |---|---|---|---|
   | ungated | **−7.7** | [−13.3, −2.3] | 49,056 |
   | gated | −10.6 | [−21.7, −0.3] | 19,717 |
   | control gate | −7.2 | [−13.6, −1.2] | 35,496 |

   **Why, and it is close to tautological:** predicted containment correlates
   **+0.70 with the range WIDTH** at entry. The model learned that wide ranges
   hold — true, and useless, because width was already measured at −15.9 bps
   *when known perfectly*. Sorting the out-of-sample trades by prediction:

   | quintile | range width | net bps | time exits | mean hold |
   |---|---|---|---|---|
   | lowest | 217 bps | −2.6 | 1% | 3.2h |
   | middle | 355 bps | −6.2 | 5% | 5.6h |
   | highest | 646 bps | −16.1 | 14% | 10.2h |

   A wide range takes longer to traverse, so it reaches the time stop 14x more
   often and pays taker to get out. The gate was not selecting sideways
   markets; it was selecting big ranges.

   **The by-product is the firmer result.** Step 9k could only say the fade was
   not distinguishable from zero on one year of ten majors. Over five years and
   35 symbols the ungated fade is **−7.7 bps per trade with an interval that
   excludes zero**. It loses money, and now there are error bars saying so.

   So all three things a range model could forecast are priced: the centre is
   unpredictable (IC +0.011, under its own control), the width is predictable
   and worth less than nothing, and containment is predictable, largely a
   restatement of width, and loses more when traded. That is the case against
   this strategy closed from three directions rather than one — and the harness
   is what makes the next candidate cheap to test instead of cheap to argue
   about.

   `simulate` gained an optional entry gate for this. Exits are deliberately
   never gated: a gate that could strand an open position is a far worse
   instrument than one that declines to open another.

9n. ~~**Stocks, commodities, and synthetic data**~~ —
   `backend\analysis\reversion_scale.py`. Three proposals with three different
   answers, and the third one found the mechanism the whole branch had been
   missing.

   ```
   python backend\analysis\reversion_scale.py --sweep
   ```

   **Tokenised equities and commodities: shorter history, worse results.**
   Binance lists 191 `TRADIFI_PERPETUAL` contracts, but the longest is XAUUSDT
   at 275 days (listed 2025-12-11) and most single stocks are 60–160 days old,
   so as a *sample* they are smaller than the crypto majors already give. Run
   the same fade on gold, silver, crude, Brent and four stocks over 270 days:
   **−17.6 bps per trade [−29.4, −6.5]** out of sample, 0 of 8 symbols
   positive, against −7.7 for crypto. Newer, thinner contracts with wider
   spreads, and equities carry session gaps on top.

   **Synthetic data cannot supply an edge.** A generator returns the
   assumptions put into it, and a mean-reverting generator makes any fade look
   brilliant. What it can do is calibrate a yardstick — Ornstein-Uhlenbeck
   paths at the majors' volatility, the identical fade, VIP 1 fees:

   | reversion half-life | variance ratio | net bps per trade |
   |---|---|---|
   | 1h | 0.055 | +56.0 |
   | 6h | 0.322 | +41.1 |
   | 24h | 0.709 | +16.6 |
   | 48h | 0.824 | +10.8 |
   | none (random walk) | 0.986 | **−0.1** |

   The random-walk row is also a check on the simulator: gross +2.6 bps
   against a standard error of ~4, fees 2.7, so it pays its costs and invents
   nothing.

   **Then the yardstick explained the last four steps.** The variance ratio
   over 24h sits below 1 for the majors — BTC 0.90, DOGE 0.61 on a 1-minute
   base — which reads as mean reversion and is precisely what a range fade
   bets on. Those figures sit in the band where the table above pays +10 to
   +25 bps. Measure the same ratio against longer base intervals:

   | | 1m base | 5m | 15m | 60m | |
   |---|---|---|---|---|---|
   | BTCUSDT | 0.904 | 0.953 | 0.985 | **1.044** | artifact |
   | SOLUSDT | 0.824 | 0.797 | 0.922 | **1.025** | artifact |
   | DOGEUSDT | 0.613 | 0.371 | 0.949 | **1.015** | artifact |
   | ADAUSDT | 0.658 | 0.401 | 0.795 | **1.041** | artifact |
   | synthetic OU, 24h half-life | 0.796 | 0.799 | 0.793 | **0.794** | scale-invariant |
   | synthetic random walk | 1.074 | 1.076 | 1.066 | 1.056 | none |

   **Genuine reversion is scale-invariant** — the OU path holds 0.79 whether
   sampled every minute or every hour. What decays as the base lengthens is
   **bid-ask bounce**: price alternating between touching bid and ask inflates
   the shortest interval's variance and nothing else. That is the Roll (1984)
   effect, it is indistinguishable from mean reversion in the statistic, and
   it is untradeable — the bounce *is* the spread, which a fade pays on the
   way in rather than harvests.

   Four of five majors show reversion at the shortest base that is gone by the
   hourly one. That is why step 9m's fade loses 7.7 bps while its headline
   variance ratio looks like a strategy that should earn +16. The statistic was
   never measuring what it appeared to measure, and this is the check to run
   **before** writing the next reversion strategy rather than after.

9o. ~~**Ranges at several timeframes, per side**~~ —
   `backend\analysis\range_multiscale.py`. Step 9l scored one symmetric
   `contained` target — did price stay inside *both* edges of *one* range —
   which collapses the two sides into a bit and throws the direction away.
   This computes the range at 4h, 1d, 3d and 7d, predicts each side of each
   scale separately, and asks whether knowing which scales break says where
   price ends up.

   ```
   python backend\analysis\range_multiscale.py --scale-hours 4,24,72,168
   ```

   **What the idea reduces to.** A break at scale L is
   `log(future_high/close) > log(range_high_L/close)`, and the left side does
   not depend on L. So nested ranges are a **discretised CDF of the same two
   quantities** — the next 24h up- and down-excursion — read at different
   thresholds. The scales re-parameterise location (the centre) and spread (the
   width) rather than adding a third thing. That is an argument for a low prior
   and not a result, and it has a real hole: these are tail *classifications*,
   and a classifier on a tail can find structure a least-squares fit on a mean
   misses. So it was measured.

   **Predicting the breaks works, and is the easy half.** On 35 symbols over
   five years: IC **0.37–0.55** at every scale on both sides, **35/35 symbols
   positive**, against controls of 0.09–0.18. A wide range holds and a narrow
   one breaks.

   **Predicting direction: it worked on one year and dissolved on five.**

   | | 10 majors, 1 yr | 35 symbols, 5 yrs |
   |---|---|---|
   | centre, 1d features alone | +0.075 (ctrl 0.062) | +0.013 (ctrl 0.016) |
   | centre, all four scales | −0.025 (ctrl 0.065) | +0.020 (ctrl 0.014) |
   | **gap between the two sides** | **+0.078** (ctrl 0.057) | **+0.008** (ctrl 0.026) |

   On one year the per-side framing looked like a real improvement: +0.078
   where regressing the centre on the same features gave −0.025, above its own
   control, and the gap scored +0.088 to +0.097 at the 1d, 3d and 7d scales.
   On the full sample it is +0.008 against a control ceiling of 0.026 — below
   the noise floor, and 22x short of the 0.177 break-even.

   **That is now the third target to decay the same way**, after the centre
   (+0.064 → +0.011) and the gated fade. A result on one year of ten majors
   that sits just above its control is what a draw looks like; the expanded
   sample is what tells the two apart, which is the whole reason step 9l built
   it.

   One thing that did NOT behave as predicted, and is worth keeping: the two
   sides come back **negatively correlated, −0.41 to −0.75**, on both samples.
   The model is not merely forecasting the size of the next move — it leans one
   way or the other, driven by the momentum and range-position features. The
   lean is simply uninformative about where price actually lands.

9p. ~~**Attention across timeframes**~~ — `backend\analysis\attention_model.py`,
   `range_multiscale.py --model attention`. Step 9o combined the scales
   linearly, which cannot rule out that the scales interact. This puts one
   token per scale and lets them read each other before anything is predicted.

   ```
   python backend\analysis\range_multiscale.py --model attention --scale-hours 4,24,72,168
   ```

   ~1,300 parameters over four tokens, in numpy with hand-derived gradients —
   torch would be a multi-GB dependency for matrices this size, in an analysis
   layer that needs numpy alone. A hand-derived backward pass is worth exactly
   what its gradient check is worth, so the first test is central finite
   differences on **every** parameter, and the second plants a cross-scale
   PRODUCT no linear model can represent, which attention must beat ridge on by
   30% of MSE.

   **That second test earned its place immediately.** The first version
   mean-pooled the tokens, and since the input projection is shared across
   scales, the head saw only the average token — it could not recover even a
   plain linear target. Concatenating the tokens fixed it. Without a
   planted-signal test, an underfitting model's failure would have been written
   up as a fact about the market.

   **Ten majors, one year (80,640 rows): worse than ridge everywhere.**
   Break targets 0.276–0.500 against ridge's 0.305–0.538; centre −0.027 against
   −0.025. A capacity sweep made it worse rather than better — d64 over 60
   epochs drops `break_up@1d` to **+0.374** against ridge's +0.534, which is
   overfitting a target whose signal is mostly linear.

   **35 symbols, five years (1,480,232 rows): parity on the learnable target,
   nothing on the one that pays.**

   | target | ridge | attention |
   |---|---|---|
   | centre | +0.020 (control 0.001) | +0.012 (control **0.024**) |
   | break_up@1d | +0.512 (control 0.102) | **+0.518** (control **0.299**) |
   | fit time, centre | 6s | 775s |

   Read the controls, not the headline. Ridge clears its own noise floor by
   **0.410** on `break_up@1d`; attention by 0.219, because the same pipeline
   fitted on shuffled labels still reaches 0.299. On the centre, attention sits
   *below* its own control.

   **The data argument was half right.** 18x more data moved attention from
   clearly worse (−0.066 on `break_up@1d`) to nominal parity (+0.006), exactly
   as "it is data-hungry" predicts. It closed that gap on the target that was
   never worth money, and closed nothing on the centre. Capacity was not the
   binding constraint at either sample size.

   One part is worth keeping: the attention map is coherent. Every scale reads
   the **1d and 3d** tokens (weights 0.29–0.43) and nearly ignores 4h
   (0.04–0.07), so the mechanism works and learned that the daily range carries
   most of what there is. There is simply nothing there to combine.

9q. **A cross-section of perpetuals, held for a week** —
   `backend\analysis\panel_daily.py` and `backend\analysis\factor_panel.py`.

   ```
   python backend\analysis\panel_daily.py
   python backend\analysis\factor_panel.py --vol-scale --hold-days 7 --top-frac 0.3
   ```

   **The observation that motivated it.** Steps 6 through 9o all forecast price
   over seconds to one day, and all of them died on cost rather than on
   prediction. The one branch that ever cleared its cost — the funding carry of
   9e-9j — cleared it by not forecasting anything: it is a cash flow held for
   30 days, so a 32 bps four-leg round trip is amortised over hundreds of bps
   of collected funding. Every dead branch and the one live one differ in the
   same two variables, and the quadrant nothing here had tested is **many
   instruments, held for weeks**. At a 7-day hold a 10 bps round trip sits
   against a cross-sectional return dispersion of several hundred bps, so the
   break-even IC is roughly 0.02 against the 0.177 that killed step 9l.

   `panel_daily.py` folds the 5.7 GB 1-minute archive into 108 perpetuals x
   1,825 days and joins Binance funding onto it. Two alignments decide the
   whole result and both are tested: a row closes at 00:00 UTC, and funding for
   day D is what ACCRUED during day D — the settlement stamped 00:00 on D+1,
   rounded to its nominal minute first, because settlements print milliseconds
   late and a raw `-1ms` leaves them on the wrong side of midnight. Bucketing
   them a day early hands every row tomorrow's carry today.

   `factor_panel.py` scores factors as money. Holding periods do not overlap,
   so the rebalance count IS the effective N and no correction is offered; the
   label is market-neutral and net of funding; the universe is point-in-time on
   both history and liquidity; cost is charged on turnover rather than per
   position; and gross is split into its **price** and **funding** legs,
   because for a carry factor that split is the result.

   **Seventeen factors, and the carry family is the one that stands up.** At a
   7-day hold, top/bottom 30%, inverse-vol sized, 10 bps per unit traded (twice
   the VIP 1 taker fee):

   | factor | net bps/period | 95% block | Sharpe | ann% | control |
   |---|---|---|---|---|---|
   | carry_7 | **+40.5** | [+15.6, +63.5] | 1.60 | +21.1 | −2.3 |
   | carry_mom (60/40 rank blend) | **+44.5** | [+18.0, +68.5] | 1.57 | +23.2 | −0.0 |
   | mom_14 | +12.5 | [−14.7, +37.7] | 0.36 | +6.5 | −7.2 |
   | carry_demeaned | +4.3 | [−23.7, +32.7] | 0.15 | +2.2 | −5.4 |
   | carry_accel | −6.5 | [−34.5, +20.5] | −0.23 | −3.4 | +3.4 |

   Four things distinguish this from the previous nineteen results, and they
   are the reason it is written up rather than discarded:

   - **The funding leg is a cash flow, and it is the same size every year.**
     Split by calendar year, carry_7's funding leg reads +18.1, +15.5, +18.1,
     +18.1, +31.3 bps per week. Its price leg reads +62.5, +4.2, −9.8, +72.5,
     +38.8. The stable half is the half that is not a forecast, which is
     exactly the shape of the only other thing that ever worked here.
   - **It replicates in 12 of 12 random halves of the universe.** Splitting the
     cross-section rather than the timeline gives as many replications as there
     are ways to cut it, over identical dates and regimes; both halves were
     positive in all six splits tried, at Sharpe 0.59 to 1.97. A handful of
     lucky symbols cannot do that.
   - **95 of 96 specification-grid cells are positive** — holds of 3/7/14/30
     days, top 10/20/30%, costs of 5/10/20/30 bps, and universes cut at $5M and
     $50M a day. The one negative is the most hostile corner of the grid. It
     degrades with cost and with liquidity exactly as it should, rather than
     having a peak where the search stopped.
   - **The signal is the LEVEL of funding, not a change in it.** Subtracting
     each coin's own 180-day mean destroys the effect (+4.3), and so does
     ranking on acceleration (−6.5). That makes it a persistent risk premium
     rather than a crowding-timing signal, which is both the more believable
     reading and the one that implies low turnover.

   **What it costs when it is wrong.** The worst week was 2024-02-26 at
   **−1,991 bps** on gross notional: SHIB +256%, PEPE +288% and BONK +180% in
   seven days, and the book was short all three because they had the highest
   funding. That is not a bug in the backtest, it is the trade — a carry book
   is short whatever is crowded, and occasionally the crowd is right and
   violent. Worst drawdown over the five years is 40% of gross notional.
   Momentum is the natural hedge (the two return series correlate **−0.06**),
   and a 60/40 blend cuts the worst week to −881 while raising Sharpe.

   **What is not established.** Everything above is Binance, and this account
   trades BloFin. Spreads are a constant rather than a history. The universe is
   the hundred most-traded contracts as of 2026-09-11, so it is
   survivorship-selected — least badly for carry, which ranks on a cash flow
   rather than on past price, but not innocently. And after seventeen factors
   and four sweeps the specification has been looked at enough times to be
   fitted whether or not anything was deliberately optimised.

   ### Pre-registration, written before the second venue was looked at

   The defence against that last point is not another control on the same
   sample. It is to freeze the specification and evaluate it once on data that
   has never been seen, so this is committed **before** `panel_blofin.py` is
   run. The frozen specification:

   | | |
   |---|---|
   | factor | `carry_7` — negated trailing 7-day mean funding, bps/day |
   | universe | point in time: >= 90 complete days, >= $5M median daily volume over the trailing 30 |
   | book | long top 30%, short bottom 30%, dollar-neutral, sized 1/vol_30, each side renormalised |
   | hold | 7 days, non-overlapping |
   | cost | 10 bps per unit of notional traded |

   The prediction, stated in advance: on BloFin's own instruments, funding and
   daily closes, **net is positive, the funding leg is positive, and the
   shuffled control is not distinguishable from zero**. A failure of any of
   those is a failure of the result, not an occasion for a different cut.

   ### The out-of-sample run, on the venue whose data had not been looked at

   `backend\analysis\panel_blofin.py` builds the identical schema from BloFin's
   own endpoints — `getCandlesticks(bar="1D")` returns 1,339 days and
   `getFundingRate()` returns all 488 instruments' rates and funding cadence in
   one call — so `factor_panel.py` reads it unchanged. 47 USDT perps clear $1M
   a day on that venue, against 108 on Binance; its cross-section is a third
   the width, and the quoted spread runs p25 1.05 / median 2.86 / p75 4.83 bps.

   Running the frozen specification once, on 174 non-overlapping weeks from
   2023-05 to 2026-09:

   | | Binance (in sample) | BloFin (out of sample) |
   |---|---|---|
   | net bps per week | +40.5 | **+44.2** |
   | 95% block interval | [+15.6, +63.5] | **[−2.8, +91.6]** |
   | Sharpe | 1.60 | 1.05 |
   | funding leg | +14.8 | **+18.2** |
   | shuffled control (best of 5) | −2.3 | +4.5 |
   | eligible names, median | 56 | 15 |

   **Against the three predictions written in advance: net positive, yes;
   funding leg positive, yes; control not distinguishable from zero, yes.**
   (The control column here is the best of five shuffles, which step 9s shows
   was the wrong statistic to report. On 200 shuffles the control MEAN is −13.3
   and the real book beats 99% of them, so the verdict holds and the number
   quoted above does not — read 9s's table instead.) The
   funding leg is positive in all four BloFin years too (+11.0, +21.6, +15.6,
   +23.9), so across two venues that is nine calendar years out of nine.

   What the table does not let anyone claim is significance. BloFin's interval
   spans zero, and it does so for a reason visible in the last row: a median of
   15 eligible names gives four or five positions a side, so the book's
   standard deviation is 303 bps a week against Binance's 239 on the same
   strategy. 2023 lost 36.8 bps a week there — and its funding leg was still
   +11.0, with the price leg taking the loss, which is the same split the
   Binance sample shows.

   Two caveats on reading this as a replication. The two venues quote funding
   on overlapping coins, so the samples are correlated rather than independent
   — what differs is the universe, three of the five years, the price series,
   the funding formula and the cap. And a one-day decision lag costs about a
   tenth of the return rather than the result (+40.5 → +36.0 on Binance, with
   the funding leg going +14.8 → +14.4), so none of this rests on acting at the
   instant of settlement.

9r. **The same coin, two venues, two funding rates** —
   `backend\analysis\funding_dispersion.py`.

   ```
   python backend\analysis\funding_dispersion.py --hold-days 30 --top 8
   ```

   Step 9e's cash-and-carry was constrained by its spot leg in three ways: no
   borrow, so only POSITIVE funding is harvestable; a spot spread of 4-50 bps
   against a perp spread under 2; and the full notional tied up in spot.
   Replace the spot leg with **the same coin's perp on another venue** and all
   three go away — the pair is delta-neutral by construction, harvests the
   funding DIFFERENCE whichever sign either leg has, and both legs are perps.

   **BloFin's alt perps charge structurally more funding than Binance's**, and
   it is not an artifact of cadence: both venues settle exactly 3.0 times a day
   on every coin checked. Over 36,282 coin-days on 42 coins:

   | coin | BloFin bps/day | Binance bps/day | difference | days BloFin higher | 2026 only |
   |---|---|---|---|---|---|
   | DOGE | +6.79 | +2.23 | **+4.55** | 84% | +4.71 (94%) |
   | SUI | +6.02 | +1.51 | **+4.51** | 89% | +4.92 (94%) |
   | ADA | +5.41 | +1.94 | +3.47 | 78% | +3.09 (81%) |
   | SOL | +4.61 | +1.36 | +3.25 | 79% | +2.65 (75%) |
   | ETH | +3.10 | +2.02 | +1.08 | 74% | +1.03 (76%) |
   | BTC | +2.66 | +2.00 | +0.66 | 64% | **−0.60 (37%)** |

   BTC is the row that makes the rest mean something, exactly as it was in step
   9c: the difference is near zero on the most arbitraged contract and turns
   negative there in 2026, so this is not a blanket offset between two data
   sources. It is concentrated in the alts, which is where a smaller,
   leverage-oriented venue would have the more crowded longs.

   Short the dearer venue's perp, long the cheaper one's, 8 pairs at a time,
   30-day non-overlapping holds:

   | cost per leg | funding | divergence | round trip | net | 95% | Sharpe | worst |
   |---|---|---|---|---|---|---|---|
   | 5 bps | +70.6 | −0.4 | −10.0 | **+60.2** | [+38.9, +86.0] | 4.77 | −27 |
   | 10 bps | +70.6 | −0.4 | −20.0 | +50.2 | [+28.9, +76.0] | 3.98 | −37 |
   | 15 bps | +70.6 | −0.4 | −30.0 | +40.2 | [+18.9, +66.0] | 3.18 | −47 |

   **The divergence leg is −0.4 bps.** Both legs are the same underlying, so
   the coin's own move cancels and what is left is the gap between two venues'
   marks — which is the entire reason the Sharpe is 3 to 5 where the
   cross-sectional book's is 1.6. The worst 30-day period in 43 is −27 bps.

   Four things that keep this from being an obvious yes:

   - **A random-pair control earns +43.0 of the +60.2.** Two thirds of the
     return is structural rather than selection: the trade is "be short this
     venue's alts and long the other's", and choosing the widest pairs adds
     about 40%. That is a weaker and more fragile claim than a selection edge,
     because it is one bet repeated, not eight.
   - **It needs a Binance futures account**, which this project does not have
     and which is not available everywhere. Everything else here needs one
     venue; this needs two, with two margin pools that cannot net.
   - **43 independent 30-day holds**, of which 2023's eleven made nothing
     (−0 / −10 / −20 across the cost column) and 2024-2026's made +82, +101 and
     +50. A premium that appeared in 2024 is not the same evidence as one
     present throughout.
   - **Holding a leveraged short on the smaller venue for a month is venue
     risk**, and no backtest prices it. The legs also margin separately, so a
     move that leaves the pair flat can still liquidate one side — the failure
     `monitor.py` was written for in 9i, now with the two halves on different
     exchanges and no single account to read.

   At 6.1%/yr on gross notional at 10 bps a leg, this is a Sharpe story rather
   than a return story: it pays to run levered or not at all, and the leverage
   is where the venue and liquidation risks live.

9s. **A third venue, a correction, and a planner** —
   `backend\analysis\panel_hyperliquid.py`, `backend\trading\strategies\carry_xs\`
   and `backend\plan_carry_xs.py`.

   ```
   python backend\analysis\panel_hyperliquid.py --top 60
   python backend\plan_carry_xs.py --notional 4000 --min-volume 2000000
   ```

   ### The correction: the control was being read as a maximum

   Steps 9q and 9r reported the BEST of five shuffled controls. That is a max
   statistic. With a standard error near 25 bps the luckiest of five draws sits
   about 1.2 standard deviations up, so "control +35" was what noise looks like
   — and on the first Hyperliquid run it read as *the control beating the
   factor*, which it was not. It misleads in both directions: a lucky draw can
   equally flatter a weak factor by making the bar look like one it cleared.

   `factor_panel.py` now reports the control MEAN and `pct`, the share of
   shuffles the real book beat, which is an empirical one-sided p-value. On 200
   shuffles per venue, at the frozen specification:

   | venue | periods | real | control mean | control sd | **pct** |
   |---|---|---|---|---|---|
   | Binance | 243 | +40.5 | −14.2 | 10.9 | **100.0%** |
   | BloFin | 174 | +44.2 | −13.3 | 21.4 | **99.0%** |
   | Hyperliquid | 138 | +31.2 | −15.8 | 25.0 | **95.0%** |

   **Step 9u supersedes the Hyperliquid row.** A pre-listing-candle bug moved it
   to +20.7 on 137 periods, and — the substantive point — the result does not
   survive varying the liquidity floor, which the other two do. Hyperliquid is
   not a replication of the total; only of the funding leg.

   The control means land near minus the cost, which is the check that the cost
   model and the turnover agree: a shuffled book pays the same turnover and
   earns nothing. The three venues are not independent — they quote funding on
   overlapping coins — but the universes, the dates, the price series, the
   funding formulas and the cadences all differ, and Hyperliquid settles
   **hourly** against the other two's three times a day.

   **One thing on Hyperliquid points the wrong way and belongs here rather than
   in a footnote.** Its funding leg decays across the sample — +30.9, +21.5,
   +12.0, +10.3 bps a week by year — and 2025 netted +4.6. On a venue that
   launched into this period and grew fast, a premium that shrinks year on year
   is what an opportunity being competed away looks like. Binance's funding leg
   over five years did not decay (+18.1, +15.5, +18.1, +18.1, +31.3), so this is
   not yet a statement about the effect as a whole. It is the single most
   important thing to keep measuring.

   ### Spreads are no longer a constant

   The one assumption left in 9q was a flat cost. `--spreads` now charges each
   instrument its own fee plus half its measured spread, from the snapshot
   `panel_blofin.py` writes. On BloFin that is a median of 6.43 bps and a max of
   12.21 per unit traded, and it does not break the result — it slightly
   improves on the flat 10 bps, which was conservative:

   | BloFin, $2M/day universe, median 22 names | net | 95% block | Sharpe | pct |
   |---|---|---|---|---|
   | flat 10 bps | +43.7 | [+6.1, +80.8] | 1.23 | 100% |
   | measured per-instrument spreads | **+46.4** | [+8.6, +83.2] | 1.30 | 100% |

   Note the universe: at the pre-registered $5M floor BloFin gives a median of
   15 names, and at $2M it gives 22 and an interval clear of zero. That is a
   post-hoc cut and is reported as one — the pre-registered result stands at
   $5M, where it passed all three predictions with a wide interval.

   ### The planner, and what running it live said

   `trading/strategies/carry_xs/plan.py` sizes the book: point-in-time
   eligibility, inverse-volatility weights capped at 25% a name, lots rounded
   down, both sides re-weighted across their survivors and then trimmed to
   whatever the weaker one can fill, liquidation priced per leg, and refusals
   plural. It imports no broker and there is no path from it to `placeOrder` —
   a grep, and now also a test that greps. There is deliberately **no
   `execute.py`**: the evidence justifies a plan, and order-sending code is
   asked for by name.

   Running it against BloFin was worth more than another backtest, because it
   contradicted an assumption immediately. At the pre-registered $5M floor the
   venue offers **eleven** instruments today, three a side, and NEAR quoting
   16.9 bps pushed the round trip to 14.0 against 9.1 bps of carry. It refused,
   correctly. At a $2M floor, 29 instruments and 12 legs:

   | | |
   |---|---|
   | carry spread | +7.03 bps/day between the two baskets |
   | expected funding | +24.6 bps per 7-day hold, per unit of gross |
   | round trip | −12.7 bps (taker + half spread, both ways) |
   | expected net | **+11.9 bps** = $4.69 on $3,965 gross |
   | break-even | 3.6 days |
   | net exposure | $-9.45, 0.24% of gross |

   Three things that table makes concrete and no backtest did. The venue's
   tradeable cross-section is **29 names, not 108**, so the book is a third the
   width the strongest evidence was measured on. The round trip is dominated by
   spread rather than fee — 12.7 bps against a 10 bps taker round trip — and the
   book wants to short exactly the wide names, because wide spreads and crowded
   longs sit on the same instruments. And at $3,965 gross against names trading
   $2-10M a day, capacity is the next binding constraint, not edge.

   **What is still missing, in order.** No executor, so none of this trades. No
   monitor, so a book that is on has no scoreboard — and unlike the two-leg
   carry, this one has twelve legs that margin separately. Capacity is
   unmeasured. And the Hyperliquid decay is a live question that only more time
   answers.

9t. **A funding ladder across three venues** —
   `backend\analysis\funding_dispersion.py`, now matching coins across venues
   that name them differently.

   ```
   python backend\analysis\funding_dispersion.py --panel-a data\panel\blofin-daily.csv --name-a BloFin --panel-b data\panel\hyperliquid-daily.csv --name-b Hyperliquid --hold-days 30
   ```

   Step 9r found BloFin's alts charging more funding than Binance's and priced
   the pair, with one disqualifying caveat: it needs a Binance futures account
   this project does not have. Hyperliquid needs none — an address is an
   account — so with a third panel the question becomes whether the same
   dispersion exists in a form this account could actually trade.

   Matching the coins is the part that had to be got right first. The three
   panels spell the same asset three ways (`1000BONKUSDT`, `kBONK`, `BONK`) and
   pick their own contract multipliers. A multiplier changes the size of a
   contract, not the percentage of funding paid, so those are the same
   underlying — but matching on the raw ticker silently drops every multiplied
   contract, which is most of the meme perps, and those are exactly where the
   dispersion is largest. `canonical()` strips quote and multiplier; a base that
   two tickers on one venue map to is dropped rather than guessed at.

   **There is a ladder, and it is the same ladder on every pair.** On the
   27,591 coin-days where all three venues quote the same 33 coins:

   | venue | mean funding, bps accrued per day | annualised cost of being long |
   |---|---|---|
   | BloFin | **+5.70** | ~20.8% |
   | Hyperliquid | +3.11 | ~11.3% |
   | Binance | +1.40 | ~5.1% |

   The smaller and more leverage-oriented the venue, the more its longs pay.
   The pairwise gaps are also stable across *different* coin subsets, which is
   the check worth making: measured on each pair's own overlap the gaps are
   +3.79 / +2.60 / +1.55, against +4.30 / +2.59 / +1.70 on the common cells.
   (The three gaps summing exactly on common cells is arithmetic, not evidence
   — it is the stability across subsets that says the ladder is a property of
   the venues rather than of which coins each one happens to list.)

   All three pairs clear four taker legs at 30-day holds, 6 pairs at a time,
   10 bps a leg:

   | pair | funding | divergence | net | 95% | Sharpe | worst | control |
   |---|---|---|---|---|---|---|---|
   | BloFin / Binance | +74.5 | −0.4 | **+54.5** | [+32.3, +82.8] | 3.99 | −34 | +35.3 |
   | BloFin / Hyperliquid | +57.5 | +0.0 | **+37.5** | [+21.6, +56.3] | 3.79 | −19 | +19.8 |
   | Hyperliquid / Binance | +51.8 | +3.8 | +31.8 | [+12.6, +45.0] | 2.11 | −54 | +15.0 |

   **BloFin / Hyperliquid is the one this account could trade**, and it is the
   best-behaved of the three on the measures that are not the headline: the
   divergence leg is +0.0 bps, the worst 30-day period in 34 is −19 bps, the hit
   rate is 88%, and selection carries nearly half the return (+37.5 against a
   random-pair control of +19.8) where on the Binance pair it carried a third.

   **Step 9x supersedes the ranking in this table.** Eight more venues were
   measured, BloFin turns out to be the dearest of eleven rather than of three,
   and at a common specification Hyperliquid returns +56.3 against Binance's
   +57.7 — so the Binance pair's apparent advantage here was an artifact of
   comparing only three venues at two different cuts.

   **And the ladder is not static, which cuts both ways.** By year, against
   Binance:

   | | 2023 | 2024 | 2025 | 2026 |
   |---|---|---|---|---|
   | BloFin − Binance | +0.58 | +4.77 | +6.03 | +3.46 |
   | Hyperliquid − Binance | +0.16 | +3.25 | +1.62 | **+1.05** |

   Hyperliquid's premium peaked in 2024 and has shrunk by two thirds. That is
   the same decay its own carry factor shows in step 9s, and the two readings
   agree on a story: a venue that launched into this period is converging on
   the most arbitraged one. BloFin's premium has not converged. Whether that is
   because it is structurally harder to arbitrage or because nobody has yet is
   the question, and it decides whether this trade has years left or months.

   **What would have to be true to trade it.** Capital on two venues that
   cannot net margin; a leveraged short held for a month on the smaller of
   them, which is venue risk no backtest prices; legs that liquidate
   independently, so a move leaving the pair flat can still take one side out;
   and 4.6% a year on gross notional, which is a Sharpe story that only pays
   levered — where the venue and liquidation risks live. Nothing here is built:
   there is no planner for the pair, and there should not be one until the
   decay question has another six months of data on it.

9u. **Sweep the liquidity floor, and the result splits in two** —
   `backend\analysis\factor_panel.py --min-volume`, `--capacity`.

   Step 9s called Hyperliquid a replication on the strength of one cut of the
   universe. Varying the liquidity floor — the same sweep Binance and BloFin
   pass comfortably — says it is not, and separates the claim into a robust
   half and a fragile one. **This supersedes 9s's reading of that venue.**

   A data bug had to be fixed first. `candleSnapshot` returns candles for days
   BEFORE a coin listed on Hyperliquid, carrying an OHLC from somewhere with
   `v` and `n` both zero: ZEC and XMR had 999 such rows each, and 13.3% of the
   whole panel was one, concentrated in 2023 when the venue was young. Those
   are not thin days, they are days the venue did not trade the coin, and a
   return across one is a move that could not have been captured. They now set
   `minutes = 0` and route through the same guard the Binance panel uses for an
   exchange outage. It changed the numbers and did not change the conclusion.

   **The funding leg is positive in all fifteen cells. The price leg is not.**

   | volume floor | Binance net / price / fund | BloFin net / price / fund | Hyperliquid net / price / fund |
   |---|---|---|---|
   | $1M | +33.9 / +32.9 / **+15.1** | +48.4 / +36.1 / **+19.9** | −5.9 / −17.9 / **+20.9** |
   | $2M | +34.6 / +33.5 / **+15.1** | +43.7 / +32.2 / **+19.5** | −32.2 / −40.6 / **+17.8** |
   | $5M | +40.5 / +39.9 / **+14.8** | +44.2 / +34.4 / **+18.2** | +20.7 / +15.3 / **+15.1** |
   | $10M | +40.4 / +40.2 / **+14.5** | +68.0 / +57.6 / **+18.7** | −18.6 / −21.2 / **+12.9** |
   | $25M | +35.9 / +35.9 / **+14.4** | +89.1 / +78.6 / **+17.9** | −69.0 / −71.0 / **+12.1** |
   | beat N of 200 shuffles | **100%** | **99%** | 95% at $5M, 5–62% elsewhere |

   Read the bold column first. **The funding a carry book collects is between
   +12.1 and +20.9 bps a week in every one of fifteen venue-by-floor
   combinations**, across three venues, two funding cadences and five liquidity
   cuts. That is the cash flow, and it is as stable as anything this project has
   measured.

   The price leg — "the coins whose longs pay most go on to underperform" — is
   the forecast half, and it splits by venue. On Binance it is +32.9 to +40.2 at
   every cut; on BloFin +32.2 to +78.6; on Hyperliquid it is **negative at four
   of five cuts**, and it drags the total with it. Hyperliquid's single positive
   cut is the $5M one, which is the floor the specification was frozen at, so
   9s reported the one cut in five that worked and called it a replication.
   That was the wrong call and this is the correction.

   No explanation for the venue difference is offered, because none has been
   tested. It would be easy to write a story about a more arbitraged user base
   and it would be a story.

   **What this leaves standing, precisely.** A cross-sectional carry book
   replicates on **two** venues out of three, and the cash-flow component of it
   replicates on all three. The two that work are the two the strategy would be
   traded on. The one that does not is the one whose funding premium is also
   decaying year on year (step 9t), and those two failures are consistent with
   each other rather than independent.

   ### Capacity, because a rate is not a size

   Every number in this branch is bps per unit of gross notional, and a rate
   says nothing about how many units there are. The binding constraint is the
   SMALLEST position: a name with weight `w` must trade `G * w` against its own
   daily volume, so the book's ceiling is `min(participation * volume / w)`.

   | venue | worst period | 10th percentile | median | at the 10th percentile |
   |---|---|---|---|---|
   | Binance, $5M floor, 17 a side | $1.5M | $2.3M | $4.3M | +$492k a year |
   | BloFin, $2M floor, 7 a side | $20k | $235k | $527k | +$54k a year |

   At 2% of a day's volume, which is conservative for a weekly rebalance that
   can be worked over hours — patience being the one genuine advantage a
   multi-day strategy has over the HFT branches this repo abandoned. **BloFin
   holds a few hundred thousand dollars of gross, not millions.** That is the
   honest size of the opportunity on the venue with the keys, and it is set by
   the venue's own 29-name cross-section rather than by the edge.

9v. **Try to hedge the tail, and find out it was not visible** —
   `backend\analysis\factor_panel.py --cluster-lookback`.

   The carry book's worst week was not three bad positions, it was one position
   held three times: short SHIB, PEPE and BONK in February 2024, when all three
   went up together. Inverse-volatility sizing made it worse rather than
   better, because it sizes on trailing volatility and those three were quiet
   right up until they were not. The obvious fix is to group names that move
   together and give each group one group's worth of money.

   **The first attempt made things worse, and the reason is worth keeping.**
   Single linkage on raw returns at a 0.75 threshold put **61 of 64 names in
   ONE cluster** (measured 2024-12-25) and left three singletons, because every
   crypto correlates through its beta to the market and single linkage chains
   A-B-C through that beta. Dividing by cluster size then handed almost the
   whole book to whichever three names happened not to chain, and it cost 0.74
   of a Sharpe point on Binance: 1.60 down to 0.86, with the worst drawdown
   rising from 2,740 bps to 4,445.

   Demeaning the cross-section first fixes the clustering — what is left is
   names moving together BEYOND their beta, and the groups become
   recognisable: `SHIB+DOGE`, `BNB+BTC+ETH+XRP`, `ARB+OP`, `AVAX+LINK`.

   **And with the clustering working, it does almost nothing.**

   | | Binance net / Sharpe / worst drawdown | BloFin net / Sharpe / worst drawdown |
   |---|---|---|
   | no clustering | +40.5 / 1.60 / 2,740 | +43.7 / 1.23 / 2,271 |
   | residual clusters, 90d | +40.5 / **1.66** / 2,711 | +44.9 / 1.28 / 2,271 |
   | residual clusters, 180d | +40.5 / 1.66 / 2,711 | +44.6 / 1.27 / 2,271 |
   | raw-return clusters, 180d | +34.1 / **0.84** / 4,445 | +53.6 / 1.20 / 1,759 |

   **Why it does nothing is the actual finding.** In the 90 and 180 days before
   2024-02-26, the residual correlations among the three memecoins were
   **+0.12, −0.13 and +0.21** — no relationship at all beyond their beta — and
   the clustering put them in three different groups. In the week that
   followed they returned **+256%, +288% and +180%**.

   The correlation that destroyed the book did not exist in the data
   beforehand. It arrived with the event. No risk control estimated from
   trailing correlation could have seen it, which is exactly why a correct
   clustering changes the drawdown by 1%.

   What follows for how this should be risk-managed: the only control that
   works against a correlation which does not yet exist is one that does not
   try to estimate it. A hard cap on any single position — `max_weight_frac`,
   25% of a side in the planner — is that control, and it is worth more here
   than any covariance model. This is also the honest reason the strategy's
   headline Sharpe of 1.6 should not be levered into a Sharpe-1.6-shaped
   position size.

   (The earlier write-up of that week quoted +127%, +136% and +103%. Those were
   log returns read as percentages; the simple returns are the ones above.)

9w. **Rank crowding instead of carry, and reject it** —
   `backend\analysis\factor_panel.py --reference-panel`.

   Step 9t found a funding ladder between venues, which raises a question
   `carry_7` does not answer. `carry_7` ranks a coin against the rest of ITS
   OWN venue's cross-section, so a venue-wide offset cancels out of it. The
   cross-venue version asks something different: which coin is crowded HERE
   specifically — funding high relative to the same coin elsewhere. That is a
   different signal rather than a rescaling, and it has an appealing story,
   which is exactly why it needed testing rather than adopting.

   `carry_rel_*` is that signal: this venue's funding minus the same coin's
   funding on a reference venue, matched canonically, trailing-averaged and
   negated. The reference funding is a FEATURE only — the label keeps using the
   venue's own funding, because a BloFin position pays BloFin's rate whatever
   Binance charges, and a test pins that.

   **On BloFin it looks like an improvement. Everywhere else it is not.**

   | | net | Sharpe | price leg | **funding leg** | pct |
   |---|---|---|---|---|---|
   | BloFin, carry_7 | +43.7 | 1.23 | +32.2 | **+19.5** | 100% |
   | BloFin, carry_rel_7 | **+48.0** | **1.43** | +43.3 | **+13.4** | 100% |
   | Hyperliquid $2M, carry_7 | −32.2 | −0.73 | −40.6 | **+17.8** | 20% |
   | Hyperliquid $2M, carry_rel_7 | −40.0 | −0.88 | −38.7 | **+9.8** | 10% |
   | Hyperliquid $5M, carry_7 | +20.7 | 0.53 | +15.3 | **+15.1** | 96% |
   | Hyperliquid $5M, carry_rel_7 | +10.1 | 0.26 | +14.5 | **+7.2** | 88% |
   | Hyperliquid $10M, carry_7 | −18.6 | −0.39 | −21.2 | **+12.9** | 44% |
   | Hyperliquid $10M, carry_rel_7 | +2.4 | 0.05 | +9.3 | **+5.4** | 76% |

   **Rejected, and the funding column is why.** The relative signal roughly
   HALVES the funding leg in every cell — 19.5 to 13.4, 17.8 to 9.8, 15.1 to
   7.2, 12.9 to 5.4 — while its total depends on a price leg that is larger and
   no more reliable. That is trading the component this project has shown to be
   stable across fifteen venue-by-floor cells for the component it has shown to
   be venue-specific, and the one cut where the total came out ahead is not
   worth that swap.

   It also does not rescue Hyperliquid, which was the real test: if ranking by
   venue-specific crowding were the better signal, it should work where ranking
   by carry level failed. It does not.

   The mechanical reason for the halved funding leg is not subtle, and it is
   the argument against the idea rather than a detail of it: subtracting
   another venue's funding removes most of the level a carry book is paid FOR.
   The residual is a crowding forecast wearing a carry factor's name.

9x. **Eleven venues, and Binance turns out not to be needed** —
   `backend\analysis\panel_venue.py`.

   ```
   python backend\analysis\panel_venue.py --list
   python backend\analysis\panel_venue.py --venue bybit --top 70
   ```

   Step 9r's pair had one partner venue and it was Binance, which is not
   available in much of the world. A result that depends on one exchange is a
   bad shape for a result to have, so this makes substituting one cheap: an
   adapter is four functions — the universe with its volume, daily candles,
   funding history, and how the venue spells a coin — and everything else,
   including both alignment rules, is shared.

   **Every venue tried answers publicly with no API key.** Mean funding in bps
   per day, ~50 settlements on ten majors, measured 2026-09-12:

   | venue | bps/day | | venue | bps/day |
   |---|---|---|---|---|
   | **BloFin** | **+3.78** | | OKX | +1.47 |
   | Aster | +2.18 | | dYdX | +1.38 |
   | Bitget | +2.09 | | MEXC | +1.33 |
   | Hyperliquid | +1.60 | | Bybit | +1.07 |
   | KuCoin | +1.56 | | Gate | +1.02 |
   | | | | **Kraken** | **−0.51** |

   **BloFin is the dearest of eleven**, which is the finding that matters more
   than any single pair: it is the short leg whichever venue ends up opposite,
   and step 9r's result was never about Binance. Kraken is cheapest by a
   distance and is actually NEGATIVE on DOGE, XRP, SUI, LINK and AVAX — its
   longs are paid to hold those.

   Then the historical pair, every partner at one identical specification —
   30-day non-overlapping holds, 6 pairs at a time, 10 bps a leg, $2M/day floor:

   | partner | coins | diff bps/day | **periods** | net/30d | 95% block | Sharpe | hit | worst | control |
   |---|---|---|---|---|---|---|---|---|---|
   | Binance | 43 | +3.79 | **44** | +57.7 | [+29.6, +91.6] | 3.40 | 77% | −58 | +30.2 |
   | Bybit | 40 | +2.94 | **44** | +34.0 | [+16.1, +52.6] | 3.17 | 73% | −39 | +19.9 |
   | **Hyperliquid** | 33 | +2.60 | **35** | **+56.3** | [+39.9, +72.1] | 5.48 | 91% | −22 | +29.1 |
   | MEXC | 32 | +4.69 | 17 | +82.0 | [+53.0, +115.7] | 6.94 | 94% | −17 | +55.3 |
   | Kraken | 24 | +4.24 | 11 | +54.9 | [+46.3, +61.0] | 8.15 | 100% | +16 | +35.9 |

   **Read the `periods` column before the Sharpe.** MEXC publishes about 18
   months of funding and Kraken about 12, so their intervals and their Sharpes
   of 6.94 and 8.15 rest on 17 and 11 independent observations. Kraken not
   having lost in eleven tries is not the same evidence as Hyperliquid winning
   32 of 35.

   **The answer to "is Binance needed": no.** Hyperliquid returns +56.3 against
   Binance's +57.7 on a comparable number of periods, and needs no account at
   all — an address is an account. That supersedes the earlier reading of step
   9t, which ranked BloFin/Binance top when Binance was the only alternative
   measured.

   Three things the table does not say. Bybit has the cheapest funding on a
   live snapshot (+1.07) and the WEAKEST historical spread against BloFin
   (+2.94) — a reminder that today's reading is not the average. Kraken's perps
   are USD-margined, so a pair against a USDT venue carries the USDT/USD basis
   on top of the funding difference, and its p99 absolute daily gap is 63.8 bps
   against Hyperliquid's 23.8. And Gate is excluded as a panel source entirely:
   it returns ~90 settlements however large a limit it is given, which is 30
   days, so `--list` says so rather than producing a short panel that looks
   like the others.

9y. **Give the book three verbs** — `backend\trading\strategies\carry_xs\`,
   `backend\run_carry_xs.py`, `backend\monitor_carry_xs.py`.

   ```
   python backend\plan_carry_xs.py --notional 10000 --min-volume 2000000
   python backend\run_carry_xs.py  --notional 10000 --min-volume 2000000 --confirm
   python backend\monitor_carry_xs.py
   python backend\run_carry_xs.py  --repair-only
   ```

   The cross-sectional book now has the same three verbs as the two-leg carry,
   with two deliberate differences.

   **One verb where the carry has two.** `reconcile` replaces open and close,
   because this book is REBALANCED weekly: "open" is the case where the
   exchange holds nothing and "close" is the case where the target is empty.
   It is idempotent — every decision comes from positions read back from the
   exchange — which matters because twelve legs is twelve chances to be
   interrupted, and the answer to "what if it dies halfway" has to be "run it
   again".

   **Repair, not unwind.** `strategies/carry` unwinds a half-open carry; this
   must not inherit that, because closing eleven positions the plan wants
   because the twelfth was rejected costs a full round trip to undo a book that
   is 92% correct. A failed leg leaves an imbalance which is reduced — always
   `reduce_only`, so the repair can only shrink risk — and reported whether or
   not the reduction worked. `--repair-only` does it without rebalancing, which
   is the right move mid-hold and is what the monitor's `not-neutral` alert
   points at.

   Orders are **interleaved** so the partly-filled book stays near neutral, and
   the running total starts from the exposure the account ALREADY carries
   rather than from zero — the two are the same thing only when the account
   begins flat, which on a rebalance it never does. Against an account holding
   $69k of leftover BTC that difference is the whole safety property: measured
   $310 of worst intermediate exposure on a $3,949 book, against $69k if the
   counter starts at zero.

   ### Running it live found three bugs no unit test had

   - **Notional was `contracts x price`, ignoring contract value.** A BloFin
     BTC contract is 0.001 BTC and a DOGE contract is 1000 DOGE, so it reported
     a real position as **$68.9 million** and a TRX leg as $0.49 — both wrong
     by exactly the multiplier, and the interleaving was balancing that.
   - **Size rules were read from PRODUCTION while orders went to DEMO.** Demo
     lists 87 instruments against production's 488 and the lot size differs on
     10 of the 87 they share: DOGE 0.01 against 0.1, ZEC 0.1 against 1. Six
     orders came back `152002 Parameter size error`. This repo already knew the
     rule — `plan_carry.py` reads margin tiers from the account's host because
     demo and production MMR differ — and it applies to anything the exchange
     VALIDATES. Prices still come from production, because demo's book is not
     the market.
   - **The repair sent un-rounded sizes**, `0.2276622802836378615352929061`
     contracts, because it sized itself as `excess / unit`. Rounding where a
     size is computed is not enough; it now happens on the way OUT, so the next
     code path that computes one does not have to remember.

   A fourth was invisible to the suite entirely: a syntax error shipped in
   `plan_carry_xs.py` and 922 tests stayed green, because the tests import the
   strategy packages and nothing imported the command-line files that wrap
   them. `test_entrypoints_import.py` now parses every entrypoint and checks
   no source file carries a stray control character, which is what a shell
   heredoc makes of `\a` in a Windows path.

   ### The monitor, and what it says about the book that is on

   `monitor.py` does not import the broker. The broker can place orders and a
   monitor holding one would be one typo from being an executor, so it defines
   its own four-method `Reader` and a test walks the AST to assert neither
   `place_perp` nor any broker import appears in it.

   Two things a cross-margined book changes. The margin ratio is
   **account-level** — all eight live positions report an identical one, which
   is what confirms they share a pool — so the risk question has a single
   answer that is read rather than modelled. And the plan's per-leg `solo liq`
   is explicitly a bound: it assumes isolated margin, which this book does not
   use, and is conservative by construction.

   Funding is derived the same way step 9i derives it, because this API version
   still publishes no bills endpoint: `funding = sum(realizedPnl) + fees`. On
   the live book, minutes after opening and before any settlement, `realizedPnl`
   summed to **−2.0241** against fees of **+2.0240** — the derivation returns
   zero when zero is the true answer, which is the same check the two-leg
   version passed. It gets the same control too: the public funding history,
   signed by side, priced at today's notional.

   The first reading was not clean, and that is the point of having one. The
   book on the demo account came back **4.15% net long** against a 2% tolerance
   — exactly as the failed run left it, with DOGE and ZEC never placed. The
   monitor raised `not-neutral` as critical, exited 2, and `--repair-only`
   priced the fix as a single trim: sell 0.21 TRX contracts, $71.30,
   `reduce_only`.

   ### The repair could not reach neutral, and the arithmetic says why

   The second live run found the first one: the repair sized a trim, sent it,
   and then reported that the book was STILL directional. It was not a fluke
   and a retry would not have helped - it was arithmetically incapable of
   succeeding. Measured 2026-09-12 on demo, $139.91 net on $3,377.27 gross
   (4.14%) against a 2% tolerance:

   ```
   sized     sell 0.21 TRX-USDT   $71.28
   landed    $68.62 net on $3,305.98 gross   2.08%   -> "STILL directional"
   ```

   Three things compounded, all in the same direction:

   - it aimed at the tolerance BOUNDARY rather than at zero, so success was
     defined as being as directional as the alarm would just barely allow;
   - it rounded the trim DOWN, so it did not even reach that boundary;
   - and trimming shrinks GROSS, so the boundary moves toward the book by
     `tolerance x trim` while the trim moves the book toward the boundary. The
     target was running away at 2% of the speed of the cure.

   The fix is one sentence: the tolerance decides WHETHER to repair, and it has
   no business deciding how much. The target is zero. The same book now:

   ```
   sized     sell 0.41 TRX-USDT   $139.17
   landed    $0.74 net on $3,238.10 gross    0.02%
   ```

   Two smaller corrections came with it. Trims round to the NEAREST lot rather
   than down, because rounding down cannot remove an imbalance smaller than one
   lot and those are precisely the ones that survive to be a problem; the
   overshoot is bounded by half a lot. And the trimming stops at a quarter of
   the tolerance band, because aiming at zero without a floor walks down the
   whole heavy side paying a taker fee per leg - the first run of the fixed
   version sized a second order for the remaining $0.74.

   That floor is a stopping rule and NOT a target, which is worth writing down
   in the same place as the bug, because the two are one careless edit apart.

   The dry run and the live run had also been sizing the trims in two separate
   copies of the same loop, so the rehearsal was a rehearsal of something else.
   They are one `repair_trims` now.

   ### Nothing to do, and which nothing

   Running the repair again against the account reported "The book already
   matches the plan", which was the opposite of the truth: the positions had
   been closed by hand in the BloFin UI between the two runs. The order history
   is unambiguous about it - eight `reduce_only` closes at 14:25:59, 23 seconds
   after the repair, carrying NO `clientOrderId`, where every order this repo
   sends has one (`xsd03c047d00`, `xre966825900`). Being able to tell one's own
   orders from a human's, after the fact, is worth the eleven characters.

   Three situations end in `already_correct` - an empty account, a book inside
   tolerance, and a book that matches the plan - and reporting all three with
   the same sentence tells the operator the opposite of the truth in the first
   case. A repair that finds no positions is not a book in good order, it is an
   account that may have been closed out from under it. Each says which now.

   Still missing: nothing writes a baseline at execution time, so the monitor
   reconstructs and freezes one on first sight (like step 9i's, and for the
   same reason). And the epoch model is untested against a real rebalance,
   because there has not been one.

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
