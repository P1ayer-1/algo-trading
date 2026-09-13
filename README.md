# Crypto Bot

Intended goal: a high-frequency, high-leverage crypto trading bot that uses
technical indicators to decide entries/exits and keeps liquidation levels at a
comfortable distance to manage risk.

## Current state (be honest with yourself here)

**The per-strategy status board is [`STRATEGIES.md`](STRATEGIES.md)**: which
strategies exist, what the evidence says about each, what is running on
which account and host, what data is being recorded and which strategy
consumes it, and the open problems. It is rewritten in place; the roadmap
below is the history it points into.

The description that follows is the original one from before any order path
existed and is kept as the architecture overview. Order paths now exist for
the carry strategies and the lead quoter (steps 9h, 9y, 9ae); all of them are
dry unless `--confirm` and demo unless `--production`, and nothing has run
against production. What exists is:

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
│   │   ├── panel_intraday.py  # 1m archive -> 15m bars per symbol (npz), once
│   │   ├── intraday.py        # a (period, symbol) grid at any bar, funding aligned
│   │   ├── intraday_factors.py # factor_panel's harness at 8h/1h bars
│   │   ├── settlement_event.py # price around a funding settlement, by rate
│   │   ├── settlement_short.py # the negative-funding settlement trade, priced
│   │   ├── extreme_move.py    # do extreme short-window moves revert? (per coin)
│   │   ├── cascade_rebound.py # the basket version: buy the market after a cascade
│   │   ├── seasonality.py     # hour-of-day and weekday, by year
│   │   ├── validate_liquidation.py  # our liq math vs the exchange's own
│   │   ├── blofin_spread_survey.py  # the same, live, on BloFin itself
│   │   ├── layout.py          # where recorded data lives; one owner
│   │   ├── replay.py          # rebuild features from raw events
│   │   ├── compact.py         # CSV -> Parquet, storage report
│   │   └── stats.py           # IC, AUC, logistic regression, purged split
│   └── tests/                 # pytest suite (998 tests)
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

998 tests covering the order book's gap handling, the OFI recursion, the
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
   position via `backend\analysis\validate_liquidation.py`: 0.0568% relative
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

9c. ~~**What are BloFin's own spreads?**~~ — `backend\analysis\blofin_spread_survey.py`.
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
   `backend\analysis\venue_compare.py` is the tool that reads the answer out:
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

9e. **Funding carry** — `backend\analysis\funding_carry.py`. Every branch so
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

9f. ~~**Backtest the carry**~~ — `backend\analysis\carry_backtest.py`. The
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

9g. ~~**Plan the carry**~~ — `backend\trading\carry.py` and
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

   ### A way out that is not the exchange's website

   ```
   python backend
un_carry_xs.py --flatten
   python backend
un_carry_xs.py --flatten --confirm
   ```

   There was no close. `reconcile` moves the book to a plan and `--repair-only`
   trims it toward neutral; neither can get you out. So the only exit was
   BloFin's own web UI, and that is not a hypothetical - it is how the demo
   book was actually closed on 2026-09-12, eight `reduce_only` orders carrying
   no `clientOrderId` because no part of this repo could send them.

   The dependency is what makes it worth a verb. `reconcile` needs a plan, and
   a plan needs a universe read, funding history for eighty instruments and a
   volatility estimate per name: a minute of API calls and a dozen ways to
   fail, sitting between an operator and the exit. `--flatten` reads positions
   and closes them. It needs prices only to ORDER the sends, so a position it
   cannot price is closed last rather than left open.

   It is `reconcile` against an empty target, deliberately and not
   incidentally: `plan_orders` already makes every close `reduce_only`, already
   exempts a close from the exchange's minimum size - which is what stops a
   small leftover leg being stranded forever - and already interleaves.
   Re-deriving any of that in the code path used when something has already
   gone wrong would be a second implementation of the rules that matter most.

   Interleaving matters as much on the way out as on the way in, and it is the
   easy part to get wrong: a neutral book closed longs-first is naked short by
   half its gross in the middle, at exactly the moment an unintended
   directional position is least affordable. Measured on the live demo book, 10
   legs and $9,656.50 gross:

   ```
   worst intermediate net exposure   $1,130.38   (11.7% of gross, one leg)
   net after the last order              $0.00
   ```

   Closing longs first would have passed through $4,113.

   Flat is read back from the exchange and never inferred from what was sent,
   because an operator told the book is closed stops looking at it. A rejected
   close is reported as "STILL open" by name, and `--flatten --repair-only`
   together is refused: those are different intentions and only one of them is
   undone by re-running it.

   Still missing: nothing writes a baseline at execution time, so the monitor
   reconstructs and freezes one on first sight (like step 9i's, and for the
   same reason). And the epoch model is untested against a real rebalance,
   because there has not been one.

9y2. **The first real epoch transition, and the $4.05 of funding that was
   not funding** - `backend\trading\strategies\carry_xs\monitor.py`.

   Step 9y shipped the epoch model untested, because no rebalance had happened
   yet. One has now: an 8-leg book was closed and a 10-leg one opened. The
   epoch machinery worked - it named DOGE and ZEC as new, refused to score the
   new book against the old forecast, and archived on request. What it did NOT
   do was stop the funding derivation running across the boundary anyway:

   ```
   derived             $+4.0505   = sum(realizedPnl) + fees
   public-rate control $+0.0000   = settlements x notional, by side
   realised rate       +87.107 bps/day on gross
   planned rate         +6.275 bps/day
   ```

   Fourteen times the forecast on a book 72 minutes old, and every cent of it
   stale fees. `funding = sum(realizedPnl) + fees` only holds when both terms
   cover the same trades, and `realizedPnl` is a property of a position that
   VANISHES when the position closes. The fee window came from the baseline, so
   it swept in $4.4844 from the previous epoch - 7 opens, a repair and 8 closes
   - whose matching PnL had already left with the positions.

   **The control is what caught it.** `funding-disagrees` fired because the
   public-rate estimate said $0.0000 and meant it: no settlement had crossed in
   72 minutes, so zero was the true answer. A derived number without an
   independent control would have reported the best week in the strategy's
   history.

   The first fix was wrong in an instructive way. Bounding each leg by its own
   `createTime` returned $0.0000 of fees on a book that had just paid $5.80 of
   them, because **every position is stamped 20-35 ms AFTER the fill that
   opened it** - BCH 27 ms, ZEC 35 ms, measured across all ten legs - so
   `ts >= created_ms` excluded every opening fill. Two clocks compared at
   millisecond precision, which is the same bug shape as the funding
   settlements that print milliseconds late and landed on the wrong side of
   midnight in `panel_daily.py`.

   The right fix uses no clock at all. Both a position and its fills carry
   `positionId`, so "did this fee belong to a position that is still open" is
   an identity, not a comparison:

   ```
   fees on current positions   $5.798122434
   sum(realizedPnl)           -$5.798122434
   derived funding             $0.000000000
   fees on closed positions    $4.050492516   <- exactly the phantom funding
   ```

   Nine decimal places, and the discarded term is precisely the number that had
   been reported as profit. The timestamp path survives only as a fallback for
   a venue reporting no position id, with 5 seconds of tolerance - 140x the
   worst observed skew.

   Two things this leaves. The derivation is now robust to a STALE baseline,
   which matters because a stale baseline is exactly when nobody is checking.
   And the new baseline froze with no forecast, because nothing carries the
   plan's expected rate into the monitor - the same gap step 9y named - so
   `funding-short` cannot fire until one is supplied by hand.

9z. **The touch formula is a moderate-distance instrument, and the book is
   the worst place to push it** - `backend\analysis\touch_calibration.py`.

   ```
   python backend\analysis\touch_calibration.py
   ```

   Step 9o left a calibrated engine - `P(touch) = 2*Phi(-d / (sigma*sqrt(H)))`,
   the reflection principle - and the obvious next move was to point it at the
   thing that makes money and replace a static risk buffer with a probability:
   "P(mark touches liquidation before you exit) = 0.4%" instead of "liquidation
   must be 15% away". Before wiring that into `carry_xs`, the assumption got
   tested, and it does not hold where it would have been used.

   The reflection principle needs BROWNIAN motion, which is stronger than a
   random walk: not just uncorrelated increments but Gaussian ones. The
   variance-ratio test that justified the formula (VR 1.00-1.13) only checked
   the first. Crypto is uncorrelated and fat-tailed at once, and every risk
   question worth asking lives in the tail the variance ratio never looked at.

   Non-overlapping 7-day returns, standardised and pooled, counted against what
   a Gaussian predicts at each distance. Two populations, because they answer
   different questions - and the book is pooled over six grid cells so the
   answer is not one specification's luck:

   ```
   distance      single assets        the book
                 (62 symbols,      (6 cells, 1,458
                 10,298 obs)            periods)
     1.0 sd            0.8x             0.7x
     2.0 sd            0.9x             0.9x
     2.5 sd            1.7x             2.2x
     3.0 sd            3.2x             7.6x
     4.0 sd           24.5x           173.2x
     5.0 sd               -        16,748.9x
   usable to           2.5 sd           2.0 sd
   ```

   **It is calibrated out to about 2 sigma and not past it.** That is the
   regime step 9o validated it in, and the result there stands - this does not
   retract it. What it retracts is the extrapolation: the formula is not a tail
   model and was never tested as one. Liquidation, ruin and "how bad can one
   week be" are 4-sigma questions, and there it is wrong by one to two orders
   of magnitude. So the headline application proposed for it - a liquidation
   gate - is precisely the use it cannot support.

   **The book's tail is seven times fatter than its own legs'.** 173x against
   24.5x at 4 sigma, excess kurtosis +9.38 against +2.42. That is the opposite
   of what diversification is supposed to do, and step 9v already named the
   mechanism: in February 2024 the book was short SHIB, PEPE and BONK because
   they had the highest funding, they returned +256%, +288% and +180% in a
   week, and in the 90 days beforehand their residual correlations were +0.12,
   -0.13 and +0.21. The correlation that did the damage did not exist in the
   data yet. A dollar-neutral cross-section is not a diversified book; it is a
   crowding bet, and crowding is exactly what shows up as a fat tail.

   What it costs to get this wrong, for a $2,000 weekly loss budget:

   ```
   Gaussian at the 1/n quantile       -532 bps  ->  $37,599 gross
   empirical 1st percentile           -444 bps  ->  $45,086 gross
   worst actually observed            -976 bps  ->  $20,500 gross
   ```

   The Gaussian would let you run 1.8x too big, and the worst observed is
   itself one draw from 243 periods, so $20,500 is not a bound either.

   There is one thing left for it. `max_weight_frac` is already the right kind
   of control - a cap that does not depend on having estimated the risk - and
   for WEIGHTING the formula adds nothing anyway: at a fixed distance
   `2*Phi(-d/(sigma*sqrt(H)))` is a monotone transform of sigma, so it ranks
   legs identically to the inverse-vol weighting `_side_weights` already does.
   It only carries information where the DISTANCE differs per leg, which under
   the cross margin this book uses it does not. Its honest remaining use is the
   1-to-2-sigma questions the monitor already asks: how likely the net drifts
   past the 2% tolerance before the rebalance, and how likely the account
   margin ratio reaches the 300 warning level. Both are worth having and
   neither is where the money is.

9aa. **The same book, held for three days instead of seven** —
   `backend\analysis\factor_panel.py --hold-days 3 --lag 1`, and
   `backend\analysis\intraday_factors.py` for the same harness at 8-hour bars.

   ```
   python backend\analysis\factor_panel.py --vol-scale --hold-days 3 --top-frac 0.3 --cost-bps 10 --lag 1
   python backend\analysis\panel_intraday.py
   python backend\analysis\intraday_factors.py --bar 480 --hold 3 --vol-scale
   ```

   The brief on 2026-09-12 was to beat step 9q's weekly book, preferably on a
   shorter horizon. Step 9q's specification grid swept holds of 3, 7, 14 and
   30 days and never looked below three, because every horizon under a day
   in this repo had died on cost. The cheapest test was therefore the one
   nobody had run: the frozen `carry_mom` blend (60/40 rank blend of
   `carry_7` and `mom_14`, top/bottom 30%, inverse-vol) at holds of one to
   three days, same cost, same controls. **Read this table against the README
   headline at `--cost-bps 10` - the harness default is `config.TAKER_FEE_BPS`
   (5.0), which flatters a daily hold by about a quarter of its result.**

   Binance, 108 perpetuals, 10 bps per unit traded, `carry_mom`:

   | hold | net bps/period | 95% block | ~per week | Sharpe | ann% | worst drawdown | shuffles beaten |
   |---|---|---|---|---|---|---|---|
   | 1 day | +7.5 | [+4.1, +10.7] | 52 | 1.91 | 27.3 | 1,426 | 100% |
   | 2 days | +17.5 | [+10.9, +24.2] | 61 | 2.29 | 31.9 | 1,111 | 100% |
   | **3 days** | **+25.8** | [+15.5, +34.8] | **60** | **2.25** | **31.3** | **759** | 100% |
   | 7 days (9q) | +44.5 | [+18.0, +68.5] | 44.5 | 1.57 | 23.2 | 1,567 | 100% |

   Same signal, a third of the hold, half the worst drawdown, and about 40%
   more per week at a Sharpe of 2.2 against 1.6. The by-year row at three
   days reads +23.2, +11.4, +26.0, +40.2, +29.1 - positive in every year
   including 2023, which the weekly book only just survived (+15.0).

   **Where the extra return comes from, and why that is the caveat.** It is
   not the carry. `carry_7` alone at a 3-day hold makes +17.7 (Sharpe 1.74)
   against +40.5 a week at seven days - the same money per week at a slightly
   better Sharpe, because the funding leg is a cash flow and does not care how
   often the book is re-ranked. What changes is the momentum half: `mom_14`
   alone goes from Sharpe 0.36 at a 7-day hold to 0.82 at three days. A
   14-day momentum rank is stale by the end of a week, and re-ranking every
   three days keeps it fresh. So the gain sits in the price leg (26.3 of a
   31.7 bps gross at three days, against a funding leg of 5.9), which is the
   half this project has learned to trust less. It is a forecast that
   replicated, not a cash flow.

   **The honest version, lagged.** A 3-day hold rebalanced from bars that
   closed one day earlier - `--lag 1`, so nobody has to act at the instant of
   the close - makes +22.4 (Sharpe 1.97) on Binance. The signal is not about
   the last few hours. A 3-day hold also has three possible rebalance
   phases, and the harness only sees one; the other two (`--start 121`,
   `--start 122`) give +17.6 (1.50) and +23.0 (1.99), with `carry_7` alone
   at +17.4 / +16.9 / +16.3 across the three.

   **It replicates where it should and is weaker where it should be.** At the
   venues' own floors (BloFin $2M with measured per-instrument spreads, the
   others at 5 bps flat, so these are not directly comparable to the Binance
   table):

   | venue, `carry_mom` | 1 day | 2 days | 3 days | 3 days, lag 1 | 7 days |
   |---|---|---|---|---|---|
   | BloFin, spreads | +8.1 (1.51) | +19.3 (1.83) | **+28.0 (1.80)** | +21.5 (1.37) | +55.9 (1.48) |
   | Hyperliquid, $5M | +6.9 (1.18) | +16.3 (1.41) | **+24.0 (1.31)** | +20.2 (1.13) | +53.5 (1.32) |
   | Bybit, $2M | +4.3 (0.94) | +9.8 (1.06) | +12.5 (0.90) | - | +41.6 (1.29) |

   (net bps per period, Sharpe in brackets.) BloFin's three-day book is
   positive in every year (+16.0, +32.6, +35.8, +20.9), and Hyperliquid - the
   venue whose weekly carry book step 9u showed does NOT survive a change of
   liquidity floor - is positive in all three of its years at a 3-day hold.
   Bybit is the weakest everywhere, as it was in step 9x: it has the cheapest
   funding of the eleven venues and so the least carry to rank on.

   Two more checks in the 9q style: six random halves of the Binance universe
   at a 2-day hold are positive on both sides in 6 of 6 splits (Sharpe 0.90
   to 2.42); and all twelve cells of holds 1/2/3 x top 20%/30% x floors
   $2M/$5M/$20M are positive, +6.4 to +30.3, with the 3-day column the best in
   every floor.

   **And it does not keep improving as the hold shrinks.** Every 8-hour bar,
   scored with the identical harness on the same 108 names: `carry_mom`
   rebalanced at every settlement earns +2.8 bps per 8 hours at 5 bps cost
   with a turnover of 0.42 per period - which at 10 bps is about +1.8, or
   roughly the weekly book's return for three times the trading. Held for
   three bars (a day) it is +9.8 (Sharpe 2.48 at 5 bps), the same as the
   daily-panel number. The sweet spot is one to three days. What is
   established is a hold, not a new signal.

   **What this does not settle, in order.** The planner (`carry_xs/plan.py`)
   ranks on carry alone; the blend needs a momentum term and a 3-day epoch,
   and neither has been built or run live. Turnover per week rises from 1.8
   to 2.8 units of gross, so the BloFin capacity in 9u shrinks in proportion
   and the round trip is paid 2.3x as often - the measured-spread column
   above already charges that, and it is why BloFin's lagged 3-day Sharpe is
   1.37 rather than 1.97. And after this sweep the specification has now been
   looked at on four venues and 96 + 12 grid cells; the 3-day hold was chosen
   from a table of three, not pre-registered, and the next out-of-sample
   evidence is the live book.

9ab. **Everything shorter than a day, measured, and all of it dead** —
   `backend\analysis\panel_intraday.py` folds the 5.7 GB 1-minute archive
   into 15-minute bars once (175,200 bars a symbol, 8 MB as `.npz`), and
   `backend\analysis\intraday.py` turns those into a `(period, symbol)` grid
   at any bar from 15 minutes up with funding placed in the bar whose
   interval it paid for. Five hypotheses were run on it on 2026-09-12; each
   has a tool, a placebo and a year table, and each is negative. Written up
   so they are not run again.

   ```
   python backend\analysis\intraday_factors.py --bar 480 --hold 1 --vol-scale
   python backend\analysis\settlement_event.py
   python backend\analysis\settlement_short.py --signal settle --hold-hours 2
   python backend\analysis\extreme_move.py --window-min 60 --z 6
   python backend\analysis\cascade_rebound.py --z 5 --delay-min 15
   python backend\analysis\seasonality.py
   ```

   **Intraday cross-sectional reversal and flow, on a hundred names.** Step 8
   found reversal at 15-240 minutes on ten majors worth 0.9 bps against a
   2.4 bps round trip, and the hope was that a cross-section ten times wider
   would make the dispersion pay for it. At 8-hour bars, one-bar holds, cost
   5 bps, every intraday factor is negative and the break-even cost is under
   one basis point:

   | factor | net bps / 8h | gross | turnover | break-even cost |
   |---|---|---|---|---|
   | `rev_1b` (fade the last 8h) | −6.4 | +0.9 | 2.91 | 0.6 |
   | `rev_1d` (fade the last day) | −4.8 | −0.5 | 1.72 | −0.6 |
   | `taker_1b` (last bar's taker-buy share) | −6.5 | +0.1 | 2.62 | 0.1 |
   | `carry_last` (the single latest settlement) | −1.1 | +2.8 | 1.55 | 3.6 |
   | `carry_7d` | +2.2 | +2.8 | 0.25 | 22.8 |

   Reversal at these horizons is real and worth less than a basis point,
   exactly as step 8 measured it, and the single latest settlement is a
   noisier version of the 7-day mean, not a faster one.

   **Price around a funding settlement.** 376,485 (symbol, settlement) events,
   the 15-minute path from four hours before to four after, bucketed by the
   rate paid, in excess of the cross-section. Positive-rate buckets show
   price continuing UP after the settlement (+14 bps in the next hour for
   rates of 5-10 bps, +67 above 10), which reads as the paying side
   re-entering - and shows the identical continuation at placebo instants
   two and four hours earlier. Those are pumps continuing, not settlements.
   The one bucket locked to the instant was rates below −5 bps: those coins
   rise in the two hours before the settlement and fall +33 ± 6.5 bps in the
   two after, and the placebos point the other way. Priced as a trade -
   short every eligible coin whose rate is below −5 at the settlement close,
   cover two hours later, 5 bps a leg - it is +23.7 bps per settlement on
   7,130 trades, t 3.64. **Then the year table:**

   | | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
   |---|---|---|---|---|---|---|
   | bps per settlement | +45 | −1.0 | −12.1 | −0.9 | +2.0 | **+59.7** |
   | settlements | 83 | 444 | 412 | 659 | 1,267 | 1,884 |

   Two symbols (RIVER and LAB, both listed this year) carry 53% of the
   total. Using the previous settlement's rate instead, a −10 threshold, a
   one- or four-hour hold, or a 15-minute entry delay all give the same
   shape: one year, a couple of names. Not a strategy.

   **Extreme moves and liquidation cascades.** A coin that has just fallen
   more than six of its own hourly standard deviations rebounds +73 bps net
   in the next hour, positive in all six years - and its excess over the
   cross-section is −6. The rebound is the market's, because these events
   are cascades hitting dozens of names in the same minutes. So the trade is
   the basket after an index-level event, and the honest entry is a bar
   after the print nobody could buy at. Index down more than 4 sd in an
   hour, long the equal-weight basket: +14.7 bps (t 1.17) entered at the
   bar's own close, **+2.4 (t 0.19) entered fifteen minutes later**, −7.3 held
   two hours, −6.6 in BTC instead. At 5 sd, 200 trades, +14.1 (t 0.74). The
   rebound is real and it is over within the bar that printed it.

   **Seasonality.** Equal-weight index return by UTC hour and weekday, by
   year. The most consistent hour (05:00, positive in 6 of 6 years) is worth
   +2.1 bps; nothing exceeds 3 bps an hour against a 10 bps round trip.

   **The cross-venue funding pair, held for days instead of a month.** Step
   9r/9t's BloFin-versus-partner pair at 3-day holds collects 8.0 bps of
   funding against four taker legs of 10, and loses; at 7 days it clears by
   +7.5 (3.9% a year); it only becomes the Sharpe-4 trade of 9x at 14-30 day
   holds. It is a long-hold trade by construction and does not shorten.

   What survives of the day, then, is 9aa: the working strategy, held for
   three days, on a lagged ranking. Everything genuinely intraday on this
   venue's data is either a spread being paid or one year's memecoins.

9ac. **Trying to improve the 3-day book, and what the planner now does** —
   `backend\analysis\factor_panel.py` (six new factors and four blends),
   `backend\analysis\tranche_book.py`, and `--momentum-weight` on
   `plan_carry_xs.py` / `run_carry_xs.py`.

   ```
   python backend\analysis\factor_panel.py --vol-scale --hold-days 3 --cost-bps 10 --lag 1
   python backend\analysis\tranche_book.py --hold-days 3 --factor carry_mom --cost-bps 10 --lag 1
   python backend\plan_carry_xs.py --notional 10000 --min-volume 2000000 --hold-days 3 --momentum-weight 0.4
   ```

   **Every existing factor at a 3-day hold**, 10 bps, lagged a day: the carry
   family clears (`carry_7` +17.4, `carry_14` +17.4, `carry_30` +19.0 at a
   turnover of only 0.47), the two blends clear (+22.4 / +22.9), momentum and
   low-vol are positive but weak (Sharpe 0.6-0.9), and everything that
   forecasts a reversal loses hard: `rev_3` −23.5, `rvshock_7` −21.6,
   `rangepos_30` −20.6, all beaten by 0% of their shuffles. At a 1-day hold
   the picture is the same, smaller.

   **Six new factors, signs stated before the run**: the lottery effect
   (`max_7`, short the biggest single-day jump), skewness, betting against
   beta, an abnormal-volume shock, and carry and momentum each divided by
   volatility. In sample, at three days: −1.8, −11.7, +7.9, −19.0, +6.3 and
   +12.7. None beats the plain versions; the volume shock is the wrong sign
   with conviction (10% of shuffles), which is a reversal signal again.

   **Four three-way blends, and this is the part worth reading.** Handing a
   fifth of the rank to low volatility raises Binance from +22.4 to +26.8
   (Sharpe 2.12); to the max-jump factor, +23.3. Both would have been adopted
   on the in-sample table. Out of sample at the same specification:

   | 3-day hold, lag 1 | Binance | BloFin, spreads | Hyperliquid | Bybit |
   |---|---|---|---|---|
   | `carry_mom` (frozen 60/40) | +22.4 | +21.5 | **+20.2** | +9.4 |
   | `carry_mom_lowvol` | **+26.8** | +21.3 | +15.3 | +9.1 |
   | `carry_mom_max` | +23.3 | +23.2 | +7.1 | +7.7 |
   | `carry_mom_flow` (taker share) | +18.5 | - | - | - |

   The extra components are Binance noise: nothing on BloFin or Bybit, and a
   third of the return gone on Hyperliquid. The frozen blend stands and the
   new factors stay in the file as the record of having been tried.

   **Staggered tranches.** A 3-day book has three rebalance phases; a live
   book can run all three at a third of the notional, rebalancing one a day.
   The phases' weekly returns correlate +0.70 to +0.74, so the gain is real
   but modest:

   | Binance, bps per week | mean | Sharpe | weekly sd | worst week | drawdown |
   |---|---|---|---|---|---|
   | best single phase | +53.6 | 2.07 | 187 | −562 | 1,119 |
   | worst single phase | +41.1 | 1.63 | 182 | −416 | 1,045 |
   | three tranches | +48.7 | **2.08** | **169** | **−357** | **809** |

   The same trades, spread over three days instead of one, and a third less
   damage in the worst week. That is also how the book should be put on for
   capacity reasons: a third of the gross crosses the spread each day.

   **No regime to filter on.** Split by the cross-section's trailing 30-day
   return at entry the book makes +14.0 / +27.3 / +26.0 across terciles; by
   trailing vol, +20.9 / +23.3 / +23.0. Flat, all positive. The only split
   that moves it is the market's move DURING the hold - +58.5 when the market
   falls, −8.8 when it rallies hard - which is not knowable at entry and is
   the signature of a book that is short the crowded names that squeeze in a
   rally. A vol or trend filter would be fitting noise.

   **The planner.** `BookConfig.momentum_weight` (default 0, the 9q book)
   rank-blends the carry score with 14-day momentum, computed from the same
   daily closes the volatility already uses; a name without 14 days of closes
   is excluded and named rather than scored flat. The funding-covers-round-trip
   gate becomes a WARNING for a blended book, because at a 3-day hold funding
   alone is not expected to cover it - the measured return is in the price
   leg, and the warning says so in those words. Run dry against BloFin on
   2026-09-12 at a $2M floor: 16 of 21 instruments eligible, five a side,
   long BTC/TRX/ETH/INJ/ZEC and short LTC/XMR/SOL/DOGE/SUI, expected funding
   +8.8 bps against a 12.4 bps round trip, and PLAN OK with the gate as a
   warning. Sixteen names is the honest width of this book on this venue.

   **A rebalance band, because turnover is a fifth of the gross.**
   `factor_panel.py --band` keeps a name the book already holds while it
   still ranks inside the top `top_frac + band` for its side, and opens only
   names inside `top_frac`. It is a cost mechanism rather than a signal, and
   it does what a cost mechanism should:

   | 3-day hold, lag 1, `carry_mom` | band 0 | band 0.1 | band 0.2 |
   |---|---|---|---|
   | Binance: net / Sharpe / turnover | +22.4 / 1.97 / 1.21 | **+23.7 / 2.10 / 0.78** | +22.9 / 2.07 / 0.57 |
   | BloFin $2M, spreads | +21.5 / 1.37 / 1.19 | +23.4 / 1.52 / 0.85 | +22.6 / 1.47 / 0.65 |
   | BloFin $1M, top 40% | +24.6 / 1.96 / 1.01 | +22.3 / 1.78 / 0.71 | +24.9 / 1.92 / 0.52 |
   | Hyperliquid $5M | +20.2 / 1.13 / 1.48 | **+23.5 / 1.31 / 1.09** | +10.7 / 0.60 / 0.83 |
   | Bybit $2M | +9.4 / 0.68 / 1.25 | +11.1 / 0.80 / 0.87 | - |

   Half the trading for the same return at 0.1, on four venues; at 0.2 the
   18-name Hyperliquid book starts holding stale positions and gives a third
   back. The three-tranche Binance book with a 0.2 band scores Sharpe 2.16 at
   +50.6 a week. **The width was chosen from this table**, so treat 0.1 as
   the pre-registered value for anything that follows and the rest as the
   sweep it came from. BloFin at a $1M floor and top 40% (24 names a
   rebalance rather than 15) is also the better BloFin book on every cut,
   which is the 9u finding again: this venue's binding constraint is width.

   **Two venues this account can reach, run as one book** —
   `backend\analysis\venue_stack.py`. BloFin and Hyperliquid rank
   overlapping coins on different funding, universes and marks, and their
   3-day books correlate at only **+0.29** (each about +0.4 with Binance).
   On the 322 common periods from 2024-01 to 2026-09, lagged a day, each
   venue at its own cost:

   | 3-day hold | net/period | Sharpe | sd | worst period | drawdown |
   |---|---|---|---|---|---|
   | BloFin alone (measured spreads) | +26.8 | 1.65 | 179 | −842 | 1,758 |
   | Hyperliquid alone | +20.9 | 1.27 | 182 | −634 | 2,092 |
   | **both, half the gross each** | +23.9 | **1.82** | **145** | −703 | **1,097** |
   | Binance, for reference | +25.6 | 2.07 | 136 | −546 | 731 |

   The stack gives up a tenth of BloFin's return for a fifth less variance
   and a drawdown 40% smaller, which is the diversification a 16-name
   cross-section cannot provide on its own. Hyperliquid needs only an
   address, so this is the most profitable *deployable* shape found so far:
   the same 3-day tranche book on both venues, each sized to its own capacity.

   **The weight is not on a cliff.** Momentum at 0.2 / 0.4 / 0.6 of the rank,
   3-day hold, lag 1, band 0.1: Binance +20.7 / +23.7 / +24.8, BloFin +20.7 /
   +22.3 / +23.7, Hyperliquid +3.5 / +23.5 / +19.0, Bybit +6.5 / +11.1 / +12.6.
   Flat on the two wide venues, a peak at 0.4 on the 18-name one, rising on
   the venue whose funding is least informative. 0.4 stays frozen.

   **Widening Hyperliquid did not widen it.** `panel_hyperliquid.py --top 120`
   returned 39 coins against the 53 in the panel every number above was
   measured on, so the wider panel is kept beside it as
   `hyperliquid-daily.top120.csv` and the 53-coin panel stays the reference.
   The venue's tradeable cross-section at $5M a day is about 18 names, and
   that - not the signal - is why its 3-day book is fragile to the floor.

   **Maker execution is not a free improvement on BloFin.** The round trip
   is 40% of the gross per 3-day period on this venue, and a 3-day hold has
   the patience to quote passively, so `passive_sim.py --source raw` was run
   on 12 hours of the recorder's own BloFin archive (2026-09-11), quoting at
   the touch every second with a 60s timeout:

   | BloFin | pessimistic net markout, 60s | fill rate |
   |---|---|---|
   | ADA-USDT | −6.4 bps (optimistic bound +0.3) | 11.0% |
   | DOGE-USDT | −7.2 | 3.2% |
   | SUI-USDT | −10.1 | 5.6% |
   | PEPE-USDT | −16.2 | 2.4% |

   A touch quote on this venue fills a few percent of the time and, when it
   does, is run over: even ADA's front-of-queue bound is a rounding error. So
   the plan keeps charging taker plus half the spread, and the honest way to
   cut the round trip is the band above, not the order type. Step 9's verdict
   on passive quoting stands on the venue it matters on.

   **What "most profitable" means here, and what it costs.** On gross
   notional the 3-day tranche book is about +49 bps a week, 25% a year
   un-compounded, at a weekly standard deviation of 169 bps. On CAPITAL at
   leverage L that is 25% x L a year against a measured worst week of
   3.6% x L and a worst drawdown of 8.1% x L - and step 9z showed the book's
   tail is seven times fatter than Gaussian at 4 sigma, so the worst observed
   is not a bound. At 3x that is ~75% a year against an observed drawdown of
   24% and a week that has already cost 11%; the February 2024 week, which a
   3-day hold shortened to −454 bps, would have cost 14% of capital. That is
   the trade, and the cap on any single name is still the only control that
   does not depend on estimating a correlation that has not happened yet.

9ad. **Eleven ways to make money inside eight hours without funding, and the
   one that survived** — `backend\analysis\lead_lag.py`, `pump_fade.py`,
   `listing_day.py`, `pair_reversion.py`, `venue_lag.py`, `fetch_bybit_oi.py`
   + `oi_factors.py` + `oi_cascade.py`, `liquidation_signal.py`,
   `fetch_premium_index.py` + `premium_signal.py`, `settlement_short.py --side long`,
   `listing_announcement.py`, and `backend\announcement_watch.py`.

   The brief on 2026-09-12: beat the carry book with something that is not a
   funding carry and holds for at most eight hours. Each idea below has a
   tool, a placebo or control, a year table, and a stated sign before the
   run. Ten are negative and are written up so they are not run again. The
   eleventh is the first sub-day result in this repo that clears its cost by
   more than a rounding error, and it is an event, not a forecast — which is
   the shape of the only other thing that ever worked here.

   ```
   python backend\analysis\listing_announcement.py --minutes --cost-bps 30 --events
   python backend\announcement_watch.py --check
   python backend\analysis\fetch_bybit_oi.py --top 80
   python backend\analysis\oi_factors.py --hold 8 --cost-bps 10
   python backend\analysis\liquidation_signal.py
   ```

   ### The ten that died, in the order they were run

   - **BTC leads the alts** (`lead_lag.py`). After a 2-3 sd BTC move over
     15 or 60 minutes, an equal-weight alt basket in BTC's direction hedged
     with BTC: gross −1 to −4 bps at every setting, placebo identical, and
     the pooled cross-correlation between BTC's last 15 minutes and the
     alts' hedged next 15 minutes sits within ±0.015 in every year. Splitting
     the basket into the alts that lagged and the ones that led changes
     nothing. The lag, if it exists, is shorter than a 15-minute bar.
   - **Fade the top of a pump** (`pump_fade.py`). `extreme_move.py`'s z ≥ 8
     bucket looked like +23 bps. Delay the entry one bar past the print, one
     event per coin per hold, and it is +5.0 ± 18.4 over four hours on 1,647
     events, +2.0 ± 26.3 over eight, with a worst single event of −3,505 bps.
     The median is positive; the mean is not, because the short is on the
     wrong side of the tail.
   - **The first hours of a new Binance perpetual** (`listing_day.py`). 73
     listings; the median path bleeds −200 to −340 bps over 48 hours, and a
     short from hour one to hour nine makes +80 ± 159 with the sign flipping
     by year. Fifteen events a year at that dispersion is not a strategy.
   - **Pairs at extremes** (`pair_reversion.py`). Each coin against its most
     correlated peer (trailing 30 days, re-chosen daily), a 24-hour gap
     beyond 2.5 sd, held four hours: 25,667 events, gross convergence +1.6
     bps against a 10 bps round trip; 8-hour holds at 3 sd, +1.0. It is step
     8's basis point again, conditioned harder.
   - **BloFin's book lags Binance** (`venue_lag.py`). BloFin's archived book
     against Binance aggTrades for the same day, one-second grid. An archive
     gap first read as a −229 bps p1 that never closed; a freshness filter
     (both venues updated within 3 s) removed it, and this is the check to
     keep. On SUI the lead is real — BloFin's mid follows a Binance-BloFin
     gap with slope 0.79 within 30 seconds — and a taker who buys the stale
     ask nets **−0.03 bps at mid and −2.5 at the touch** after 30 seconds.
     The lead exists and pays exactly the fee. ADA's Binance tick is 4.8 bps,
     so its gap is quantisation.
   - **Open interest** (`fetch_bybit_oi.py`, `oi_factors.py`). Bybit is the
     one venue serving hourly OI history (back to 2022, ~80 names), the
     first positioning series in any panel here. Ten factors — OI growth over
     1/4/24 h, its z-score, new-longs (sign of return × OI change), OI level
     against volume, OI churn — scored with `run_factor` at 8-hour holds, 10
     bps, 20 shuffles: **every gross leg within ±2 bps** (best `doi_24_z`
     +0.87 price, worst `rev_24` −2.00), controls at −14. The same at four
     hours. Positioning at this resolution carries no cross-sectional
     information a taker can pay for.
   - **Liquidation levels** (`liquidation_signal.py`). Built for the
     Hyperliquid archive, which on 2026-09-12 held 13.1 hours of BTC and
     ETH: 787 snapshots each, coverage 7% of long and 15% of short open
     interest, within-1% notional a median $44k on BTC. The magnet
     correlation comes back negative (t −2 to −5) on **14 hourly clusters**,
     and the overshoot test found no band swept. Fourteen clusters is a
     smoke test; the tool is written to be re-run in weeks, and widening the
     recorder past BTC/ETH is the operator's call (the running process was
     left alone).
   - **The settlement tail, honestly** (`settlement_short.py --side long`).
     Step 9ab's +49 bps hour after a >10 bps settlement priced as a trade:
     +53.5 per settlement (t 2.63) entered at the instant, **−3.9 with a
     15-minute delay**. And the −4 h "placebo" that dismissed it reads
     +72.8 — because the settled rate at T−4h contains four hours of future
     premium. The effect is real and lives inside the first quarter hour.
   - **The live premium index** (`fetch_premium_index.py`,
     `premium_signal.py`). Binance Vision publishes the perp-index premium at
     15 minutes, so the settlement study's lookahead can be removed. Ranked
     on the negated premium at the bar close, 106 names, 10 bps: the price
     leg is **+1.3 bps per hour and +2.2 per four hours** (IC +0.028, the
     largest intraday IC this repo has measured, and every year positive)
     against a turnover of 2.1 units — net −9.1 and −8.4. The tails are
     worse: a 1h premium beyond ±10 bps, held one or four hours, has an
     excess of about −1 bp either side on 78k and 106k events. The premium
     reverts, by one basis point, which is what a basis is.
   - **Intraday cascades read off OI** (`oi_cascade.py`). The hours where OI
     fell with price (a forced flush) against the hours where it rose (new
     shorts) or stayed flat, rebound priced at 1/4/8 hours from the close:
     the flush rebounds −11.6 ± 6.0 excess at one hour and −7.7 ± 10.3 at
     four on 6,600-6,900 events, indistinguishable from the placebo a day
     later and from the OI-flat rows. The distinction OI was meant to add
     adds nothing at hourly resolution; at one-hour holds the four OI
     factors' price legs are −0.37 to +0.16.

   ### The one that survived: the Binance listing announcement

   `listing_announcement.py` reads Binance's own announcement catalogue
   (2,253 articles, each with a millisecond release stamp), classifies each
   title as a spot listing ("Binance Will List X (X)") or a futures launch
   ("Binance Futures Will Launch USDⓈ-M XUSDT Perpetual"), and — because the
   coin cannot be bought on Binance yet — reads what it did on **Bybit**, at
   one minute, from 30 minutes before to nearly three hours after, with BTC
   over the same window as the market term. 378 coins named since 2022, 157
   of which Bybit already traded two hours before the release. Every number
   is net of a **30 bps taker leg** (the spread of a thin coin in that minute
   is the whole question, so the stress is the headline) and in excess of
   BTC, from the close of the minute the article landed in — up to 59
   seconds late.

   | spot listing, n 46 | +1m | +5m | +15m | +60m | +120m |
   |---|---|---|---|---|---|
   | long from the announcement minute's close | +105 | +96 | +244 | **+416 ± 250** | **+551 ± 265** |
   | median | −40 | +47 | +53 | +426 | +372 |
   | entered one minute later | −60 | −69 | +79 | +251 | +386 |

   The announcement minute itself is +1,339 (median +843) and nobody gets
   it. What is left after it is four to five percent over the next two
   hours, and it was there in 2023 (+859 at 60 m), 2024 (+575) and 2025
   (+560); in 2026 it is +157 with a negative median on 15 events, which is
   what a crowd of faster bots looks like and is the caveat on this leg.

   | futures launch, n 111 | +1m | +5m | +15m | +60m | +120m |
   |---|---|---|---|---|---|
   | long from the announcement minute's close | −180 ± 44 | −281 ± 88 | **−313 ± 108** | −288 ± 116 | −249 |
   | median | −140 | −212 | −180 | −346 | −274 |
   | hit rate of the long | 27% | 31% | 33% | 30% | 39% |

   The opposite trade, and the more robust one: the launch announcement
   jumps the coin +707 (median +514) in its minute and gives a third of it
   back within fifteen. **Short at the close of that minute, cover fifteen
   minutes later, nets +190 to +250 bps** at 30 bps a leg, 67-70% of the
   time, in 2024 (34 events), 2025 (62) and 2026 (15) alike. Bybit removes
   delisted contracts from its API, so any survivorship here cuts *against*
   this leg — the coins that went to zero afterwards are the ones missing.

   **As one strategy on capital**, one unit per event, long spot listings
   for 60 minutes and short futures launches for 15, daily P&L with zeros
   on the 90% of days with no event:

   | 30 bps/leg | events | bps/yr on capital | Sharpe (daily) | worst event |
   |---|---|---|---|---|
   | 2024 | 46 | +8,958 | 1.14 | |
   | 2025 | 74 | +22,657 | 1.75 | |
   | 2026 (to Sep) | 30 | +5,721 | 1.42 | |
   | all, 2022-2026 | 157 | **+8,694** | **1.13** | −4,054 |
   | futures leg alone | 111 | +8,514 | 1.09 | −3,128 |
   | spot leg alone | 46 | +4,114 | 0.80 | −4,054 |
   | at 50 bps/leg, both | 157 | +6,745 / +3,719 by leg | 0.87 / 0.73 | |

   **Does it beat the carry book?** Not on Sharpe: 1.1-1.5 by year against
   1.6, on 157 events rather than 243 weeks, and the interval on that is
   wide. On return per unit of capital it is not close — 85% a year at 1x
   against the carry book's 21% on gross — and it holds capital for minutes
   a few times a week, so it is an overlay on the carry book's margin rather
   than a competitor for it. That is the honest claim: the only sub-day idea
   of eleven that clears a 60 bps round trip by a multiple, positive in
   every year with more than five events, with a single-event tail of −40%
   that sets the position size.

   **What is not established, in order.** Every price is Bybit's; this
   account trades BloFin, which lists 35 of the 54 spot-listed coins and 99
   of the 174 futures-launched coins since 2024, and whose book in the
   minute after an announcement has never been observed. The 30 bps leg is
   a guess at that book. Reaction time matters: the futures leg is worth
   +180 in the first minute and the tables are for an order sent within 60
   seconds of the article. And the spot leg is decaying in 2026.

   So the next thing is not an executor. `backend\announcement_watch.py`
   polls the catalogue every ten seconds and, for every new listing or
   launch article naming a coin BloFin lists, samples BloFin's top of book
   and BTC-USDT's every two seconds for 150 minutes to
   `data/announcements/`, article beside it, and prints the spread at the
   first sample and the mid at +1/2/5/15/60/120 minutes. It has no order
   path. When a dozen of those CSVs agree with the Bybit tables, the
   executor is the thing to ask for by name.

9ae. **High frequency: be the stale side's counterparty, not its taker** —
   `backend\analysis\venue_lag_passive.py`.

   ```
   python backend\analysis\venue_lag_passive.py --date 2026-09-11 --instruments SUI-USDT,DOGE-USDT --binance-dir <aggTrades dir> --edge-bps 7 --stop-bps 3 --passive-exit --exit-at fair
   python backend\analysis\venue_lag_passive.py --date 2026-09-11 --instruments SUI-USDT --binance-dir <dir> --unconditional 10
   ```

   Step 9ad's `venue_lag.py` left one live number behind: BloFin's mid
   follows Binance with a slope of 0.79 inside thirty seconds, and a taker
   who lifts the stale ask earns exactly the fee. The brief on 2026-09-12
   was something high-frequency, and this is the same lead used from inside
   the book. No Binance account is involved anywhere: Binance is the public
   trade feed, and every order is on BloFin.

   **The mechanism.** When Binance's mid is `edge` bps above BloFin's
   `ask - tick`, post a bid at `ask - tick`. Nobody on BloFin quotes there
   yet, so the order is alone and first at its level; the next BloFin seller
   who has not seen the Binance print hits it, at the maker fee, at a price
   already known to be below fair. The simulation is event by event on the
   recorder's own book-and-trade archive with Binance aggTrades (plus 150 ms
   of feed latency) as the leader: a fill is the first sell-aggressor print
   at or below the order, or the ask coming down through it; the order is
   cancelled when the bid overtakes it, when Binance comes back, or after
   10 s. One order or position per side. The control posts the same orders
   on a ten-second clock with no signal — step 9ac's plain touch quote.

   **The lead turns a passive fill from adverse to favourable.** 2026-09-11,
   marked out at BloFin's mid, net of the 0.6 bps maker fee:

   | | fill rate | fills/day | markout at 30 s | 60 s |
   |---|---|---|---|---|
   | SUI, signal (edge 3) | 24.7% | 405 | **+2.99 ± 0.74** | +4.23 |
   | SUI, clock | 11.5% | 1,152 | −2.09 ± 0.40 | −2.05 |
   | DOGE, signal | 21.5% | 128 | +1.98 ± 2.27 | +2.89 |
   | DOGE, clock | 9.0% | 984 | −2.54 ± 0.37 | −2.70 |
   | AVAX, signal | 14.4% | 152 | +3.50 ± 1.78 | +2.84 |
   | AVAX, clock | 5.5% | 632 | −2.63 ± 0.50 | −2.44 |

   Five to six basis points per fill separate the two rows on every
   instrument, and the signalled quote fills twice as often. That is the
   whole edge, and every version of the exit below is an attempt not to give
   it back.

   **The exit decides the sign, and it took three versions.** A taker exit
   at the far touch after 30 s pays 5 bps plus half a spread and nets −3.7
   to −5.3. A maker exit one tick past entry fills at once for +0.5 and
   leaves the 10–30% of fills that go wrong to a forced taker exit at −12 to
   −27; net −0.3 to −4.5. A maker exit at the **Binance-implied fair** (the
   leader's mid rounded to the tick, never below entry + tick, pessimistic
   on queue position) nets +4 to +7 on the 60–88% that fill and still loses
   the rest at −11 to −35. What fixes it is using the leader on the way out
   too: **cross out at once if Binance moves `stop` bps through the entry**,
   so the informed fills are cut at −6 to −10 instead of −20. Three days,
   four instruments, pessimistic queue, 120 s maximum hold, net bps per fill:

   | edge / stop | SUI 11 / 10 / 09 | DOGE | AVAX | BTC | positive cells |
   |---|---|---|---|---|---|
   | 3 / 3 | −0.2 / −1.7 / −2.8 | −1.6 / +1.7 / +0.4 | −1.1 / −2.6 / −2.7 | +0.9 / +0.5 / +0.4 | 6 / 12 |
   | 5 / 3 | +4.4 / +2.9 / +0.5 | +0.1 / +3.7 / +2.0 | +1.8 / −0.7 / +0.3 | +4.0 / +1.9 / +3.9 | 11 / 12 |
   | **7 / 3** | **+7.1 / +3.4 / +1.4** | **+2.1 / +6.8 / +5.1** | **+3.3 / +0.8 / +3.8** | **+7.4 / +2.3 / +5.2** | **12 / 12** |

   The return per fill rises monotonically with the edge — 0, +2, +4 bps —
   which is what a real signal does and what a peak found by search does
   not. At 7 / 3 the fill counts are 14–175 a day per instrument (fill rate
   24–33%, median wait under a second, median hold 2–18 s) and the three
   days sum to about **290 bps a day per instrument on the notional of one
   order**. The optimistic queue bound at 5 / 3 adds about +1.5 per fill.

   **Where it does not work, and what it needs.** ADA, whose tick is 4.8
   bps, loses −3 a fill on all three days: a one-tick step is a whole spread
   there, and "fair rounded to the tick" is not a price. LTC is zero. And
   the edge is a latency edge: rerun with 500 ms instead of 150 ms of feed
   latency, SUI falls to +0.5, DOGE to −5.7 and AVAX to −3.5, with only BTC
   holding. The strategy exists at a Tokyo-hosted feed and does not exist
   from a home connection, and nothing here has measured which of those
   this machine is.

   What is still assumed, in order of how much it could move the result:
   the order is filled in full by any print at its price (size is not
   modelled, and the flow being caught is thin); BloFin's feed latency to
   here is the 150 ms the recorder measured as `t - ts`, and its order
   latency is zero; the parameters were chosen from three edges and two
   stops on the same three days; and three days is three days. The next
   thing is not a bigger backtest — the archive grows a day per day — it is
   a paper quoter that posts and cancels on demo with real latency and logs
   what fills, which is the executor question this repo asks for by name.

   ### The paper quoter — `backend\trading\strategies\lead_quote\`, `backend\run_lead_quote.py`

   ```
   python backend\run_lead_quote.py --instruments SUI-USDT --measure-only
   python backend\run_lead_quote.py --instruments SUI-USDT --minutes 60
   python backend\run_lead_quote.py --instruments SUI-USDT --minutes 60 --confirm
   python backend\run_lead_quote.py --instruments SUI-USDT --confirm --probe 10
   python backend\run_lead_quote.py --summary data\SUI-USDT\lead_quote\<run>.jsonl
   ```

   Three verbs, as the others. `quoter.py` is the backtest's rule set as an
   event-driven object with no I/O and no path to the order endpoint (a
   test greps that); `plan.py` gates on the two things 9ae died of — a tick
   over 2.5 bps of price, a feed lag over 350 ms, with a warning above 200;
   `execute.py` runs PRODUCTION feeds (Binance's public `bookTicker` and
   BloFin's books and trades — Binance is a data source, never an account)
   and fills the quoter from the production tape on paper, which is the
   reference result because the demo book is not the market; `monitor.py`
   reads the log back. `--confirm` additionally mirrors every intent to the
   demo account as a real `post_only` order, cancel or reduce-only exit and
   times each acknowledgement, and `--probe` sends far-from-touch orders and
   cancels them to time the venue with no fill possible: the number to
   compare a Germany box with a Tokyo one.

   **First runs, 2026-09-12, from the development machine:**

   | | |
   |---|---|
   | leader feed lag (Binance event time to receipt) | p50 81–98 ms |
   | follower feed lag (BloFin `ts` to receipt) | p50 78–90 ms, p99 up to 546 |
   | demo order acknowledgement, post_only and cancel | **p50 186 ms, p90 199, p99 257** (n 20) |
   | five-minute paper run | 1 post, 1 fill, maker exit **+8.51 bps** |
   | five-minute demo run | 0 posts: spread sat under two ticks the whole time |

   Both feed lags include whatever clock offset this machine carries, so
   they are upper bounds on the network; the order round trip is a real
   measurement and it is the one the backtest could not make. At 186 ms a
   post arrives roughly when the study's 150 ms feed assumption says the
   opportunity is half gone, which is the case for a host near the venues
   and the number to beat from one.

   **The first demo run closed a position it did not open.** Shutdown read
   the account's positions and sent a reduce-only market order for a SUI
   short of 1,677 contracts left over from earlier demo work. Demo money, no
   harm, and precisely the behaviour `strategies/__init__.py` exists to
   forbid. The runner now accumulates its own fills from the venue's order
   stream and closes only that quantity; anything else the account holds is
   reported as a `PROBLEM` and left alone, and a test holds it there.

   What this does not yet say: whether the paper fill rate and net per fill
   over hours match the backtest's 24–33% and +2 to +7 (one fill is one
   fill), and whether a demo `post_only` at `ask - tick` is ever filled by
   the demo book at all. Run it for a day; read it back with `--summary`.

   **requests vs aiohttp for the order path, 2026-09-12: 4–6 ms of ~185.**
   The claim was that `requests` adds latency. The Binance side is a
   websocket and no HTTP library touches it; what an HTTP client can affect is
   the BloFin REST order round trip. The SDK now has `blofin.async_client.AsyncClient`
   (same signing, `TradingAPI` unchanged, calls awaited on the loop) and the
   runner takes `--http aiohttp|requests`, default aiohttp.

   | same host, same hour | requests (in a thread) | aiohttp |
   |---|---|---|
   | `--probe 10`, 4 runs each, alternated: post + cancel acks (n 80 each) | p50 184 ms, p90 200, mean 187.9 | p50 180 ms, p90 193, mean 184.2 |
   | signed GET, interleaved per request, 150 rounds | p50 207.2 ms, p90 215.8 | p50 200.0 ms, p90 208.2 |
   | paired per round | | median **−6.5 ms**, faster in 120 / 150 |

   Of that, `requests`' own Python work (total minus `response.elapsed`) is
   p50 1.6 ms and the thread hop about 1 ms; the rest is on the wire side
   of `response.elapsed` and was not decomposed. Real and consistent, and
   about 3% of the round trip: the ~180 ms is distance to the venue, and
   the host is still the lever.

   The A/B turned up two older bugs. `--probe` never read the order id from
   the REST ack (`probe_post` was missing from the kinds that capture it), so
   a probe cancelled only when the WS order stream beat the ack, and the
   "post+cancel" row above it was mostly posts. And on Python 3.11
   `asyncio.wait_for` can swallow a cancel that lands as `recv()` completes:
   the Binance feed kept running after shutdown was requested, the process
   hung for nine minutes and left seven probe bids resting on demo
   (cancelled by hand). Both feeds now use `asyncio.timeout`, and a test holds
   the probe to cancel by the ack's order id.

   **From Tokyo, 2026-09-13 00:10 UTC: the venue is 30 ms away.** Noah ran
   the probe on an AWS Lightsail box in ap-northeast-1 (`--confirm --probe 10`,
   aiohttp), same code, same demo account:

   | host | leader feed lag | follower feed lag | order ack, post_only + cancel |
   |---|---|---|---|
   | development machine (2026-09-12) | p50 81–98 ms | p50 78–90 ms | p50 186 ms, p90 199, p99 257 (n 20) |
   | Tokyo (2026-09-13) | **p50 1 ms**, p90 2, p99 2 (n 29) | **p50 11 ms**, p90 24, p99 77 (n 27) | **p50 30 ms**, p90 35, p99 50 (n 20) |

   Plan gate `OK` (tick 1.38 bps, spread 4.14 bps at the time); 20 orders
   sent, none filled, no rejections, nothing foreign on the account. The
   leader lag of 1 ms says Binance's futures matching engine and this box
   share a region and a clock; the follower's 11 ms says BloFin's is close
   too. Six times faster on the order path and eighty on the feeds: the
   whole 9ae backtest was run at an assumed 150 ms of feed lag, so this host
   sits well inside the study's assumptions rather than at their edge, where
   the development machine was. The Germany box was not measured; at these
   numbers there is no reason to. What remains is the run that measures the
   edge rather than the plumbing: hours of paper fills from this host,
   compared with the backtest's 24–33% fill rate and +2 to +7 bps net per
   fill.

   **The eight-hour Tokyo run, first 70 minutes (2026-09-13):** 9 posts,
   1 paper fill at **+8.45 bps** (maker exit), feed lags unchanged, live
   order acks p50 29 ms; 17 demo orders, 1 filled, 1 rejected. The fill and
   the rejection are one story: the demo book filled a bid the production
   tape never did, the paper side cancelled, and the cancel came back
   rejected as already filled — leaving one contract open on demo for the
   rest of the run (shutdown would have closed it, being this run's own
   fill). The runner now closes such an orphan reduce-only the moment the
   fill is known, in either ordering of fill notice and cancel, logs it as
   `demo_orphan`, and the summary counts them; a rejected ack also keeps
   the message from the response envelope, which is where BloFin puts it
   (the summary had printed an empty one). Three tests hold the three
   cases. A demo fill without a paper fill is expected, not a signal: the
   demo book is thinner and stiller than production, so a resting bid one
   tick under its ask is hit more often there than it would be for real.

   **Where the remaining latency is, and what was done about it
   (2026-09-13).** Noah asked what could still be cut. Reading the code
   against the Tokyo log:

   - *The feed that produced an intent awaited the venue's acknowledgement.*
     `handle()` called `mirror()` inline, so the leader task read nothing
     for the 28 ms (p99 258) a post took to acknowledge - the moment the
     market was moving. Intents now go on a queue drained in order by one
     worker task; a feed never waits on REST. A test pins `handle()` to
     touch no broker.
   - *Most posts paid a TLS handshake.* Orders came ~3.6 minutes apart and
     an idle HTTPS connection does not live that long, so the p90 67 / p99
     258 tail is consistent with connection setup before the request even
     left (inferred from the spacing, not decomposed). A `keep_warm` task
     now sends a signed positions GET every 15 s on the same session and
     logs its round trip as `warm`; the summary reports it beside the acks,
     and splits acks by kind (post / cancel / exit), so a cold-post tail is
     visible rather than averaged away.
   - *The follower book is batched by the venue.* The SDK documents the
     `books` channel as incremental updates every 100 ms, so the 11-12 ms
     follower lag is receipt minus the batch stamp and the book itself is up
     to 100 ms behind the matching engine. Nothing on our side cuts that;
     `trades` push per print and are what the paper fill uses.
   - *The spikes in the follower lag* (28-44 ms among 11s in Noah's excerpt)
     are on the Python side or the host: a Lightsail instance is burstable
     with shared cores. Two cheap levers remain untested: `uvloop` (the
     runner installs it when present) and a dedicated-core EC2 instance in
     the same region.
   - Not available as far as known: order entry over the private websocket
     (BloFin documents REST only; unverified beyond the SDK) and anything
     below the venue's own matching latency.

   These change the demo mirror's timing and the paper quoter's feed
   handling, not the strategy; the running eight-hour log was left on the
   old code.

   **Two eight-hour runs, Tokyo, 2026-09-13.** Both on SUI-USDT at edge 7 /
   stop 3, paper fills from the production tape, demo orders mirrored. The
   Lightsail box ran the old code (mirror awaited inline, cold connections)
   from 00:2x UTC; the EC2 box ran the new code from 02:1x. About six of the
   eight hours overlap.

   | | Lightsail, old code | EC2 (uvloop, queued mirror, keep-warm) |
   |---|---|---|
   | posts | 53 | 67 |
   | paper fills | 6 (11%) | 13 (19%) |
   | net per paper fill | **+6.33 ±4.51 bps** | **+3.88 ±2.75 bps** |
   | passive (maker) exits | 67% | 69% |
   | fills / day, bps / day on one order | 18, ~114 | 39, ~151 |
   | post ack p50 / max | 55 / 339 ms | 36 / 68 ms |
   | cancel ack p50 / max | 24 / 48 ms | 29 / 51 ms |
   | keep-warm GET | — | p50 33 ms (n 1917) |
   | follower feed lag p50 / p99 | 12 / 80 ms | 12 / 99 ms |
   | demo fills | 8 | 0 |

   Pooled: 120 posts, 19 fills (16%), **+4.65 bps per fill**, roughly ±2.4,
   19 fills. Against the backtest's 24–33% fill rate and +2 to +7 net per
   fill: the net is inside the band, the fill rate is under it, and the
   daily rate on one order's notional is 40–50% of the backtest's ~290 bps
   — the ordinary paper haircut, and with a standard error that still
   admits zero. The sign is right on both hosts and both exits; that is
   what 19 fills can say.

   The keep-warm did what it was for: the post ack's median fell from 55 ms
   to 36 and its maximum from 339 to 68, while cancels (always sent onto a
   warm connection) stayed at 24–29. The extra posts on EC2 (67 vs 53 over
   overlapping hours) are consistent with the leader feed no longer
   stalling during its own acks, though the hours differ. The 15 rejected
   cancels on each host were all code 102068 on `post_only` bids the demo
   book had cancelled on arrival — the demo ask often sits under
   production's, so a bid at production's `ask - tick` would cross it.
   The summary now counts those separately from rejections.

   Next: more fills, not more hosts. SUI alone yields 20–40 fills a day;
   the backtest's other survivors (DOGE, AVAX, BTC) run as separate
   processes on the same box and triple the sample, and a week gives the
   standard error a chance to close.

   **Four instruments on the EC2 box, first 5.6 hours (Sunday 2026-09-13,
   ~09:30–15:10 UTC):**

   | | posts | paper fills | net per fill | maker exits |
   |---|---|---|---|---|
   | SUI-USDT | 23 | 4 | −3.46 ±3.73 bps | 25% |
   | DOGE-USDT | 14 | 2 | −0.43 ±5.17 | 50% |
   | AVAX-USDT | 15 | 0 | | |
   | BTC-USDT | 0 | 0 | | |

   A losing window, and the first: three of SUI's four fills were crossed
   out by the leader stop or the hold limit. SUI cumulative is now 143
   posts, 23 fills, about **+3.2 bps per fill**; all instruments pooled 172
   posts, 25 fills, about **+2.9**, still positive and still within a
   standard error or two of zero. Sunday afternoon is the quietest tape of
   the week, the backtest's three days were Wednesday to Friday, and the
   post rate shows it: 4 an hour on SUI against 8 on the weekday-overlap
   runs. BTC never posted at all — its tick is 0.01 bps of price and its
   BloFin book follows Binance inside the 100 ms batch, so the 7 bps gap
   with an empty level never appeared on a quiet day. Nothing is being
   tuned on 25 fills; the rule set in 9ae stands until there are a hundred,
   and the decision then is on the sign of the pooled net with its standard
   error, per instrument and together.

   **More pairs: the screen (`backend\analysis\lead_quote_universe.py`,
   2026-09-13).** Noah asked to test more pairs. The backtest needed three
   archived days per instrument; the paper quoter needs only production
   feeds, so the question for a new pair is whether it passes the quoter's
   own gate and has a BloFin tape to fill from. The screen reads BloFin's
   instrument and ticker lists and Binance's 24h futures tickers (all
   public, nothing sent) and applies: tick between 0.3 and 2.5 bps of price
   (above: ADA; below: BTC, whose BloFin book never lags a whole edge),
   median spread of at least two ticks over `--samples` reads (one read is
   not enough — LINK read 1, 2 and 6 ticks in three reads a minute apart),
   BloFin volume of at least $1M a day (AVAX at $0.5M gave 0 fills in
   5.6 h), a Binance USDT-M future of the same name with at least $20M a
   day. The six 9ae instruments carry their backtest verdict as a note.

   Of 464 USDT swaps, ten passed on Sunday afternoon, by BloFin volume:

   | | tick bps | spread ticks (room) | BloFin $M/day | Binance $M/day |
   |---|---|---|---|---|
   | XRP | 0.74 | 3 (80%) | 12.2 | 422 |
   | LINK | 0.87 | 3 (67%) | 3.8 | 89 |
   | DOGE | 1.19 | 3 (100%) | 3.4 | 235 |
   | SUI | 1.39 | 3 (87%) | 2.7 | 152 |
   | FLOCK | 1.44 | 7 (100%) | 1.7 | 161 |
   | BCH | 0.45 | 3 (87%) | 1.7 | 46 |
   | USELESS | 0.46 | 17 (80%) | 1.1 | 63 |
   | INJ | 1.63 | 8 (100%) | 1.1 | 39 |
   | UAI | 2.00 | 3 (67%) | 1.1 | 59 |
   | IOST | 1.29 | 3 (93%) | 1.0 | 46 |

   Notable failures: ETH and SOL (tick 0.04 bps; SOL's spread is one tick),
   WLD (tick 2.55), AVAX (volume), BNB (tick and volume). XRP is the one
   to watch: four times SUI's BloFin tape at half the tick. The wide-spread
   names (FLOCK, USELESS, INJ) are a different animal — a bid one tick
   under an ask that sits 8–17 ticks off the bid is far from fair, and the
   exit at Binance fair may be many ticks away; they are in the list because
   they pass the gate, not because the study says anything about them. The
   screen ranks; the week of `--summary` decides.

   **Sunday evening, 2026-09-13 (~21:00 UTC): the first hours of the wider
   set, and the SUI/DOGE runs to 7.5 h.**

   | | hours | posts | paper fills | net per fill | maker exits |
   |---|---|---|---|---|---|
   | SUI-USDT (13:31 run) | 7.5 | 29 | 6 | −1.82 ±2.96 | 33% |
   | DOGE-USDT | 7.5 | 16 | 2 | −0.43 ±5.17 | 50% |
   | BTC-USDT | 6.0 | 0 | 0 | | |
   | INJ-USDT | 1.55 | 14 | 3 | −3.04 ±5.22 | 33% |
   | LINK-USDT | 1.55 | 4 | 1 | −5.60 | 0% |
   | BCH-USDT | 1.55 | 1 | 0 | | |
   | XRP-USDT | 1.55 | **0** | 0 | | |
   | FLOCK, USELESS | | | | no log: not on the demo host | |

   Cumulative, every run so far: **SUI 149 posts, 25 fills, +3.1 bps per
   fill; all instruments 199 posts, 31 fills, +2.0 bps per fill**, standard
   error about 2. The two Saturday-night-to-Sunday-morning runs were the
   positive ones (+4.65 on 19 fills); every window since, Sunday afternoon
   and evening, has been negative, and the maker-exit share has fallen from
   two thirds to one third — the fills that do happen are the ones the
   market keeps running through. 31 fills cannot separate "small positive
   edge, noisy" from "zero edge and a good first sample"; the weekday
   sessions the backtest was measured on are what will.

   Three things learned about the machinery rather than the edge:

   - **XRP posted zero times** from the biggest BloFin tape on the list,
     and the runner could not say why: the leader never 7 bps past the
     post price, or it was and the level was occupied. `Quoter` now counts
     episodes of the leader beyond the edge and how many began with the
     level occupied, the runner writes them every minute (`quoter_stats`,
     so a killed run still has its last minute), and the summary prints
     "leader past the edge N times (max gap X), M with the level occupied".
     A test walks the three cases by hand.
   - **The demo host lists 87 of production's 488 swaps.** FLOCK and
     USELESS passed the screen, started under `--confirm`, and exited on the
     demo instrument lookup before a log existed. The screen now fetches the
     demo list, shows a `demo` column, and prints two launch lines: one with
     `--confirm` for pairs the demo mirror can take, one paper-only for the
     rest (FLOCK, USELESS, UAI tonight; LTC and FIL joined the passing list
     as the evening's spreads widened).
   - **A BTC position of 15.7 contracts** sat on the demo account that the
     BTC run did not open (the carry book's, on the shared demo account);
     the run reported it and left it alone, as designed after 2026-09-12.

   **The demo host's tick is not always production's (FIL-USDT,
   2026-09-13).** The FIL run started under `--confirm` at 21:15 UTC had
   36 of its first 44 demo posts rejected, code 102016 "Precision does not
   match: 0.001". The runner priced demo orders on the tick it read from
   production, and the two hosts' instrument lists disagree. Read from both
   at 21:40 UTC for the nine pairs running with `--confirm`:

   | | XRP | DOGE | LINK | SUI | LTC | BCH | FIL | UNI | INJ |
   |---|---|---|---|---|---|---|---|---|---|
   | production tick | 0.0001 | 0.00001 | 0.001 | 0.0001 | 0.01 | 0.01 | 0.0001 | 0.001 | 0.001 |
   | demo tick | 0.0001 | 0.00001 | 0.001 | 0.0001 | 0.01 | 0.01 | **0.001** | 0.001 | 0.001 |

   FIL is the only mismatch, and the 8 posts demo accepted were exactly the
   8 whose price happened to be a multiple of 0.001. Lot sizes differ too
   (demo lists XRP and DOGE at 0.1 against production's 0.01, LINK at 1
   against 0.1), but that was already harmless: the runner has always
   taken its order size from the demo host, and none of those logs has a
   size rejection. The paper side never saw demo, so FIL's paper fills
   stand; its demo mirror measured eight acks and nothing else.

   The runner now reads the tick from the demo host along with the size,
   logs it in the `start` row as `demo_tick`, prints a line when it differs
   from production's, and prices every demo order on it. The paper price
   is snapped to production's grid first (it arrives as
   0.9873999999999999), then a buy is rounded down and a sell up to demo's,
   so a `post_only` entry or maker exit never sits closer to the touch than
   the quote it mirrors; rounding to nearest would have sent some of them a
   whole demo tick more aggressive. The quoter itself stays on production's
   tick. On FIL one demo tick is ~10 bps of price, so a demo bid can now
   rest 0 to ~9 bps behind the paper bid: FIL's demo acks still time the
   order path, but its demo fills and its count of `post_only` cancelled on
   arrival are not comparable with the other pairs'. A test walks a
   FIL-shaped bid, its exit and an ask by hand; it fails on the old code,
   on nearest-rounding to the demo tick, and on rounding the exit the wrong
   way. FIL was stopped at 21:45 UTC (clean shutdown: its 9 accepted demo
   posts all ended cancelled, no demo fills, nothing to close) and
   relaunched at 21:52 on the fix: `demo_tick` "0.001" in the start row,
   and 10 of 10 demo posts and 10 of 10 cancels accepted in its first four
   minutes. The other eight needed nothing.

   **Two signed requests in one millisecond (2026-09-13).** At 21:39 UTC a
   SUI demo post came back 152407 "Repeated nonce" — the only such
   rejection in any lead-quote log, against ~9,700 signed requests since
   13:31. The vendored SDK made its REST nonce from the millisecond clock
   and used the timestamp itself as the websocket login nonce, while nine
   `--confirm` processes sign with one API key; BloFin's docs ask for a
   generator that never repeats within the server's window, "such as UUID".
   Re-sending one fixed nonce on the demo host reproduces it on both a POST
   (a cancel of a non-existent order id) and a GET. Which request the SUI
   post collided with is not in the logs — none of the logged requests
   started within 5 ms of it; the unlogged ones are websocket logins and
   anything else on the same key. The post was refused before the book (no
   demo order row for its client id) and the paper side never noticed.

   `blofin-sdk-python` now sends a UUID for both (a commit in the
   submodule, with two tests that freeze the clock and require distinct
   nonces; both fail on the old code). Checked on the demo host before
   merging: signed GET and POST over `requests` and aiohttp, and the
   private websocket login, all accept it. Running processes keep the SDK
   they imported; each picks the fix up at its next start, and at one
   rejection in ~9,700 none needs restarting for it.

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
