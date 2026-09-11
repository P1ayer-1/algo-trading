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
│   │   ├── fetch_klines.py    # bulk 1m klines, with the listing dates made visible
│   │   ├── validate_liquidation.py  # our liq math vs the exchange's own
│   │   ├── blofin_spread_survey.py  # the same, live, on BloFin itself
│   │   ├── layout.py          # where recorded data lives; one owner
│   │   ├── replay.py          # rebuild features from raw events
│   │   ├── compact.py         # CSV -> Parquet, storage report
│   │   └── stats.py           # IC, AUC, logistic regression, purged split
│   └── tests/                 # pytest suite (747 tests)
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

496 tests covering the order book's gap handling, the OFI recursion, the
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
   | centre IC | +0.064 (t 2.12) | **+0.011** (t 1.55) |
   | control ceiling | +0.062 | +0.039 |
   | symbols positive | 8/10 | 21/35 |

   More data made the apparent skill *shrink toward zero*, which is what a
   small-sample draw does and is the opposite of what a real effect does. The
   ten-major +0.064 was noise that happened to point one way, and it never
   separated from its control even then.

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
