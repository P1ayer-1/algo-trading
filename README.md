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
│   │   ├── check_features.py  # do the features predict anything?
│   │   ├── train_model.py     # LightGBM + shuffled-label control + paired test
│   │   ├── replay.py          # rebuild features from raw events
│   │   ├── compact.py         # CSV -> Parquet, storage report
│   │   └── stats.py           # IC, AUC, logistic regression, purged split
│   └── tests/                 # pytest suite (219 tests)
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
| `BLOFIN_RECORD_FEATURES` | Write labelled features to `data/` | `true` |
| `BLOFIN_FEATURE_SAMPLE_MS` | How often to persist a row | `1000` |
| `BLOFIN_LABEL_HORIZONS` | Forward label horizons, seconds | `300,900,1800` |
| `BLOFIN_LABEL_THRESHOLD_BPS` | Move size counted as a signal | `10` |
| `BLOFIN_MAKER_FEE_RATE` / `BLOFIN_TAKER_FEE_RATE` | Fee schedule, per side | `0.00006` / `0.0005` |
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

219 tests covering the order book's gap handling, the OFI recursion, the
recorder's lookahead guard, the liquidation math (against hand-computed
values), the unrealized-drawdown breakers, the reduce-only close path, the
raw-archive round trip, and the evaluation statistics. They need
no network, credentials, or SDK.

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
8. **Better features, then regime detection** — the honest next step. The bar
   feature set is bars, open interest, positioning and carry; it has no
   cross-asset, no order-flow at the trading horizon, and no event data. Also
   replace the percentile-based `vol_regime` placeholder with a fitted model.
9. **Execution engine** — adaptive limit orders, wired to the risk engine's
   `check_order()`. This is where the existing kill switch finally guards
   something real. Demo account only.
10. **Backtesting / paper trading** — with realistic fees, queue position and
   slippage. Expect the paper results to be considerably worse than the
   backtest; that gap is the honest measure of the model.

Given the stated leverage profile: this stays on the BloFin demo environment
until steps 5-9 are done and the liquidation math has been checked against the
exchange's own numbers. Leverage magnifies model *and* execution errors, and
the risk engine's default limits (5x, 15% liquidation buffer) are deliberately
far more conservative than the project's stated ambition.
