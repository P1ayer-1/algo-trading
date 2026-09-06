# Crypto Bot

Intended goal: a high-frequency, high-leverage crypto trading bot that uses
technical indicators to decide entries/exits and keeps liquidation levels at a
comfortable distance to manage risk.

## Current state (be honest with yourself here)

Right now this repo does **not** trade anything. What exists is a read-only
live chart viewer for BloFin:

- It pulls BTC-USDT candles and the live ticker from BloFin (REST + public
  websocket).
- It computes simple swing-based support/resistance levels from recent
  candles.
- It serves a local webpage with a live candlestick chart and a
  support/resistance panel, over a local HTTP server + websocket.

There is currently **no order placement, no position sizing, no leverage
management, no liquidation-distance logic, and no indicator-driven strategy**
(RSI/MACD/etc. are not implemented). Treat this as the market-data/visualization
foundation the actual trading logic still needs to be built on top of.

## Layout

```
.
├── backend/
│   └── live-chart.py       # fetches BloFin data, computes S/R levels, serves the chart
├── frontend/
│   ├── live-chart.html     # the chart page (lightweight-charts)
│   ├── package.json        # npm dep: lightweight-charts
│   └── node_modules/       # installed by `npm install` in frontend/ (gitignored)
├── blofin-sdk-python/      # vendored BloFin API SDK (its own git repo/history)
├── .env                    # BloFin API credentials (gitignored, never commit this)
└── .gitignore
```

## Setup

1. Python deps (from the repo root):
   ```
   python -m pip install -r blofin-sdk-python\requirements.txt
   ```
2. Frontend deps:
   ```
   cd frontend
   npm install
   ```
3. Fill in `.env` at the repo root with your BloFin API credentials
   (`API_KEY`, `SECRET`, `PASSPHRASE`). This file is gitignored — never commit
   real keys.

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

See the top of `backend/live-chart.py` for the full list.

## Roadmap toward the actual bot

The project description is "high frequency, high leverage trading, using
indicators to make decisions and comfortable liquidation levels to minimize
risk." None of that exists yet. Rough next steps, in order:

1. **Indicator layer** — compute the signals that will actually drive entries
   (e.g. RSI, EMA/MACD crossovers, volume/volatility filters) on top of the
   candle data this backend already fetches.
2. **Risk/liquidation module** — given an intended leverage and entry price,
   compute the liquidation price and enforce a minimum safety buffer before
   any order is allowed to go out. This should be a standalone, testable unit
   — high leverage is exactly where a bug here is most costly.
3. **Order execution** — wire a strategy layer to `blofin-sdk-python`'s
   trading REST/WS client to actually place and manage orders, starting on
   BloFin's demo environment.
4. **Position/state tracking** — track open positions, PnL, and stop
   conditions across restarts (this repo currently has no persistence at
   all).
5. **Backtesting** — before risking money at high leverage, validate the
   indicator + risk logic against historical data.

Given the stated leverage/risk profile, steps 1-2 and a demo-only trial of
step 3 should come well before anything touches a live account.
