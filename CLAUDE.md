# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A crypto trading research stack, mostly for BloFin, plus a Hyperliquid
recorder. Despite the README's stated goal (an HFT, high-leverage bot), almost
everything here **records data or evaluates it**. The order paths belong to
the funding-carry strategies (`run_carry.py` / `close_carry.py`,
`run_carry_xs.py`) and the lead-quote paper quoter (`run_lead_quote.py`, which
mirrors paper intents to demo as `post_only` orders); every one is dry unless
`--confirm`, demo unless `--production`, and refuses `--production --confirm`
together. `README.md`'s roadmap is the lab notebook: each step
records what was built, what it measured, and the verdict. Read it before
proposing strategy work — most obvious ideas have already been run and died on
execution cost. `STRATEGIES.md` is the status board on top of it: one row per
strategy (verdict, what is running where, next step) and one row per dataset
(writer, live or not, which strategy consumes it). When a roadmap step changes
a row, update `STRATEGIES.md` in the same commit; its "live?" column is a
measurement (process list plus newest file mtime), so re-check it rather than
carrying it forward.

## Commands

Interpreter (written as `python` in docs): `D:\Micromamba\micromambaenv\envs\crypto_bot\python.exe`
(`micromamba activate crypto_bot`). Entrypoints are run from the repo root;
tests from `backend/`.

```
cd backend
python -m pytest                                   # whole suite (pytest.ini already adds -q)
python -m pytest tests/test_risk.py                # one file
python -m pytest tests/test_risk.py::test_name     # one test
```

There is no lint or build config. Frontend: `cd frontend && npm install`.

```
python backend\record.py --instruments BTC-USDT,ADA-USDT      # BloFin, headless, N instruments
python backend\record.py --list-cost --instruments A,B         # disk arithmetic only
python backend\record_oi.py --channel mark-price --match-running   # add a REST channel to a live run
python backend\record_hyperliquid.py --check                   # validate coins live, no recording
python backend\record_hyperliquid.py --coins BTC,ETH,SOL,HYPE  # Hyperliquid + liquidation map
python backend\live-chart.py                                   # chart + one instrument's recorder
python backend\analysis\check_features.py --horizon 900 --data-dir data\BTC-USDT
python backend\plan_carry.py --instrument SUI-USDT --notional 2000 --leverage 3   # sends nothing
python backend\analysis\panel_daily.py                                            # build the daily panel
python backend\analysis\factor_panel.py --vol-scale --hold-days 7 --top-frac 0.3   # score the factors
python backend\analysis\factor_panel.py --vol-scale --hold-days 3 --top-frac 0.3 --cost-bps 10 --lag 1  # the 3-day book, honestly lagged
python backend\analysis\panel_intraday.py                                         # 1m archive -> 15m npz per symbol, once
python backend\analysis\intraday_factors.py --bar 480 --hold 3 --vol-scale        # the same harness at 8h bars
python backend\analysis\tranche_book.py --hold-days 3 --band 0.1 --lag 1              # three staggered tranches + regime table
python backend\analysis\venue_stack.py --hold-days 3 --band 0.1 --lag 1               # BloFin + Hyperliquid as one book
python backend\plan_carry_xs.py --hold-days 3 --momentum-weight 0.4 --min-volume 2000000  # the 9aa book, dry
python backend\plan_carry_xs.py --notional 4000 --min-volume 2000000               # sends nothing
python backend\analysis\fetch_bybit_oi.py --top 80                                 # hourly OI + klines from Bybit, once
python backend\analysis\oi_factors.py --hold 4 --cost-bps 10                       # positioning factors on that grid
python backend\analysis\listing_announcement.py --minutes --cost-bps 30 --events   # the announcement event study (9ad)
python backend\announcement_watch.py --check                                       # then without --check: record BloFin after each announcement
python backend\analysis\liquidation_signal.py                                      # Hyperliquid liquidation levels vs forward return; needs weeks of archive
python backend\analysis\venue_lag_passive.py --date 2026-09-11 --instruments SUI-USDT --binance-dir <aggTrades dir> --edge-bps 7 --stop-bps 3 --passive-exit --exit-at fair   # the HF maker study (9ae)
python backend\run_lead_quote.py --instruments SUI-USDT --measure-only               # feed lag on this host vs the 9ae gates
python backend\run_lead_quote.py --instruments SUI-USDT --minutes 60                 # paper quote from production feeds, sends nothing
python backend\run_lead_quote.py --instruments SUI-USDT --minutes 60 --confirm       # ...and mirror intents to DEMO as post_only orders
python backend\run_lead_quote.py --summary data\SUI-USDT\lead_quote\<run>.jsonl      # read a run back
python backend\analysis\lead_quote_universe.py --top 12                           # which pairs pass the 9ae gate, by BloFin volume; sends nothing
```

`.env` at the repo root holds `API_KEY`/`SECRET`/`PASSPHRASE` and overrides such
as `BLOFIN_VIP_TIER=1`. Recording and all market data need no keys. Every
setting lives in `backend/config.py`.

## Architecture: the rules that span files

**The raw archive is the irreplaceable half.** `trading/rawlog.py` writes every
message verbatim to `data/<INST-ID>/raw/<day>/<channel>-<HH>.jsonl.gz` as
`{"t": receive_ms, "n": per-process counter, "m": message}`. Anything derived
(feature CSVs, maps) is regenerable — `analysis/replay.py` rebuilds features
through the same `OrderBook`/`TradeTape`/`FeatureEngine` as live. Merge a
websocket channel's events on `(t, n)`. REST-polled channels (open-interest,
mark-price) run in other processes whose `n` is unrelated, so as-of join them
on `t` only. Never subscribe to less than you might later need: unrecorded
data is gone. A writer never appends to an existing file: a restart inside an
hour writes `<channel>-<HH>.r001.jsonl.gz` beside it, because appending after
a hard kill made the whole hour unreadable (2026-09-11). Read through
`iter_events`, which resumes at the next gzip member past a torn one;
`analysis/audit_raw.py` reports which hours were affected.

**One directory per instrument, enforced.** Rows for two symbols are
structurally identical, so layout is the only thing separating them.
`analysis/layout.py` owns path knowledge (`INSTRUMENT_DIR` matches
`BASE-QUOTE` directly under `data/`); `check_features.py` refuses to span
instruments. Hyperliquid lives under `data/hyperliquid/<COIN>/` and
`data/hyperliquid/_accounts/` precisely so BloFin tools cannot see it.

**One writer per channel per instrument.** Two writers used to share and
destroy an hourly gzip file (measured: 0 of 40 records); files are now created
exclusively, so a second writer would duplicate rather than destroy. Snapshot
pollers (`trading/openinterest.py` `SnapshotPoller`, `markprice.py`) still
take an exclusive pid lock per instrument *per channel*;
`record_hyperliquid.py` locks the whole venue directory. Stale locks from dead
pids are taken over.

**Long runs must not die or go silent.** Every loop runs under
`server.supervise` (restart on exception, re-raise `CancelledError`). Loops catch
their own per-iteration errors and count failures instead of raising. Feeds
await each message with a deadline (`ingest._stream`,
`HyperliquidMarketFeed._stream`), because a stalled socket raises nothing.
Archive first, parse second.

**Import boundaries are deliberate.** `trading/risk.py` imports nothing from
the package (the veto cannot depend on what it vetoes) and uses `Decimal`; the
book and features use `float`. `trading/__init__.py` loads `ingest` lazily so
the package imports without the vendored BloFin SDK. Tests must pass with no
SDK, network or credentials — `tests/conftest.py` does not add the SDK to the
path, and SDK-dependent test modules `pytest.importorskip("blofin")`. HTTP
clients take an injectable `opener`; clocks and sockets are injected fakes.
`config.py` inserts `blofin-sdk-python/src` (a vendored separate git repo) onto
`sys.path`; entrypoints insert `backend/`.

**Hosts matter.** Public market data is always read from production,
regardless of `BLOFIN_USE_DEMO`, so a dataset never describes the demo book.
Margin tiers (`trading/margin_tiers.py`) are fetched from the *account's* host,
because demo and production MMR differ (measured 0.0065 vs 0.0050).

**Strategies: three verbs, one directory each** (`trading/strategies/<name>/`).
`plan.py` computes orders and has no path to `placeOrder` (grep-checkable),
returning every failing gate, not just the first. `execute.py` is dry unless
told otherwise and owns leg ordering and unwinds. `monitor.py` is read-only and
verifies against the exchange rather than the plan. The lifecycle is Protocols
in `strategies/__init__.py`, satisfied structurally; there is intentionally no
`Strategy` base class. `tests/test_strategy_contract.py` pins this.

**The panels are one schema, three venues.** `analysis/panel_daily.py`
(Binance, from the 1m archive), `panel_blofin.py` and `panel_hyperliquid.py`
all emit the same daily CSV columns, so `factor_panel.py` and
`funding_dispersion.py` read any of them unchanged. Two rules hold across all
three or the comparisons are meaningless: a row closes at 00:00 UTC, and
funding for day D is what ACCRUED during day D - the settlement stamped 00:00
on D+1, rounded to its nominal minute first because settlements print
milliseconds late. Venues name coins differently (`1000BONKUSDT`, `kBONK`,
`BONK`); `funding_dispersion.canonical()` owns that matching. A daily funding
TOTAL is the comparable unit, never a per-settlement rate - Hyperliquid funds
hourly and the others three times a day.

**Cross-sectional results are scored as money, with non-overlapping holds.**
`factor_panel.py` never overlaps holding periods, so the rebalance count IS the
effective N; it charges cost on turnover rather than per position; it splits
gross into its price and funding legs, which for a carry factor is the whole
result; and its control is the MEAN of many shuffles plus the percentile the
real book beat, never the best draw. The control mean should land near minus
the cost - if it does not, the cost model and the turnover disagree.

**Fees decide every verdict.** `config.VIP_TIERS` / `SPOT_VIP_TIERS` are hand
transcribed; unconfirmed tiers are absent, not interpolated; an import-time
check rejects a non-monotonic ladder. Default tier is VIP 0 on purpose.

**Labels never see the future.** `trading/recorder.py` writes a row only after
its longest forward horizon (default 1800s) has elapsed, so a fresh run writes
no features for 30 minutes. Evaluation tools measure edge as excess over the
sample's drift, correct for overlapping-window effective sample size, and use
purged splits and shuffled-label controls.

**Hyperliquid liquidation levels are read, not modelled.** Every trade names
both accounts, so `trading/hyperliquid.py` discovers accounts from the trade
feed and reads `clearinghouseState` (whose positions carry the exchange's own
`liquidationPx`) under a REST weight budget (1,200/min per IP, 2 per read).
Accounts that just traded are read first; aged holders are refreshed by
`age * sqrt(notional)`. `trading/liquidation_map.py` bands them by distance
from mark and reports coverage vs open interest, null-liquidation-price size,
crossed levels and reading age beside the levels. BloFin exposes no one else's
positions, so there any such map would be a reconstruction.

## Conventions in this codebase

- Module docstrings explain *why*, with measured numbers and the date they were
  measured. Keep them accurate when behaviour changes; mark unverified
  assumptions as such rather than stating them as fact.
- Refusals are `SystemExit` with every reason listed and what to do next.
- Tests assert hand-computed values, not whatever the code returns, and their
  docstrings state the failure they guard against.
- New results go in the README roadmap in the existing style, including
  negative ones.
