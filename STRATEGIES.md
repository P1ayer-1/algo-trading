# Strategy board

**As of 2026-09-13 15:00 UTC.** This file is the *status* view: what each
strategy is, what the evidence says, what is on right now, what data it eats,
and what the next step is. It is rewritten in place. The *history* - what was
built, what it measured, why it died - stays in `README.md`'s roadmap (steps
9a-9ae), and every row here points at its steps. When a roadmap step lands,
update the row here in the same commit.

Nothing in this repo trades production. Every order path is dry unless
`--confirm`, demo unless `--production`, and refuses both flags together.

## 1. Strategies

| # | strategy | README | code | evidence verdict | on right now | next |
|---|---|---|---|---|---|---|
| S1 | **Cross-sectional funding carry** - long the perps whose funding is lowest, short the highest, dollar-neutral, rebalanced on a fixed hold | 9q-9ac (9y2 for the monitor) | `trading/strategies/carry_xs/`, `plan_carry_xs.py`, `run_carry_xs.py`, `monitor_carry_xs.py` | **The one that works.** Funding leg +12 to +21 bps/week in all 15 venue-by-floor cells. Deployable shape (9ac): `carry_mom` 60/40 blend, 3-day hold, lag 1, band 0.1, momentum weight 0.4, three staggered tranches; BloFin +23.4/period Sharpe 1.5, BloFin+Hyperliquid stacked Sharpe 1.8. BloFin capacity a few hundred $k gross. | Demo book, 7 legs, $8.4k gross, opened 2026-09-12 14:44 UTC as a 7-day single epoch. **Broken**: SUI leg vanished, book 13% net long vs 2% tolerance, monitor exits critical. | Housekeeping first (§4). Then build what the backtest has and the executor lacks: 3-day tranche schedule, rebalance band, forecast carried into the monitor's baseline, a Hyperliquid executor for the stack. |
| S2 | **Two-leg funding carry** - long spot, short the perp, collect funding | 9e-9j | `trading/strategies/carry/`, `plan_carry.py`, `run_carry.py`, `monitor_carry.py`, `close_carry.py` | Cleared cost on paper (SUI +170 bps/30d forecast) but **superseded by S1**: no borrow so long-spot only, spot spread is the binding cost, one instrument at a time. Kept as the reference for the three-verb lifecycle and the funding derivation. | Demo, SUI-USDT, opened 2026-09-09. **Broken**: perp leg gone, 253.746 SUI spot held unhedged, funding was negative the whole hold (-1.9 bps/day vs +6.54 planned). No `closed.json`. | Close or re-hedge the spot (§4). Do not open another; S1 replaces it. |
| S3 | **Lead quote** - post a BloFin maker bid at `ask - tick` when Binance's mid has moved `edge` bps through it; cut if Binance moves `stop` bps back | 9ad (`venue_lag.py`), 9ae | `trading/strategies/lead_quote/`, `run_lead_quote.py`, `analysis/venue_lag_passive.py` | Backtest 12/12 instrument-days positive at edge 7 / stop 3, +2 to +7 bps/fill, ~290 bps/day on one order's notional. It is a **latency edge**: exists at ~150 ms feed lag, gone at 500 ms. Not ADA (tick too wide), not LTC. | Paper + demo mirror from Tokyo (EC2, ap-northeast-1). Runs so far, 2026-09-13: two 8-hour SUI runs (+4.65 bps/fill on 19 fills, Saturday night to Sunday morning), then SUI/DOGE/BTC/AVAX and from 19:28 UTC XRP/LINK/BCH/INJ, every later window negative (Sunday afternoon and evening). **Cumulative: SUI 149 posts / 25 fills / +3.1 bps per fill; all instruments 199 / 31 / +2.0, SE ~2.** Maker-exit share fell from 2/3 to 1/3 in the later windows. XRP: 0 posts in 1.55 h (the runner now counts leader-past-edge episodes to say why). BTC: 0 posts. FLOCK/USELESS: not on the demo host, no log. **FIL's demo mirror measured nothing** (9ae, 21:40 UTC): demo lists FIL at tick 0.001 against production's 0.0001 and rejected every post off that grid (36 of 44, code 102016). The other eight ticks match. The runner now prices demo orders on the demo tick (bids down, asks up) and logs `demo_tick`; FIL was relaunched on it at 21:52 UTC (10/10 posts accepted in its first 4 min; its demo fills stay non-comparable, since a demo bid rests up to ~9 bps behind the paper one). One SUI post rejected 152407 "Repeated nonce" (21:39): the SDK's millisecond-clock nonce, shared by nine processes on one key; the SDK now sends a UUID, picked up by each process at its next start. Order ack p50 24-34 ms, feeds 2 / 10-15 ms. Nothing runs from this machine. | Keep the nine demo-listed pairs the screen passes running through the weekdays (XRP, DOGE, LINK, SUI, LTC, BCH, FIL, UNI, INJ with `--confirm`; FLOCK, USELESS, UAI paper-only); drop BTC and AVAX. Read XRP's `quoter_stats` line first. Decide at ~100 pooled fills on the sign of net/fill with its standard error, per pair and together; no parameter changes before then. |
| S4 | **Listing-announcement event** - long a coin for 60 min after Binance announces a spot listing; short for 15 min after a futures-launch announcement | 9ad | `analysis/listing_announcement.py`, `announcement_watch.py` (recorder only, no order path) | On Bybit prices, 157 events 2022-2026, 30 bps/leg: futures-launch short +190 to +250 bps net, 67-70% hit rate, every year; spot-listing long +416 at 60 min but decaying in 2026. As one strategy ~+87%/yr on capital at Sharpe 1.1-1.5, worst event -40%. **BloFin's book after an announcement has never been observed.** | `announcement_watch.py` running since 2026-09-12 13:06 local, polling every 10 s. **Zero events captured** so far (`data/announcements/` does not exist yet). | Wait for a dozen event CSVs and compare to the Bybit tables. If they agree, ask for the executor by name. |
| S5 | Range fade (buy the bottom of the last N hours' range, sell the top) | 9k-9p | `trading/strategies/range_trade/levels.py`, `analysis/range_backtest.py` | **Dead.** Every timeframe, side and gate priced below cost; the attention model (9p) did not rescue it. | Nothing. | Nothing. Do not re-run. |
| - | Everything sub-day that is not S3/S4 | 9ab, 9ad (the ten), 9w, 9z | `analysis/*` | **Dead**, written up so they are not run again: intraday factors, settlement tail after a 15-min delay, premium index, OI factors and cascades, BTC-leads-alts, pump fade, listing day, pair reversion, liquidation-level magnets (smoke test only), cross-venue crowding rank, touch quoting on BloFin. | Nothing. | Re-run only `liquidation_signal.py`, once the Hyperliquid archive has weeks (it needs the recorder in §2 alive). |

**How the four live ones relate.** S1 is the book; it holds capital for days
at a Sharpe near 2 and is capacity-bound on BloFin. S4 is an overlay on the
same margin: minutes of exposure a few times a week, high return per unit of
capital, fat single-event tail. S3 is a separate HFT process that needs a
host near the venues and one order's notional per instrument. S2 is
retired. None has been run in production.

## 2. Data being collected, and why

Rule from `CLAUDE.md`: the raw archive is irreplaceable, everything else is
regenerable from it or re-fetchable. "Live" below was checked against
process list and file mtimes at 2026-09-13 14:52 UTC.

### Live recorders

| dataset | path | writer (process on this machine) | since | size | live? | feeds |
|---|---|---|---|---|---|---|
| BloFin `books`, `trades`, `funding-rate` (websocket) + `open-interest` (REST) for 15 instruments: BTC, ADA, DOGE, LTC, AVAX, PUMP, PEPE, 1000BONK, WLD, SUI, INJ, IOST, ATOM, CNPY, FARTCOIN | `data/<INST>/raw/<day>/<channel>-<HH>.jsonl.gz` | `record.py` (pid 19472, since 2026-09-11 14:20 local); this process also owns the `open-interest` lock | BTC 2026-09-07, rest 2026-09-09 | ~3.7 GB, ~1.35 GB/day | **yes**, all five channels wrote in the current hour | S3 backtest (`venue_lag_passive.py`) and its paper fills; S1 spread snapshot (`panel_blofin.py`); `passive_sim.py`; `replay.py` feature rebuilds; `audit_raw.py` |
| BloFin `mark-price` (REST, mark + index for all instruments) | same layout, channel `mark-price` | `record_oi.py --channel mark-price` (pid 88728, since 2026-09-11) | 2026-09-10 | in the above | **yes** | the basis (`conv` cost in `funding_carry.py`), liquidation price checks in S1/S2 monitors |
| Hyperliquid `l2Book`, `trades`, `activeAssetCtx` for BTC, ETH; `clearinghouseState` per discovered account; derived liquidation map | `data/hyperliquid/<COIN>/raw/`, `data/hyperliquid/_accounts/raw/`, `data/hyperliquid/<COIN>/liquidation-levels-<day>.jsonl` | `record_hyperliquid.py` (lock holder pid 38180) | 2026-09-12 | 777 MB (690 MB is `_accounts`) | **NO - dead.** pid 38180 is not running; last write 2026-09-13 14:19 UTC. Stale lock will be taken over on restart. | `liquidation_signal.py` (needs weeks; had 13 h at last run); the Hyperliquid half of the S1 stack has no executor yet and reads none of this |
| Binance listing announcements, and BloFin top-of-book for 150 min after each | `data/announcements/<stamp>-<instId>.csv` (not yet created), `data/announcements.log` | `announcement_watch.py` (pid 97768, since 2026-09-12 13:06 local) | 2026-09-12 | 0 | **yes**, 0 events so far | S4, the only thing that can say what BloFin's book does in the minute after an announcement |
| Lead-quote run logs: feed lags, intents, paper fills, demo acks | `data/<INST>/lead_quote/<run>.jsonl` | `run_lead_quote.py`, per run; the Tokyo runs live on the Lightsail / EC2 boxes, not here | 2026-09-12 | small | no run active from this machine | S3 scoreboard via `lead_quote_board.py` (pooled; `--follow` live), one run via `--summary` |
| S1 / S2 position state | `data/carry_xs/demo/{baseline,epoch-*}.json`, `data/SUI-USDT/carry/{baseline.json,snapshots.jsonl}` | the monitors, on first sight of a book | 2026-09-09 / 09-12 | tiny | written on each monitor run | the S1/S2 monitors' frozen forecast and scoreboard |

There is also a `predkit.record_series` process (pid 77364, `pred_kit` env,
Polymarket/Kalshi BTC series) running on this machine. It belongs to the
prediction-market toolkit, not this repo, and writes nowhere under this
`data/`.

### Fetched or derived (regenerable, re-fetchable)

| dataset | path | tool | used by |
|---|---|---|---|
| Binance 1m/1h klines + funding, 118 symbols | `data/cache/` (6 GB) | `fetch_klines.py`, `bars_import.py`, `cross_sectional_import.py` | the Binance daily panel and everything intraday (9ab, 9ad) |
| Binance premium index at 15 min | `data/cache/premium/` | `fetch_premium_index.py` | 9ad premium study (dead) |
| Daily panels, one schema, six venues: Binance, BloFin, Hyperliquid (53-coin reference + top120), Bybit, Kraken, MEXC; BloFin spread snapshot; per-symbol daily sources | `data/panel/*.csv`, `data/panel/daily/` | `panel_daily.py`, `panel_blofin.py`, `panel_hyperliquid.py`, `panel_venue.py` | `factor_panel.py`, `tranche_book.py`, `venue_stack.py`, `funding_dispersion.py` - i.e. all S1 evidence |
| 15m intraday panel per symbol | `data/panel/intraday/*.npz` (109) | `panel_intraday.py` | `intraday_factors.py` (9aa/9ab) |
| Bybit hourly OI + klines, ~80 names | `data/panel/bybit_1h/` (160 files) | `fetch_bybit_oi.py` | `oi_factors.py`, `oi_cascade.py` (dead) |
| Binance aggTrades / bookTicker zips (2024-03-01) | `data/binance/zips/` | `binance_import.py` | `venue_lag*.py` leader tape (S3) - the day used in 9ae is passed by `--binance-dir` |
| Tardis Binance-futures book snapshots (2026-09-01) | `data/tardis/` (400 MB) | `tardis_import.py` | `venue_compare.py`, `spread_survey.py` (step 9a-9c era) |
| Labelled feature CSVs from klines | `data/bars/`, `data/cross/` | `bars_import.py`, `cross_sectional_import.py` | `check_features.py`, `train_model.py` (steps 4-8, pre-carry) |
| Replayed features from raw | `data/replayed/`, `data/<INST>/features-*.csv` | `replay.py`, `recorder.py` | `check_features.py` |

## 3. Accounts and hosts

| | |
|---|---|
| BloFin demo | holds the S1 book and the S2 remnant. Also used by S3's demo mirror from Tokyo. Three strategies on one cross-margined account, and S1 and S2 both had a SUI-USDT perp leg on it - the likely reason both lost it. Keep one strategy per demo account or per instrument until the shared-account problem is solved. |
| BloFin production | keys in `.env`; public market data is always read from here. No strategy has run against it. |
| Hyperliquid | no account. Only an address is needed; nothing here sends to it. |
| Tokyo (AWS Lightsail + EC2, ap-northeast-1) | the only hosts S3 exists on: feeds 1 / 11 ms, order ack p50 30 ms. Run logs are on those boxes. |
| This machine | recorders and the announcement watcher. Feeds ~100 ms, ack ~186 ms: fine for recording, out of spec for S3. |

## 4. Open problems, in order

1. **S2 remnant on demo**: 253.746 SUI spot with no hedge. `close_carry.py --instrument SUI-USDT --confirm` handles this case: it sells the spot, reports the perp leg as "nothing - no open position", and writes `closed.json`. The funding it captures will be zero, because the `realizedPnl` it is derived from left with the perp; the snapshots up to 2026-09-10 are the only record of that hold's funding (negative).
2. **S1 demo book is off-spec**: run `monitor_carry_xs.py --archive-epoch`, then either `run_carry_xs.py --repair-only --confirm` or `--flatten --confirm`. Nothing in the repo did the SUI close, so the account was touched from outside again (see 9y).
3. **Hyperliquid recorder is down** since 2026-09-13 14:19 UTC. `liquidation_signal.py` needs weeks of archive; every hour down is gone. Restart with `record_hyperliquid.py --coins BTC,ETH` (or widen; that is the operator's call).
4. **S1 executor lags the evidence**: single epoch, no tranche schedule, no band, baseline frozen without the plan's forecast. The measured book (9ac) is not the one the code can put on.
5. **S3 sample**: 19 fills. Needs the multi-instrument week from Tokyo.
6. **S4 sample**: 0 events. Nothing to do but leave the watcher alive.
7. `README.md`'s "Current state" section still says the repo cannot send an order. It can; it points here now.

## 5. Keeping this file honest

- A roadmap step that changes a row's verdict, what is running, or the next
  step updates the row in the same commit.
- "Live?" in §2 is a measurement (process list + newest file mtime), not a
  belief. Re-check it when touching this file; put the check time at the top.
- Dead strategies stay in the table with a pointer, so the next person does
  not re-run them.
