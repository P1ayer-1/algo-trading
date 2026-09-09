# backend/trading — microstructure, features, risk

What exists, what it does, and — just as importantly — what it does **not** do.

## Status: no orders are sent

Nothing in this package can place, modify, or cancel an order. There is no
exchange write path at all. The risk engine is built and tested, but it is
currently guarding a door that does not yet open. That is intentional: the
data foundation has to exist before there is anything worth executing on.

## Layers

| File | Does | Depends on |
|---|---|---|
| `orderbook.py` | Maintains an L2 book from BloFin's `books` channel, with sequence-gap detection | nothing |
| `tape.py` | Rolling window of aggressive trades | nothing |
| `features.py` | OBI, OFI, microprice, spread, returns, realised volatility | orderbook, tape |
| `recorder.py` | Writes feature rows + forward labels to CSV | features |
| `ingest.py` | The live websocket loop wiring the above together, with a stall watchdog | all of the above, BloFin SDK |
| `risk.py` | Liquidation math, position sizing, hard limits | **nothing** |

`risk.py` importing nothing from this package is a deliberate constraint: the
risk engine must never become dependent on the thing it is supposed to veto.
`ingest.py` is imported lazily by `__init__.py`, so `risk` and `features` stay
importable and testable without the SDK, network, or credentials.

## Two number types, on purpose

`float` in the book and features — they are ratios and differences on a hot
path running ~50 computations/second, and Decimal there is ~50x slower for no
benefit. `Decimal` in `risk.py` — those are money values compared against an
exchange's own margin engine, where float error is not acceptable.

## Data recording

With `BLOFIN_RECORD_FEATURES=true` (default), running the bot writes one CSV
per UTC day to `data/`:

```
data/features-2026-09-06.csv
```

Each row is a feature vector plus forward-looking labels at 300s / 900s /
1800s: `fwd_ret_bps_900s` (the realised move in bps) and `label_900s` (-1/0/+1
against `BLOFIN_LABEL_THRESHOLD_BPS`).

Those horizons are minutes rather than seconds because a 30-second BTC move
has a standard deviation of 2.41bps and a round trip costs 1.2-10bps. See
`backend/config.py` for the measurement and the sqrt(T) scaling that follows
from it. The practical consequence: nothing reaches disk for the first 30
minutes of a run.

**Rows are only written after their forward window has actually elapsed.** A
row inside its horizon sits in a pending buffer and is never written, so the
file cannot contain a label built from a price the features could not have
seen. This is the one thing in the recorder that must not break — a lookahead
bug produces a model with a wonderful backtest and no live edge.

Two things to know before training on it:

- **Filter on `history_seconds`.** During warmup, returns over a *feature*
  window longer than the available history are reported as `0.0`. That is
  padding, not a measured zero. Drop rows where `history_seconds` is below the
  slowest feature window (`rv_60s`, so 60 seconds). Note this is the backward
  feature lookback, **not** the forward label horizon — `check_features.py`
  used to conflate the two and silently discarded every row at any horizon
  beyond FeatureEngine's 300s of retained history.
- **Set `threshold_bps` from your real costs.** It now defaults to
  `config.ROUND_TRIP_COST_BPS` (10bps, taker both sides at VIP 1). A model
  trained to predict 3bps moves against a 10bps round trip is being trained to
  lose money.

## Liquidation math — verify before trusting

`liquidation_price()` implements the standard flat-MMR linear-perp formula
(derivation is in the docstring). Real exchanges differ: MMR is **tiered** by
size, closing fees and funding are debited from equity, and cross margin
depends on the whole account.

Before sizing anything with leverage, open a small position on the BloFin demo
account, read back the exchange's own reported liquidation price, and run:

```python
from trading.risk import compare_to_exchange, liquidation_price, Side
from decimal import Decimal as D

estimate = liquidation_price(
    entry_price=D("100000"), leverage=D("10"), side=Side.LONG,
    maintenance_margin_rate=D("0.005"),   # <- the tier for YOUR size
)
print(compare_to_exchange(estimate=estimate, exchange_reported=D("...")))
```

If the relative error is above ~1%, the MMR tier or fee assumptions are wrong.
Fix them before increasing leverage, not after.

## Tests

```
cd backend
python -m pytest
```

113 tests. The ones that matter most:

- `test_orderbook.py` — a sequence gap must mark the book **stale**, and the
  bad update must not be applied.
- `test_features.py` — the OFI recursion, case by case. A sign error there
  would invert the direction of every trade.
- `test_risk.py` — liquidation prices checked against hand-computed values,
  not against whatever the code currently returns.
- `test_ingest.py` — desync recovery, and the stall watchdog: silence past
  `stall_timeout_s` must end the stream, a trickle under it must not, and the
  two must be distinguishable from a desync (skipped if the SDK isn't
  installed).

## A silent feed is worse than a crashed one

`_stream()` awaits each message with a deadline rather than looping over
`client.listen()` directly. The failure it exists for raises nothing: the
socket stays open, pings are answered, and the subscription simply stops
delivering. `supervise` restarts loops that fail, and a loop waiting on a
silent socket has not failed — so without this, the recorder writes nothing
until a human notices, and the raw archive is the irreplaceable half.

The deadline is per message, not per stream. A deadline on the whole stream
would reconnect a healthy slow feed on a fixed cycle, discarding the book
snapshot every time.

## Not built yet

The prediction layer (LightGBM), the regime model (HMM), the signal combiner,
and the execution engine. All four need the dataset `recorder.py` produces, so
the honest ordering is: record data first, establish that the features contain
predictive information, then build a model — not the other way round.
