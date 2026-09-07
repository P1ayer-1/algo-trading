"""All configuration for the live chart backend in one place.

Everything here is read from environment variables (with sane defaults),
which are themselves loaded from the repo-root `.env` file. If you're trying
to change how the bot behaves (which instrument, which timeframe, how wide
the support/resistance search range is, which ports it runs on), this is the
file to look at first.
"""

import os
import sys
from decimal import Decimal
from pathlib import Path

# This file lives in backend/, one level below the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"

# Make the vendored BloFin SDK importable without installing it as a package.
SDK_SRC = REPO_ROOT / "blofin-sdk-python" / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))


def load_local_env(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file, if it exists.

    Existing environment variables always win (setdefault), so real env vars
    set by your shell/OS take priority over the .env file.
    """
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env(REPO_ROOT / ".env")


def env_bool(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


# --- Instrument / candle settings -----------------------------------------
INST_ID = os.getenv("BLOFIN_INST_ID", "BTC-USDT")
BAR = os.getenv("BLOFIN_SUPPORT_BAR", "15m")
CANDLE_LIMIT = os.getenv("BLOFIN_SUPPORT_CANDLE_LIMIT", "500")

# --- Support/resistance detection settings ---------------------------------
LOOKBACK = int(os.getenv("BLOFIN_SUPPORT_LOOKBACK", "3"))
CLUSTER_PCT = Decimal(os.getenv("BLOFIN_SUPPORT_CLUSTER_PCT", "0.001"))
SUPPORT_DEFAULT_BELOW = Decimal(os.getenv("BLOFIN_SUPPORT_DEFAULT_BELOW", "5000"))
RESISTANCE_DEFAULT_ABOVE = Decimal(os.getenv("BLOFIN_RESISTANCE_DEFAULT_ABOVE", "5000"))
SUPPORT_LEVEL_COUNT = int(os.getenv("BLOFIN_SUPPORT_LEVEL_COUNT", "8"))
RESISTANCE_LEVEL_COUNT = int(os.getenv("BLOFIN_RESISTANCE_LEVEL_COUNT", str(SUPPORT_LEVEL_COUNT)))
RECALCULATE_SECONDS = int(os.getenv("BLOFIN_LEVEL_RECALCULATE_SECONDS", "30"))

# --- Polling / networking settings -----------------------------------------
TICKER_POLL_SECONDS = float(os.getenv("BLOFIN_TICKER_POLL_SECONDS", "1"))
HTTP_PORT = int(os.getenv("BLOFIN_CHART_HTTP_PORT", "8765"))
WS_PORT = int(os.getenv("BLOFIN_CHART_WS_PORT", "8766"))
HOST = os.getenv("BLOFIN_CHART_HOST", "127.0.0.1")
USE_DEMO = env_bool("BLOFIN_USE_DEMO", "false")

# --- Microstructure feed (backend/trading) ---------------------------------
# The order book + trade tape feed that produces the OBI/OFI/microprice
# features. Runs as its own websocket connection, independent of the chart's
# ticker/candle stream, so one can fail without taking the other down.
MICRO_ENABLED = env_bool("BLOFIN_MICRO_ENABLED", "true")

# "books" = 200 levels with incremental updates (what you want for real
# depth features). "books5" = 5 levels, full snapshot each time — lighter,
# and it cannot desync, but obi_20 becomes meaningless.
BOOK_DEPTH = os.getenv("BLOFIN_BOOK_DEPTH", "books")

# Rolling trade-tape window, seconds. Must exceed the longest tfi_* horizon.
TAPE_WINDOW_SECONDS = float(os.getenv("BLOFIN_TAPE_WINDOW_SECONDS", "60"))

# --- Fees and execution cost ------------------------------------------------
# THE most load-bearing numbers in this project. Every "is there an edge?"
# question is really "is the predicted move bigger than these?", so they get
# defined once, here, and everything else imports them.
#
# BloFin futures schedule, VIP 1 (confirmed 2026-09-07). Fractions of
# notional, per side.
MAKER_FEE_RATE = Decimal(os.getenv("BLOFIN_MAKER_FEE_RATE", "0.00006"))  # 0.0060%
TAKER_FEE_RATE = Decimal(os.getenv("BLOFIN_TAKER_FEE_RATE", "0.00050"))  # 0.0500%


def _bps(rate: Decimal) -> Decimal:
    return rate * Decimal("10000")


# Round-trip cost in bps for the three ways a position can open and close.
# The spread is not included: measured median spread on BTC-USDT is 0.013bps,
# three orders of magnitude below the taker fee, so fees are the whole story.
MAKER_FEE_BPS = _bps(MAKER_FEE_RATE)                    # 0.6
TAKER_FEE_BPS = _bps(TAKER_FEE_RATE)                    # 5.0
COST_MAKER_MAKER_BPS = MAKER_FEE_BPS * 2                # 1.2  passive in, passive out
COST_MAKER_TAKER_BPS = MAKER_FEE_BPS + TAKER_FEE_BPS    # 5.6  passive in, market out
COST_TAKER_TAKER_BPS = TAKER_FEE_BPS * 2                # 10.0 crossing both ways

# The cost the risk engine's edge gate and the analysis tooling assume by
# default. Taker/taker deliberately: it is a *veto* threshold, and the
# conservative assumption is that you cross the spread on both sides. Set
# BLOFIN_ROUND_TRIP_COST_BPS to COST_MAKER_MAKER_BPS (1.2) to see what a
# fully passive execution engine would need to clear -- but only once such an
# engine exists and its fill rate has been measured, not before.
ROUND_TRIP_COST_BPS = Decimal(
    os.getenv("BLOFIN_ROUND_TRIP_COST_BPS", str(COST_TAKER_TAKER_BPS))
)

# --- Feature recording ------------------------------------------------------
# Writes labelled feature rows to DATA_DIR for later model training. This is
# the prerequisite for any ML in the roadmap: no recording, no dataset.
RECORD_FEATURES = env_bool("BLOFIN_RECORD_FEATURES", "true")
DATA_DIR = Path(os.getenv("BLOFIN_DATA_DIR", str(REPO_ROOT / "data")))

# Archive the raw websocket messages to data/raw/ as gzipped JSONL.
# Strongly recommended: the feature CSV only contains features you thought of
# today, whereas the raw log lets any FUTURE feature be recomputed over all
# your history via analysis/replay.py. Costs roughly 10-40 MB/hour compressed.
RECORD_RAW = env_bool("BLOFIN_RECORD_RAW", "true")

# How often to persist a feature row (ms). The feature engine still computes
# on every event; this only controls disk volume. At a 300s minimum horizon,
# 250ms sampling produced 1,200 near-identical overlapping rows per
# independent observation — disk volume without information. 1s still leaves
# 300 rows per window, and anything finer is recoverable from data/raw/ via
# analysis/replay.py anyway.
FEATURE_SAMPLE_INTERVAL_MS = int(os.getenv("BLOFIN_FEATURE_SAMPLE_MS", "1000"))

# Forward horizons (seconds) to label, and the move size that counts as a
# signal.
#
# These were 1/5/30 SECONDS, and that was the reason the bot could not work.
# Measured on BTC-USDT, the standard deviation of the forward move is 0.45bps
# at 1s, 0.97bps at 5s and 2.41bps at 30s, against a round-trip cost of
# 1.2bps (maker) to 10bps (taker). At 30 seconds an oracle with perfect
# knowledge of the sign, capturing a full standard deviation, still loses
# money at taker fees. No model fixes that; only a longer horizon does.
#
# Price scales as a near-perfect random walk here (measured exponent 0.495),
# so sigma(T) ~= 2.41bps * sqrt(T/30):
#
#     300s  (5 min)   ~7.6bps      900s (15 min)  ~13.2bps
#    1800s (30 min)  ~18.7bps     3600s (60 min)  ~26.4bps
#
# A model that captures ~0.2 sigma (which is what the 5s and 30s runs
# actually achieved) needs sigma >= 6bps to clear maker fees and >= 50bps to
# clear taker fees. 300/900/1800 brackets that range.
#
# Consequence to know about: the recorder cannot write a row until its
# forward window has closed, so nothing lands on disk for the first
# max(horizon) = 30 minutes. That is the lookahead guard, not a hang.
LABEL_HORIZONS = tuple(
    float(part)
    for part in os.getenv("BLOFIN_LABEL_HORIZONS", "300,900,1800").split(",")
    if part.strip()
)
# Defaults to the round-trip cost: labelling a move smaller than fees as "up"
# trains the model to chase edges it cannot capture.
LABEL_THRESHOLD_BPS = float(
    os.getenv("BLOFIN_LABEL_THRESHOLD_BPS", str(ROUND_TRIP_COST_BPS))
)

# --- Risk limits ------------------------------------------------------------
# Nothing sends orders yet, but these are the values the risk engine will
# enforce when execution is built. Defaults are deliberately conservative:
# they are sized for a demo account, not a tuned production system.
MAX_POSITION_BASE = Decimal(os.getenv("BLOFIN_MAX_POSITION_BASE", "0.05"))
MAX_NOTIONAL = Decimal(os.getenv("BLOFIN_MAX_NOTIONAL", "5000"))
MAX_LEVERAGE = Decimal(os.getenv("BLOFIN_MAX_LEVERAGE", "5"))
MAX_ORDER_BASE = Decimal(os.getenv("BLOFIN_MAX_ORDER_BASE", "0.01"))
MAX_OPEN_ORDERS = int(os.getenv("BLOFIN_MAX_OPEN_ORDERS", "4"))
MAX_DAILY_LOSS = Decimal(os.getenv("BLOFIN_MAX_DAILY_LOSS", "100"))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("BLOFIN_MAX_CONSECUTIVE_LOSSES", "4"))
MAX_SPREAD_BPS = Decimal(os.getenv("BLOFIN_MAX_SPREAD_BPS", "5"))
MAX_SLIPPAGE_BPS = Decimal(os.getenv("BLOFIN_MAX_SLIPPAGE_BPS", "5"))

# The liquidation guard. 0.15 = refuse any position whose estimated
# liquidation price is less than a 15% adverse move away. At 20x leverage the
# natural buffer is ~5%, so this limit will (correctly) block it.
MIN_LIQUIDATION_BUFFER_PCT = Decimal(
    os.getenv("BLOFIN_MIN_LIQ_BUFFER_PCT", "0.15")
)
# Maintenance margin rate. VERIFY THIS against BloFin's tier table for the
# instrument and size you intend to trade — it is tiered, not flat, and the
# wrong value makes every liquidation estimate optimistic.
MAINTENANCE_MARGIN_RATE = Decimal(os.getenv("BLOFIN_MMR", "0.005"))

# Unrealized-loss limits. These exist because every other circuit breaker in
# risk.py keys on *realized* PnL, which means a strategy that simply refuses
# to close a loser ("it's an uptrend, we can wait") never trips any of them.
# An open position is a loss whether or not it has been booked.
MAX_UNREALIZED_LOSS = Decimal(os.getenv("BLOFIN_MAX_UNREALIZED_LOSS", "50"))
# Trip if an already-open position drifts this close to liquidation.
# Deliberately looser than MIN_LIQUIDATION_BUFFER_PCT (which gates *opening*):
# a position that has moved against you should be closed well before the
# margin engine does it for you.
MIN_OPEN_LIQ_BUFFER_PCT = Decimal(os.getenv("BLOFIN_MIN_OPEN_LIQ_BUFFER_PCT", "0.08"))
