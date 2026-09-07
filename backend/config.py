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
# on every event; this only controls disk volume. 250ms ~= 350k rows/day.
FEATURE_SAMPLE_INTERVAL_MS = int(os.getenv("BLOFIN_FEATURE_SAMPLE_MS", "250"))

# Forward horizons (seconds) to label, and the move size that counts as a
# signal. Set the threshold from your real round-trip cost — labelling moves
# smaller than fees trains a model to chase edges it cannot capture.
LABEL_HORIZONS = tuple(
    float(part)
    for part in os.getenv("BLOFIN_LABEL_HORIZONS", "1,5,30").split(",")
    if part.strip()
)
LABEL_THRESHOLD_BPS = float(os.getenv("BLOFIN_LABEL_THRESHOLD_BPS", "3"))

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
ROUND_TRIP_COST_BPS = Decimal(os.getenv("BLOFIN_ROUND_TRIP_COST_BPS", "6"))
