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
