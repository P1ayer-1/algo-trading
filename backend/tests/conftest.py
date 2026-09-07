"""Put `backend/` on sys.path so tests can `import trading.*` directly.

Deliberately does NOT add the BloFin SDK to the path: the unit tests must keep
passing without the SDK, the network, or API credentials. If a test ever needs
the SDK, that is a signal it has stopped being a unit test.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
