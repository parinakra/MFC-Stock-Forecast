"""Competition rules and file locations.

Keeping dates and paths in one file means you only change them here
if the club ever runs a new 5-week contest.
"""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# New York time, because US stock exchanges use Eastern Time.
ET = ZoneInfo("America/New_York")

# Market open on the first day, market close on the last day.
# Oct 20, 2026 is a Tuesday; Nov 20, 2026 is a Friday.
COMPETITION_START = datetime(2026, 10, 20, 9, 30, tzinfo=ET)
COMPETITION_END = datetime(2026, 11, 20, 16, 0, tzinfo=ET)

# The "market" everyone is compared against. SPY is a fund that tracks the S&P 500.
BENCHMARK_TICKER = "SPY"
BENCHMARK_NAME = "S&P 500 (SPY)"

# Folder that contains this file: .../MFC Stock Monitor/src
SRC_DIR = Path(__file__).resolve().parent
# One level up is the project folder itself.
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PICKS_CSV = DATA_DIR / "picks.csv"

# How often Streamlit should refresh Yahoo prices (seconds).
PRICE_CACHE_SECONDS = 120
