"""Checks for the forecast log rules (no internet needed).

    python tests/test_forecast_log.py
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import ET
from src.forecast import run_forecast
from src.forecast_log import append_forecast, build_log_rows, load_log, market_status


def check(name: str, condition: bool) -> None:
    if not condition:
        raise SystemExit(f"FAIL {name}")
    print(f"OK  {name}")


def sample_rows(when: datetime) -> pd.DataFrame:
    forecast = pd.DataFrame(
        {"return_so_far": [0.05, 0.01], "win_prob": [0.7, 0.3], "p05": [-0.1, -0.2], "median": [0.05, 0.0], "p95": [0.2, 0.3]},
        index=["AAA", "BBB"],
    )
    return build_log_rows(forecast, {"AAA": "A.A.", "BBB": "B.B."}, when, 10, "test-v1", 100, 42, 0.01)


def main() -> None:
    monday_close = datetime(2026, 10, 26, 16, 30, tzinfo=ET)
    tuesday_close = datetime(2026, 10, 27, 16, 30, tzinfo=ET)

    check("4:30pm counts as after close", market_status(monday_close) == "after close")
    check("11am counts as intraday", market_status(datetime(2026, 10, 26, 11, 0, tzinfo=ET)) == "intraday")
    check("Saturday is a weekend", market_status(datetime(2026, 10, 24, 17, 0, tzinfo=ET)) == "weekend")

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "log.csv"

        saved, _ = append_forecast(sample_rows(monday_close), path)
        check("first forecast is saved", saved)

        saved, _ = append_forecast(sample_rows(monday_close), path)
        check("same day is NOT saved twice", not saved)
        check("log still has one day", load_log(path)["forecast_date"].nunique() == 1)

        saved, _ = append_forecast(sample_rows(tuesday_close), path)
        check("next day is added", saved and load_log(path)["forecast_date"].nunique() == 2)

        other_model = sample_rows(tuesday_close)
        other_model["model"] = "other-v1"
        saved, _ = append_forecast(other_model, path)
        check("a different model on the same day IS added", saved and len(load_log(path)) == 6)

        saved, _ = append_forecast(sample_rows(datetime(2026, 10, 24, 17, 0, tzinfo=ET)), path)
        check("weekend forecast is refused", not saved)

        saved, _ = append_forecast(sample_rows(monday_close), path, force=True)
        log = load_log(path)
        check("--force replaces just that day and model", saved and len(log) == 6 and log["forecast_date"].nunique() == 2)

    # Same inputs + same seed = same odds (reproducible).
    history = pd.DataFrame({"AAA": [100 * 1.01**i for i in range(60)], "BBB": [100 + (i % 3) for i in range(60)]})
    current = pd.Series({"AAA": 0.0, "BBB": 0.0})
    first = run_forecast(current, history, days_left=5, n_sims=500, seed=7)
    second = run_forecast(current, history, days_left=5, n_sims=500, seed=7)
    check("same seed gives identical forecast", first.equals(second))

    print("\nAll forecast-log checks passed.")


if __name__ == "__main__":
    main()
