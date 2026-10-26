"""One place that turns prices into a Monte Carlo forecast.

Both the app (Monte Carlo tab) and the daily logging script (log_forecast.py)
use these functions, so the forecast you see on screen is exactly the
forecast that gets written to the log.
"""

from datetime import datetime, timedelta

import pandas as pd

from src.config import COMPETITION_END, COMPETITION_START
from src.monte_carlo import outcome_ranges, simulate_final_returns, trading_days_left, win_probabilities

# Bump this whenever the model changes (e.g. "bootstrap-v2", "fat-tails-v1"),
# so old and new forecasts in the log can be told apart.
MODEL_VERSION = "bootstrap-v1"
DEFAULT_SIMS = 10_000
# A fixed seed makes the simulation repeatable: same inputs -> same odds.
DEFAULT_SEED = 42


def days_left_for(now: datetime) -> int:
    """Trading days still to play in the contest, as of `now`."""
    if now < COMPETITION_START:
        # Before the start, the whole contest is still ahead.
        return trading_days_left(COMPETITION_START.date() - timedelta(days=1), COMPETITION_END.date())
    return trading_days_left(now.date(), COMPETITION_END.date())


def current_returns(picks: list[dict], snapshot: dict, year_closes: pd.DataFrame) -> pd.Series:
    """Each pick's total return so far (0 if not started), for tickers that have history."""
    return pd.Series(
        {
            p["ticker"]: (snapshot.get(p["ticker"], {}).get("total_return") or 0.0)
            for p in picks
            if p["ticker"] in year_closes.columns
        },
        dtype=float,
    )


def run_forecast(
    current: pd.Series,
    year_closes: pd.DataFrame,
    days_left: int,
    n_sims: int = DEFAULT_SIMS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Win chance and likely-finish range for every ticker.

    Returns one row per ticker with columns:
    return_so_far, win_prob, p05, median, p95
    """
    daily = year_closes[current.index].pct_change().iloc[1:]
    final = simulate_final_returns(current, daily, days_left, n_sims=n_sims, seed=seed)
    ranges = outcome_ranges(final)
    ranges.columns = ["p05", "median", "p95"]
    result = pd.DataFrame({"return_so_far": current, "win_prob": win_probabilities(final)}).join(ranges)
    return result.sort_values("win_prob", ascending=False)
