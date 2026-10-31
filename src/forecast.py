"""One place that turns prices into a Monte Carlo forecast.

Both the app and the daily logging script (log_forecast.py) use these
functions, so the forecast you see on screen is exactly the forecast that
gets written to the log.
"""

from datetime import datetime, timedelta

import pandas as pd

from src.config import BENCHMARK_TICKER, COMPETITION_END, COMPETITION_START
from src.models import MODELS
from src.monte_carlo import outcome_ranges, trading_days_left, win_probabilities

# The "official" model: the one shown on the Monte Carlo tab and quoted in posts.
OFFICIAL_MODEL = "bootstrap"
# Kept for older code that refers to it.
MODEL_VERSION = MODELS[OFFICIAL_MODEL][1]
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
    model: str = OFFICIAL_MODEL,
) -> pd.DataFrame:
    """Win chance and likely-finish range for every ticker, using one model.

    Returns one row per ticker with columns:
    return_so_far, win_prob, p05, median, p95
    """
    daily = year_closes.pct_change().iloc[1:]
    market = daily[BENCHMARK_TICKER] if BENCHMARK_TICKER in daily.columns else None
    simulate = MODELS[model][2]
    final = simulate(current, daily[current.index], days_left, n_sims=n_sims, seed=seed, market=market)
    ranges = outcome_ranges(final)
    ranges.columns = ["p05", "median", "p95"]
    result = pd.DataFrame({"return_so_far": current, "win_prob": win_probabilities(final)}).join(ranges)
    return result.sort_values("win_prob", ascending=False)


def compare_models(
    current: pd.Series,
    year_closes: pd.DataFrame,
    days_left: int,
    n_sims: int = DEFAULT_SIMS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Win chance under every model, side by side: one row per ticker, one column per model."""
    columns = {
        key: run_forecast(current, year_closes, days_left, n_sims, seed, model=key)["win_prob"] for key in MODELS
    }
    return pd.DataFrame(columns)
