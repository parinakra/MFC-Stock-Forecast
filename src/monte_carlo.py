"""Monte Carlo simulation: who is most likely to win by the last day?

The idea in plain words
-----------------------
We don't know the future, but we know how these stocks moved over the past year.
So we play out the rest of the contest thousands of times:

  1. For each remaining trading day, pick a random REAL day from the past year.
  2. Apply every stock's move from that day (so stocks that tend to move
     together, like NVDA and MU, still move together in the simulation).
  3. At the end, see who finished first.

After 10,000 runs, "Annie won 2,300 of them" means about a 23% chance.
This is called a bootstrap Monte Carlo, because it resamples real history
instead of assuming a bell curve.

Limits (worth saying out loud in a presentation): the past year may not look
like the next few weeks, and news like earnings can move a stock far more
than a typical day.
"""

from datetime import date

import numpy as np
import pandas as pd


def trading_days_left(today: date, end: date) -> int:
    """Weekdays after today up to and including the end date.

    (No US market holidays fall between late October and Nov 20, 2026;
    Thanksgiving is Nov 26, after the contest.)
    """
    if today >= end:
        return 0
    days = pd.bdate_range(start=pd.Timestamp(today) + pd.Timedelta(days=1), end=pd.Timestamp(end))
    return len(days)


def simulate_final_returns(
    current_returns: pd.Series,
    daily_returns: pd.DataFrame,
    days_left: int,
    n_sims: int = 10_000,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Simulated end-of-contest total returns.

    current_returns: each ticker's total return so far (e.g. 0.05 for +5%)
    daily_returns:   past daily returns, one column per ticker, one row per day
    Returns a table with one row per simulation and one column per ticker.
    """
    tickers = [t for t in current_returns.index if t in daily_returns.columns]
    current = current_returns[tickers].to_numpy(dtype=float)

    if days_left <= 0 or not tickers:
        return pd.DataFrame(np.tile(current, (n_sims, 1)), columns=tickers)

    # Rows = past days, columns = tickers. A missing value means the stock
    # didn't trade (e.g. it listed recently); treat that day as "no change".
    history = daily_returns[tickers].fillna(0.0).to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    # For every simulation, pick `days_left` random past days.
    picks = rng.integers(0, len(history), size=(n_sims, days_left))
    # Growth of $1 over those days, for every ticker at once.
    growth = np.prod(1.0 + history[picks], axis=1)  # shape: (n_sims, tickers)

    final = (1.0 + current) * growth - 1.0
    return pd.DataFrame(final, columns=tickers)


def win_probabilities(final_returns: pd.DataFrame) -> pd.Series:
    """Share of simulations each ticker finished first."""
    if final_returns.empty:
        return pd.Series(dtype=float)
    winners = final_returns.idxmax(axis=1)
    counts = winners.value_counts()
    return (counts / len(final_returns)).reindex(final_returns.columns, fill_value=0.0)


def outcome_ranges(final_returns: pd.DataFrame) -> pd.DataFrame:
    """Pessimistic / typical / optimistic finish for each ticker.

    5th percentile:  only 1 in 20 runs did worse
    Median:          the middle outcome
    95th percentile: only 1 in 20 runs did better
    """
    return pd.DataFrame(
        {
            "Bad case (5th pct)": final_returns.quantile(0.05),
            "Typical (median)": final_returns.median(),
            "Good case (95th pct)": final_returns.quantile(0.95),
        }
    )
