"""Five ways to simulate the rest of the contest, side by side.

Every model answers the same question: "how might each stock move over the
trading days that are left?" They differ in their ASSUMPTIONS about how
stock returns behave. If they mostly agree, the forecast is robust. If they
disagree, the answer depends on the assumption, and that's worth knowing.
Risk teams call this "model risk".

All models use the same inputs (each pick's return so far + one year of daily
returns) and the same output: one row per simulation, one column per ticker,
holding that ticker's simulated total return on the last day.

  bootstrap        Replay random real days from the past year. (The original model.)
  gbm              Bell curve. Textbook "geometric Brownian motion".
  student_t        Fat tails. Like the bell curve, but extreme days happen more often.
  block_bootstrap  Replay random real WEEKS, so calm and stormy stretches stay together.
  market_factor    Each stock = beta x the market's move + its own separate noise.
"""

import numpy as np
import pandas as pd

from src.monte_carlo import simulate_final_returns

TRADING_DAYS_PER_WEEK = 5
STUDENT_T_DEGREES = 4  # lower = fatter tails; 4 is a common choice for daily stock returns


def fill_gaps(daily_returns: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Fill days before a stock started trading with random moves from its own history.

    (Same rule as the original model, so every model treats new listings like SpaceX the same way.)
    """
    history = daily_returns.to_numpy(dtype=float).copy()
    for column in range(history.shape[1]):
        gaps = np.isnan(history[:, column])
        observed = history[~gaps, column]
        if gaps.any():
            history[gaps, column] = rng.choice(observed, size=gaps.sum()) if observed.size else 0.0
    return history


def _finish(current: np.ndarray, growth: np.ndarray, tickers: list[str]) -> pd.DataFrame:
    """Combine where each pick is now with how much it grows from here."""
    return pd.DataFrame((1.0 + current) * growth - 1.0, columns=tickers)


def _setup(current_returns: pd.Series, daily_returns: pd.DataFrame):
    tickers = [t for t in current_returns.index if t in daily_returns.columns]
    current = current_returns[tickers].to_numpy(dtype=float)
    return tickers, current


# ---------- 1. Historical days (bootstrap) ----------
def bootstrap(current_returns, daily_returns, days_left, n_sims=10_000, seed=42, market=None):
    """Pick random real days from the past year. Every stock gets that day's real move."""
    return simulate_final_returns(current_returns, daily_returns, days_left, n_sims=n_sims, seed=seed)


# ---------- 2. Bell curve (GBM) ----------
def gbm(current_returns, daily_returns, days_left, n_sims=10_000, seed=42, market=None):
    """Assume daily log-returns follow a bell curve (multivariate normal).

    Uses each stock's average and volatility from the past year, plus how
    every pair of stocks moved together (the covariance matrix).
    Because bell curves add up to bell curves, we can jump straight to the end:
    total log-return over N days ~ Normal(N x mean, N x covariance).
    """
    tickers, current = _setup(current_returns, daily_returns)
    if days_left <= 0 or not tickers:
        return _finish(current, np.ones((n_sims, len(tickers))), tickers)
    rng = np.random.default_rng(seed)
    log_r = np.log1p(fill_gaps(daily_returns[tickers], rng))
    mean = log_r.mean(axis=0)
    cov = np.cov(log_r, rowvar=False)
    total = rng.multivariate_normal(days_left * mean, days_left * cov, size=n_sims, method="eigh")
    return _finish(current, np.exp(total), tickers)


# ---------- 3. Fat tails (Student-t) ----------
def student_t(current_returns, daily_returns, days_left, n_sims=10_000, seed=42, market=None):
    """Like the bell curve, but with fatter tails: big up or down days are more common.

    Same average, same volatility, same correlations as the bell curve, so any
    difference in the odds comes purely from how often extreme days happen.
    A Student-t draw = a normal draw divided by a random "calm or stormy" factor.
    """
    tickers, current = _setup(current_returns, daily_returns)
    if days_left <= 0 or not tickers:
        return _finish(current, np.ones((n_sims, len(tickers))), tickers)
    rng = np.random.default_rng(seed)
    df = STUDENT_T_DEGREES
    log_r = np.log1p(fill_gaps(daily_returns[tickers], rng))
    mean = log_r.mean(axis=0)
    cov = np.cov(log_r, rowvar=False)

    normal = rng.multivariate_normal(np.zeros(len(tickers)), cov, size=(n_sims, days_left), method="eigh")
    stormy = np.sqrt(rng.chisquare(df, size=(n_sims, days_left, 1)) / df)
    # Rescale so the volatility matches the bell curve exactly (a t-distribution is wider by sqrt(df/(df-2))).
    daily = mean + normal / stormy * np.sqrt((df - 2) / df)
    return _finish(current, np.exp(daily.sum(axis=1)), tickers)


# ---------- 4. Historical weeks (block bootstrap) ----------
def block_bootstrap(current_returns, daily_returns, days_left, n_sims=10_000, seed=42, market=None):
    """Replay random real 5-day stretches instead of single days.

    Markets have moods: a wild day is often followed by more wild days.
    Picking whole weeks keeps those streaks, which single-day sampling breaks up.
    """
    tickers, current = _setup(current_returns, daily_returns)
    if days_left <= 0 or not tickers:
        return _finish(current, np.ones((n_sims, len(tickers))), tickers)
    rng = np.random.default_rng(seed)
    history = fill_gaps(daily_returns[tickers], rng)
    block = TRADING_DAYS_PER_WEEK
    n_blocks = int(np.ceil(days_left / block))
    starts = rng.integers(0, len(history) - block + 1, size=(n_sims, n_blocks))
    day_index = (starts[:, :, None] + np.arange(block)).reshape(n_sims, -1)[:, :days_left]
    growth = np.prod(1.0 + history[day_index], axis=1)
    return _finish(current, growth, tickers)


# ---------- 5. Market + own noise (factor model) ----------
def market_factor(current_returns, daily_returns, days_left, n_sims=10_000, seed=42, market=None):
    """Each stock's day = its usual drift + beta x the market's day + its own random noise.

    beta: how strongly the stock follows the S&P 500 (from a regression on the past year).
    The market's days are replayed from history; each stock's "own" noise is drawn
    separately. So stocks are linked ONLY through the market. Chip stocks moving
    together for chip-specific reasons is ignored, which tests whether that matters.
    """
    tickers, current = _setup(current_returns, daily_returns)
    if days_left <= 0 or not tickers or market is None:
        return _finish(current, np.ones((n_sims, len(tickers))), tickers)
    rng = np.random.default_rng(seed)
    aligned = pd.concat([daily_returns[tickers], market.rename("__market__")], axis=1).dropna(subset=["__market__"])
    stocks = fill_gaps(aligned[tickers], rng)
    mkt = aligned["__market__"].to_numpy(dtype=float)

    # Regression of each stock on the market: stock = alpha + beta x market + residual.
    mkt_centered = mkt - mkt.mean()
    beta = (mkt_centered @ (stocks - stocks.mean(axis=0))) / (mkt_centered @ mkt_centered)
    alpha = stocks.mean(axis=0) - beta * mkt.mean()
    residuals = stocks - alpha - np.outer(mkt, beta)

    n_days = len(mkt)
    market_days = mkt[rng.integers(0, n_days, size=(n_sims, days_left))]
    # Each stock draws its own noise from its own residual history, independently.
    own_noise = np.stack(
        [residuals[rng.integers(0, n_days, size=(n_sims, days_left)), j] for j in range(len(tickers))], axis=2
    )
    daily = alpha + market_days[:, :, None] * beta + own_noise
    growth = np.prod(1.0 + daily, axis=1)
    return _finish(current, growth, tickers)


# Display name, short code for the log, function, plain-English one-liner.
MODELS = {
    "bootstrap": ("Historical days", "bootstrap-v1", bootstrap,
                  "Replays random real days from the past year (the original model)."),
    "gbm": ("Bell curve", "gbm-v1", gbm,
            "Assumes returns follow a normal distribution. The textbook model."),
    "student_t": ("Fat tails", "student-t-v1", student_t,
                  "Same as the bell curve, but extreme days happen more often."),
    "block_bootstrap": ("Historical weeks", "block-bootstrap-v1", block_bootstrap,
                        "Replays random real 5-day stretches, keeping calm and stormy streaks."),
    "market_factor": ("Market + own noise", "market-factor-v1", market_factor,
                      "Stocks are linked only through the S&P 500; everything else is independent."),
}
