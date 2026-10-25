"""Risk and performance statistics from daily closing prices.

No internet here - just math on a column of prices, so it's easy to test.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def daily_returns(closes: pd.Series) -> pd.Series:
    """Percent change from one day's close to the next.

    Example: 100 -> 102 is +2% (0.02).
    """
    return closes.dropna().pct_change().dropna()


def annualized_volatility(closes: pd.Series) -> float | None:
    """How much the stock typically swings, scaled to a year.

    volatility = standard deviation of daily returns × sqrt(252)

    Why sqrt(252)? Day-to-day randomness adds up with the square root of time.
    A stock at 30% means a typical year lands within about ±30% of its average.
    """
    returns = daily_returns(closes)
    if len(returns) < 2:
        return None
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(closes: pd.Series) -> float | None:
    """The worst fall from a previous high, as a negative percent.

    Example: 100 -> 120 -> 90 gives (90 - 120) / 120 = -25%.
    """
    closes = closes.dropna()
    if closes.empty:
        return None
    running_high = closes.cummax()
    drawdowns = closes / running_high - 1
    return float(drawdowns.min())


def period_return(closes: pd.Series) -> float | None:
    """Return from the first to the last price in the series."""
    closes = closes.dropna()
    if len(closes) < 2 or closes.iloc[0] <= 0:
        return None
    return float(closes.iloc[-1] / closes.iloc[0] - 1)


def risk_table(closes: pd.DataFrame) -> pd.DataFrame:
    """One row of stats per ticker."""
    rows = []
    for ticker in closes.columns:
        series = closes[ticker]
        returns = daily_returns(series)
        rows.append(
            {
                "Ticker": ticker,
                "1-year return": period_return(series),
                "Volatility (yearly)": annualized_volatility(series),
                "Max drawdown": max_drawdown(series),
                "Best day": float(returns.max()) if not returns.empty else None,
                "Worst day": float(returns.min()) if not returns.empty else None,
            }
        )
    return pd.DataFrame(rows)
