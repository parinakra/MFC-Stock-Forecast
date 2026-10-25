"""Download US stock prices and dividends from Yahoo Finance using yfinance.

yfinance reads the public Yahoo Finance data that people also see in a
browser. For each ticker we use:
  - start price:   the regular-session Open on Oct 20, 2026
  - current price: Yahoo's latest last price during the contest
  - end price:     the Close on Nov 20, 2026 once the contest is over
  - dividends:     cash dividends with an ex-dividend date after Oct 20
                   and on or before Nov 20

Prices are NOT dividend-adjusted (auto_adjust=False). We add dividends
ourselves so you can see the price part and the dividend part separately.
"""

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from src.config import COMPETITION_END, COMPETITION_START, ET
from src.portfolio import dividend_return, price_return, total_return


def now_et() -> datetime:
    return datetime.now(tz=ET)


def competition_has_started(when: datetime | None = None) -> bool:
    return (when or now_et()) >= COMPETITION_START


def competition_has_ended(when: datetime | None = None) -> bool:
    return (when or now_et()) >= COMPETITION_END


def _history_end_date() -> str:
    """yfinance treats end as exclusive, so we request the day after Nov 20."""
    return (COMPETITION_END.date() + timedelta(days=1)).isoformat()


def fetch_competition_history(tickers: list[str]) -> pd.DataFrame:
    """Daily bars (Open, Close, Dividends, ...) from the start date through the last contest day."""
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=tickers,
        start=COMPETITION_START.date().isoformat(),
        end=_history_end_date(),
        auto_adjust=False,  # raw prices; we add dividends ourselves
        actions=True,       # include the Dividends column
        progress=False,
        threads=True,
        group_by="ticker",
    )
    if data is None:
        return pd.DataFrame()
    return data


def _column(history: pd.DataFrame, ticker: str, field: str) -> pd.Series | None:
    """yfinance uses a MultiIndex when several tickers are downloaded."""
    if history.empty:
        return None

    if isinstance(history.columns, pd.MultiIndex):
        if (ticker, field) in history.columns:
            return history[(ticker, field)].dropna()
        if (field, ticker) in history.columns:
            return history[(field, ticker)].dropna()
        return None

    if field in history.columns:
        return history[field].dropna()
    return None


def start_price_from_history(history: pd.DataFrame, ticker: str) -> float | None:
    """First available Open on or after the contest start date."""
    opens = _column(history, ticker, "Open")
    if opens is None or opens.empty:
        return None
    return float(opens.iloc[0])


def end_close_from_history(history: pd.DataFrame, ticker: str) -> float | None:
    """Last available Close on or before the contest end date."""
    closes = _column(history, ticker, "Close")
    if closes is None or closes.empty:
        return None
    return float(closes.iloc[-1])


def dividends_from_history(history: pd.DataFrame, ticker: str) -> float:
    """Total cash dividends per share with an ex-date AFTER the first contest day.

    Why "after"? If the ex-date were Oct 20 itself, the Oct 20 open price
    already excludes that dividend, so someone buying at the open would not get it.
    """
    dividends = _column(history, ticker, "Dividends")
    if dividends is None or dividends.empty:
        return 0.0
    first_day = dividends.index[0]
    paid = dividends[(dividends.index > first_day) & (dividends > 0)]
    return float(paid.sum())


def total_return_series(history: pd.DataFrame, ticker: str, start_price: float | None) -> pd.Series | None:
    """Total return at each day's close, for the chart.

    Each day: (close - start + dividends paid so far) / start
    """
    closes = _column(history, ticker, "Close")
    if closes is None or closes.empty or start_price is None or start_price <= 0:
        return None

    dividends = _column(history, ticker, "Dividends")
    if dividends is None or dividends.empty:
        paid_so_far = pd.Series(0.0, index=closes.index)
    else:
        dividends = dividends.reindex(closes.index).fillna(0.0)
        dividends.iloc[0] = 0.0  # an ex-date on day one doesn't count (see above)
        paid_so_far = dividends.cumsum()

    return (closes - start_price + paid_so_far) / start_price


def fetch_last_price(ticker: str) -> float | None:
    """Most recent market price Yahoo has (live during regular hours)."""
    try:
        last = yf.Ticker(ticker).fast_info.last_price
        if last is None or pd.isna(last):
            return None
        return float(last)
    except Exception:
        return None


def fallback_start_price(ticker: str) -> float | None:
    """If a ticker did not trade on Oct 20, use the last close before the contest."""
    lookback = (COMPETITION_START.date() - timedelta(days=30)).isoformat()
    try:
        hist = yf.Ticker(ticker).history(
            start=lookback,
            end=COMPETITION_START.date().isoformat(),
            auto_adjust=False,
        )
    except Exception:
        return None
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    closes = hist["Close"].dropna()
    if closes.empty:
        return None
    return float(closes.iloc[-1])


def snapshot_for_tickers(tickers: list[str]) -> dict[str, dict]:
    """Everything the app needs for each ticker: prices, dividends, returns, a daily
    total-return series for the chart, and a "problem" note if Yahoo had no data."""
    ended = competition_has_ended()
    started = competition_has_started()
    # Yahoo has no bars for future dates. Wait until Oct 20 to download history.
    history = fetch_competition_history(tickers) if started else pd.DataFrame()
    snapshot: dict[str, dict] = {}

    for ticker in tickers:
        start = start_price_from_history(history, ticker) if started else None
        if started and start is None:
            start = fallback_start_price(ticker)

        if ended:
            current = end_close_from_history(history, ticker)
        else:
            current = fetch_last_price(ticker)
            if current is None:
                current = end_close_from_history(history, ticker)

        dividends = dividends_from_history(history, ticker) if started else 0.0

        # Ticker check: no start AND no current price almost always means
        # Yahoo doesn't know this symbol (typo, foreign listing, or delisted).
        problem = None
        if current is None:
            problem = "Yahoo Finance has no prices for this ticker. Check the spelling in data/picks.csv."
        elif started and start is None:
            problem = "No start price found for Oct 20. The ticker may not have traded that day."

        snapshot[ticker] = {
            "start_price": start,
            "current_price": current,
            "dividends": dividends,
            "price_return": price_return(start, current),
            "dividend_return": dividend_return(start, dividends) if current is not None else None,
            "total_return": total_return(start, current, dividends),
            "series": total_return_series(history, ticker, start) if started else None,
            "problem": problem,
        }
    return snapshot
