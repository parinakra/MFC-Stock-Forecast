"""Extra Yahoo Finance data for Version 2: quotes, valuation, and a year of history.

Only fields Yahoo Finance actually provides are used. Some fields are often
missing, and the app shows "—" for them instead of guessing:
  - P/E is blank for companies with no profits (you can't divide by negative earnings).
  - Bid and ask are often 0 outside market hours (9:30am–4pm ET), so the spread
    is only meaningful while the market is open.
"""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf

# Yahoo field name -> column name shown in the app.
QUOTE_FIELDS = {
    "currentPrice": "Price",
    "bid": "Bid",
    "ask": "Ask",
    "bidSize": "Bid size",
    "askSize": "Ask size",
    "trailingPE": "P/E (trailing)",
    "forwardPE": "P/E (forward)",
    "marketCap": "Market cap",
    "beta": "Beta",
    "fiftyTwoWeekLow": "52-wk low",
    "fiftyTwoWeekHigh": "52-wk high",
    "sector": "Sector",
}


def _clean(value):
    """Turn Yahoo's 'nothing here' values (None, 0 for prices, NaN) into None."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def bid_ask_spread(bid: float | None, ask: float | None) -> tuple[float | None, float | None]:
    """Spread in dollars and as a percent of the midpoint.

    spread     = ask - bid
    midpoint   = (bid + ask) / 2
    spread %   = spread / midpoint

    Example: bid 99.90, ask 100.10 -> spread 0.20, mid 100.00, spread % 0.20%
    Returns (None, None) when the quote is missing or doesn't make sense
    (Yahoo shows 0 bid/ask when the market is closed).
    """
    if not bid or not ask or bid <= 0 or ask <= 0 or ask < bid:
        return None, None
    spread = ask - bid
    midpoint = (bid + ask) / 2
    return spread, spread / midpoint


def fetch_quote(ticker: str) -> dict:
    """One row of quote + valuation data for a ticker."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    row = {"Ticker": ticker}
    for yahoo_name, column in QUOTE_FIELDS.items():
        row[column] = _clean(info.get(yahoo_name))

    # Yahoo uses 0 to mean "no quote right now" for bid/ask.
    for column in ("Bid", "Ask", "Bid size", "Ask size"):
        if row[column] == 0:
            row[column] = None

    spread, spread_pct = bid_ask_spread(row["Bid"], row["Ask"])
    row["Spread ($)"] = spread
    row["Spread (%)"] = spread_pct
    return row


def fetch_quotes(tickers: list[str]) -> pd.DataFrame:
    """Quotes for many tickers at once (in parallel, because each is a separate web request)."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(fetch_quote, tickers))
    return pd.DataFrame(rows)


def fetch_year_of_closes(tickers: list[str]) -> pd.DataFrame:
    """One year of daily closing prices: one column per ticker, one row per trading day.

    auto_adjust=True folds dividends and splits into the price, which is what
    you want for measuring risk and running the Monte Carlo simulation.
    """
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers=tickers,
        period="1y",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    if data is None or data.empty:
        return pd.DataFrame()

    closes = {}
    for ticker in tickers:
        if isinstance(data.columns, pd.MultiIndex):
            if (ticker, "Close") in data.columns:
                closes[ticker] = data[(ticker, "Close")]
        elif "Close" in data.columns and len(tickers) == 1:
            closes[ticker] = data["Close"]
    frame = pd.DataFrame(closes)
    # Drop tickers Yahoo returned nothing for.
    return frame.dropna(axis=1, how="all")
