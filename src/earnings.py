"""Earnings dates and past earnings-day moves, for the earnings-jump model.

The idea
--------
On an earnings day a stock can move far more than on a normal day. The other
models don't know when earnings are coming, so they treat that day like any
other. The earnings-jump model looks up:

  1. the NEXT earnings date for each pick (if it falls before the contest ends), and
  2. how the stock moved on its PAST earnings days (up to ~8 quarters),

and on that one simulated day it uses a move the size of one of the stock's
past earnings reactions, up or down with equal chance, instead of an ordinary day.

Which day reacts?
-----------------
Companies report either before the market opens (e.g. 7am) or after it closes
(e.g. 4:05pm). A before-open report moves the stock that same day; an
after-close report moves it the NEXT trading day. Yahoo gives the time, so we
use it. If the time is missing (shown as midnight), we assume before the open.
"""

from datetime import date, datetime

import pandas as pd
import yfinance as yf

MAX_PAST_EVENTS = 8   # about two years of quarterly reports
MIN_PAST_MOVES = 2    # need at least this many past reactions to model a jump


def reaction_day(announced: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """The first trading day whose close reflects the announcement."""
    day = pd.Timestamp(announced.date())
    # 4pm or later = after the close -> next trading day reacts.
    # Earlier (including "midnight" = time unknown) -> that same day reacts.
    after_close = announced.hour >= 16
    candidates = trading_days[trading_days > day] if after_close else trading_days[trading_days >= day]
    return candidates[0] if len(candidates) else None


def _announcements(ticker: str) -> list[pd.Timestamp]:
    """All earnings timestamps Yahoo knows for this ticker (past and upcoming)."""
    stamps: list[pd.Timestamp] = []
    try:
        table = yf.Ticker(ticker).get_earnings_dates(limit=12)
        if table is not None and not table.empty:
            stamps += [pd.Timestamp(t).tz_convert("America/New_York").tz_localize(None) for t in table.index]
    except Exception:
        pass
    if not stamps:
        # Fallback: the calendar endpoint only has the next date, without a time.
        try:
            upcoming = (yf.Ticker(ticker).calendar or {}).get("Earnings Date") or []
            stamps += [pd.Timestamp(d) for d in upcoming]
        except Exception:
            pass
    return sorted(set(stamps))


def fetch_earnings_events(tickers: list[str], today: date, contest_end: date) -> dict:
    """Everything the earnings-jump model needs.

    Returns {"window": [the trading days still to simulate],
             "tickers": {ticker: {"next_reaction": date or None,
                                  "past_reactions": [dates], "moves": [returns]}}}
    """
    window = list(pd.bdate_range(pd.Timestamp(today) + pd.Timedelta(days=1), pd.Timestamp(contest_end)))
    try:
        prices = yf.download(tickers=tickers, period="3y", auto_adjust=True, progress=False, group_by="ticker")
    except Exception:
        prices = pd.DataFrame()

    result = {"window": window, "tickers": {}}
    for ticker in tickers:
        info = {"next_reaction": None, "past_reactions": [], "moves": []}
        closes = None
        if isinstance(prices.columns, pd.MultiIndex) and (ticker, "Close") in prices.columns:
            closes = prices[(ticker, "Close")].dropna()
        if closes is not None:
            closes.index = pd.DatetimeIndex(closes.index).tz_localize(None)
            trading_days = closes.index.append(pd.DatetimeIndex(window)).unique().sort_values()
        else:
            trading_days = pd.DatetimeIndex(window)

        now = pd.Timestamp(datetime.combine(today, datetime.max.time()))
        for stamp in _announcements(ticker):
            day = reaction_day(stamp, trading_days)
            if day is None:
                continue
            if stamp > now or day > pd.Timestamp(today):
                # Upcoming: keep the first one that lands inside the remaining contest.
                if info["next_reaction"] is None and day in window:
                    info["next_reaction"] = day.date()
            elif closes is not None and day in closes.index:
                position = closes.index.get_loc(day)
                if position > 0:
                    info["past_reactions"].append(day.date())
                    info["moves"].append(float(closes.iloc[position] / closes.iloc[position - 1] - 1))

        info["past_reactions"] = info["past_reactions"][-MAX_PAST_EVENTS:]
        info["moves"] = info["moves"][-MAX_PAST_EVENTS:]
        result["tickers"][ticker] = info
    return result
