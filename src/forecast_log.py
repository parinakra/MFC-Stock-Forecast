"""The forecast log: a running record of what the model predicted, and when.

Why keep a log?
  A forecast is only useful if you can check it later. On Nov 20 we'll know
  who actually won. With a daily log we can ask: did the model favor the
  eventual winner? How early? Were 20% predictions right about 20% of the time?
  Without a log, we can only say what the model thinks *now*, and it's too
  late to recreate what it thought on Oct 27.

Rules that keep the log honest:
  1. One forecast per model per trading day.
  2. Never edit old rows. If a day was logged, it stays as it was, even if
     the model later turns out to be wrong. (--force exists only for fixing a
     run that was clearly broken, and it replaces that one day.)
  3. Record everything needed to reproduce the forecast: model version,
     number of simulations, random seed, and when it was made.

File format: data/forecast_log.csv, one row per student per day.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR

LOG_PATH = DATA_DIR / "forecast_log.csv"

COLUMNS = [
    "forecast_date",   # the trading day this forecast is for (YYYY-MM-DD)
    "logged_at_et",    # exact time it was made, New York time
    "market_status",   # "after close" (official) or "intraday"
    "model",           # model version, e.g. bootstrap-v1
    "n_sims",          # number of simulations
    "seed",            # random seed (makes it reproducible)
    "days_left",       # trading days left in the contest
    "student",
    "ticker",
    "return_so_far",   # total return at the time of the forecast
    "win_prob",        # chance of finishing first
    "p05",             # bad-case finish (5th percentile)
    "median",          # typical finish
    "p95",             # good-case finish (95th percentile)
    "spy_return_so_far",  # S&P 500 benchmark at the same moment
    "earnings_day",       # the pick's next earnings reaction day inside the contest, if known
]


def market_status(now: datetime) -> str:
    """'after close' once the regular session (9:30am–4pm ET) has ended."""
    if now.weekday() >= 5:
        return "weekend"
    if now.hour >= 16:
        return "after close"
    if (now.hour, now.minute) >= (9, 30):
        return "intraday"
    return "before open"


def build_log_rows(
    forecast: pd.DataFrame,
    student_of: dict[str, str],
    now: datetime,
    days_left: int,
    model: str,
    n_sims: int,
    seed: int,
    spy_return: float | None,
    earnings_days: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Turn a forecast table (one row per ticker) into log rows."""
    rows = forecast.reset_index().rename(columns={"index": "ticker"})
    rows.insert(0, "forecast_date", now.date().isoformat())
    rows.insert(1, "logged_at_et", now.strftime("%Y-%m-%d %H:%M"))
    rows.insert(2, "market_status", market_status(now))
    rows.insert(3, "model", model)
    rows.insert(4, "n_sims", n_sims)
    rows.insert(5, "seed", seed)
    rows.insert(6, "days_left", days_left)
    rows["student"] = rows["ticker"].map(student_of)
    rows["spy_return_so_far"] = spy_return
    rows["earnings_day"] = rows["ticker"].map(earnings_days or {})
    for column in ["return_so_far", "win_prob", "p05", "median", "p95", "spy_return_so_far"]:
        rows[column] = pd.to_numeric(rows[column]).round(6)
    return rows[COLUMNS]


def load_log(path: Path = LOG_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"forecast_date": str})


def append_forecast(rows: pd.DataFrame, path: Path = LOG_PATH, force: bool = False) -> tuple[bool, str]:
    """Add one day's forecast to the log. Returns (saved?, message)."""
    if rows.empty:
        return False, "Nothing to log: the forecast is empty."

    day = rows["forecast_date"].iloc[0]
    model = rows["model"].iloc[0]
    status = rows["market_status"].iloc[0]
    if status == "weekend":
        return False, f"{day} is a weekend. Markets are closed, so there's nothing new to forecast."

    log = load_log(path)
    same = (log["forecast_date"] == day) & (log["model"] == model)
    if same.any():
        if not force:
            return False, f"A {model} forecast for {day} is already in the log. Old forecasts are never overwritten."
        log = log[~same]

    combined = pd.concat([log, rows], ignore_index=True) if not log.empty else rows
    combined = combined.sort_values(["forecast_date", "model", "win_prob"], ascending=[True, True, False])
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)

    note = " (intraday: prices will still move before the close)" if status != "after close" else ""
    return True, f"Saved the {model} forecast for {day}{note}."
