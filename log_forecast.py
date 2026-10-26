"""Save today's Monte Carlo forecast to data/forecast_log.csv.

Run it from the project folder, ideally after the market closes (4pm ET / 1pm PT):

    python log_forecast.py            # log today's forecast
    python log_forecast.py --force    # replace today's entry (only if a run was broken)

GitHub Actions also runs this automatically every weekday after the close
(see .github/workflows/log_forecast.yml).
"""

import argparse
import sys

from src.config import BENCHMARK_TICKER
from src.data_store import load_picks
from src.forecast import DEFAULT_SEED, DEFAULT_SIMS, MODEL_VERSION, current_returns, days_left_for, run_forecast
from src.forecast_log import LOG_PATH, append_forecast, build_log_rows, market_status
from src.market_data import fetch_year_of_closes
from src.prices import now_et, snapshot_for_tickers


def main() -> int:
    parser = argparse.ArgumentParser(description="Log today's contest forecast.")
    parser.add_argument("--force", action="store_true", help="replace today's entry if it already exists")
    args = parser.parse_args()

    now = now_et()
    if market_status(now) == "weekend":
        print(f"{now:%A %b %d} is a weekend. Nothing to log.")
        return 0

    picks = load_picks()
    tickers = sorted({p["ticker"] for p in picks} | {BENCHMARK_TICKER})

    print("Downloading prices from Yahoo Finance...")
    snapshot = snapshot_for_tickers(tickers)
    year = fetch_year_of_closes(tickers)
    if year.empty:
        print("ERROR: no price history came back from Yahoo Finance. Try again in a few minutes.")
        return 1

    current = current_returns(picks, snapshot, year)
    missing = [p["ticker"] for p in picks if p["ticker"] not in current.index]
    if missing:
        print(f"Warning: no price history for {', '.join(missing)}. They are left out of this forecast.")

    days_left = days_left_for(now)
    forecast = run_forecast(current, year, days_left, n_sims=DEFAULT_SIMS, seed=DEFAULT_SEED)

    rows = build_log_rows(
        forecast,
        student_of={p["ticker"]: p["student"] for p in picks},
        now=now,
        days_left=days_left,
        model=MODEL_VERSION,
        n_sims=DEFAULT_SIMS,
        seed=DEFAULT_SEED,
        spy_return=snapshot.get(BENCHMARK_TICKER, {}).get("total_return"),
    )
    saved, message = append_forecast(rows, force=args.force)
    print(message)

    print(f"\nTop 5 as of {now:%b %d %H:%M} ET ({days_left} trading days left):")
    for _, row in rows.head(5).iterrows():
        print(f"  {row['student']:<7} {row['ticker']:<5}  win {row['win_prob']:6.1%}   so far {row['return_so_far']:+.2%}")
    if saved:
        print(f"\nLog file: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
