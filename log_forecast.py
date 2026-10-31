"""Save today's Monte Carlo forecasts to data/forecast_log.csv.

Every model (see src/models.py) is logged each trading day, so on Nov 20 we
can score them against each other on the same days.

Run it from the project folder, ideally after the market closes (4pm ET / 1pm PT):

    python log_forecast.py            # log today's forecasts
    python log_forecast.py --force    # replace today's entries (only if a run was broken)

GitHub Actions also runs this automatically every weekday after the close
(see .github/workflows/log_forecast.yml).
"""

import argparse
import sys

from src.config import BENCHMARK_TICKER
from src.data_store import load_picks
from src.forecast import DEFAULT_SEED, DEFAULT_SIMS, OFFICIAL_MODEL, current_returns, days_left_for, run_forecast
from src.forecast_log import LOG_PATH, append_forecast, build_log_rows, market_status
from src.market_data import fetch_year_of_closes
from src.models import MODELS
from src.prices import now_et, snapshot_for_tickers


def main() -> int:
    parser = argparse.ArgumentParser(description="Log today's contest forecasts.")
    parser.add_argument("--force", action="store_true", help="replace today's entries if they already exist")
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
    student_of = {p["ticker"]: p["student"] for p in picks}
    spy_return = snapshot.get(BENCHMARK_TICKER, {}).get("total_return")

    for key, (name, version, _, _) in MODELS.items():
        forecast = run_forecast(current, year, days_left, n_sims=DEFAULT_SIMS, seed=DEFAULT_SEED, model=key)
        rows = build_log_rows(
            forecast,
            student_of=student_of,
            now=now,
            days_left=days_left,
            model=version,
            n_sims=DEFAULT_SIMS,
            seed=DEFAULT_SEED,
            spy_return=spy_return,
        )
        _, message = append_forecast(rows, force=args.force)
        top = rows.iloc[0]
        print(f"{name:<20} favorite {top['student']:<6} {top['ticker']:<5} {top['win_prob']:6.1%}   {message}")

        if key == OFFICIAL_MODEL:
            official = rows

    print(f"\nOfficial model, top 5 as of {now:%b %d %H:%M} ET ({days_left} trading days left):")
    for _, row in official.head(5).iterrows():
        print(f"  {row['student']:<7} {row['ticker']:<5}  win {row['win_prob']:6.1%}   so far {row['return_so_far']:+.2%}")
    print(f"\nLog file: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
