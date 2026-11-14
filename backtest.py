"""Run the backtest: score the models on past pretend contests.

Run from the project folder (takes a few minutes):

    python backtest.py              # default: a new contest every 10 trading days, 4,000 simulations
    python backtest.py --step 5     # more windows (slower)
    python backtest.py --sims 2000  # faster, a bit noisier

Writes three files the app's Backtest tab reads:
    data/backtest_results.csv      one row per window, checkpoint and model
    data/backtest_summary.csv      averages per model and checkpoint
    data/backtest_calibration.csv  predicted vs actual win rates
"""

import argparse
import sys

import pandas as pd
import yfinance as yf

from src import recalibration
from src.backtest import BACKTEST_MODELS, CALIBRATION_BINS, calibration, run_backtest, summarize
from src.config import BENCHMARK_TICKER, DATA_DIR
from src.data_store import load_picks
from src.models import MODELS


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest the forecasting models on past windows.")
    parser.add_argument("--years", type=int, default=5, help="years of price history to download")
    parser.add_argument("--step", type=int, default=10, help="trading days between pretend contests")
    parser.add_argument("--sims", type=int, default=4000, help="simulations per forecast")
    args = parser.parse_args()

    tickers = sorted({p["ticker"] for p in load_picks()} | {BENCHMARK_TICKER})
    print(f"Downloading {args.years} years of prices for {len(tickers)} tickers...")
    data = yf.download(tickers=tickers, period=f"{args.years}y", auto_adjust=True, progress=False, group_by="ticker")
    closes = pd.DataFrame(
        {t: data[(t, "Close")] for t in tickers if isinstance(data.columns, pd.MultiIndex) and (t, "Close") in data.columns}
    ).dropna(how="all")
    if closes.empty or BENCHMARK_TICKER not in closes.columns:
        print("ERROR: no prices came back from Yahoo Finance. Try again in a few minutes.")
        return 1
    print(f"Got {len(closes)} trading days, {closes.index[0].date()} to {closes.index[-1].date()}.")

    def progress(i, total, day):
        if i == 1 or i % 10 == 0 or i == total:
            print(f"  window {i}/{total} (starting {day.date()})", flush=True)

    results, pairs = run_backtest(closes, BENCHMARK_TICKER, step=args.step, n_sims=args.sims, progress=progress)
    print()
    if results.empty:
        print("ERROR: not enough history to build any past windows.")
        return 1

    summary = summarize(results)
    calib = calibration(pairs)
    DATA_DIR.mkdir(exist_ok=True)
    results.to_csv(DATA_DIR / "backtest_results.csv", index=False)
    summary.to_csv(DATA_DIR / "backtest_summary.csv", index=False)
    calib.to_csv(DATA_DIR / "backtest_calibration.csv", index=False)
    pairs.to_csv(DATA_DIR / "backtest_forecasts.csv", index=False)

    print("\nFitting recalibration (older half of contests) and testing it (newer half)...")
    recal, recal_tables = {}, []
    for model in BACKTEST_MODELS:
        model_forecasts = pairs[pairs["model"] == model]
        params = recalibration.fit_and_test(model_forecasts)
        recal[model] = params
        test = model_forecasts[model_forecasts["window_start"] >= params["test_windows"][0]]
        before = recalibration.calibration_table(test, 1.0, 0.0, CALIBRATION_BINS).assign(model=model, version="raw")
        after = recalibration.calibration_table(
            test, params["train_temperature"], params["train_weight"], CALIBRATION_BINS
        ).assign(model=model, version="recalibrated")
        recal_tables += [before, after]
    recalibration.save(recal, DATA_DIR / "recalibration.json")
    pd.concat(recal_tables, ignore_index=True).to_csv(DATA_DIR / "backtest_calibration_test.csv", index=False)

    names = {k: MODELS[k][0] for k in BACKTEST_MODELS}
    windows = results["window_start"].nunique()
    print(f"\n{windows} past contests, {results['window_start'].min()} to {results['window_end'].max()}")
    print(f"{'Model':<20}{'Days left':>10}{'Winner prob':>13}{'Blind guess':>13}{'Favorite won':>14}{'Leader won':>12}{'Brier skill':>13}")
    for _, r in summary.iterrows():
        print(
            f"{names[r['model']]:<20}{r['days_left']:>10}{r['avg_winner_prob']:>13.1%}{r['blind_guess_prob']:>13.1%}"
            f"{r['favorite_won']:>14.0%}"
            + (f"{r['leader_won']:>12.0%}" if pd.notna(r["leader_won"]) else f"{'—':>12}")
            + f"{r['brier_skill']:>13.1%}"
        )
    print(f"\n{'Model':<20}{'Temperature':>12}{'Floor':>8}{'Test log loss':>16}{'Test Brier':>14}   (before -> after, newer half only)")
    for model, r in recal.items():
        print(
            f"{names[model]:<20}{r['temperature']:>12.2f}{r['weight']:>8.1%}"
            f"   {r['test_log_loss_before']:.3f} -> {r['test_log_loss_after']:.3f}"
            f"   {r['test_brier_before']:.4f} -> {r['test_brier_after']:.4f}"
            f"   {'helps' if r['helps'] else 'does NOT help'}"
        )
    print("\nSaved data/backtest_*.csv and data/recalibration.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
