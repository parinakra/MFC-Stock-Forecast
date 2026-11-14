"""Checks for the backtest (no internet needed).

    python tests/test_backtest.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.backtest import calibration, forecast_window, run_backtest, score, summarize, window_starts


def check(name: str, condition: bool) -> None:
    if not condition:
        raise SystemExit(f"FAIL {name}")
    print(f"OK  {name}")


def fake_prices(days: int = 700, stocks: int = 8, seed: int = 0) -> pd.DataFrame:
    """Random-walk prices: a market plus stocks with different volatilities."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=days)
    market = rng.normal(0.0004, 0.01, days)
    data = {"SPY": 100 * np.cumprod(1 + market)}
    for i in range(stocks):
        vol = 0.01 + 0.005 * i
        data[f"S{i}"] = 50 * np.cumprod(1 + market + rng.normal(0, vol, days))
    return pd.DataFrame(data, index=dates)


def main() -> None:
    # --- Window positions ---
    starts = window_starts(400, contest_days=24, step=10, history_days=252)
    check("first window starts after a year of history", starts[0] == 252)
    check("last window ends inside the data", starts[-1] + 24 <= 399)

    # --- Scoring math ---
    s = score(pd.Series({"A": 0.5, "B": 0.3, "C": 0.2}), "B")
    check("winner probability", abs(s["winner_prob"] - 0.3) < 1e-12)
    check("Brier score = 0.25 + 0.49 + 0.04", abs(s["brier"] - 0.78) < 1e-12)
    check("log loss = -ln(0.3)", abs(s["log_loss"] + np.log(0.3)) < 1e-12)
    check("favorite A did not win", s["hit"] is False)
    check("blind-guess Brier = 1 - 1/3", abs(s["uniform_brier"] - 2 / 3) < 1e-12)

    # --- No peeking at the future ---
    prices = fake_prices()
    start, days_left = 300, 14
    checkpoint = start + 24 - days_left
    before = forecast_window(prices, start, days_left, "bootstrap", "SPY", n_sims=1000)
    tampered = prices.copy()
    tampered.iloc[checkpoint + 1: start + 24] *= 3.0  # wildly change every day after the checkpoint (not the final day)
    after = forecast_window(tampered, start, days_left, "bootstrap", "SPY", n_sims=1000)
    check("forecast ignores prices after the checkpoint", before["probs"].equals(after["probs"]))

    # --- A stock with too little history is left out ---
    young = prices.copy()
    young.loc[: young.index[260], "S0"] = np.nan  # S0 starts trading late
    result = forecast_window(young, 300, 24, "bootstrap", "SPY", n_sims=500)
    check("stocks without enough history are excluded", "S0" not in result["probs"].index)

    # --- Full run on fake data ---
    results, pairs = run_backtest(prices, "SPY", models=["bootstrap", "gbm"], checkpoints=(24, 5), step=20, n_sims=800)
    check("results have one row per window x checkpoint x model",
          len(results) == results["window_start"].nunique() * 2 * 2)
    check("probabilities in each forecast add up to 1",
          np.allclose(pairs.groupby(np.arange(len(pairs)) // 8)["prob"].sum(), 1.0, atol=1e-6))

    summary = summarize(results)
    near_end = summary[(summary["model"] == "bootstrap") & (summary["days_left"] == 5)].iloc[0]
    check("5 days before the end, the model beats blind guessing", near_end["brier_skill"] > 0)
    check("5 days before the end, the favorite wins more than 1 in 8", near_end["favorite_won"] > 1 / 8)

    table = calibration(pairs)
    check("calibration table has predicted and observed rates",
          {"mean_pred", "observed", "count"} <= set(table.columns) and table["count"].sum() == len(pairs))

    print("\nAll backtest checks passed.")


if __name__ == "__main__":
    main()
