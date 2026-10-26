"""Checks for the Version 2 math: spread, risk stats, and the Monte Carlo engine.

Run from the project folder with your virtual environment on:

    python tests/test_analytics.py
"""

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.market_data import bid_ask_spread
from src.monte_carlo import simulate_final_returns, trading_days_left, win_probabilities
from src.risk import annualized_volatility, max_drawdown, period_return


def check(name: str, actual, expected, places: int = 10) -> None:
    if actual is None and expected is None:
        print(f"OK  {name}")
        return
    if actual is None or expected is None or round(actual, places) != round(expected, places):
        raise SystemExit(f"FAIL {name}: got {actual}, expected {expected}")
    print(f"OK  {name}")


def main() -> None:
    # --- Bid/ask spread ---
    spread, spread_pct = bid_ask_spread(99.90, 100.10)
    check("spread in dollars", spread, 0.20)
    check("spread as % of midpoint", spread_pct, 0.002)
    check("market closed (bid/ask 0) -> no spread", bid_ask_spread(0, 0)[0], None)

    # --- Risk stats ---
    check("drawdown 100 -> 120 -> 90", max_drawdown(pd.Series([100, 120, 90])), -0.25)
    check("never falls -> drawdown 0", max_drawdown(pd.Series([100, 101, 102])), 0.0)
    check("period return 100 -> 125", period_return(pd.Series([100, 110, 125])), 0.25)
    check("flat price -> zero volatility", annualized_volatility(pd.Series([50.0] * 10)), 0.0)

    # --- Trading days left ---
    # Fri Nov 13 -> Mon 16 .. Fri 20 = 5 trading days
    check("days left from Fri Nov 13", trading_days_left(date(2026, 11, 13), date(2026, 11, 20)), 5)
    check("contest over -> 0 days", trading_days_left(date(2026, 11, 20), date(2026, 11, 20)), 0)

    # --- Monte Carlo ---
    # Stock A always rises 1% a day; B is flat. A should win every simulation.
    history = pd.DataFrame({"A": [0.01] * 50, "B": [0.0] * 50})
    final = simulate_final_returns(pd.Series({"A": 0.0, "B": 0.0}), history, days_left=3, n_sims=500)
    check("3 days of +1% compounds to +3.0301%", float(final["A"].iloc[0]), 1.01**3 - 1)
    check("A wins 100% of runs", float(win_probabilities(final)["A"]), 1.0)

    # No days left -> everyone finishes exactly where they are now.
    final = simulate_final_returns(pd.Series({"A": 0.05, "B": 0.02}), history, days_left=0, n_sims=10)
    check("no days left keeps current return", float(final["A"].iloc[0]), 0.05)

    # A recently listed stock (missing early history) must keep its volatility:
    # gaps are filled from its own moves, not treated as "no change".
    import numpy as np
    new_stock = [None] * 40 + [0.05, -0.05] * 5
    history = pd.DataFrame({"OLD": [0.0] * 50, "NEW": new_stock}, dtype=float)
    final = simulate_final_returns(pd.Series({"OLD": 0.0, "NEW": 0.0}), history, days_left=1, n_sims=2000)
    share_flat = float((np.abs(final["NEW"]) < 1e-12).mean())
    check("new stock never gets a fake 0% day", share_flat, 0.0)

    print("\nAll analytics checks passed.")


if __name__ == "__main__":
    main()
