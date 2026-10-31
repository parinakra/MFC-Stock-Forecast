"""Checks for the five forecasting models (no internet needed).

    python tests/test_models.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.models import MODELS


def check(name: str, condition: bool) -> None:
    if not condition:
        raise SystemExit(f"FAIL {name}")
    print(f"OK  {name}")


def sample_history(days: int = 250, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """A fake year: a market, two stocks that follow it, one independent stock, one new listing."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0005, 0.01, days)
    data = pd.DataFrame(
        {
            "FOLLOW1": 1.5 * market + rng.normal(0, 0.01, days),
            "FOLLOW2": 1.2 * market + rng.normal(0, 0.01, days),
            "LONER": rng.standard_t(3, days) * 0.02,
            "NEWBIE": np.r_[[np.nan] * 150, rng.normal(0, 0.03, days - 150)],
        }
    )
    return data, pd.Series(market, name="SPY")


def main() -> None:
    daily, market = sample_history()
    current = pd.Series({"FOLLOW1": 0.05, "FOLLOW2": 0.0, "LONER": -0.02, "NEWBIE": 0.01})

    for key, (name, _, simulate, _) in MODELS.items():
        final = simulate(current, daily, 10, n_sims=2000, seed=1, market=market)
        again = simulate(current, daily, 10, n_sims=2000, seed=1, market=market)
        check(f"{name}: one column per stock, one row per simulation", final.shape == (2000, 4))
        check(f"{name}: no missing values (new listing handled)", not final.isna().any().any())
        check(f"{name}: same seed gives identical results", final.equals(again))
        same = simulate(current, daily, 0, n_sims=50, seed=1, market=market)
        check(f"{name}: zero days left keeps today's returns", np.allclose(same.iloc[0].values, current.values))

    # Fat tails vs bell curve: over ONE day, Student-t should produce more extreme moves
    # than the bell curve, even though both are scaled to the same volatility.
    gbm = MODELS["gbm"][2](current * 0, daily, 1, n_sims=40_000, seed=3)
    fat = MODELS["student_t"][2](current * 0, daily, 1, n_sims=40_000, seed=3)

    def extreme_share(frame):
        values = frame["FOLLOW1"]
        return float((values.abs() > 4 * values.std()).mean())

    check("fat tails produce more 4-sigma days than the bell curve", extreme_share(fat) > 2 * extreme_share(gbm))
    check(
        "fat tails keep the same volatility as the bell curve (within 10%)",
        abs(fat["FOLLOW1"].std() / gbm["FOLLOW1"].std() - 1) < 0.10,
    )

    # Factor model: stocks that follow the market should stay correlated; the loner shouldn't.
    factor = MODELS["market_factor"][2](current * 0, daily, 5, n_sims=20_000, seed=4, market=market)
    check("factor model: market followers move together", factor["FOLLOW1"].corr(factor["FOLLOW2"]) > 0.3)
    check("factor model: independent stock stays independent", abs(factor["FOLLOW1"].corr(factor["LONER"])) < 0.1)

    print("\nAll model checks passed.")


if __name__ == "__main__":
    main()
