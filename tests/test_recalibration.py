"""Checks for recalibration (no internet needed).

    python tests/test_recalibration.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.recalibration import apply, fit, fit_and_test, log_loss


def check(name: str, condition: bool) -> None:
    if not condition:
        raise SystemExit(f"FAIL {name}")
    print(f"OK  {name}")


def overconfident_forecasts(n_forecasts: int = 400, picks: int = 20, seed: int = 0) -> pd.DataFrame:
    """Fake forecasts from a model that is too sure of itself.

    True chances q are random; the 'model' reports q squared (rescaled), which
    exaggerates favorites exactly like temperature T = 2 would. Winners are drawn from q.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_forecasts):
        q = rng.dirichlet(np.full(picks, 0.5))
        p = q**2 / (q**2).sum()
        winner = rng.choice(picks, p=q)
        for j in range(picks):
            rows.append({"forecast_id": i, "window_start": f"2024-{1 + i * 12 // n_forecasts:02d}-01",
                         "prob": p[j], "won": int(j == winner)})
    return pd.DataFrame(rows)


def main() -> None:
    probs = pd.Series({"A": 0.6, "B": 0.3, "C": 0.1, "D": 0.0})

    same = apply(probs, 1.0, 0.0)
    check("T = 1 and no floor leaves odds unchanged", np.allclose(same, probs))
    flat = apply(probs, 2.0, 0.0)
    check("recalibrated odds still add up to 100%", abs(flat.sum() - 1) < 1e-12)
    check("T > 1 lowers the favorite", flat["A"] < probs["A"])
    check("T > 1 raises the underdogs", flat["C"] > probs["C"])
    check("ranking is preserved", list(flat.sort_values(ascending=False).index[:3]) == ["A", "B", "C"])
    floored = apply(probs, 1.0, 0.04)
    check("the floor gives a 0% pick a small chance (4% / 4 picks = 1%)", abs(floored["D"] - 0.01) < 1e-12)

    data = overconfident_forecasts()
    temperature, weight = fit(data)
    check(f"fit finds the model is overconfident (T = {temperature:.2f}, true value 2)", 1.6 < temperature < 2.5)
    check("recalibration lowers log loss", log_loss(data, temperature, weight) < log_loss(data))

    result = fit_and_test(data)
    check("it also helps on the held-out newer half",
          result["test_log_loss_after"] < result["test_log_loss_before"])

    print("\nAll recalibration checks passed.")


if __name__ == "__main__":
    main()
