"""Checks for the earnings-jump model (no internet needed).

    python tests/test_earnings.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.earnings import reaction_day
from src.models import MODELS


def check(name: str, condition: bool) -> None:
    if not condition:
        raise SystemExit(f"FAIL {name}")
    print(f"OK  {name}")


def main() -> None:
    days = pd.bdate_range("2026-11-02", "2026-11-20")

    # --- Which day reacts to the announcement ---
    check("7am report moves the same day", reaction_day(pd.Timestamp("2026-11-06 07:00"), days) == pd.Timestamp("2026-11-06"))
    check("4:05pm report moves the next day", reaction_day(pd.Timestamp("2026-11-18 16:05"), days) == pd.Timestamp("2026-11-19"))
    check("Friday after-close report moves Monday", reaction_day(pd.Timestamp("2026-11-06 16:30"), days) == pd.Timestamp("2026-11-09"))
    check("unknown time (midnight) = same day", reaction_day(pd.Timestamp("2026-11-10 00:00"), days) == pd.Timestamp("2026-11-10"))

    # --- The model ---
    simulate = MODELS["earnings_jump"][2]
    history_days = pd.bdate_range("2025-11-01", periods=250)
    calm = pd.DataFrame({"AAA": [0.0] * 250, "BBB": [0.0] * 250}, index=history_days)
    calm.loc[history_days[100], "AAA"] = 1.0  # one past earnings day where AAA doubled
    current = pd.Series({"AAA": 0.0, "BBB": 0.0})
    window = list(pd.bdate_range("2026-11-03", periods=10))

    # Past earnings day removed from the normal-day pool: AAA never doubles on a normal day.
    events = {"window": window, "tickers": {"AAA": {"next_reaction": None, "past_reactions": [history_days[100].date()],
                                                    "moves": [1.0, 1.0]}}}
    final = simulate(current, calm, 10, n_sims=5000, seed=1, events=events)
    check("past earnings days are not reused as normal days", float(final["AAA"].max()) < 0.5)

    # Upcoming earnings inside the window: every simulation gets a jump of that size on that day.
    events["tickers"]["AAA"]["next_reaction"] = window[4].date()
    events["tickers"]["AAA"]["moves"] = [0.20, 0.20, 0.20]
    final = simulate(current, calm, 10, n_sims=2000, seed=1, events=events)
    check("earnings jump is applied (always a 20% move)", np.allclose(final["AAA"].abs(), 0.20))
    up_share = float((final["AAA"] > 0).mean())
    check("direction is a coin flip (about half up)", 0.45 < up_share < 0.55)
    check("other stocks unaffected", np.allclose(final["BBB"], 0.0))

    # Earnings after the contest ends -> no jump.
    events["tickers"]["AAA"]["next_reaction"] = pd.Timestamp("2026-12-01").date()
    final = simulate(current, calm, 10, n_sims=2000, seed=1, events=events)
    check("earnings after the contest is ignored", np.allclose(final["AAA"], 0.0))

    # Too few past reactions -> no jump (not enough evidence).
    events["tickers"]["AAA"]["next_reaction"] = window[4].date()
    events["tickers"]["AAA"]["moves"] = [0.20]
    final = simulate(current, calm, 10, n_sims=2000, seed=1, events=events)
    check("needs at least 2 past earnings moves", np.allclose(final["AAA"], 0.0))

    # No earnings data at all -> still runs.
    final = simulate(current, calm, 10, n_sims=500, seed=1, events=None)
    check("works without earnings data", final.shape == (500, 2))

    print("\nAll earnings checks passed.")


if __name__ == "__main__":
    main()
