"""Backtest: would the models have given sensible odds in PAST contests?

The idea in plain words
-----------------------
We pretend the same picks ran the same 24-trading-day contest many times in
the past (one contest starting every ~2 weeks over several years). For each
pretend contest we ask each model for its odds at three moments:

    at the start (24 days left), in the middle (14 left), near the end (5 left)

using ONLY prices up to that moment (no peeking at the future). Then we look
at who actually won that past contest and score the odds.

How odds are scored
-------------------
- Winner probability: the chance the model gave to the pick that actually won.
  Higher is better. Guessing blindly gives 1 / (number of picks).
- Hit rate: how often the model's favorite actually won.
- Brier score: squared error of all the probabilities (0 = perfect). We report
  "skill" = how much better than blind guessing (0% = no better, 100% = perfect).
- Log loss: punishes being confidently wrong. Lower is better.
- Calibration: of all the times the model said "about 10%", did those picks
  win about 10% of the time? That's the most important test of a forecast.

Honest limits
-------------
- Windows overlap and share the same stocks, so they aren't fully independent.
- These stocks were picked in 2026, partly because of how they had been doing,
  so the history isn't a random sample of stocks.
- The earnings model isn't backtested (Yahoo doesn't reliably give old earnings dates).
"""

import numpy as np
import pandas as pd

from src.models import MODELS
from src.monte_carlo import win_probabilities

CONTEST_DAYS = 24              # Oct 20 – Nov 20, 2026 is 24 trading days
CHECKPOINTS = (24, 14, 5)      # days left when each forecast is made
HISTORY_DAYS = 252             # one year of past days, like the live models
MIN_HISTORY = 120              # a pick needs at least this much history to be included
BACKTEST_MODELS = ["bootstrap", "gbm", "student_t", "block_bootstrap", "market_factor"]
CALIBRATION_BINS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0]
FLOOR = 1e-4                   # smallest probability used in log loss (avoids log(0))


def window_starts(n_days: int, contest_days: int = CONTEST_DAYS, step: int = 10,
                  history_days: int = HISTORY_DAYS) -> list[int]:
    """Positions (row numbers) where a pretend contest can start.

    Needs a year of history before the start and a full contest after it.
    """
    first = history_days
    last = n_days - 1 - contest_days
    return list(range(first, last + 1, step))


def forecast_window(closes: pd.DataFrame, start: int, days_left: int, model: str, benchmark: str,
                    n_sims: int = 4000, seed: int = 42, contest_days: int = CONTEST_DAYS) -> dict | None:
    """One model's forecast for one past contest at one checkpoint, plus the real outcome.

    closes: daily closing prices (rows = days, columns = tickers, including the benchmark)
    start:  row of the contest's starting close
    """
    end = start + contest_days
    checkpoint = end - days_left                     # the day the forecast is made
    picks = [t for t in closes.columns if t != benchmark]

    returns = closes.pct_change()
    # History = the year of daily returns up to and INCLUDING the checkpoint day, nothing after.
    history = returns.iloc[checkpoint - HISTORY_DAYS + 1: checkpoint + 1]

    eligible = [
        t for t in picks
        if pd.notna(closes[t].iloc[start]) and pd.notna(closes[t].iloc[end])
        and history[t].notna().sum() >= MIN_HISTORY
    ]
    if len(eligible) < 2:
        return None

    current = closes[eligible].iloc[checkpoint] / closes[eligible].iloc[start] - 1
    final_actual = closes[eligible].iloc[end] / closes[eligible].iloc[start] - 1
    market = history[benchmark] if benchmark in history.columns else None

    simulate = MODELS[model][2]
    final_sim = simulate(current, history[eligible], days_left, n_sims=n_sims, seed=seed, market=market)
    probs = win_probabilities(final_sim).reindex(eligible).fillna(0.0)

    return {
        "probs": probs,
        "winner": final_actual.idxmax(),
        "leader": current.idxmax() if days_left < contest_days else None,
        "window_start": closes.index[start],
        "window_end": closes.index[end],
    }


def score(probs: pd.Series, winner: str) -> dict:
    """Scores for one forecast."""
    n = len(probs)
    outcome = (probs.index == winner).astype(float)
    p_win = float(probs[winner])
    return {
        "n_picks": n,
        "winner_prob": p_win,
        "brier": float(((probs.values - outcome) ** 2).sum()),
        "log_loss": float(-np.log(max(p_win, FLOOR))),
        "hit": bool(probs.idxmax() == winner),
        "uniform_prob": 1.0 / n,
        "uniform_brier": 1.0 - 1.0 / n,      # Brier score of giving everyone 1/n
        "uniform_log_loss": float(np.log(n)),
    }


def run_backtest(closes: pd.DataFrame, benchmark: str, models=BACKTEST_MODELS, checkpoints=CHECKPOINTS,
                 step: int = 10, n_sims: int = 4000, seed: int = 42, progress=None):
    """Run every model on every past window and checkpoint.

    Returns (results, pairs):
      results: one row per (window, checkpoint, model) with its scores
      pairs:   one row per (window, checkpoint, model, pick) with the predicted
               probability and whether that pick won, used for calibration
    """
    rows, pairs = [], []
    starts = window_starts(len(closes), step=step)
    for i, start in enumerate(starts):
        if progress:
            progress(i + 1, len(starts), closes.index[start])
        for days_left in checkpoints:
            for model in models:
                result = forecast_window(closes, start, days_left, model, benchmark, n_sims=n_sims, seed=seed)
                if result is None:
                    continue
                s = score(result["probs"], result["winner"])
                rows.append({
                    "window_start": result["window_start"].date(),
                    "window_end": result["window_end"].date(),
                    "days_left": days_left,
                    "model": model,
                    "winner": result["winner"],
                    "favorite": result["probs"].idxmax(),
                    "leader": result["leader"],
                    "leader_won": (result["leader"] == result["winner"]) if result["leader"] else None,
                    **s,
                })
                forecast_id = f"{result['window_start'].date()}|{days_left}|{model}"
                for ticker, p in result["probs"].items():
                    pairs.append({"forecast_id": forecast_id, "window_start": str(result["window_start"].date()),
                                  "model": model, "days_left": days_left, "ticker": ticker, "prob": float(p),
                                  "won": int(ticker == result["winner"])})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """One row per model and checkpoint: how good were the odds?"""
    grouped = results.groupby(["model", "days_left"])
    summary = pd.DataFrame({
        "windows": grouped.size(),
        "avg_winner_prob": grouped["winner_prob"].mean(),
        "blind_guess_prob": grouped["uniform_prob"].mean(),
        "favorite_won": grouped["hit"].mean(),
        "leader_won": grouped["leader_won"].apply(lambda s: s.dropna().astype(float).mean() if s.notna().any() else np.nan),
        "brier_skill": 1 - grouped["brier"].mean() / grouped["uniform_brier"].mean(),
        "log_loss": grouped["log_loss"].mean(),
        "blind_guess_log_loss": grouped["uniform_log_loss"].mean(),
    }).reset_index()
    return summary.sort_values(["days_left", "brier_skill"], ascending=[False, False])


def calibration(pairs: pd.DataFrame, bins=CALIBRATION_BINS) -> pd.DataFrame:
    """For each model and probability range: average predicted chance vs how often those picks won."""
    if pairs.empty:
        return pd.DataFrame(columns=["model", "bin", "mean_pred", "observed", "count"])
    labels = [f"{int(lo * 100)}–{int(hi * 100)}%" for lo, hi in zip(bins[:-1], bins[1:])]
    data = pairs.assign(bin=pd.cut(pairs["prob"], bins=bins, labels=labels, include_lowest=True))
    table = (
        data.groupby(["model", "bin"], observed=True)
        .agg(mean_pred=("prob", "mean"), observed=("won", "mean"), count=("won", "size"))
        .reset_index()
    )
    table["bin"] = table["bin"].astype(str)
    return table
