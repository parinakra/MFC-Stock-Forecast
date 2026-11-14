"""Recalibration: turn the model's odds into odds that match how often things really happened.

The backtest showed the model is overconfident: picks it gave ~26% won ~19% of
the time, and long shots it gave ~0.4% won ~1%. We fix that with two knobs:

  T (temperature)  > 1 flattens the odds: favorites come down, others go up.
                   Each probability p becomes p^(1/T), then everything is rescaled to add to 100%.
  w (floor)        blends in a little of "everyone has an equal chance", so no pick is
                   ever at a true 0%. That accounts for surprises like INO's FDA approval.

    calibrated = (1 - w) * p^(1/T) / sum(p^(1/T))  +  w / (number of picks)

The two knobs are chosen to make the backtest's forecasts as accurate as possible
(lowest log loss). To keep it honest, they're fitted on the OLDER half of past
contests and tested on the NEWER half, which the fit never saw.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FLOOR = 1e-4
TEMPERATURES = np.round(np.geomspace(0.6, 4.0, 40), 4)
WEIGHTS = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]


def apply(probs: pd.Series, temperature: float, weight: float) -> pd.Series:
    """Recalibrate one forecast (one probability per pick, adding to 1)."""
    values = np.clip(probs.to_numpy(dtype=float), 0.0, None) ** (1.0 / temperature)
    total = values.sum()
    sharpened = values / total if total > 0 else np.full(len(values), 1.0 / len(values))
    calibrated = (1 - weight) * sharpened + weight / len(values)
    return pd.Series(calibrated, index=probs.index)


def _winner_probs(forecasts: pd.DataFrame, temperature: float, weight: float) -> np.ndarray:
    """Recalibrated probability of the actual winner, for every forecast."""
    out = []
    for _, group in forecasts.groupby("forecast_id", sort=False):
        calibrated = apply(group["prob"].reset_index(drop=True), temperature, weight)
        out.append(float(calibrated[group["won"].to_numpy() == 1].sum()))
    return np.array(out)


def log_loss(forecasts: pd.DataFrame, temperature: float = 1.0, weight: float = 0.0) -> float:
    return float(-np.log(np.maximum(_winner_probs(forecasts, temperature, weight), FLOOR)).mean())


def brier(forecasts: pd.DataFrame, temperature: float = 1.0, weight: float = 0.0) -> float:
    scores = []
    for _, group in forecasts.groupby("forecast_id", sort=False):
        calibrated = apply(group["prob"].reset_index(drop=True), temperature, weight).to_numpy()
        scores.append(float(((calibrated - group["won"].to_numpy()) ** 2).sum()))
    return float(np.mean(scores))


def fit(forecasts: pd.DataFrame) -> tuple[float, float]:
    """Find the (temperature, weight) with the lowest log loss on these forecasts.

    A coarse-to-fine grid search: simple, transparent, and plenty for two knobs.
    """
    groups = [
        (g["prob"].to_numpy(dtype=float), g["won"].to_numpy() == 1)
        for _, g in forecasts.groupby("forecast_id", sort=False)
    ]

    def loss(temperature: float, weight: float) -> float:
        total = 0.0
        for p, won in groups:
            v = np.clip(p, 0, None) ** (1.0 / temperature)
            s = v.sum()
            v = v / s if s > 0 else np.full(len(v), 1.0 / len(v))
            q = (1 - weight) * v + weight / len(v)
            total += -np.log(max(float(q[won].sum()), FLOOR))
        return total / len(groups)

    best = min(((loss(t, w), t, w) for t in TEMPERATURES for w in WEIGHTS))
    return float(best[1]), float(best[2])


def calibration_table(forecasts: pd.DataFrame, temperature: float, weight: float, bins) -> pd.DataFrame:
    """Predicted vs actual win rates after recalibration."""
    rows = []
    for _, group in forecasts.groupby("forecast_id", sort=False):
        calibrated = apply(group["prob"].reset_index(drop=True), temperature, weight)
        rows.append(pd.DataFrame({"prob": calibrated.to_numpy(), "won": group["won"].to_numpy()}))
    data = pd.concat(rows, ignore_index=True)
    labels = [f"{int(lo * 100)}–{int(hi * 100)}%" for lo, hi in zip(bins[:-1], bins[1:])]
    data["bin"] = pd.cut(data["prob"], bins=bins, labels=labels, include_lowest=True)
    table = (
        data.groupby("bin", observed=True)
        .agg(mean_pred=("prob", "mean"), observed=("won", "mean"), count=("won", "size"))
        .reset_index()
    )
    table["bin"] = table["bin"].astype(str)
    return table


def fit_and_test(forecasts: pd.DataFrame) -> dict:
    """Fit on the older half of past contests, test on the newer half, then refit on everything.

    The refit on all contests gives the knobs used live; the test-half numbers
    show how well recalibration works on contests the fit never saw.
    """
    windows = sorted(forecasts["window_start"].unique())
    split = windows[len(windows) // 2]
    train = forecasts[forecasts["window_start"] < split]
    test = forecasts[forecasts["window_start"] >= split]

    t_train, w_train = fit(train)
    t_all, w_all = fit(forecasts)
    result = {
        "temperature": t_all,
        "weight": w_all,
        "train_temperature": t_train,
        "train_weight": w_train,
        "train_windows": [str(windows[0]), str(windows[len(windows) // 2 - 1])],
        "test_windows": [str(split), str(windows[-1])],
        "test_log_loss_before": log_loss(test),
        "test_log_loss_after": log_loss(test, t_train, w_train),
        "test_brier_before": brier(test),
        "test_brier_after": brier(test, t_train, w_train),
    }
    # Only use recalibration live if it actually improved forecasts on contests the fit never saw.
    result["helps"] = bool(result["test_log_loss_after"] < result["test_log_loss_before"])
    return result


def save(params: dict, path: Path) -> None:
    path.write_text(json.dumps(params, indent=2))


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None
