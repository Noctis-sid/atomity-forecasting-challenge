"""Simple, leakage-safe baselines.

All ``*_steps`` functions take only the *history* (values strictly before the forecast
origin) and an array of forecast ``steps`` (1 = the next 5-minute slot) and return one
forecast per step. NaNs in history (empty grid slots) are handled by forward-filling, which
only ever uses past information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEASON = 12  # 12 x 5 min = 1 hour


def _ffill(history) -> np.ndarray:
    h = pd.Series(np.asarray(history, dtype=float)).ffill().bfill().to_numpy()
    if h.size == 0 or not np.isfinite(h).any():
        raise ValueError("history has no finite values")
    return h


# --- starter API (kept for backwards compatibility) ---------------------------------------
def persistence_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    if train.empty:
        raise ValueError("train must not be empty")
    last_value = float(train.iloc[-1][target])
    return pd.Series(last_value, index=template.index, dtype=float)


def seasonal_naive_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, season_length: int) -> pd.Series:
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    values = train[target].astype(float).to_numpy()
    if len(values) < season_length:
        raise ValueError("not enough history for seasonal-naive forecast")
    forecast = [float(values[-season_length + (i % season_length)]) for i in range(len(template))]
    return pd.Series(forecast, index=template.index, dtype=float)


# --- step-based baselines used by the pipeline ---------------------------------------------
def persistence_steps(history, steps) -> np.ndarray:
    h = _ffill(history)
    return np.full(len(steps), h[-1])


def moving_average_steps(history, steps, window: int = 12) -> np.ndarray:
    h = _ffill(history)
    return np.full(len(steps), h[-window:].mean())


def drift_steps(history, steps, window: int = 24) -> np.ndarray:
    """Linear trend through the last ``window`` points, extrapolated."""
    h = _ffill(history)[-window:]
    if len(h) < 3:
        return np.full(len(steps), h[-1])
    x = np.arange(len(h))
    slope, intercept = np.polyfit(x, h, 1)
    return intercept + slope * (len(h) - 1 + np.asarray(steps, dtype=float))


def seasonal_naive_steps(history, steps, season: int = SEASON) -> np.ndarray:
    h = _ffill(history)
    if len(h) < season:
        raise ValueError("not enough history for seasonal-naive")
    steps = np.asarray(steps)
    k = np.ceil(steps / season).astype(int)
    idx = len(h) - 1 + steps - k * season  # most recent past slot with the same phase
    return h[idx]


def phase_profile(history, season: int = SEASON, shrink: float = 2.0) -> np.ndarray:
    """Average within-hour deviation from a centred 1-hour mean, by phase.

    Uses only the supplied history. Phase 0 is the first history slot. The profile is shrunk
    towards zero by n/(n+shrink) where n is the number of cycles observed per phase.
    """
    h = _ffill(history)
    if len(h) < 2 * season:
        return np.zeros(season)
    level = pd.Series(h).rolling(season, center=True, min_periods=season // 2).mean().to_numpy()
    dev = h - level
    phases = np.arange(len(h)) % season
    prof = np.zeros(season)
    for p in range(season):
        d = dev[phases == p]
        d = d[np.isfinite(d)]
        if d.size:
            prof[p] = d.mean() * d.size / (d.size + shrink)
    return prof - prof.mean()


def seasonal_profile_steps(history, steps, window: int = 12, season: int = SEASON) -> np.ndarray:
    """Recent de-seasonalised level + average within-hour profile."""
    h = _ffill(history)
    prof = phase_profile(h, season)
    n = len(h)
    phases_hist = np.arange(n) % season
    level = (h - prof[phases_hist])[-window:].mean()
    target_phase = (n - 1 + np.asarray(steps)) % season
    return level + prof[target_phase]


def _ses_sse(x: np.ndarray, alpha: float) -> float:
    level, sse = x[0], 0.0
    for v in x[1:]:
        sse += (v - level) ** 2
        level = alpha * v + (1 - alpha) * level
    return sse


def ses_level(x, alpha: float) -> float:
    level = x[0]
    for v in x[1:]:
        level = alpha * v + (1 - alpha) * level
    return float(level)


def fit_ses_alpha(x, grid=np.linspace(0.02, 1.0, 50)) -> float:
    """Smoothing weight chosen by in-sample one-step-ahead SSE on the history only."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return 1.0
    return float(min(grid, key=lambda a: _ses_sse(x, a)))


def ses_profile_steps(history, steps, season: int = SEASON) -> np.ndarray:
    """Additive local-level + fixed within-hour profile (an ETS(A,N,A)-style model).

    1. estimate the within-hour profile from the history (``phase_profile``);
    2. de-seasonalise; 3. fit simple exponential smoothing (alpha by in-sample one-step SSE);
    4. forecast = smoothed level + profile at the target phase (flat level, no trend).
    """
    h = _ffill(history)
    prof = phase_profile(h, season)
    n = len(h)
    x = h - prof[np.arange(n) % season]
    level = ses_level(x, fit_ses_alpha(x))
    return level + prof[(n - 1 + np.asarray(steps)) % season]
