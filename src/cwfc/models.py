"""Model registry and the training-only view handed to every forecaster."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import baselines as B
from .panel import PanelForecaster


@dataclass
class History:
    """Everything a model may see at a forecast origin: data strictly before it."""

    timestamps: np.ndarray            # grid timestamps of the history
    count: np.ndarray                 # active_vm_count history
    cpu: np.ndarray                   # aggregate_cpu_utilization history (NaN = empty slot)
    X: np.ndarray                     # VM x time avg_cpu matrix (NaN = no reading)
    static: np.ndarray                # VM static features
    cache: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.timestamps)

    def upto(self, n: int) -> "History":
        """History truncated to its first ``n`` slots (used for rolling-origin backtests)."""
        return History(self.timestamps[:n], self.count[:n], self.cpu[:n], self.X[:, :n], self.static)


ModelFn = Callable[[History, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Model:
    name: str
    target: str
    fn: ModelFn
    min_history: Callable[[int], int]
    is_baseline: bool = False
    description: str = ""


def _series(hist: History, target: str) -> np.ndarray:
    return hist.count if target == "active_vm_count" else hist.cpu


def _simple(name, target, f, minh, desc, baseline=False, **kw) -> Model:
    return Model(name, target, lambda hist, steps: f(_series(hist, target), steps, **kw),
                 lambda H: minh, baseline, desc)


def _panel(learner: str) -> ModelFn:
    def fn(hist: History, steps):
        key = ("panel", learner, len(hist), tuple(steps))
        if key not in hist.cache:
            hist.cache[key] = PanelForecaster(learner=learner).forecast(hist.X, hist.static, steps)
        return hist.cache[key]
    return fn


def _ensemble(hist: History, steps):
    a = _panel("gbm")(hist, steps)
    b = B.ses_profile_steps(hist.cpu, steps)
    return 0.5 * (a + b)


def registry() -> list[Model]:
    cpu, cnt = "aggregate_cpu_utilization", "active_vm_count"
    models = [
        _simple("persistence", cpu, B.persistence_steps, 1, "last observed value", True),
        _simple("moving_average_6", cpu, B.moving_average_steps, 6, "mean of last 30 min", True, window=6),
        _simple("moving_average_12", cpu, B.moving_average_steps, 12, "mean of last hour", True, window=12),
        _simple("moving_average_24", cpu, B.moving_average_steps, 24, "mean of last 2 h", True, window=24),
        _simple("seasonal_naive_1h", cpu, B.seasonal_naive_steps, 12, "same 5-min phase, previous hour", True),
        _simple("drift_24", cpu, B.drift_steps, 24, "linear trend over last 2 h", True, window=24),
        _simple("seasonal_profile", cpu, B.seasonal_profile_steps, 24,
                "de-seasonalised 1 h mean level + within-hour profile"),
        _simple("ses_profile", cpu, B.ses_profile_steps, 24,
                "exponentially smoothed level (alpha fit in-sample) + within-hour profile"),
        Model("panel_ridge", cpu, _panel("ridge"), PanelForecaster.min_history, False,
              "pooled VM-level ridge, aggregated bottom-up"),
        Model("panel_gbm", cpu, _panel("gbm"), PanelForecaster.min_history, False,
              "pooled VM-level gradient boosting, aggregated bottom-up"),
        Model("ensemble_gbm_ses", cpu, _ensemble, PanelForecaster.min_history, False,
              "mean of panel_gbm and ses_profile"),
        _simple("persistence", cnt, B.persistence_steps, 1, "last observed count", True),
        _simple("moving_average_12", cnt, B.moving_average_steps, 12, "mean count over last hour", True, window=12),
        _simple("drift_24", cnt, B.drift_steps, 24, "linear trend in count over last 2 h", True, window=24),
    ]
    return models


def models_for(target: str) -> list[Model]:
    return [m for m in registry() if m.target == target]


def build_history(train_targets: pd.DataFrame, readings: pd.DataFrame, metadata) -> History:
    """Construct the model-visible history from TRAINING rows only."""
    from .data import vm_matrix
    from .panel import static_features

    ts = train_targets["timestamp"].to_numpy()
    r = readings[readings["timestamp"] <= ts[-1]]
    W = vm_matrix(r, ts)
    return History(
        timestamps=ts,
        # An empty grid slot is treated as missing telemetry, not as "zero VMs", for modelling.
        count=train_targets["active_vm_count"].where(train_targets["observed"]).to_numpy(dtype=float),
        cpu=train_targets["aggregate_cpu_utilization"].to_numpy(dtype=float),
        X=W.to_numpy(dtype=float),
        static=static_features(W.index, metadata),
    )
