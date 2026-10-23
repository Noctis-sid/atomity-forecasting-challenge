"""Turning a point forecast + residual quantiles into a provisioning decision.

Policy ("forecast-plus-quantile headroom"):

    provisioned(t+h) = forecast(t+h) + Q_tau(residual at step h)

``Q_tau`` is the empirical tau-quantile of backtest residuals (actual - forecast) at step h,
estimated only from rolling-origin folds inside the training window. Under a linear cost
c_under per unit of shortfall and c_over per unit of idle headroom, the cost-minimising choice
is the newsvendor quantile tau* = c_under / (c_under + c_over).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import monotone_bands


def newsvendor_tau(cost_under: float, cost_over: float) -> float:
    return cost_under / (cost_under + cost_over)


def provision(forecast, residual_q) -> np.ndarray:
    return np.asarray(forecast, dtype=float) + np.asarray(residual_q, dtype=float)


def evaluate(actual, provisioned, cost_under: float = 1.0, cost_over: float = 1.0) -> dict:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(provisioned, dtype=float)
    short = np.maximum(a - p, 0)
    idle = np.maximum(p - a, 0)
    return {
        "shortfall_rate": float(np.mean(short > 0)),
        "mean_shortfall": float(short.mean()),
        "max_shortfall": float(short.max()),
        "mean_headroom": float(idle.mean()),
        "cost": float(cost_under * short.sum() + cost_over * idle.sum()) / len(a),
    }


def residual_quantile_table(errors: pd.DataFrame, model: str, taus, horizon: int, smooth: int = 3) -> pd.DataFrame:
    e = errors[errors["model"] == model]
    rows = []
    for s in range(1, horizon + 1):
        r = e.loc[(e["step"] - s).abs() <= smooth, "error"].to_numpy()
        rows.append({"step": s, **{tau: float(np.quantile(r, tau)) for tau in taus}})
    return monotone_bands(pd.DataFrame(rows).set_index("step"))


def policy_table(actual, forecast, steps, qtable: pd.DataFrame, cost_under=1.0, cost_over=1.0,
                 extra: dict | None = None) -> pd.DataFrame:
    """Compare quantile-headroom policies (and optional fixed policies in ``extra``)."""
    rows = []
    for tau in qtable.columns:
        prov = provision(forecast, qtable.loc[steps, tau].to_numpy())
        rows.append({"policy": f"forecast + Q{tau:.2f}", **evaluate(actual, prov, cost_under, cost_over)})
    for name, prov in (extra or {}).items():
        rows.append({"policy": name, **evaluate(actual, prov, cost_under, cost_over)})
    return pd.DataFrame(rows)
