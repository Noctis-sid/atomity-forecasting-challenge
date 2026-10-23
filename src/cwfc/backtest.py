"""Rolling-origin (expanding-window) backtesting inside the training window only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics as M
from .models import History, Model


def origins(n_train: int, horizon: int, n_folds: int = 8, min_train: int | None = None) -> list[int]:
    """Expanding-window origins (= number of history slots) such that origin + horizon <= n_train."""
    last = n_train - horizon
    if min_train is None:
        min_train = max(horizon + 24, n_train // 3)
    min_train = min(min_train, last)
    if last < 2:
        return []
    return sorted(set(np.linspace(min_train, last, n_folds).round().astype(int).tolist()))


@dataclass
class BacktestResult:
    errors: pd.DataFrame      # one row per (model, origin, step)
    summary: pd.DataFrame     # one row per model
    origins: list[int]


def run_backtest(hist: History, models: list[Model], target: str, horizon: int,
                 n_folds: int = 8, min_train: int | None = None) -> BacktestResult:
    y = hist.count if target == "active_vm_count" else hist.cpu
    folds = origins(len(hist), horizon, n_folds, min_train)
    steps = np.arange(1, horizon + 1)
    recs = []
    for o in folds:
        sub = hist.upto(o)
        sub.cache = hist.cache  # share panel fits between the panel model and the ensemble
        actual = y[o:o + horizon]
        for m in models:
            if o < m.min_history(horizon):
                continue
            pred = np.asarray(m.fn(sub, steps), dtype=float)
            for s, a, p in zip(steps, actual, pred):
                if np.isfinite(a):
                    recs.append((m.name, o, int(s), int(hist.timestamps[o + s - 1]), float(a), float(p)))
    err = pd.DataFrame(recs, columns=["model", "origin", "step", "timestamp", "actual", "pred"])
    if err.empty:
        return BacktestResult(err, pd.DataFrame(), folds)
    err["error"] = err["actual"] - err["pred"]
    return BacktestResult(err, summarise(err, [m for m in models]), folds)


def summarise(err: pd.DataFrame, models: list[Model]) -> pd.DataFrame:
    # Only compare models on folds every model could run on (common support).
    n_models = err["model"].nunique()
    common = err.groupby("origin")["model"].nunique()
    common = common[common == n_models].index
    e = err[err["origin"].isin(common)] if len(common) else err
    base = e[e["model"] == "persistence"].set_index(["origin", "step"])["pred"]
    peak_thr = e.drop_duplicates(["origin", "step"])["actual"].quantile(0.9)
    rows = []
    for name, g in e.groupby("model"):
        b = base.reindex(pd.MultiIndex.from_frame(g[["origin", "step"]])).to_numpy()
        fold_mae = g.groupby("origin").apply(lambda d: M.mae(d.actual, d.pred), include_groups=False)
        pk = g[g["actual"] >= peak_thr]
        rows.append({
            "model": name,
            "baseline": next((m.is_baseline for m in models if m.name == name), False),
            "folds": g["origin"].nunique(),
            "mae": M.mae(g.actual, g.pred),
            "rmse": M.rmse(g.actual, g.pred),
            "wape_pct": M.wape(g.actual, g.pred),
            "bias": float(np.mean(g.actual - g.pred)),
            "mae_vs_persistence": M.normalized_mae(g.actual, g.pred, b) if np.isfinite(b).all() else np.nan,
            "mae_fold_std": float(fold_mae.std(ddof=0)),
            "worst_fold_mae": float(fold_mae.max()),
            "peak_mae": M.mae(pk.actual, pk.pred) if len(pk) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def select_model(summary: pd.DataFrame, models: list[Model], tol: float = 0.01) -> Model:
    """Lowest backtest MAE; within ``tol`` relative MAE, prefer the simpler (earlier) model."""
    order = {m.name: i for i, m in enumerate(models)}
    best = summary["mae"].min()
    near = summary[summary["mae"] <= best * (1 + tol) + 1e-12].copy()
    near["order"] = near["model"].map(order)
    name = near.sort_values("order").iloc[0]["model"]
    return next(m for m in models if m.name == name)


def residual_quantiles(err: pd.DataFrame, model: str, horizon: int,
                       qs=(0.05, 0.1, 0.5, 0.9, 0.95), smooth: int = 3) -> pd.DataFrame:
    """Empirical quantiles of (actual - pred) per step, pooled over neighbouring steps."""
    e = err[err["model"] == model]
    out = []
    for s in range(1, horizon + 1):
        r = e.loc[(e["step"] - s).abs() <= smooth, "error"].to_numpy()
        if r.size == 0:
            r = e["error"].to_numpy()
        row = {"step": s}
        for q in qs:
            row[f"q{int(q * 100):02d}"] = float(np.quantile(r, q)) if r.size else 0.0
        out.append(row)
    return monotone_bands(pd.DataFrame(out).set_index("step")).reset_index()


def monotone_bands(q: pd.DataFrame) -> pd.DataFrame:
    """Uncertainty should not shrink with lead time: widen upper quantiles with a running max
    and lower quantiles with a running min over steps (guards against small-sample artefacts)."""
    q = q.copy()
    for col in q.columns:
        level = float(str(col).lstrip("q")) / (100 if str(col).startswith("q") else 1)
        if level > 0.5:
            q[col] = q[col].cummax()
        elif level < 0.5:
            q[col] = q[col].cummin()
    return q
