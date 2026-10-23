"""Score a predictions file against the public holdout. REPORTING ONLY.

    python -m cwfc.evaluate --data challenge_data --predictions predictions.csv

This module is the only place that reads holdout target values. It is not imported by the
training / prediction path and must be run after ``predictions.csv`` has been written.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import metrics as M
from .baselines import persistence_steps
from .data import TARGETS, load_dataset, split_holdout


def score(data_dir, predictions: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    ds = load_dataset(data_dir)
    sp = split_holdout(ds.targets, horizon)
    act = sp.holdout.set_index("timestamp").loc[predictions["timestamp"]]
    rows = []
    for t in TARGETS:
        y = act[t].to_numpy(dtype=float)
        p = predictions[t].to_numpy(dtype=float)
        hist = sp.train[t].where(sp.train["observed"]).to_numpy(dtype=float)
        base = persistence_steps(hist, np.arange(1, len(p) + 1))
        thr = np.quantile(y, 0.75)
        pk = y >= thr
        rows.append({
            "target": t,
            "mae": M.mae(y, p), "rmse": M.rmse(y, p), "wape_pct": M.wape(y, p),
            "bias": float(np.mean(y - p)),
            "persistence_mae": M.mae(y, base),
            "mae_vs_persistence": M.normalized_mae(y, p, base),
            "peak_mae_top25pct": M.mae(y[pk], p[pk]),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="challenge_data")
    ap.add_argument("--predictions", default="predictions.csv")
    ap.add_argument("--horizon", type=int, default=24)
    a = ap.parse_args(argv)
    print(score(a.data, pd.read_csv(a.predictions), a.horizon).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
