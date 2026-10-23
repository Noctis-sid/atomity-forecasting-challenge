"""End-to-end CLI: data -> targets -> split -> backtest/selection -> final forecast.

    python -m cwfc.predict --data challenge_data --output predictions.csv

Leakage guard: after the split, only ``split.train`` (timestamps strictly before the first
holdout timestamp) and readings up to the last training timestamp are passed to any model,
backtest or selection step. Holdout *timestamps* are used (they are known in advance: they are
the slots we must forecast); holdout *target values* are never read here. Scoring against the
holdout lives in ``cwfc.evaluate`` and is run only after predictions are written.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import residual_quantiles, run_backtest, select_model
from .data import CADENCE_S, TARGETS, load_dataset, split_holdout
from .models import build_history, models_for


def forecast_all(data_dir: str | Path, horizon: int = 24, n_folds: int = 16, min_train: int | None = None,
                 skip_panel: bool = False, verbose: bool = True) -> dict:
    t0 = time.time()
    ds = load_dataset(data_dir)
    split = split_holdout(ds.targets, horizon)
    hist = build_history(split.train, ds.readings, ds.metadata)
    assert hist.timestamps.max() < split.holdout_start  # leakage guard

    last_ts = int(hist.timestamps[-1])
    steps = ((split.holdout_timestamps - last_ts) // CADENCE_S).astype(int)
    max_step = int(steps.max())

    out = pd.DataFrame({"timestamp": split.holdout_timestamps})
    intervals = out.copy()
    selection, summaries, errors = {}, {}, {}
    for target in TARGETS:
        cands = models_for(target)
        if skip_panel:
            cands = [m for m in cands if "panel" not in m.name and "ensemble" not in m.name]
        bt = run_backtest(hist, cands, target, max_step, n_folds=n_folds, min_train=min_train)
        if bt.summary.empty:
            chosen = next(m for m in cands if m.name == "persistence")
            reason = "too little history for backtesting; fell back to persistence"
        else:
            chosen = select_model(bt.summary, cands)
            reason = f"lowest mean backtest MAE over {len(bt.origins)} rolling origins"
        if len(hist) < chosen.min_history(max_step):
            chosen = next(m for m in cands if m.name == "persistence")
            reason += "; insufficient history for chosen model, fell back to persistence"
        all_steps = np.arange(1, max_step + 1)
        pred_all = np.asarray(chosen.fn(hist, all_steps), dtype=float)
        pred = pred_all[steps - 1]

        if bt.errors.empty:
            q = pd.DataFrame({"step": all_steps, "q05": 0.0, "q10": 0.0, "q50": 0.0, "q90": 0.0, "q95": 0.0})
        else:
            q = residual_quantiles(bt.errors, chosen.name, max_step)
        q = q.set_index("step").reindex(steps).reset_index()

        lo, hi = (0.0, 100.0) if target == "aggregate_cpu_utilization" else (0.0, np.inf)
        pred = np.clip(pred, lo, hi)
        if target == "active_vm_count":
            pred = np.round(pred).astype(int)
        out[target] = pred
        for col in ("q05", "q10", "q50", "q90", "q95"):
            intervals[f"{target}_{col}"] = np.clip(pred + q[col].to_numpy(), lo, hi)
        intervals[target] = pred
        selection[target] = {"model": chosen.name, "description": chosen.description, "reason": reason,
                             "backtest_origins": bt.origins}
        summaries[target] = bt.summary
        errors[target] = bt.errors
        if verbose:
            print(f"[{target}] selected {chosen.name}: {reason}")
            if not bt.summary.empty:
                print(bt.summary[["model", "mae", "rmse", "wape_pct", "mae_vs_persistence", "peak_mae"]]
                      .round(4).to_string(index=False))
    if verbose:
        print(f"done in {time.time() - t0:.1f}s")
    return {"predictions": out, "intervals": intervals, "selection": selection, "summaries": summaries,
            "errors": errors, "split": split, "dataset": ds, "history": hist}


def validate_submission(df: pd.DataFrame, holdout_ts: np.ndarray) -> None:
    assert list(df.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"], df.columns
    assert len(df) == len(holdout_ts)
    assert (df["timestamp"].to_numpy() == holdout_ts).all()
    assert df["timestamp"].is_monotonic_increasing and df["timestamp"].is_unique
    vals = df[["active_vm_count", "aggregate_cpu_utilization"]].to_numpy(dtype=float)
    assert np.isfinite(vals).all()
    assert (df["active_vm_count"] >= 0).all()
    assert df["aggregate_cpu_utilization"].between(0, 100).all()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="challenge_data", help="directory with vm_cpu_readings.csv(.gz)")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--artifacts", default="artifacts", help="where backtest tables/intervals are written")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--folds", type=int, default=16, help="number of rolling backtest origins")
    ap.add_argument("--min-train", type=int, default=None, help="history length of the first backtest origin")
    ap.add_argument("--skip-panel", action="store_true", help="skip the slower VM-level ML candidates")
    a = ap.parse_args(argv)

    res = forecast_all(a.data, a.horizon, a.folds, a.min_train, a.skip_panel)
    preds = res["predictions"]
    validate_submission(preds, res["split"].holdout_timestamps)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(a.output, index=False)

    art = Path(a.artifacts)
    art.mkdir(parents=True, exist_ok=True)
    res["intervals"].to_csv(art / "prediction_intervals.csv", index=False)
    for t in TARGETS:
        res["summaries"][t].to_csv(art / f"backtest_summary_{t}.csv", index=False)
        res["errors"][t].to_csv(art / f"backtest_errors_{t}.csv", index=False)
    meta = {"selection": res["selection"], "data_quality": res["dataset"].report.as_dict(),
            "n_train_slots": int(len(res["history"])), "holdout_start": res["split"].holdout_start}
    (art / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=int))
    print(f"wrote {a.output} ({len(preds)} rows) and artifacts to {art}/")


if __name__ == "__main__":
    main()
