"""Loading, validation and target construction.

Design rules
------------
* A *valid* CPU observation is a row whose ``avg_cpu`` is finite and inside [0, 100].
* Duplicate ``(vm_id, timestamp)`` rows are collapsed to one (mean of the valid readings),
  so a VM is counted at most once per timestamp.
* The target series lives on a regular 5-minute grid spanning the first to the last observed
  timestamp. Grid slots with no valid readings are kept as explicit rows:
  ``active_vm_count = 0`` (that is literally what was observed) and
  ``aggregate_cpu_utilization = NaN`` (we do not invent a mean of nothing).
* The holdout is the final ``horizon`` *observed* timestamps. Everything before the first
  holdout timestamp is the training window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CADENCE_S = 300
TARGETS = ("active_vm_count", "aggregate_cpu_utilization")
CPU_COLS = ("min_cpu", "max_cpu", "avg_cpu")


@dataclass
class QualityReport:
    rows_raw: int = 0
    rows_invalid: int = 0
    rows_duplicate: int = 0
    n_vms: int = 0
    n_timestamps_observed: int = 0
    n_grid_slots: int = 0
    n_empty_grid_slots: int = 0
    off_grid_timestamps: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _find(data_dir: Path, stem: str) -> Path | None:
    for name in (f"{stem}.csv.gz", f"{stem}.csv"):
        p = data_dir / name
        if p.exists():
            return p
    return None


def load_readings(data_dir: str | Path) -> pd.DataFrame:
    path = _find(Path(data_dir), "vm_cpu_readings")
    if path is None:
        raise FileNotFoundError(f"vm_cpu_readings.csv(.gz) not found in {data_dir}")
    df = pd.read_csv(path, dtype={"vm_id": str})
    missing = {"timestamp", "vm_id", "avg_cpu"} - set(df.columns)
    if missing:
        raise ValueError(f"readings file is missing columns: {sorted(missing)}")
    return df


def load_metadata(data_dir: str | Path) -> pd.DataFrame | None:
    """Metadata is optional for the pipeline (segment features are skipped without it)."""
    path = _find(Path(data_dir), "vm_metadata")
    if path is None:
        return None
    return pd.read_csv(path, dtype={"vm_id": str})


def clean_readings(df: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    rep = QualityReport(rows_raw=len(df))
    out = df.copy()
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    for c in CPU_COLS:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    valid = out["timestamp"].notna() & out["vm_id"].notna() & np.isfinite(out["avg_cpu"])
    valid &= out["avg_cpu"].between(0, 100)
    rep.rows_invalid = int((~valid).sum())
    out = out[valid]
    out["timestamp"] = out["timestamp"].astype(np.int64)

    dup = out.duplicated(["vm_id", "timestamp"], keep=False)
    rep.rows_duplicate = int(out.duplicated(["vm_id", "timestamp"]).sum())
    if dup.any():
        cols = [c for c in CPU_COLS if c in out]
        out = out.groupby(["vm_id", "timestamp"], as_index=False)[cols].mean()
        rep.notes.append(f"collapsed {rep.rows_duplicate} duplicate (vm_id, timestamp) rows by mean")
    out = out.sort_values(["timestamp", "vm_id"]).reset_index(drop=True)

    if out.empty:
        raise ValueError("no valid CPU readings after cleaning")
    t0 = int(out["timestamp"].min())
    off = (out["timestamp"] - t0) % CADENCE_S != 0
    rep.off_grid_timestamps = int(out.loc[off, "timestamp"].nunique())
    if off.any():
        # Snap to the 5-minute slot that *contains* the reading (floor), never forward in time.
        out["timestamp"] = t0 + ((out["timestamp"] - t0) // CADENCE_S) * CADENCE_S
        cols = [c for c in CPU_COLS if c in out]
        out = out.groupby(["vm_id", "timestamp"], as_index=False)[cols].mean()
        rep.notes.append("snapped off-grid timestamps down to their 5-minute slot")
    rep.n_vms = int(out["vm_id"].nunique())
    rep.n_timestamps_observed = int(out["timestamp"].nunique())
    return out, rep


def build_targets(readings: pd.DataFrame, rep: QualityReport | None = None) -> pd.DataFrame:
    """Aggregate VM-level readings to the two required targets on a regular grid."""
    g = readings.groupby("timestamp")
    agg = pd.DataFrame(
        {
            "active_vm_count": g["vm_id"].nunique(),
            "aggregate_cpu_utilization": g["avg_cpu"].mean(),
        }
    )
    grid = np.arange(agg.index.min(), agg.index.max() + 1, CADENCE_S, dtype=np.int64)
    agg = agg.reindex(grid)
    agg.index.name = "timestamp"
    agg["observed"] = agg["active_vm_count"].notna()
    agg["active_vm_count"] = agg["active_vm_count"].fillna(0).astype(int)
    if rep is not None:
        rep.n_grid_slots = len(grid)
        rep.n_empty_grid_slots = int((~agg["observed"]).sum())
    return agg.reset_index()


def vm_matrix(readings: pd.DataFrame, timestamps: np.ndarray, value: str = "avg_cpu") -> pd.DataFrame:
    """Wide VM x timestamp matrix (NaN where a VM has no valid reading)."""
    wide = readings.pivot(index="vm_id", columns="timestamp", values=value)
    return wide.reindex(columns=timestamps)


@dataclass
class Split:
    """Chronological split. ``train`` never contains a holdout timestamp."""

    train: pd.DataFrame
    holdout: pd.DataFrame
    holdout_timestamps: np.ndarray
    holdout_start: int


def split_holdout(targets: pd.DataFrame, horizon: int = 24) -> Split:
    obs_ts = targets.loc[targets["observed"], "timestamp"].to_numpy()
    if len(obs_ts) <= horizon:
        raise ValueError(f"need more than {horizon} observed timestamps, got {len(obs_ts)}")
    hold_ts = obs_ts[-horizon:]
    start = int(hold_ts[0])
    train = targets[targets["timestamp"] < start].reset_index(drop=True)
    holdout = targets[targets["timestamp"].isin(hold_ts)].reset_index(drop=True)
    return Split(train=train, holdout=holdout, holdout_timestamps=hold_ts, holdout_start=start)


@dataclass
class Dataset:
    readings: pd.DataFrame
    metadata: pd.DataFrame | None
    targets: pd.DataFrame
    report: QualityReport


def load_dataset(data_dir: str | Path) -> Dataset:
    raw = load_readings(data_dir)
    readings, rep = clean_readings(raw)
    targets = build_targets(readings, rep)
    meta = load_metadata(data_dir)
    if meta is not None:
        unknown = set(readings["vm_id"].unique()) - set(meta["vm_id"])
        if unknown:
            rep.notes.append(f"{len(unknown)} VMs have no metadata row (segment features set to 'missing')")
    return Dataset(readings=readings, metadata=meta, targets=targets, report=rep)
