import numpy as np
import pandas as pd

from cwfc.backtest import origins
from cwfc.data import load_dataset, split_holdout
from cwfc.policy import newsvendor_tau
from cwfc.predict import forecast_all, main


def test_cli_on_smoke_dataset(smoke_dir, tmp_path):
    out = tmp_path / "predictions.csv"
    main(["--data", str(smoke_dir), "--output", str(out), "--artifacts", str(tmp_path / "art"), "--folds", "4"])
    p = pd.read_csv(out)
    assert list(p.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert len(p) == 24 and p.timestamp.is_monotonic_increasing and p.timestamp.is_unique
    assert np.isfinite(p.iloc[:, 1:].to_numpy()).all()
    assert (p.active_vm_count >= 0).all() and p.aggregate_cpu_utilization.between(0, 100).all()
    sp = split_holdout(load_dataset(smoke_dir).targets, 24)
    assert (p.timestamp.to_numpy() == sp.holdout_timestamps).all()


def test_holdout_values_cannot_influence_predictions(smoke_dir, tmp_path):
    """Leakage test: scrambling every reading in the holdout window must not change the forecast."""
    base = forecast_all(smoke_dir, n_folds=4, verbose=False)["predictions"]
    df = pd.read_csv(smoke_dir / "vm_cpu_readings.csv.gz")
    start = split_holdout(load_dataset(smoke_dir).targets, 24).holdout_start
    late = df.timestamp >= start
    rng = np.random.default_rng(1)
    df.loc[late, "avg_cpu"] = rng.uniform(0, 100, late.sum())
    df = df[~(late & (rng.random(len(df)) < 0.3))]  # also change which VMs are active
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    df.to_csv(tampered / "vm_cpu_readings.csv.gz", index=False)
    (tampered / "vm_metadata.csv.gz").write_bytes((smoke_dir / "vm_metadata.csv.gz").read_bytes())
    after = forecast_all(tampered, n_folds=4, verbose=False)["predictions"]
    pd.testing.assert_frame_equal(base, after)


def test_runs_without_metadata(tmp_path):
    from conftest import make_smoke
    make_smoke(tmp_path / "nometa", with_metadata=False)
    p = forecast_all(tmp_path / "nometa", n_folds=3, verbose=False)["predictions"]
    assert len(p) == 24


def test_backtest_origins_stay_inside_training():
    o = origins(110, 24, n_folds=8)
    assert max(o) + 24 <= 110 and min(o) >= 24


def test_newsvendor():
    assert newsvendor_tau(9, 1) == 0.9
