import numpy as np
import pandas as pd

from cwfc.data import build_targets, clean_readings, split_holdout


def _df(rows):
    return pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])


def test_targets_dedupe_invalid_and_gaps():
    raw = _df([
        (0, "a", 0, 20, 10.0), (0, "a", 0, 20, 30.0),   # duplicate -> averaged, counted once
        (0, "b", 0, 5, 2.0), (0, "c", 0, 5, np.nan),     # invalid reading -> not active
        (300, "a", 0, 5, 4.0), (300, "b", 0, 5, 120.0),  # out of range -> dropped
        (900, "a", 0, 5, 6.0),                            # slot 600 missing entirely
    ])
    clean, rep = clean_readings(raw)
    t = build_targets(clean, rep).set_index("timestamp")
    assert rep.rows_invalid == 2 and rep.rows_duplicate == 1
    assert t.loc[0, "active_vm_count"] == 2
    assert t.loc[0, "aggregate_cpu_utilization"] == (20.0 + 2.0) / 2
    assert t.loc[300, "active_vm_count"] == 1
    assert t.loc[600, "active_vm_count"] == 0 and np.isnan(t.loc[600, "aggregate_cpu_utilization"])
    assert not t.loc[600, "observed"] and rep.n_empty_grid_slots == 1


def test_split_is_chronological_and_uses_observed_timestamps():
    ts = np.arange(0, 40 * 300, 300)
    tg = pd.DataFrame({"timestamp": ts, "active_vm_count": 5, "aggregate_cpu_utilization": 1.0, "observed": True})
    tg.loc[35, "observed"] = False  # empty slot inside the final stretch
    sp = split_holdout(tg, horizon=4)
    assert list(sp.holdout_timestamps) == list(ts[[34, 36, 37, 38, 39]][-4:])
    assert sp.train["timestamp"].max() < sp.holdout_start
