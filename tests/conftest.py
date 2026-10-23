import numpy as np
import pandas as pd
import pytest


def make_smoke(path, n_vms=120, n_slots=100, seed=0, gap_slot=40, with_metadata=True):
    """Tiny synthetic dataset with the awkward cases a reviewer's smoke set might contain:
    churn (VMs starting/stopping), one fully missing grid slot, a duplicate row, an invalid
    reading, an hourly cycle, and (optionally) no metadata file."""
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_vms):
        start = rng.integers(0, 10) if v % 7 == 0 else 0
        stop = n_slots - rng.integers(0, 15) if v % 11 == 0 else n_slots
        base = rng.gamma(2, 4)
        for t in range(start, stop):
            if t == gap_slot:
                continue
            cpu = np.clip(base + 2 * (t % 12 == 5) + rng.normal(0, 1), 0, 100)
            rows.append((t * 300 + 600, f"vm{v}", max(cpu - 1, 0), min(cpu + 3, 100), cpu))
    df = pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])
    df = pd.concat([df, df.iloc[[5]], pd.DataFrame([(900, "vm1", 0, 0, np.nan)], columns=df.columns)])
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / "vm_cpu_readings.csv.gz", index=False)
    if with_metadata:
        meta = pd.DataFrame({
            "vm_id": [f"vm{v}" for v in range(n_vms - 3)],  # a few VMs deliberately lack metadata
            "vm_category": rng.choice(["Delay-insensitive", "Interactive", "Unknown"], n_vms - 3),
            "virtual_core_bucket": rng.choice(["2", "4", ">24"], n_vms - 3),
            "memory_gb_bucket": rng.choice(["2", "8", ">64"], n_vms - 3),
        })
        meta.to_csv(path / "vm_metadata.csv.gz", index=False)
    return df


@pytest.fixture
def smoke_dir(tmp_path):
    make_smoke(tmp_path / "smoke")
    return tmp_path / "smoke"
