import numpy as np

from cwfc import baselines as B


def test_seasonal_naive_steps_uses_same_phase():
    h = np.arange(24, dtype=float)
    assert list(B.seasonal_naive_steps(h, np.array([1, 12, 13]), season=12)) == [12.0, 23.0, 12.0]


def test_ses_profile_recovers_flat_level_plus_cycle():
    rng = np.random.default_rng(0)
    cycle = np.tile([0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0], 10).astype(float)
    y = 10 + cycle + rng.normal(0, 0.05, cycle.size)
    steps = np.arange(1, 25)
    f = B.ses_profile_steps(y, steps)
    truth = 10 + np.tile(cycle[:12], 12)[cycle.size % 12:][:24]
    assert np.abs(f - truth).mean() < 0.15
    assert np.abs(f - truth).mean() < np.abs(B.persistence_steps(y, steps) - truth).mean()


def test_baselines_handle_missing_history():
    h = np.array([1.0, np.nan, 3.0, np.nan])
    assert B.persistence_steps(h, np.arange(1, 3)).tolist() == [3.0, 3.0]
