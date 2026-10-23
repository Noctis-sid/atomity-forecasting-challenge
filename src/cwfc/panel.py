"""Global ("pooled") VM-level model, aggregated bottom-up.

The aggregate series has only ~110 training points, which is too few for a flexible ML model.
But every one of those points is the mean of ~10,000 VM series. We therefore train ONE model
across all VMs that predicts each VM's ``avg_cpu`` ``h`` steps ahead from that VM's own recent
history (direct multi-horizon strategy, ``h`` is a feature), then average the VM forecasts.

Leakage rules
-------------
* Only the VM x time matrix *up to the forecast origin* is ever passed in.
* Every training row uses features computed at an anchor ``a`` and a target at ``a + h``,
  with ``a + h`` strictly before the origin.
* Static metadata used as features: ``vm_category``, ``virtual_core_bucket``,
  ``memory_gb_bucket`` (known when a VM is provisioned). The metadata CPU statistics
  (``avg_cpu``, ``max_cpu``, ``p95_max_cpu``) and ``timestamp_vm_deleted`` summarise the VM's
  whole 30-day life, i.e. the future, and are deliberately NOT used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEASON = 12
LOOKBACK = 24
CORE_ORDER = {"2": 2, "4": 4, "8": 8, "24": 24, ">24": 32}
MEM_ORDER = {"2": 2, "4": 4, "8": 8, "32": 32, "64": 64, ">64": 96}
CAT_ORDER = {"Delay-insensitive": 0, "Interactive": 1, "Unknown": 2}

FEATURES = [
    "h", "target_phase", "last", "lag1", "lag2", "m3", "m6", "m12", "m24", "mall",
    "s12", "seas1", "seas_mean", "slope12", "ewm10", "ewm30", "ewm60", "cat", "cores", "mem",
]
EWM_ALPHAS = (0.1, 0.3, 0.6)
BASE_FEATURE = "ewm30"  # the model learns a correction to this per-VM smoothed level


def static_features(vm_ids: pd.Index, metadata: pd.DataFrame | None) -> np.ndarray:
    """(n_vm, 3) array of category / core / memory codes; -1 when unknown."""
    out = np.full((len(vm_ids), 3), -1.0)
    if metadata is None:
        return out
    m = metadata.drop_duplicates("vm_id").set_index("vm_id").reindex(vm_ids)
    if "vm_category" in m:
        out[:, 0] = m["vm_category"].map(CAT_ORDER).fillna(-1).to_numpy()
    if "virtual_core_bucket" in m:
        out[:, 1] = m["virtual_core_bucket"].astype(str).map(CORE_ORDER).fillna(-1).to_numpy()
    if "memory_gb_bucket" in m:
        out[:, 2] = m["memory_gb_bucket"].astype(str).map(MEM_ORDER).fillna(-1).to_numpy()
    return out


def _ffill_rows(X: np.ndarray) -> np.ndarray:
    return pd.DataFrame(X).ffill(axis=1).to_numpy()


def _window_mean(cs: np.ndarray, a: int, w: int) -> np.ndarray:
    lo = max(a + 1 - w, 0)
    return (cs[:, a + 1] - cs[:, lo]) / (a + 1 - lo)


def _ewm_matrices(Xf: np.ndarray) -> list[np.ndarray]:
    """Per-VM exponentially weighted means; column t uses slots <= t only."""
    out = []
    X0 = np.where(np.isfinite(Xf), Xf, np.nan)
    first = pd.DataFrame(X0).bfill(axis=1).to_numpy()[:, 0]
    for alpha in EWM_ALPHAS:
        E = np.empty_like(X0)
        level = np.nan_to_num(first)
        for t in range(X0.shape[1]):
            v = X0[:, t]
            level = np.where(np.isfinite(v), alpha * v + (1 - alpha) * level, level)
            E[:, t] = level
        out.append(E)
    return out


def _features_at(Xf: np.ndarray, cs: np.ndarray, cs2: np.ndarray, a: int, h: int, static: np.ndarray,
                 ewm: list[np.ndarray]) -> np.ndarray:
    """Features for every VM at anchor ``a`` (last known slot) for step ``h``."""
    n = Xf.shape[0]
    last = Xf[:, a]
    m12 = _window_mean(cs, a, 12)
    sq12 = _window_mean(cs2, a, 12)
    s12 = np.sqrt(np.maximum(sq12 - m12**2, 0))
    k = int(np.ceil(h / SEASON))
    i1 = a + h - k * SEASON
    seas1 = Xf[:, i1] if i1 >= 0 else last
    idx = [a + h - (k + j) * SEASON for j in range(4)]
    idx = [i for i in idx if i >= 0] or [a]
    seas_mean = np.nanmean(Xf[:, idx], axis=1)
    lo = max(a - 11, 0)
    slope12 = (Xf[:, a] - Xf[:, lo]) / max(a - lo, 1)
    cols = [
        np.full(n, h, float),
        np.full(n, (a + h) % SEASON, float),
        last,
        Xf[:, max(a - 1, 0)],
        Xf[:, max(a - 2, 0)],
        _window_mean(cs, a, 3),
        _window_mean(cs, a, 6),
        m12,
        _window_mean(cs, a, 24),
        cs[:, a + 1] / (a + 1),
        s12,
        seas1,
        seas_mean,
        slope12,
        *[E[:, a] for E in ewm],
        static[:, 0],
        static[:, 1],
        static[:, 2],
    ]
    return np.column_stack(cols)


@dataclass
class PanelForecaster:
    learner: str = "gbm"  # "gbm" or "ridge"
    max_vms: int = 2500
    anchor_stride: int = 2
    seed: int = 0

    def _model(self):
        if self.learner == "ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
            l2_regularization=1.0, random_state=self.seed,
        )

    @staticmethod
    def min_history(max_step: int) -> int:
        return LOOKBACK + max_step + 8

    def forecast(self, X: np.ndarray, static: np.ndarray, steps, return_vm: bool = False):
        """X: (n_vm, T) history matrix ending at the origin. Returns aggregate forecasts."""
        steps = np.asarray(steps, dtype=int)
        n_vm, T = X.shape
        H = int(steps.max())
        if T < self.min_history(H):
            raise ValueError(f"panel model needs >= {self.min_history(H)} history slots, got {T}")
        Xf = _ffill_rows(X)
        observed = np.isfinite(X)
        Xf0 = np.nan_to_num(Xf, nan=0.0)
        cs = np.concatenate([np.zeros((n_vm, 1)), np.cumsum(Xf0, axis=1)], axis=1)
        cs2 = np.concatenate([np.zeros((n_vm, 1)), np.cumsum(Xf0**2, axis=1)], axis=1)
        ewm = _ewm_matrices(Xf)
        bi = FEATURES.index(BASE_FEATURE)

        rng = np.random.default_rng(self.seed)
        train_vms = np.arange(n_vm)
        if n_vm > self.max_vms:
            train_vms = np.sort(rng.choice(n_vm, self.max_vms, replace=False))

        rows, ys = [], []
        for h in range(1, H + 1):
            anchors = np.arange(T - 1 - h, LOOKBACK - 2, -self.anchor_stride)
            for a in anchors:
                tgt_ok = observed[train_vms, a + h] & np.isfinite(Xf[train_vms, a])
                if not tgt_ok.any():
                    continue
                v = train_vms[tgt_ok]
                F = _features_at(Xf[v], cs[v], cs2[v], a, h, static[v], [E[v] for E in ewm])
                rows.append(F)
                ys.append(X[v, a + h] - F[:, bi])
        Xtr = np.nan_to_num(np.vstack(rows), nan=-1.0)
        ytr = np.concatenate(ys)
        model = self._model().fit(Xtr, ytr)

        # Forecast: VMs seen in the last hour are assumed to persist over the horizon.
        recent = observed[:, max(T - SEASON, 0):].any(axis=1)
        a = T - 1
        preds = np.empty(len(steps))
        vm_preds = np.full((n_vm, len(steps)), np.nan)
        for j, h in enumerate(steps):
            F = np.nan_to_num(_features_at(Xf[recent], cs[recent], cs2[recent], a, int(h), static[recent],
                                           [E[recent] for E in ewm]), nan=-1.0)
            p = np.clip(model.predict(F) + F[:, bi], 0, 100)
            vm_preds[recent, j] = p
            preds[j] = p.mean()
        self.model_ = model
        self.n_train_rows_ = len(ytr)
        return (preds, vm_preds) if return_vm else preds
