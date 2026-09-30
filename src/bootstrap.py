"""Bootstrap mean interval. The resampling loop from _scratch/thesis_diag_modal.py `_bootstrap_a8`.

That routine also computed LPIPS, SSIM, and a 10-subset FID. Those steps stayed
in the Modal diagnostic and are not wired here. This function is only the
resample of per-tile means: numpy Generator.integers, n_boot draws, percentiles
2.5 and 97.5.
"""
from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    arrays: dict[str, np.ndarray],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    keys = list(arrays)
    if not keys:
        return {}
    n = len(np.asarray(arrays[keys[0]]))
    boot = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for k in keys:
            boot[k].append(float(np.asarray(arrays[k], dtype=np.float64)[idx].mean()))
    ci = {}
    for k in keys:
        b = np.asarray(boot[k])
        arr = np.asarray(arrays[k], dtype=np.float64)
        ci[k] = {
            "mean": float(arr.mean()),
            "ci95_lo": float(np.percentile(b, 2.5)),
            "ci95_hi": float(np.percentile(b, 97.5)),
        }
    return ci
