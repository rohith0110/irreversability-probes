"""Statistics shared by every analysis (brief 7.4)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import binomtest

N_BOOT = 1000


def bootstrap_ci(metric: Callable[[np.ndarray], float], n: int, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of ``metric(indices)`` over 1,000 resamples of n items."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        value = metric(idx)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test for paired binary outcomes a and b (same tasks)."""
    a, b = a.astype(bool), b.astype(bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return {"a_only": only_a, "b_only": only_b, "p_value": float(p)}


def rate_with_ci(values: np.ndarray, seed: int = 0) -> dict[str, float]:
    values = values.astype(float)
    if len(values) == 0:
        return {"rate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    low, high = bootstrap_ci(lambda idx: float(values[idx].mean()), len(values), seed)
    return {"rate": float(values.mean()), "ci_low": low, "ci_high": high, "n": int(len(values))}
