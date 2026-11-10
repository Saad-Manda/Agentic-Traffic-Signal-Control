"""Seed-level summary statistics: mean ± std and 95 % CI (evaluation protocol §7).

The CI uses Student's t with n − 1 degrees of freedom. A small table of two-sided 97.5 %
quantiles replaces a scipy dependency (n − 1 = 1…30; beyond that the normal value 1.96).
If every value is equal (rule-based baselines are deterministic across seeds) the result is
flagged ``deterministic`` and the CI collapses to the value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
    17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}  # fmt: skip


def t975(df: int) -> float:
    if df < 1:
        raise ValueError(f"need at least 1 degree of freedom, got {df}")
    return T975.get(df, 1.96)


def summarise(values: Sequence[float]) -> dict[str, float | int | bool]:
    """``{mean, std, ci95_low, ci95_high, n, deterministic}`` of one metric over seeds."""
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        raise ValueError("no values to summarise")
    mean = sum(xs) / n
    deterministic = all(x == xs[0] for x in xs)
    if n == 1:
        std = half = math.nan
    elif deterministic:
        std = half = 0.0
    else:
        std = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
        half = t975(n - 1) * std / math.sqrt(n)
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "n": n,
        "deterministic": deterministic and n > 1,
    }
