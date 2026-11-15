"""Helpers that read finished evaluation runs (they never run a simulation)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from trafficrl.utils.config import apply_overrides
from trafficrl.utils.stats import summarise


def fmt(stats: dict, digits: int = 1) -> str:
    """``mean ± std [low, high]`` or ``value (deterministic)``."""
    if stats["deterministic"]:
        return f"{stats['mean']:.{digits}f} (deterministic)"
    return (
        f"{stats['mean']:.{digits}f} ± {stats['std']:.{digits}f} "
        f"[{stats['ci95_low']:.{digits}f}, {stats['ci95_high']:.{digits}f}]"
    )


def condition_table(runs: Mapping[tuple[str, str], Path]) -> pd.DataFrame:
    """One row per (model, condition) from finished evaluation runs (``scripts/robustness.py``).

    ``runs`` maps ``(model, condition)`` to a run directory holding ``metrics.csv`` (one row
    per seed). Travel time: mean, std, 95 % CI over seeds; max wait and fallback rate: mean.
    """
    rows = []
    for (model, condition), run in runs.items():
        m = pd.read_csv(Path(run) / "metrics.csv")
        att = summarise(m["avg_travel_time_s"].tolist())
        rows.append(
            {
                "model": model,
                "condition": condition,
                "avg_travel_time_mean": att["mean"],
                "avg_travel_time_std": att["std"],
                "avg_travel_time_ci95_low": att["ci95_low"],
                "avg_travel_time_ci95_high": att["ci95_high"],
                "deterministic": att["deterministic"],
                "max_waiting_time_mean": float(m["max_waiting_time_s"].mean()),
                "fallback_rate_mean": float(m["fallback_rate"].mean()),
                "n_seeds": att["n"],
            }
        )
    return pd.DataFrame(rows)


def condition_name(cond: Mapping) -> str:
    """Explicit ``name``, else the label of the ``robustness.*`` keys (``clean``, ``noise0.1``)."""
    from trafficrl.eval.runner import condition_label

    if cond.get("name"):
        return str(cond["name"])
    overrides = [f"{k}={v}" for k, v in cond.items() if k != "name"]
    return condition_label(apply_overrides({"robustness": {}}, overrides))


def finished_run(job_dir: Path) -> Path | None:
    """Latest run directory under ``job_dir`` that holds a ``metrics.csv``."""
    runs = sorted(p for p in Path(job_dir).glob("*") if (p / "metrics.csv").exists())
    return runs[-1] if runs else None
