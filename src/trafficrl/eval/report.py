"""Collect finished runs into tables / curves for reports (Phase 5 outputs, plan D10).

Pure readers of ``results/``: they never run a simulation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from trafficrl.utils.config import apply_overrides
from trafficrl.utils.stats import summarise, t975


def latest_eval_run(eval_dir: Path, agent: str, dataset: str, split: str) -> Path:
    """Most recent ``results/eval/<agent>_<dataset>-<split>_clean_<stamp>`` directory."""
    runs = sorted(eval_dir.glob(f"{agent}_{dataset}-{split}_clean_*"))
    if not runs:
        raise FileNotFoundError(f"no clean {dataset}/{split} evaluation of {agent} in {eval_dir}")
    return runs[-1]


def training_summaries(train_dir: Path, agent: str, tag: str, seeds: Sequence[int]) -> pd.DataFrame:
    rows = []
    for s in seeds:
        f = train_dir / f"{agent}_{tag}_s{s}" / "summary.json"
        if f.exists():
            rows.append(json.loads(f.read_text()))
    return pd.DataFrame(rows)


def val_curves(train_dir: Path, agent: str, tag: str, seeds: Sequence[int]) -> pd.DataFrame:
    """Per training episode: mean, 95 % CI (Student t) and n of the val travel time over seeds."""
    frames = []
    for s in seeds:
        f = train_dir / f"{agent}_{tag}_s{s}" / "val_log.csv"
        if f.exists():
            df = pd.read_csv(f)[["episode", "avg_travel_time_s"]]
            frames.append(df.assign(seed=s))
    if not frames:
        raise FileNotFoundError(f"no val logs for {agent}_{tag}_s* in {train_dir}")
    allv = pd.concat(frames)
    rows = []
    for ep, g in allv.groupby("episode"):
        v = g["avg_travel_time_s"].to_numpy(float)
        n = len(v)
        sd = float(v.std(ddof=1)) if n > 1 else 0.0
        half = t975(n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
        rows.append(
            {
                "episode": int(ep),
                "mean": v.mean(),
                "std": sd,
                "ci95_low": v.mean() - half,
                "ci95_high": v.mean() + half,
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def fmt(stats: dict, digits: int = 1) -> str:
    """``mean ± std [low, high]`` or ``value (deterministic)``."""
    if stats["deterministic"]:
        return f"{stats['mean']:.{digits}f} (deterministic)"
    return (
        f"{stats['mean']:.{digits}f} ± {stats['std']:.{digits}f} "
        f"[{stats['ci95_low']:.{digits}f}, {stats['ci95_high']:.{digits}f}]"
    )


def metric_stats(metrics_csv: Path, metric: str) -> dict:
    return summarise(pd.read_csv(metrics_csv)[metric].tolist())


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


# --- model × condition studies (scripts/robustness.py, scripts/final_tables.py) -------------


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


def study_runs(spec: Mapping, study_dir: Path) -> dict[tuple[str, str], Path | None]:
    """``{(model, condition): run directory or None}`` in the spec's table order."""
    return {
        (model, condition_name(dict(c or {}))): finished_run(
            Path(study_dir) / "eval" / f"{model}__{condition_name(dict(c or {}))}"
        )
        for model in spec["models"]
        for c in spec["conditions"]
    }


def load_study(spec: Mapping, study_dir: Path) -> pd.DataFrame:
    """Per-seed metrics of every finished job: ``model, condition, run_dir, seed, <metrics>``."""
    frames, absent = [], []
    for (model, cond), run in study_runs(spec, study_dir).items():
        if run is None:
            absent.append(f"{model}__{cond}")
            continue
        m = pd.read_csv(run / "metrics.csv")
        m.insert(0, "run_dir", str(run))
        m.insert(0, "condition", cond)
        m.insert(0, "model", model)
        frames.append(m)
    if absent:
        raise FileNotFoundError(f"no finished run for {absent} in {study_dir}")
    return pd.concat(frames, ignore_index=True)
