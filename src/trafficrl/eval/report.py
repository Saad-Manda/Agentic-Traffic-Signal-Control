"""Collect finished runs into tables / curves for reports (Phase 5 outputs, plan D10).

Pure readers of ``results/``: they never run a simulation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

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
