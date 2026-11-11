"""Random-search tuning and multi-seed runs (ROADMAP 5.5; plan phase5-rl-baselines D8).

The search space lives in ``configs/search/*.yaml`` and is committed before the first trial.
Each trial is one ``scripts/train.py`` process (train flow, checkpoint on val); trials run
as parallel processes, one simulator thread and one torch thread each. Test / held-out flows
are never touched: ``train.py`` accepts only the train split.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trafficrl.data.splits import REPO_ROOT
from trafficrl.utils.config import read_yaml

TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


def load_search(path: str | Path) -> dict[str, Any]:
    spec = read_yaml(path)
    for key in ("study", "seed", "sampler_seed", "trials", "space", "selection"):
        if key not in spec:
            raise ValueError(f"{path}: missing key {key!r}")
    return spec


def sample_trials(space: Mapping[str, Sequence[Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """``n`` distinct points of the grid ``space``, drawn uniformly without replacement."""
    keys = list(space)
    grid = list(itertools.product(*(space[k] for k in keys)))
    if n > len(grid):
        raise ValueError(f"{n} trials requested, the space has {len(grid)} points")
    pick = np.random.default_rng(seed).choice(len(grid), size=n, replace=False)
    return [dict(zip(keys, grid[i], strict=True)) for i in pick]


def overrides(params: Mapping[str, Any]) -> list[str]:
    return [f"{k}={v}" for k, v in params.items()]


def select_best(trials: pd.DataFrame, metric: str, tie_tolerance: float) -> pd.Series:
    """Lowest ``metric``; trials within ``tie_tolerance`` of it -> lowest ``train.lr``."""
    ok = trials.dropna(subset=[metric])
    if ok.empty:
        raise ValueError("no finished trial")
    near = ok[ok[metric] <= ok[metric].min() + tie_tolerance]
    return near.sort_values(["train.lr", metric], kind="stable").iloc[0]


def run_parallel(
    jobs: Sequence[tuple[str, list[str]]], workers: int, log_dir: Path
) -> dict[str, int]:
    """Run ``(name, argv)`` jobs, ``workers`` at a time; stdout+stderr -> ``log_dir/name.log``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    pending, running, codes = list(jobs), {}, {}
    while pending or running:
        while pending and len(running) < workers:
            name, argv = pending.pop(0)
            log = (log_dir / f"{name}.log").open("w")
            running[name] = (
                subprocess.Popen(
                    argv, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT, env=env
                ),
                log,
            )
            print(f"start  {name}", flush=True)
        for name, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                codes[name] = proc.returncode
                del running[name]
                print(f"finish {name} (exit {proc.returncode})", flush=True)
        if running:
            try:
                next(iter(running.values()))[0].wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
    return codes


def train_argv(config: Path, run_name: str, extra: Sequence[str], note: str) -> list[str]:
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(config),
        "--run-name",
        run_name,
        "--quiet",
        "--note",
        note,
        *extra,
    ]


def collect(results_dir: Path, runs: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """One row per run: its parameters + summary.json (NaN metrics if the run failed)."""
    rows = []
    for name, params in runs.items():
        f = results_dir / name / "summary.json"
        s = json.loads(f.read_text()) if f.exists() else {}
        rows.append(
            {
                "run_name": name,
                **params,
                "best_val_avg_travel_time_s": s.get("best_val_avg_travel_time_s"),
                "best_episode": s.get("best_episode"),
                "last_val_avg_travel_time_s": s.get("last_val_avg_travel_time_s"),
                "training_time_s": s.get("training_time_s"),
                "params": s.get("params"),
            }
        )
    return pd.DataFrame(rows)
