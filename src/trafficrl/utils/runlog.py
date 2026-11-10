"""Experiment log and environment record (guidelines G8, ROADMAP 5.4).

Every simulator / training / evaluation run

* appends one row to ``docs/experiment_log.csv`` (tracked, mentor-facing), and
* saves its software versions and hardware (``env.json``) next to its results.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from collections.abc import Mapping
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from trafficrl.data.splits import REPO_ROOT

EXPERIMENT_LOG = REPO_ROOT / "docs" / "experiment_log.csv"
COLUMNS = (
    "run_id",
    "date",
    "split",
    "seed",
    "model",
    "hyperparameters",
    "metric_name",
    "metric_value",
    "validation_metric",
    "runtime_s",
    "git_commit",
    "notes",
)


def git_commit(repo: Path = REPO_ROOT) -> str:
    """Short commit hash, with ``-dirty`` if the working tree has uncommitted changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{head}-dirty" if dirty else head
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def env_info() -> dict[str, Any]:
    """Software versions and hardware of the current machine."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "torch": _version("torch"),
        "cityflow": _version("CityFlow") or _version("cityflow"),
        "cuda": None,
        "gpu": None,
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "git_commit": git_commit(),
    }
    try:
        import torch

        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def save_env_info(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env_info(), indent=2) + "\n")
    return path


def append_experiment_row(
    *,
    run_id: str,
    split: str,
    seed: int,
    model: str,
    hyperparameters: Mapping[str, Any],
    metric_name: str,
    metric_value: float | None,
    runtime_s: float,
    validation_metric: float | None = None,
    notes: str = "",
    path: Path = EXPERIMENT_LOG,
) -> dict[str, Any]:
    """Append one run to the experiment log (the header is written if the file is new).

    ``validation_metric`` is filled only for runs evaluated on the validation split;
    ``metric_name`` / ``metric_value`` hold the run's own headline metric on ``split``.
    """
    row = {
        "run_id": run_id,
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "split": split,
        "seed": seed,
        "model": model,
        "hyperparameters": json.dumps(dict(hyperparameters), sort_keys=True),
        "metric_name": metric_name,
        "metric_value": "" if metric_value is None else metric_value,
        "validation_metric": "" if validation_metric is None else validation_metric,
        "runtime_s": round(runtime_s, 2),
        "git_commit": git_commit(),
        "notes": notes,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row
