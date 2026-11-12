"""Lock manifest of the models selected on validation (plan phase7 P8).

``models/manifest.json`` lists, for every learned model and seed, the SHA-256 of the files a
final evaluation reads: ``checkpoints/best.pt``, the run's ``config.yaml`` and, for gated
UG-MARL variants, ``gate.json`` — plus τ, p and the best validation score for reference. It is
written and committed **before** the test flow is opened; ``scripts/robustness.py --final``
verifies it, so a checkpoint or τ changed afterwards cannot be evaluated silently.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

MODELS = ("dqn", "idqn", "ugmarl", "ugmarl-noattn", "ugmarl-noens")
GATED = ("ugmarl", "ugmarl-noattn")  # variants that need a gate.json
FILES = ("checkpoints/best.pt", "config.yaml")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_name(model: str, tag: str, seed: int) -> str:
    return f"{model}_{tag}_s{seed}"


def build_manifest(
    train_dir: Path, models: Sequence[str], seeds: Sequence[int], tag: str = "final"
) -> dict[str, Any]:
    """Hash the locked files; raises ``FileNotFoundError`` if one is missing."""
    runs: dict[str, Any] = {}
    for model in models:
        for seed in seeds:
            name = run_name(model, tag, int(seed))
            d = train_dir / name
            names = [*FILES, *(["gate.json"] if model in GATED else [])]
            missing = [n for n in names if not (d / n).is_file()]
            if missing:
                raise FileNotFoundError(f"{name}: missing {missing}")
            entry: dict[str, Any] = {
                "model": model,
                "seed": int(seed),
                "files": {n: sha256(d / n) for n in names},
            }
            if model in GATED:
                gate = json.loads((d / "gate.json").read_text())
                entry.update(tau=gate["tau"], p=gate.get("p"))
            summary = d / "summary.json"
            if summary.is_file():
                s = json.loads(summary.read_text())
                entry.update(
                    best_val_avg_travel_time_s=s.get("best_val_avg_travel_time_s"),
                    best_episode=s.get("best_episode"),
                    params=s.get("params"),
                )
            runs[name] = entry
    return {"tag": tag, "models": list(models), "seeds": [int(s) for s in seeds], "runs": runs}


def verify_manifest(manifest: dict[str, Any], train_dir: Path) -> list[str]:
    """Problems found (empty list = every locked file is present and unchanged)."""
    problems = []
    for name, entry in manifest.get("runs", {}).items():
        for rel, digest in entry["files"].items():
            f = train_dir / name / rel
            if not f.is_file():
                problems.append(f"{name}/{rel}: missing")
            elif sha256(f) != digest:
                problems.append(f"{name}/{rel}: changed since the lock")
    return problems


def uncovered(manifest: dict[str, Any], run_names: Sequence[str]) -> list[str]:
    """Training runs a study needs that the manifest does not list."""
    return [n for n in run_names if n not in manifest.get("runs", {})]
