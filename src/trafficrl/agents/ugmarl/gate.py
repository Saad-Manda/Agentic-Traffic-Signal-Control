"""Uncertainty gate: σ(s), percentile thresholds and gate.json."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

GATE_FILE = "gate.json"


def sigma_from_q(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``q`` ``(..., K, A)`` -> mean Q ``(..., A)``, greedy index a* ``(...)``, σ ``(...)``."""
    mu = q.mean(dim=-2)
    a_star = mu.argmax(dim=-1)
    idx = a_star[..., None, None].expand(*q.shape[:-1], 1)
    q_star = q.gather(-1, idx).squeeze(-1)
    sigma = q_star.std(dim=-1, unbiased=False)
    return mu, a_star, sigma


def percentile_thresholds(sigmas: Sequence[float], percentiles: Sequence[float]) -> dict:
    """``{p: τ_p}`` with τ_p = p-th percentile (numpy linear interpolation) of ``sigmas``."""
    s = np.asarray(sigmas, dtype=float)
    if s.size == 0:
        raise ValueError("no σ values")
    return {float(p): float(np.percentile(s, p)) for p in percentiles}


def select_percentile(scores: pd.DataFrame, tie_tolerance: float) -> float:
    """``scores``: columns ``p`` and ``score`` (lower is better). Candidates within
    ``tie_tolerance`` of the best -> the **smallest** p (more fallback, the safer choice)."""
    ok = scores.dropna(subset=["score"])
    if ok.empty:
        raise ValueError("no scored percentile")
    near = ok[ok["score"] <= ok["score"].min() + tie_tolerance]
    return float(near["p"].min())


def save_gate(path: str | Path, gate: Mapping[str, Any]) -> Path:
    if "tau" not in gate:
        raise ValueError("gate file needs 'tau'")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(gate), indent=2, default=float) + "\n")
    return path


def load_gate(path: str | Path) -> dict[str, Any]:
    gate = json.loads(Path(path).read_text())
    if "tau" not in gate:
        raise ValueError(f"{path}: no 'tau'")
    return gate


def gate_file_for(checkpoint: str | Path) -> Path:
    """``results/train/<run>/checkpoints/best.pt`` -> ``results/train/<run>/gate.json``."""
    ckpt = Path(checkpoint)
    run_dir = ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent
    return run_dir / GATE_FILE
