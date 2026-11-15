"""Seeding for reproducible runs."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch; optionally force deterministic kernels."""
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError(f"seed must be an int in [0, 2**32), got {seed!r}")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def seed_cityflow(engine: Any, seed: int) -> None:
    """Seed a ``cityflow.Engine`` instance (vehicle generation / lane-change randomness)."""
    engine.set_random_seed(seed)


def make_rng(seed: int) -> np.random.Generator:
    """Return an independent NumPy Generator, e.g. for noise injection or ε-greedy draws.

    Prefer explicit generators over the global NumPy state inside library code.
    """
    return np.random.default_rng(seed)
