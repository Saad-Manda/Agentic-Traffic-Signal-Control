"""Seeding for reproducible runs (ROADMAP 0.3).

Every run is driven by one integer seed from its config. ``seed_everything`` seeds Python,
NumPy and PyTorch (CPU + CUDA) and, by default, switches PyTorch to deterministic kernels so
that the same config + seed reproduces the same run on the same machine and software stack.

CityFlow has its own RNG: the env wrapper passes the same seed in the engine config (``"seed"``
key) and/or calls ``seed_cityflow(engine, seed)`` after creating the engine.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

# Required by cuBLAS for deterministic matmuls on CUDA >= 10.2. Must be set before the first
# CUDA call, so it is set at import time (without overriding a user-provided value).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch; optionally force deterministic PyTorch kernels.

    Args:
        seed: Non-negative integer seed (fits in 32 bits for NumPy compatibility).
        deterministic: If True, enable ``torch.use_deterministic_algorithms`` and the cuDNN
            deterministic flags. Ops without a deterministic implementation then raise
            instead of silently producing run-to-run noise.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError(f"seed must be an int in [0, 2**32), got {seed!r}")

    random.seed(seed)
    np.random.seed(seed)
    # PYTHONHASHSEED only affects interpreters started after this point (e.g. worker
    # subprocesses); record it so those inherit the run seed.
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:  # torch-free tools (e.g. data audit) can still seed
        return

    torch.manual_seed(seed)  # also seeds all CUDA devices
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
