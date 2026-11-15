"""Uniform replay buffer with one ring buffer per group."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        groups: int,
        capacity: int,
        state_shape: tuple[int, ...],
        rng: np.random.Generator,
        extras: Mapping[str, tuple[tuple[int, ...], type | np.dtype]] | None = None,
    ) -> None:
        if groups < 1 or capacity < 1:
            raise ValueError("groups and capacity must be >= 1")
        self.groups = groups
        self.capacity = capacity
        self.state_shape = tuple(state_shape)
        self.rng = rng
        shape = (groups, capacity)
        self.state = np.zeros(shape + self.state_shape, dtype=np.float32)
        self.next_state = np.zeros(shape + self.state_shape, dtype=np.float32)
        self.action = np.zeros(shape, dtype=np.int64)
        self.reward = np.zeros(shape, dtype=np.float32)
        self.duration = np.zeros(shape, dtype=np.float32)
        self.done = np.zeros(shape, dtype=bool)
        self.extras = {
            name: np.zeros(shape + tuple(eshape), dtype=dtype)
            for name, (eshape, dtype) in (extras or {}).items()
        }
        self.pos = np.zeros(groups, dtype=np.int64)
        self.size = np.zeros(groups, dtype=np.int64)

    def add(
        self,
        group: int,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        duration_s: float,
        done: bool,
        **extras: np.ndarray,
    ) -> None:
        if set(extras) != set(self.extras):
            raise ValueError(f"extras {sorted(extras)} != buffer columns {sorted(self.extras)}")
        k = int(self.pos[group])
        for name, value in extras.items():
            self.extras[name][group, k] = value
        self.state[group, k] = state
        self.next_state[group, k] = next_state
        self.action[group, k] = action
        self.reward[group, k] = reward
        self.duration[group, k] = duration_s
        self.done[group, k] = done
        self.pos[group] = (k + 1) % self.capacity
        self.size[group] = min(self.size[group] + 1, self.capacity)

    def __len__(self) -> int:
        """Fill of the emptiest group (sampling needs every group non-empty)."""
        return int(self.size.min())

    @property
    def total(self) -> int:
        return int(self.size.sum())

    def sample(self, batch: int) -> dict[str, np.ndarray]:
        if len(self) == 0:
            raise ValueError("cannot sample: a group is empty")
        idx = np.stack([self.rng.integers(0, self.size[g], size=batch) for g in range(self.groups)])
        g = np.arange(self.groups)[:, None]
        out = {
            "state": self.state[g, idx],
            "action": self.action[g, idx],
            "reward": self.reward[g, idx],
            "next_state": self.next_state[g, idx],
            "duration": self.duration[g, idx],
            "done": self.done[g, idx],
        }
        out.update({name: col[g, idx] for name, col in self.extras.items()})
        return out
