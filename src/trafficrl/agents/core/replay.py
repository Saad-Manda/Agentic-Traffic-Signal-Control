"""Grouped uniform replay buffer (ROADMAP 5.1; plan phase5-rl-baselines D3).

One ring buffer per *group*: the shared-parameter DQN uses a single group that receives every
intersection's transitions; IDQN uses one group per intersection. :meth:`ReplayBuffer.sample`
draws ``batch`` transitions uniformly **from each group** and returns arrays shaped
``(groups, batch, ...)``, which is the layout the Q-networks take.

Rewards are stored raw (−|P_i|); the learner applies the constant reward scale. ``duration_s``
(15 s or 20 s; shorter at the episode cut) is kept for the SMDP discount. ``done`` marks the
time-limit cut and is stored for logging only (the learner always bootstraps).
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        groups: int,
        capacity: int,
        state_shape: tuple[int, ...],
        rng: np.random.Generator,
    ) -> None:
        if groups < 1 or capacity < 1:
            raise ValueError("groups and capacity must be >= 1")
        self.groups = groups
        self.capacity = capacity  # per group
        self.state_shape = tuple(state_shape)
        self.rng = rng
        shape = (groups, capacity)
        self.state = np.zeros(shape + self.state_shape, dtype=np.float32)
        self.next_state = np.zeros(shape + self.state_shape, dtype=np.float32)
        self.action = np.zeros(shape, dtype=np.int64)
        self.reward = np.zeros(shape, dtype=np.float32)
        self.duration = np.zeros(shape, dtype=np.float32)
        self.done = np.zeros(shape, dtype=bool)
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
    ) -> None:
        k = int(self.pos[group])
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
        return {
            "state": self.state[g, idx],
            "action": self.action[g, idx],
            "reward": self.reward[g, idx],
            "next_state": self.next_state[g, idx],
            "duration": self.duration[g, idx],
            "done": self.done[g, idx],
        }
