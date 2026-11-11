"""State encoders: env observations -> network input (plan phase5-rl-baselines D2).

:class:`LocalEncoder` keeps each intersection's own 44 values (DQN, IDQN, UG-MARL
−attention). Phase 6 adds a neighbourhood encoder (own + neighbour rows + mask) with the
same two methods, so the replay buffer stores whatever ``encode_transition`` returns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


class LocalEncoder:
    def __init__(self, intersection_ids: Sequence[str], obs_dim: int) -> None:
        self.intersection_ids = list(intersection_ids)
        self.state_shape: tuple[int, ...] = (obs_dim,)

    def encode_all(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        """``(n_intersections, obs_dim)`` in ``intersection_ids`` order."""
        return np.stack([obs[iid] for iid in self.intersection_ids]).astype(np.float32)

    def encode_transition(self, tr) -> tuple[np.ndarray, np.ndarray]:
        """(state, next_state) of intersection ``tr.iid`` only."""
        return tr.obs[tr.iid], tr.next_obs[tr.iid]
