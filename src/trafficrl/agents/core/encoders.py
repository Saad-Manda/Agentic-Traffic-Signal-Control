"""State encoders from env observations to network input."""

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


class NeighborhoodEncoder:
    """Stack own observation and the four neighbours (E, N, W, S) with a presence flag.

    Output per intersection is ``(5, obs_dim + 1)``; absent neighbours are zero rows.
    """

    SLOTS = ("self", "E", "N", "W", "S")

    def __init__(
        self,
        intersection_ids: Sequence[str],
        obs_dim: int,
        index: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        self.intersection_ids = list(intersection_ids)
        n = len(self.intersection_ids)
        self.index = np.asarray(index, dtype=np.int64).reshape(n, 4)
        self.mask = np.asarray(mask, dtype=bool).reshape(n, 4)
        self.obs_dim = int(obs_dim)
        self.state_shape: tuple[int, ...] = (5, obs_dim + 1)
        self._pos = {iid: k for k, iid in enumerate(self.intersection_ids)}

    @classmethod
    def from_env(cls, env) -> NeighborhoodEncoder:
        index, mask = env.neighbor_index()
        return cls(env.intersection_ids, env.obs_dim, index, mask)

    def _rows(self, own: np.ndarray, k: np.ndarray | slice) -> np.ndarray:
        """Encode intersections ``k`` given ``own`` = all observations ``(n, obs_dim)``."""
        idx, present = self.index[k], self.mask[k]
        out = np.zeros(idx.shape[:-1] + self.state_shape, dtype=np.float32)
        out[..., 0, :-1] = own[k]
        out[..., 0, -1] = 1.0
        out[..., 1:, :-1] = own[idx] * present[..., None]
        out[..., 1:, -1] = present
        return out

    def _own(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.stack([obs[iid] for iid in self.intersection_ids]).astype(np.float32)

    def encode_all(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        """``(n_intersections, 5, obs_dim + 1)`` in ``intersection_ids`` order."""
        return self._rows(self._own(obs), slice(None))

    def encode_transition(self, tr) -> tuple[np.ndarray, np.ndarray]:
        """(state, next_state) of ``tr.iid``; ``tr.obs`` / ``tr.next_obs`` hold every
        intersection, so the neighbours are read from the same snapshot."""
        k = self._pos[tr.iid]
        return self._rows(self._own(tr.obs), k), self._rows(self._own(tr.next_obs), k)
