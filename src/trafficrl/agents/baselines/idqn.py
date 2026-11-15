"""Multi-agent baseline: one independent DQN per intersection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trafficrl.agents.core import LocalEncoder, QLearner, StackedMLPQNet


class IDQNAgent(QLearner):
    name = "idqn"

    def __init__(self, env: Any, params: Mapping[str, Any] | None = None, seed: int = 0):
        p = dict(params or {})
        hidden = tuple(p.get("hidden", (128, 128)))
        n = len(env.intersection_ids)
        net = StackedMLPQNet(n, env.obs_dim, len(env.phase_ids), hidden)
        super().__init__(
            net,
            LocalEncoder(env.intersection_ids, env.obs_dim),
            env.intersection_ids,
            env.phase_ids,
            env.delta_t,
            shared=False,
            params=p,
            seed=seed,
        )
