"""Multi-agent RL baseline: independent DQN learners (ROADMAP 5.3).

One Double-DQN network **and** one replay buffer per intersection (Tan 1993 independent
learners; the IDQN baseline of CoLight / PressLight). The 12 networks are stored as one
:class:`~trafficrl.agents.core.StackedMLPQNet` and evaluated in a single batched call; their
parameters are independent. Each intersection acts on its own 44-value state only.

IDQN is tied to the training network's intersections (one network per intersection), so it
cannot be applied zero-shot to a city with a different layout (Hangzhou, 7.4).
"""

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
