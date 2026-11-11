"""Single-agent RL baseline: shared-parameter DQN (ROADMAP 5.2; Decision Log 27 Oct).

One Double-DQN network (MLP 44 → 128 → 128 → 8) and one replay buffer shared by every
intersection; each intersection acts on its **own** 44-value state only — no coordination
signal. Same Q-learning core as IDQN and UG-MARL (:class:`~trafficrl.agents.core.QLearner`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trafficrl.agents.core import LocalEncoder, MLPQNet, QLearner


class DQNAgent(QLearner):
    name = "dqn"

    def __init__(self, env: Any, params: Mapping[str, Any] | None = None, seed: int = 0):
        p = dict(params or {})
        hidden = tuple(p.get("hidden", (128, 128)))
        net = MLPQNet(env.obs_dim, len(env.phase_ids), hidden)
        super().__init__(
            net,
            LocalEncoder(env.intersection_ids, env.obs_dim),
            env.intersection_ids,
            env.phase_ids,
            env.delta_t,
            shared=True,
            params=p,
            seed=seed,
        )
