"""Agents sharing one interface: baselines and the proposed UG-MARL."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trafficrl.agents.base import Agent, RuleAgent
from trafficrl.agents.baselines.dqn import DQNAgent
from trafficrl.agents.baselines.fixed_time import FixedTimeAgent
from trafficrl.agents.baselines.idqn import IDQNAgent
from trafficrl.agents.baselines.max_pressure import MaxPressureAgent

AGENTS = ("fixed_time", "max_pressure", "dqn", "idqn")
LEARNED_AGENTS = ("dqn", "idqn")


def make_agent(name: str, env: Any, cfg: Mapping[str, Any]) -> Agent:
    """Build agent ``name`` for ``env``.

    Parameters come from ``cfg['agents'][name]``; learned agents also read the ``train:``
    block (learning hyperparameters) and ``seed`` (exploration / replay RNG streams).
    """
    params = dict((cfg.get("agents", {}) or {}).get(name, {}) or {})
    if name == "fixed_time":
        return FixedTimeAgent(env.phase_ids, env.delta_t, **params)
    if name == "max_pressure":
        return MaxPressureAgent(env.features, **params)
    if name in LEARNED_AGENTS:
        learn = {**dict(cfg.get("train", {}) or {}), **params}
        cls = DQNAgent if name == "dqn" else IDQNAgent
        return cls(env, learn, seed=int(cfg.get("seed", 0)))
    raise ValueError(f"unknown agent {name!r}; one of {AGENTS}")


__all__ = [
    "AGENTS",
    "LEARNED_AGENTS",
    "Agent",
    "DQNAgent",
    "FixedTimeAgent",
    "IDQNAgent",
    "MaxPressureAgent",
    "RuleAgent",
    "make_agent",
]
