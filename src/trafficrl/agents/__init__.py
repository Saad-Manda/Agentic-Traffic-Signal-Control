"""Agents sharing one interface: baselines and the proposed UG-MARL."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trafficrl.agents.base import Agent, RuleAgent
from trafficrl.agents.baselines.dqn import DQNAgent
from trafficrl.agents.baselines.fixed_time import FixedTimeAgent
from trafficrl.agents.baselines.idqn import IDQNAgent
from trafficrl.agents.baselines.max_pressure import MaxPressureAgent
from trafficrl.agents.ugmarl import UGMARLAgent, variant_name

AGENTS = ("fixed_time", "max_pressure", "dqn", "idqn", "ugmarl")
LEARNED_AGENTS = ("dqn", "idqn", "ugmarl")


def make_agent(name: str, env: Any, cfg: Mapping[str, Any]) -> Agent:
    """Build agent ``name`` for ``env`` from ``cfg['agents'][name]``.

    Learned agents also read the ``train`` block and ``seed``.
    """
    params = dict((cfg.get("agents", {}) or {}).get(name, {}) or {})
    if name == "fixed_time":
        return FixedTimeAgent(env.phase_ids, env.delta_t, **params)
    if name == "max_pressure":
        return MaxPressureAgent(env.features, **params)
    if name in LEARNED_AGENTS:
        learn = {**dict(cfg.get("train", {}) or {}), **params}
        cls = {"dqn": DQNAgent, "idqn": IDQNAgent, "ugmarl": UGMARLAgent}[name]
        return cls(env, learn, seed=int(cfg.get("seed", 0)))
    raise ValueError(f"unknown agent {name!r}; one of {AGENTS}")


def model_label(name: str, cfg: Mapping[str, Any]) -> str:
    """Name used in run directories and the experiment log: the agent name, or the UG-MARL
    variant (``ugmarl-noattn`` …) given by the ``agents.ugmarl`` ablation flags."""
    if name == "ugmarl":
        return variant_name((cfg.get("agents", {}) or {}).get("ugmarl", {}))
    return name


__all__ = [
    "AGENTS",
    "LEARNED_AGENTS",
    "Agent",
    "DQNAgent",
    "FixedTimeAgent",
    "IDQNAgent",
    "MaxPressureAgent",
    "RuleAgent",
    "UGMARLAgent",
    "make_agent",
    "model_label",
]
