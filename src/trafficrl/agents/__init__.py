"""Agents sharing one interface: baselines and the proposed UG-MARL."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trafficrl.agents.base import Agent, RuleAgent
from trafficrl.agents.baselines.fixed_time import FixedTimeAgent
from trafficrl.agents.baselines.max_pressure import MaxPressureAgent

AGENTS = ("fixed_time", "max_pressure")


def make_agent(name: str, env: Any, cfg: Mapping[str, Any]) -> Agent:
    """Build agent ``name`` for ``env`` with its parameters from ``cfg['agents'][name]``."""
    params = dict((cfg.get("agents", {}) or {}).get(name, {}) or {})
    if name == "fixed_time":
        return FixedTimeAgent(env.phase_ids, env.delta_t, **params)
    if name == "max_pressure":
        return MaxPressureAgent(env.features, **params)
    raise ValueError(f"unknown agent {name!r}; one of {AGENTS}")


__all__ = ["AGENTS", "Agent", "FixedTimeAgent", "MaxPressureAgent", "RuleAgent", "make_agent"]
