"""Shared agent interface (``.claude/reference/api.md``).

Every controller — rule-based baseline, DQN, IDQN, UG-MARL and its ablations — implements
:class:`Agent`, so one evaluation loop (``trafficrl.eval.runner``) serves all of them.

``act`` receives the observations of **all** intersections plus the env ``info`` dict and
returns a phase for every intersection; the env applies only the actions of the
intersections listed in ``info["ready"]``. Computing all intersections every call keeps the
measured decision latency comparable across agents ("batch of all intersections",
evaluation protocol §5).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Agent(Protocol):
    name: str

    def act(
        self, obs: dict[str, np.ndarray], info: dict[str, Any], explore: bool = False
    ) -> dict[str, int]: ...

    def observe(self, transitions: Sequence[Any]) -> None: ...

    def update(self) -> dict[str, float]: ...

    def save(self, path: str | Path) -> None: ...

    def load(self, path: str | Path) -> None: ...

    def last_explanations(self) -> dict[str, Any]: ...


class RuleAgent:
    """Base for non-learning controllers: everything except :meth:`act` is a no-op."""

    name = "rule"

    def act(
        self, obs: dict[str, np.ndarray], info: dict[str, Any], explore: bool = False
    ) -> dict[str, int]:
        raise NotImplementedError

    def observe(self, transitions: Sequence[Any]) -> None:
        return None

    def update(self) -> dict[str, float]:
        return {}

    def save(self, path: str | Path) -> None:
        return None

    def load(self, path: str | Path) -> None:
        return None

    def last_explanations(self) -> dict[str, Any]:
        return {}
