"""Fixed-Time controller (ROADMAP 4.4; protocol §4).

Every intersection cycles through the green phases in order 1 → 8, each held for
``green_seconds`` (30 s), with the env's 5 s yellow between phases. The agent works inside
the Δt decision loop: at a ready intersection it keeps the phase while the green has run for
less than ``green_seconds`` and moves to the next phase otherwise, so with Δt = 15 s it keeps
at 15 s and switches at 30 s. Observations are ignored (the controller is blind by design).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from trafficrl.agents.base import RuleAgent


class FixedTimeAgent(RuleAgent):
    name = "fixed_time"

    def __init__(self, phase_ids: tuple[int, ...], delta_t: int, green_seconds: int = 30):
        if green_seconds <= 0 or green_seconds % delta_t:
            raise ValueError(
                f"green_seconds ({green_seconds}) must be a positive multiple of Δt ({delta_t})"
            )
        self.phase_ids = tuple(phase_ids)
        self.green_seconds = int(green_seconds)

    def act(
        self, obs: dict[str, np.ndarray], info: dict[str, Any], explore: bool = False
    ) -> dict[str, int]:
        if info["t"] == 0:
            return {iid: self.phase_ids[0] for iid in obs}
        out = {}
        for iid in obs:
            phase = info["phase"][iid]
            if info["green_elapsed"][iid] >= self.green_seconds:
                phase = self.phase_ids[(self.phase_ids.index(phase) + 1) % len(self.phase_ids)]
            out[iid] = phase
        return out
