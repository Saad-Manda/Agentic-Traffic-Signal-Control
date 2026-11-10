"""Max-Pressure controller (ROADMAP 4.5; Varaiya 2013; protocol §4).

At every decision each intersection serves the green phase with the largest pressure
Σ over the phase's movements of (x_in − x_out), from the **observed** lane vehicle counts
(``info["lane_count"]``, the same detectors the RL agents read). Ties → lowest phase id.
The same rule is the UG-MARL fallback when the uncertainty gate fires (Phase 6).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from trafficrl.agents.base import RuleAgent
from trafficrl.envs.features import FeatureBuilder


class MaxPressureAgent(RuleAgent):
    name = "max_pressure"

    def __init__(self, features: FeatureBuilder):
        self.features = features
        self.phase_ids = tuple(features.phase_ids)

    def choose(self, iid: str, lane_count: Mapping[str, float]) -> int:
        pressure = self.features.phase_pressure(iid, lane_count)
        return max(self.phase_ids, key=pressure.__getitem__)  # max() keeps the first: lowest id

    def act(
        self, obs: dict[str, np.ndarray], info: dict[str, Any], explore: bool = False
    ) -> dict[str, int]:
        counts = info["lane_count"]
        return {iid: self.choose(iid, counts) for iid in obs}
