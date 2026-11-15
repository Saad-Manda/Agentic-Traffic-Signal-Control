"""Max-Pressure controller (Varaiya 2013)."""

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
        return max(self.phase_ids, key=pressure.__getitem__)

    def act(
        self, obs: dict[str, np.ndarray], info: dict[str, Any], explore: bool = False
    ) -> dict[str, int]:
        counts = info["lane_count"]
        return {iid: self.choose(iid, counts) for iid in obs}
