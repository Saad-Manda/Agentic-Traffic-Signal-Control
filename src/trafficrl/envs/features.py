"""Observation features and pressure definitions (ROADMAP 3.4, guidelines G6).

For a signalised intersection with 12 incoming lanes ``l``, 12 movements ``m`` and lane
capacity ``C_l`` (vehicles that fit on the lane, from the roadnet):

* occupancy   x̃_l = min(x_l / C_l, 1)   x_l = vehicles on the lane
* queue       q̃_l = min(q_l / C_l, 1)   q_l = waiting vehicles (speed < 0.1 m/s)
* pressure    p_m = mean x̃ over the movement's incoming lanes
                    − mean x̃ over its outgoing lanes            ∈ [−1, 1]
* phase       one-hot of the current green phase (8)

Observation = concatenation of the configured groups, default
``[queue (12) | count (12) | pressure (12) | phase (8)]`` = 44 values, lanes in the
canonical order of :mod:`trafficrl.data.roadnet`.

**No fitted statistics.** Normalisation uses only lane capacities, which come from road
lengths in the roadnet and a constant vehicle spacing. The builder never reads a flow file,
so there is nothing that could leak from validation, test or held-out demand, and the same
code applies unchanged to another city's roadnet.

Raw (un-normalised) pressures used by Max-Pressure and the reward are defined here too so
that every agent and the environment share one definition.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from trafficrl.data.roadnet import Roadnet, lane_capacity

GROUPS = ("queue", "count", "pressure", "phase")
NORMALISATIONS = ("lane_capacity",)


@dataclass(frozen=True)
class FeatureSpec:
    groups: tuple[str, ...] = GROUPS
    vehicle_space_m: float = 7.5  # vehicle length 5.0 m + minimum gap 2.5 m
    clip: bool = True
    normalisation: str = "lane_capacity"

    def __post_init__(self) -> None:
        if not self.groups or len(set(self.groups)) != len(self.groups):
            raise ValueError(f"feature groups must be non-empty and unique, got {self.groups}")
        unknown = [g for g in self.groups if g not in GROUPS]
        if unknown:
            raise ValueError(f"unknown feature group(s) {unknown}; choose from {GROUPS}")
        if self.normalisation not in NORMALISATIONS:
            raise ValueError(
                f"normalisation must be one of {NORMALISATIONS}, got {self.normalisation!r}"
            )

    @classmethod
    def from_config(cls, cfg: Mapping) -> FeatureSpec:
        """Build from the ``features:`` block of an experiment config."""
        f = cfg.get("features", {})
        return cls(
            groups=tuple(f.get("groups", GROUPS)),
            vehicle_space_m=float(f.get("vehicle_space_m", 7.5)),
            clip=bool(f.get("clip", True)),
            normalisation=f.get("normalisation", "lane_capacity"),
        )


class FeatureBuilder:
    """Turns raw per-lane simulator readings into per-intersection observation vectors."""

    def __init__(self, roadnet: Roadnet, spec: FeatureSpec | None = None) -> None:
        self.roadnet = roadnet
        self.spec = spec or FeatureSpec()
        self.capacity = lane_capacity(roadnet, self.spec.vehicle_space_m)
        self.phase_ids = roadnet.green_phase_ids
        first = next(iter(roadnet.intersections.values()))
        self._sizes = {
            "queue": len(first.in_lanes),
            "count": len(first.in_lanes),
            "pressure": len(first.movements),
            "phase": len(self.phase_ids),
        }

    @property
    def dim(self) -> int:
        return sum(self._sizes[g] for g in self.spec.groups)

    def slices(self) -> dict[str, slice]:
        """Position of each feature group inside the observation vector."""
        out, start = {}, 0
        for g in self.spec.groups:
            out[g] = slice(start, start + self._sizes[g])
            start += self._sizes[g]
        return out

    def _norm(self, lanes: tuple[str, ...], values: Mapping[str, float]) -> np.ndarray:
        x = np.array([values.get(ln, 0) / self.capacity[ln] for ln in lanes], dtype=np.float32)
        return np.minimum(x, 1.0) if self.spec.clip else x

    def observation(
        self,
        iid: str,
        lane_count: Mapping[str, float],
        lane_waiting: Mapping[str, float],
        green_phase: int,
    ) -> np.ndarray:
        """Observation of one intersection (float32, length :attr:`dim`).

        ``green_phase`` is the current green phase id (during a transition: the phase being
        switched to). ``lane_count`` / ``lane_waiting`` are the engine's per-lane dictionaries.
        """
        inter = self.roadnet.intersections[iid]
        parts = []
        for g in self.spec.groups:
            if g == "queue":
                parts.append(self._norm(inter.in_lanes, lane_waiting))
            elif g == "count":
                parts.append(self._norm(inter.in_lanes, lane_count))
            elif g == "pressure":
                parts.append(
                    np.array(
                        [
                            self._norm(m.in_lanes, lane_count).mean()
                            - self._norm(m.out_lanes, lane_count).mean()
                            for m in inter.movements
                        ],
                        dtype=np.float32,
                    )
                )
            else:  # phase
                if green_phase not in self.phase_ids:
                    raise ValueError(f"{iid}: {green_phase} is not a green phase {self.phase_ids}")
                one_hot = np.zeros(len(self.phase_ids), dtype=np.float32)
                one_hot[self.phase_ids.index(green_phase)] = 1.0
                parts.append(one_hot)
        return np.concatenate(parts)

    def observations(
        self,
        lane_count: Mapping[str, float],
        lane_waiting: Mapping[str, float],
        green_phases: Mapping[str, int],
    ) -> dict[str, np.ndarray]:
        """Observations of all signalised intersections, keyed by intersection id."""
        return {
            iid: self.observation(iid, lane_count, lane_waiting, green_phases[iid])
            for iid in self.roadnet.intersections
        }

    # --- raw pressures (vehicles), shared by Max-Pressure and the reward -------------------

    def phase_pressure(self, iid: str, lane_count: Mapping[str, float]) -> dict[int, float]:
        """Pressure of each green phase: Σ over its movements of (x_in − x_out)."""
        inter = self.roadnet.intersections[iid]
        mov = [
            sum(lane_count.get(ln, 0) for ln in m.in_lanes)
            - sum(lane_count.get(ln, 0) for ln in m.out_lanes)
            for m in inter.movements
        ]
        return {k: float(sum(mov[i] for i in ms)) for k, ms in inter.green_phases.items()}

    def intersection_pressure(self, iid: str, lane_count: Mapping[str, float]) -> float:
        """P_i = Σ vehicles on incoming lanes − Σ vehicles on outgoing lanes."""
        inter = self.roadnet.intersections[iid]
        return float(
            sum(lane_count.get(ln, 0) for ln in inter.in_lanes)
            - sum(lane_count.get(ln, 0) for ln in inter.out_lanes)
        )
