"""A minimal stand-in for ``cityflow.Engine`` used by fast env tests.

Lane counts are whatever the test puts in ``counts`` / ``waiting``; every ``set_tl_phase``
call is recorded with the simulated time.
"""

from __future__ import annotations

from trafficrl.data.roadnet import Roadnet


class FakeEngine:
    def __init__(self, roadnet: Roadnet) -> None:
        self.lanes = sorted(roadnet.lane_length_m)
        self.counts = {ln: 0 for ln in self.lanes}
        self.waiting = {ln: 0 for ln in self.lanes}
        self.speeds: dict[str, float] = {}
        self.time = 0
        self.calls: list[tuple[int, str, int]] = []
        self.seeds: list[int] = []
        self.resets = 0

    def next_step(self) -> None:
        self.time += 1

    def set_tl_phase(self, iid: str, phase: int) -> None:
        self.calls.append((self.time, iid, phase))

    def get_lane_vehicle_count(self) -> dict[str, int]:
        return dict(self.counts)

    def get_lane_waiting_vehicle_count(self) -> dict[str, int]:
        return dict(self.waiting)

    def get_vehicle_speed(self) -> dict[str, float]:
        return dict(self.speeds)

    def get_average_travel_time(self) -> float:
        return 123.0

    def get_vehicle_count(self) -> int:
        return len(self.speeds)

    def set_random_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def reset(self) -> None:
        self.time = 0
        self.calls.clear()
        self.resets += 1
