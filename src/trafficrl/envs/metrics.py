"""Episode metrics (ROADMAP 4.2; definitions frozen in docs/evaluation_protocol.md §5).

:class:`EpisodeTracker` is fed by :class:`trafficrl.envs.traffic_env.TrafficEnv`:

* ``on_second(speeds)`` after every simulated second, with ``engine.get_vehicle_speed()``
  (keys = vehicles currently running in the network) — only at ``track="full"``;
* ``on_step(lane_waiting)`` at every env step (decision step), with the **true** waiting
  counts;
* ``on_switch()`` whenever a yellow phase is inserted.

``track="light"`` skips the per-second vehicle reads (used for RL training, where only the
primary metric and the queue are needed); the vehicle-level metrics are then ``None``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

TRACK_LEVELS = ("full", "light")
WAIT_SPEED = 0.1  # m/s — "waiting" threshold, same as CityFlow's lane waiting count


class EpisodeTracker:
    def __init__(self, in_lanes: Iterable[str], track: str = "full") -> None:
        if track not in TRACK_LEVELS:
            raise ValueError(f"track must be one of {TRACK_LEVELS}, got {track!r}")
        self.track = track
        self.in_lanes = tuple(in_lanes)
        self.entered: set[str] = set()
        self.left: set[str] = set()
        self._inside: set[str] = set()
        self.wait_s: dict[str, int] = defaultdict(int)
        self.queue_sum = 0.0
        self.steps = 0
        self.switches = 0

    @property
    def full(self) -> bool:
        return self.track == "full"

    def on_second(self, speeds: Mapping[str, float]) -> None:
        current = set(speeds)
        self.entered |= current
        self.left |= self._inside - current
        self._inside = current
        for vid, v in speeds.items():
            if v < WAIT_SPEED:
                self.wait_s[vid] += 1

    def on_step(self, lane_waiting: Mapping[str, float]) -> None:
        """Mean waiting vehicles per incoming lane (all signalised intersections)."""
        self.queue_sum += sum(lane_waiting.get(ln, 0) for ln in self.in_lanes) / len(self.in_lanes)
        self.steps += 1

    def on_switch(self, n: int = 1) -> None:
        self.switches += n

    def result(self, avg_travel_time_s: float, in_network: int) -> dict[str, float | None]:
        out: dict[str, float | None] = {
            "avg_travel_time_s": round(float(avg_travel_time_s), 4),
            "avg_queue_veh": round(self.queue_sum / self.steps, 4) if self.steps else 0.0,
            "phase_switches": self.switches,
            "in_network_at_end": int(in_network),
            "avg_waiting_time_s": None,
            "max_waiting_time_s": None,
            "throughput_veh": None,
            "vehicles_entered": None,
        }
        if self.full:
            waits = [self.wait_s.get(v, 0) for v in self.entered]
            out.update(
                avg_waiting_time_s=round(sum(waits) / len(waits), 4) if waits else 0.0,
                max_waiting_time_s=max(waits, default=0),
                throughput_veh=len(self.left),
                vehicles_entered=len(self.entered),
            )
        return out
