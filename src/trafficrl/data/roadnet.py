"""CityFlow road-network model (ROADMAP 3.4).

One parser shared by the data audit, the feature builder, the EDA and the environment
wrapper. It reads only the roadnet file (the map), never a flow file, so everything derived
from it (lane order, phases, neighbours, lane capacity) is free of demand information.

Conventions (dataset card §4.2):

* Lane id = ``"<road id>_<lane index>"``; lane 0 = left turn, 1 = through, 2 = right turn.
* Light phase 0 is the 5 s transition ("yellow"); phases 1..8 are the green phases the
  agent chooses from.
* Directions are numbered 0 = east, 1 = north, 2 = west, 3 = south (direction of travel).

**Canonical lane order** of an intersection: incoming roads sorted by direction of travel
(E, N, W, S), then lane index. Every intersection therefore exposes its 12 incoming lanes
in the same geometric order, which lets one network share parameters across intersections.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

YELLOW_PHASE = 0
DIRECTIONS = ("E", "N", "W", "S")
N_APPROACHES = 4
MOVEMENT_KINDS = ("go_straight", "turn_left", "turn_right")


class RoadnetError(ValueError):
    """The roadnet does not have the structure the project assumes."""


@dataclass(frozen=True)
class Movement:
    """One ``roadLink``: an allowed movement from an incoming road to an outgoing road."""

    kind: str
    start_road: str
    end_road: str
    in_lanes: tuple[str, ...]
    out_lanes: tuple[str, ...]


@dataclass(frozen=True)
class Intersection:
    id: str
    in_roads: tuple[str, ...]  # canonical order (E, N, W, S direction of travel)
    out_roads: tuple[str, ...]
    in_lanes: tuple[str, ...]  # canonical order
    out_lanes: tuple[str, ...]
    movements: tuple[Movement, ...]  # roadnet order (indices used by the light phases)
    green_phases: dict[int, tuple[int, ...]]  # engine phase id -> movement indices
    neighbors: tuple[str | None, ...]  # signalised neighbour towards E, N, W, S


@dataclass(frozen=True)
class Roadnet:
    intersections: dict[str, Intersection]  # signalised only, sorted by id
    virtual_ids: frozenset[str]
    road_start: dict[str, str]  # road id -> start intersection id
    road_end: dict[str, str]
    road_length_m: dict[str, float]
    road_links: frozenset[tuple[str, str]]  # (start road, end road) pairs that are connected
    lane_length_m: dict[str, float]
    lane_max_speed: dict[str, float]

    @property
    def road_ids(self) -> frozenset[str]:
        return frozenset(self.road_start)

    @property
    def green_phase_ids(self) -> tuple[int, ...]:
        """Green phase ids shared by all intersections (the action set)."""
        sets = {tuple(sorted(i.green_phases)) for i in self.intersections.values()}
        if len(sets) != 1:
            raise RoadnetError("intersections do not share one set of green phases")
        return next(iter(sets))


def _direction(points: list[dict]) -> int:
    """Direction of travel of a road from its polyline: 0 E, 1 N, 2 W, 3 S."""
    dx = points[-1]["x"] - points[0]["x"]
    dy = points[-1]["y"] - points[0]["y"]
    if abs(dx) >= abs(dy):
        return 0 if dx > 0 else 2
    return 1 if dy > 0 else 3


def _length(points: list[dict]) -> float:
    return sum(math.dist((a["x"], a["y"]), (b["x"], b["y"])) for a, b in pairwise(points))


def _lanes(road_id: str, n: int) -> list[str]:
    return [f"{road_id}_{k}" for k in range(n)]


def parse_roadnet(raw: dict) -> Roadnet:
    """Build a :class:`Roadnet` from the decoded roadnet JSON."""
    roads = {r["id"]: r for r in raw["roads"]}
    direction = {rid: _direction(r["points"]) for rid, r in roads.items()}
    n_lanes = {rid: len(r["lanes"]) for rid, r in roads.items()}
    virtual = frozenset(i["id"] for i in raw["intersections"] if i["virtual"])

    intersections: dict[str, Intersection] = {}
    links: set[tuple[str, str]] = set()
    for inter in sorted(raw["intersections"], key=lambda i: i["id"]):
        if inter["virtual"]:
            continue
        iid = inter["id"]
        in_roads = sorted(
            (r for r in inter["roads"] if roads[r]["endIntersection"] == iid),
            key=direction.__getitem__,
        )
        out_roads = sorted(
            (r for r in inter["roads"] if roads[r]["startIntersection"] == iid),
            key=direction.__getitem__,
        )
        for name, rs in (("incoming", in_roads), ("outgoing", out_roads)):
            if [direction[r] for r in rs] != list(range(N_APPROACHES)):
                raise RoadnetError(f"{iid}: needs one {name} road per direction, got {rs}")

        movements = []
        for link in inter["roadLinks"]:
            if link["type"] not in MOVEMENT_KINDS:
                raise RoadnetError(f"{iid}: unknown movement type {link['type']!r}")
            start, end = link["startRoad"], link["endRoad"]
            links.add((start, end))
            movements.append(
                Movement(
                    kind=link["type"],
                    start_road=start,
                    end_road=end,
                    in_lanes=tuple(
                        sorted({f"{start}_{ll['startLaneIndex']}" for ll in link["laneLinks"]})
                    ),
                    out_lanes=tuple(
                        sorted({f"{end}_{ll['endLaneIndex']}" for ll in link["laneLinks"]})
                    ),
                )
            )
        if len(movements) != 3 * N_APPROACHES:
            raise RoadnetError(f"{iid}: expected 12 movements, got {len(movements)}")

        phases = inter["trafficLight"]["lightphases"]
        if len(phases) < 2:
            raise RoadnetError(f"{iid}: needs a transition phase and at least one green phase")
        green = {}
        for k, p in enumerate(phases):
            if any(not 0 <= m < len(movements) for m in p["availableRoadLinks"]):
                raise RoadnetError(f"{iid}: phase {k} references an unknown movement")
            if k != YELLOW_PHASE:
                green[k] = tuple(p["availableRoadLinks"])

        nb = []
        for r in out_roads:  # one per direction, E N W S
            other = roads[r]["endIntersection"]
            nb.append(None if other in virtual else other)

        intersections[iid] = Intersection(
            id=iid,
            in_roads=tuple(in_roads),
            out_roads=tuple(out_roads),
            in_lanes=tuple(ln for r in in_roads for ln in _lanes(r, n_lanes[r])),
            out_lanes=tuple(ln for r in out_roads for ln in _lanes(r, n_lanes[r])),
            movements=tuple(movements),
            green_phases=green,
            neighbors=tuple(nb),
        )

    length = {rid: _length(r["points"]) for rid, r in roads.items()}
    return Roadnet(
        intersections=intersections,
        virtual_ids=virtual,
        road_start={rid: r["startIntersection"] for rid, r in roads.items()},
        road_end={rid: r["endIntersection"] for rid, r in roads.items()},
        road_length_m=length,
        road_links=frozenset(links),
        lane_length_m={ln: length[rid] for rid in roads for ln in _lanes(rid, n_lanes[rid])},
        lane_max_speed={
            f"{rid}_{k}": lane["maxSpeed"]
            for rid, r in roads.items()
            for k, lane in enumerate(r["lanes"])
        },
    )


def load_roadnet(path: str | Path) -> Roadnet:
    return parse_roadnet(json.loads(Path(path).read_text()))


def lane_capacity(roadnet: Roadnet, vehicle_space_m: float) -> dict[str, int]:
    """Vehicles that fit on each lane bumper to bumper: ``floor(length / vehicle_space_m)``.

    ``vehicle_space_m`` = vehicle length + minimum gap (5.0 + 2.5 m in both datasets).
    """
    if vehicle_space_m <= 0:
        raise ValueError(f"vehicle_space_m must be positive, got {vehicle_space_m}")
    return {
        ln: max(1, int(length // vehicle_space_m)) for ln, length in roadnet.lane_length_m.items()
    }
