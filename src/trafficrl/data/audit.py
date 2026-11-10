"""Structural data audit (ROADMAP 3.0, guidelines G3).

Pure functions that *count* problems in a flow file or a roadnet. Nothing is fitted and
nothing is repaired: the flow files are hash-pinned benchmark data, so a defect is reported
and handled by a recorded decision, never by silent filtering.

The counts are structural (is each record valid?), so they are safe to compute on every
split including the held-out city. Demand *distributions* and plots are produced for the
train split only (``scripts/audit_data.py``, ``database.md`` rules 7 and 9).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from numbers import Real

import numpy as np

from trafficrl.data.roadnet import MOVEMENT_KINDS, N_APPROACHES, YELLOW_PHASE, Roadnet

FLOW_KEYS = ("vehicle", "route", "interval", "startTime", "endTime")
VEHICLE_KEYS = (
    "length",
    "width",
    "maxPosAcc",
    "maxNegAcc",
    "usualPosAcc",
    "usualNegAcc",
    "minGap",
    "maxSpeed",
    "headwayTime",
)
YELLOW_SECONDS = 5
N_PHASES = 9

# Checks whose count must be 0 for the file to be used as-is.
FLOW_DEFECTS = (
    "missing_field",
    "missing_vehicle_field",
    "wrong_type",
    "duplicate_records",
    "start_time_out_of_range",
    "start_time_not_integer",
    "end_before_start",
    "multi_vehicle_records",
    "route_too_short",
    "unknown_road",
    "disconnected_route",
    "route_not_from_boundary",
    "route_repeats_road",
    "nonpositive_vehicle_value",
    "speed_above_lane_limit",
    "unexpected_field",
    "unexpected_string_value",
)
# Reported for information; not defects.
#   route_ends_inside_network: the last road ends at a signalised intersection instead of a
#     boundary node. Valid for CityFlow (the vehicle is removed at the end of its last road),
#     so it is a property of the recorded trips, not an impossible value.
#   route_revisits_intersection: the route passes the same intersection more than once.
FLOW_INFO = (
    "records",
    "arrival_burst_bins",
    "arrival_low_bins",
    "same_second_same_entry_lane",
    "route_ends_inside_network",
    "route_revisits_intersection",
    "vehicle_profiles",
)

ROADNET_DEFECTS = (
    "intersection_without_4_in_roads",
    "intersection_without_4_out_roads",
    "intersection_without_12_movements",
    "intersection_without_9_phases",
    "transition_phase_not_5s",
    "invalid_phase_movement_index",
    "unknown_movement_type",
    "lane_link_index_out_of_range",
    "road_with_unknown_intersection",
    "movement_with_unknown_road",
    "nonpositive_road_length",
    "nonpositive_lane_value",
)
ROADNET_INFO = ("intersections_signalised", "intersections_virtual", "roads", "lanes")


def _is_number(x: object) -> bool:
    return isinstance(x, Real) and not isinstance(x, bool)


def arrival_counts(flows: Sequence[dict], episode_seconds: int, bin_seconds: int) -> np.ndarray:
    """Vehicles entering the network per ``bin_seconds`` window over the episode."""
    n_bins = -(-episode_seconds // bin_seconds)
    counts = np.zeros(n_bins, dtype=int)
    for v in flows:
        t = v.get("startTime")
        if _is_number(t) and 0 <= t < episode_seconds:
            counts[int(t) // bin_seconds] += 1
    return counts


def burst_threshold(counts: np.ndarray) -> float:
    """Tukey upper fence Q3 + 1.5·IQR of the per-bin arrival counts."""
    q1, q3 = np.percentile(counts, [25, 75])
    return float(q3 + 1.5 * (q3 - q1))


def low_threshold(counts: np.ndarray) -> float:
    """Tukey lower fence Q1 − 1.5·IQR of the per-bin arrival counts."""
    q1, q3 = np.percentile(counts, [25, 75])
    return float(q1 - 1.5 * (q3 - q1))


def audit_flow(
    flows: Sequence[dict], roadnet: Roadnet, episode_seconds: int, bin_seconds: int = 60
) -> dict[str, int]:
    """Count structural problems in one flow file. Keys: ``FLOW_DEFECTS`` + ``FLOW_INFO``."""
    c: Counter[str] = Counter({k: 0 for k in FLOW_DEFECTS + FLOW_INFO})
    c["records"] = len(flows)
    road_ids = roadnet.road_ids
    max_lane_speed = max(roadnet.lane_max_speed.values())
    seen: set[str] = set()
    entry: Counter[tuple] = Counter()
    profiles: set[str] = set()

    for v in flows:
        key = json.dumps(v, sort_keys=True)
        c["duplicate_records"] += key in seen
        seen.add(key)

        c["missing_field"] += sum(k not in v or v[k] is None for k in FLOW_KEYS)
        c["unexpected_field"] += sum(k not in FLOW_KEYS for k in v)

        veh = v.get("vehicle")
        if isinstance(veh, dict):
            profiles.add(json.dumps(veh, sort_keys=True))
            c["missing_vehicle_field"] += sum(k not in veh or veh[k] is None for k in VEHICLE_KEYS)
            c["unexpected_field"] += sum(k not in VEHICLE_KEYS for k in veh)
            for k in VEHICLE_KEYS:
                x = veh.get(k)
                if x is None:
                    continue
                if not _is_number(x):
                    c["wrong_type"] += 1
                    c["unexpected_string_value"] += isinstance(x, str)
                elif x <= 0:
                    c["nonpositive_vehicle_value"] += 1
            speed = veh.get("maxSpeed")
            c["speed_above_lane_limit"] += _is_number(speed) and speed > max_lane_speed + 1e-9
        elif veh is not None:
            c["wrong_type"] += 1

        start, end = v.get("startTime"), v.get("endTime")
        for t in (start, end, v.get("interval")):
            if t is not None and not _is_number(t):
                c["wrong_type"] += 1
                c["unexpected_string_value"] += isinstance(t, str)
        if _is_number(start):
            c["start_time_out_of_range"] += not 0 <= start < episode_seconds
            c["start_time_not_integer"] += start != int(start)
            if _is_number(end):
                c["end_before_start"] += end < start
                c["multi_vehicle_records"] += end > start

        route = v.get("route")
        if route is None:
            continue
        if not isinstance(route, list) or not all(isinstance(r, str) for r in route):
            c["wrong_type"] += 1
            continue
        c["route_too_short"] += len(route) < 2
        unknown = [r for r in route if r not in road_ids]
        c["unknown_road"] += len(unknown)
        c["route_repeats_road"] += len(set(route)) != len(route)
        if unknown or not route:
            continue
        c["disconnected_route"] += any(
            (a, b) not in roadnet.road_links for a, b in zip(route, route[1:], strict=False)
        )
        c["route_not_from_boundary"] += roadnet.road_start[route[0]] not in roadnet.virtual_ids
        c["route_ends_inside_network"] += roadnet.road_end[route[-1]] not in roadnet.virtual_ids
        nodes = [roadnet.road_end[r] for r in route[:-1]]
        c["route_revisits_intersection"] += len(set(nodes)) != len(nodes)
        if _is_number(start) and len(route) >= 2:
            # the movement at the first intersection fixes the entry lane
            entry[(start, route[0], route[1])] += 1

    counts = arrival_counts(flows, episode_seconds, bin_seconds)
    c["arrival_burst_bins"] = int((counts > burst_threshold(counts)).sum())
    c["arrival_low_bins"] = int((counts < low_threshold(counts)).sum())
    c["same_second_same_entry_lane"] = sum(n - 1 for n in entry.values() if n > 1)
    c["vehicle_profiles"] = len(profiles)
    return {k: int(c[k]) for k in FLOW_DEFECTS + FLOW_INFO}


def audit_roadnet(raw: dict) -> dict[str, int]:
    """Count structural problems in a roadnet. Keys: ``ROADNET_DEFECTS`` + ``ROADNET_INFO``."""
    c: Counter[str] = Counter({k: 0 for k in ROADNET_DEFECTS + ROADNET_INFO})
    inters = {i["id"]: i for i in raw["intersections"]}
    roads = {r["id"]: r for r in raw["roads"]}
    c["roads"] = len(roads)
    c["lanes"] = sum(len(r["lanes"]) for r in roads.values())

    for r in roads.values():
        ends = (r["startIntersection"], r["endIntersection"])
        c["road_with_unknown_intersection"] += any(e not in inters for e in ends)
        p = r["points"]
        c["nonpositive_road_length"] += (p[0]["x"], p[0]["y"]) == (p[-1]["x"], p[-1]["y"])
        c["nonpositive_lane_value"] += sum(
            ln["width"] <= 0 or ln["maxSpeed"] <= 0 for ln in r["lanes"]
        )

    for i in inters.values():
        if i["virtual"]:
            c["intersections_virtual"] += 1
            continue
        c["intersections_signalised"] += 1
        known = [r for r in i["roads"] if r in roads]
        n_in = sum(roads[r]["endIntersection"] == i["id"] for r in known)
        n_out = sum(roads[r]["startIntersection"] == i["id"] for r in known)
        c["intersection_without_4_in_roads"] += n_in != N_APPROACHES
        c["intersection_without_4_out_roads"] += n_out != N_APPROACHES
        links = i["roadLinks"]
        c["intersection_without_12_movements"] += len(links) != 3 * N_APPROACHES
        for link in links:
            c["unknown_movement_type"] += link["type"] not in MOVEMENT_KINDS
            ok = link["startRoad"] in roads and link["endRoad"] in roads
            c["movement_with_unknown_road"] += not ok
            if ok:
                n_start = len(roads[link["startRoad"]]["lanes"])
                n_end = len(roads[link["endRoad"]]["lanes"])
                c["lane_link_index_out_of_range"] += sum(
                    not (0 <= ll["startLaneIndex"] < n_start and 0 <= ll["endLaneIndex"] < n_end)
                    for ll in link["laneLinks"]
                )
        phases = i["trafficLight"]["lightphases"]
        c["intersection_without_9_phases"] += len(phases) != N_PHASES
        if phases:
            c["transition_phase_not_5s"] += phases[YELLOW_PHASE]["time"] != YELLOW_SECONDS
        c["invalid_phase_movement_index"] += sum(
            not 0 <= m < len(links) for p in phases for m in p["availableRoadLinks"]
        )
    return {k: int(c[k]) for k in ROADNET_DEFECTS + ROADNET_INFO}


def defects(counts: dict[str, int]) -> dict[str, int]:
    """The non-zero defect counts of an audit result (empty dict = clean)."""
    keys = set(FLOW_DEFECTS) | set(ROADNET_DEFECTS)
    return {k: v for k, v in counts.items() if k in keys and v}
