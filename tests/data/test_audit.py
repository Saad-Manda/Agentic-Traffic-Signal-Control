import copy

import pytest

from tests.data.toy import toy_roadnet, toy_vehicle
from trafficrl.data.audit import (
    FLOW_DEFECTS,
    ROADNET_DEFECTS,
    arrival_counts,
    audit_flow,
    audit_roadnet,
    burst_threshold,
    defects,
)
from trafficrl.data.roadnet import parse_roadnet

RN = parse_roadnet(toy_roadnet())
EPISODE = 3600


def clean_flow():
    """One vehicle every 30 s, alternating between two valid routes."""
    routes = [("in_0", "out_0"), ("in_1", "out_2")]
    return [toy_vehicle(t, routes[(t // 30) % 2]) for t in range(0, EPISODE, 30)]


def audit(flows):
    return audit_flow(flows, RN, EPISODE)


def test_clean_flow_has_no_defects():
    res = audit(clean_flow())
    assert set(res) >= set(FLOW_DEFECTS)
    assert defects(res) == {}
    assert res["records"] == 120
    assert res["vehicle_profiles"] == 1
    assert res["arrival_burst_bins"] == 0
    assert res["same_second_same_entry_lane"] == 0
    assert res["route_ends_inside_network"] == 0
    assert res["route_revisits_intersection"] == 0


def test_route_ending_inside_network_is_information_not_a_defect():
    flows = clean_flow()
    flows[5]["route"] = ["in_0"]  # ends at the signalised node C
    flows[5]["startTime"] = flows[5]["endTime"] = 151
    res = audit(flows)
    assert res["route_ends_inside_network"] == 1
    assert defects(res) == {"route_too_short": 1}


def _mutated(fn):
    flows = clean_flow()
    fn(flows)
    return defects(audit(flows))


def _set(i, key, value):
    def fn(flows):
        flows[i][key] = value

    return fn


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda f: f[0].pop("interval"), {"missing_field": 1}),
        (lambda f: f[0]["vehicle"].pop("minGap"), {"missing_vehicle_field": 1}),
        (lambda f: f.append(copy.deepcopy(f[3])), {"duplicate_records": 1}),
        (_set(0, "startTime", -1), {"start_time_out_of_range": 1, "multi_vehicle_records": 1}),
        (_set(5, "startTime", 150.5), {"start_time_not_integer": 1, "end_before_start": 1}),
        (_set(5, "endTime", 100), {"end_before_start": 1}),
        (_set(5, "endTime", 200), {"multi_vehicle_records": 1}),
        (_set(5, "route", ["in_0"]), {"route_too_short": 1}),
        (_set(5, "route", ["in_0", "nope"]), {"unknown_road": 1}),
        (_set(5, "route", ["in_0", "out_2"]), {"disconnected_route": 1}),  # U-turn: no link
        (
            _set(5, "route", ["out_0", "in_0"]),
            {"disconnected_route": 1, "route_not_from_boundary": 1},
        ),
        (lambda f: f[5]["vehicle"].update(maxSpeed=0), {"nonpositive_vehicle_value": 1}),
        (lambda f: f[5]["vehicle"].update(maxSpeed=50.0), {"speed_above_lane_limit": 1}),
        (_set(5, "plate", "ABC123"), {"unexpected_field": 1}),
        (_set(5, "startTime", "08:05"), {"wrong_type": 1, "unexpected_string_value": 1}),
    ],
)
def test_each_defect_is_counted(mutate, expected):
    assert _mutated(mutate) == expected


def test_repeated_road_is_counted():
    # a route cannot loop on the toy network, so only the repeat check is asserted
    got = _mutated(_set(5, "route", ["in_0", "out_0", "in_0", "out_0"]))
    assert got["route_repeats_road"] == 1


def test_burst_and_same_lane_insertions_are_reported_not_defects():
    flows = clean_flow() + [toy_vehicle(600) for _ in range(20)]
    for k, v in enumerate(flows[-20:]):
        v["interval"] = 1.0 + (k + 1) * 1e-6  # keep the records distinct (not duplicates)
    res = audit(flows)
    assert res["arrival_burst_bins"] == 1
    assert res["same_second_same_entry_lane"] == 20  # 21 vehicles on one lane in one second
    assert defects(res) == {}


def test_arrival_counts_and_threshold():
    counts = arrival_counts(clean_flow(), EPISODE, 60)
    assert counts.shape == (60,) and set(counts) == {2}
    assert burst_threshold(counts) == 2.0


def test_roadnet_audit():
    raw = toy_roadnet()
    res = audit_roadnet(raw)
    assert set(res) >= set(ROADNET_DEFECTS)
    assert defects(res) == {}
    assert (res["intersections_signalised"], res["intersections_virtual"]) == (1, 4)
    assert (res["roads"], res["lanes"]) == (8, 24)

    bad = copy.deepcopy(raw)
    c = bad["intersections"][0]
    c["roadLinks"].pop()
    c["trafficLight"]["lightphases"][0]["time"] = 3
    c["trafficLight"]["lightphases"].pop()
    bad["roads"][0]["lanes"] = [{"width": 4, "maxSpeed": 0}] * 3
    got = defects(audit_roadnet(bad))
    assert got["intersection_without_12_movements"] == 1
    assert got["transition_phase_not_5s"] == 1
    assert got["intersection_without_9_phases"] == 1
    assert got["nonpositive_lane_value"] == 3
    assert got["invalid_phase_movement_index"] >= 1  # phases still point at the removed link
