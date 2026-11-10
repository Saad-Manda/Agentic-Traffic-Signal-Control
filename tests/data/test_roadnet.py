import copy
from collections import Counter

import pytest

from tests.data.test_splits import needs_raw
from tests.data.toy import toy_roadnet
from trafficrl.data.roadnet import (
    YELLOW_PHASE,
    RoadnetError,
    lane_capacity,
    load_roadnet,
    parse_roadnet,
)
from trafficrl.data.splits import load_splits

# --- hand-built roadnet --------------------------------------------------------------------


def test_canonical_lane_order():
    c = parse_roadnet(toy_roadnet()).intersections["C"]
    assert c.in_roads == ("in_0", "in_1", "in_2", "in_3")
    assert c.out_roads == ("out_0", "out_1", "out_2", "out_3")
    assert c.in_lanes[:4] == ("in_0_0", "in_0_1", "in_0_2", "in_1_0")
    assert len(c.in_lanes) == len(c.out_lanes) == 12


def test_movements_and_phases():
    rn = parse_roadnet(toy_roadnet())
    c = rn.intersections["C"]
    assert Counter(m.kind for m in c.movements) == {
        "go_straight": 4,
        "turn_left": 4,
        "turn_right": 4,
    }
    left = next(m for m in c.movements if m.start_road == "in_0" and m.kind == "turn_left")
    assert left.in_lanes == ("in_0_0",)
    assert left.out_lanes == ("out_1_0", "out_1_1", "out_1_2")  # east-bound turning left = north
    assert YELLOW_PHASE not in c.green_phases
    assert rn.green_phase_ids == (1, 2, 3, 4, 5, 6, 7, 8)
    assert ("in_0", "out_1") in rn.road_links and ("in_0", "out_2") not in rn.road_links


def test_only_signalised_intersections_and_no_neighbours():
    rn = parse_roadnet(toy_roadnet())
    assert list(rn.intersections) == ["C"]
    assert rn.virtual_ids == {"E", "N", "W", "S"}
    assert rn.intersections["C"].neighbors == (None, None, None, None)


def test_lane_capacity():
    rn = parse_roadnet(toy_roadnet())
    cap = lane_capacity(rn, 7.5)
    assert set(cap.values()) == {40}  # 300 m / 7.5 m
    assert len(cap) == 24
    with pytest.raises(ValueError):
        lane_capacity(rn, 0)


def test_malformed_intersection_raises():
    raw = toy_roadnet()
    bad = copy.deepcopy(raw)
    bad["intersections"][0]["roadLinks"].pop()
    with pytest.raises(RoadnetError, match="12 movements"):
        parse_roadnet(bad)

    bad = copy.deepcopy(raw)
    bad["intersections"][0]["roads"].remove("in_3")
    with pytest.raises(RoadnetError, match="incoming"):
        parse_roadnet(bad)

    bad = copy.deepcopy(raw)
    bad["intersections"][0]["trafficLight"]["lightphases"][1]["availableRoadLinks"] = [99]
    with pytest.raises(RoadnetError, match="unknown movement"):
        parse_roadnet(bad)


# --- the real road networks ----------------------------------------------------------------


@needs_raw
@pytest.mark.parametrize(
    ("dataset", "n", "capacities", "neighbour_counts"),
    [
        ("jinan", 12, {53, 106}, {2: 4, 3: 6, 4: 2}),
        ("hangzhou", 16, {80, 106}, {2: 4, 3: 8, 4: 4}),
    ],
)
def test_real_roadnets(dataset, n, capacities, neighbour_counts):
    rn = load_roadnet(load_splits()[dataset]["roadnet"].path)
    assert len(rn.intersections) == n
    for inter in rn.intersections.values():
        assert len(inter.in_lanes) == len(inter.out_lanes) == 12
        assert len(inter.movements) == 12
        # lane index convention: 0 left, 1 through, 2 right
        for m in inter.movements:
            want = {"turn_left": "0", "go_straight": "1", "turn_right": "2"}[m.kind]
            assert {ln.rsplit("_", 1)[1] for ln in m.in_lanes} == {want}
        # road ids end in the direction of travel, matching the geometric order
        assert [r.rsplit("_", 1)[1] for r in inter.in_roads] == ["0", "1", "2", "3"]
    assert rn.green_phase_ids == (1, 2, 3, 4, 5, 6, 7, 8)
    assert set(lane_capacity(rn, 7.5).values()) == capacities
    got = Counter(sum(x is not None for x in i.neighbors) for i in rn.intersections.values())
    assert dict(got) == neighbour_counts
    # neighbour relation is symmetric
    for inter in rn.intersections.values():
        for d, other in enumerate(inter.neighbors):
            if other is not None:
                assert rn.intersections[other].neighbors[(d + 2) % 4] == inter.id
