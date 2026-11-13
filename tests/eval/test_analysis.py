"""Phase 8 error-analysis functions on hand-made logs (plan phase8-analysis)."""

import numpy as np
import pandas as pd
import pytest

from trafficrl.eval import analysis as A


def trace(times, rows):
    """rows: {iid: [(phase, ready, q_0, q_3), ...]} — one tuple per time; other lanes 0."""
    out = []
    for iid, seq in rows.items():
        for t, (phase, ready, q0, q3) in zip(times, seq, strict=True):
            r = {"t": t, "iid": iid, "phase": phase, "in_yellow": False, "ready": ready}
            r.update({f"q_{k}": 0 for k in range(12)})
            r.update({f"n_{k}": 0 for k in range(12)})
            r["q_0"], r["q_3"] = q0, q3
            out.append(r)
    return pd.DataFrame(out)


FLOW = [
    {"route": ["road_0_1_0", "road_1_1_0"], "startTime": 0},
    {"route": ["road_0_1_0", "road_1_1_1", "road_1_2_1"], "startTime": 10},
    {"route": ["road_0_1_0", "road_1_1_0", "road_2_1_0", "road_3_1_3", "road_3_0_3"],
     "startTime": 20},
]  # fmt: skip
VEH = pd.DataFrame(
    {
        "vehicle": ["flow_0_0", "flow_1_0"],
        "enter_t": [1, 41],
        "leave_t": [101.0, float("nan")],
        "wait_s": [5, 7],
    }
)


def test_vehicle_times_counts_vehicles_inside_to_the_horizon():
    v = A.vehicle_times(VEH, horizon=200)
    assert v["travel_s"].tolist() == [99.0, 159.0]
    assert v["completed"].tolist() == [True, False]


def test_trips_use_scheduled_start_and_keep_vehicles_that_never_entered():
    t = A.trips(VEH, FLOW, horizon=200, max_entry_delay_s=60).set_index("flow")
    assert t["travel_s"].tolist() == [100.0, 190.0, 180.0]  # last: never entered, 200 − 20
    assert t["entry_delay_s"].tolist() == [0.0, 30.0, 180.0]
    assert t["entered"].tolist() == [True, True, False]
    assert t["wait_s"].tolist() == [5, 7, 0]
    assert t["length_group"].tolist() == ["1", "2", "4+"]
    assert t["turn_group"].tolist() == ["straight", "turning", "turning"]
    e = A.entry_summary(A.trips(VEH, FLOW, horizon=200, max_entry_delay_s=60)).iloc[0]
    assert e["never_entered"] == 1 and e["blocked_over_60s"] == 1


@pytest.mark.parametrize(
    "vehicles, msg",
    [
        (VEH.assign(vehicle=["flow_0_0", "car_7"]), "form"),
        (VEH.assign(vehicle=["flow_0_0", "flow_9_0"]), "past the end"),
        (VEH.assign(vehicle=["flow_0_0", "flow_0_1"]), "more than one"),
        (VEH.assign(vehicle=["flow_2_0", "flow_1_0"]), "before"),  # seen at 1, starts at 20
        (VEH.assign(enter_t=[50, 90]), "median entry delay"),
    ],
)
def test_trips_refuse_an_inconsistent_join(vehicles, msg):
    with pytest.raises(A.RouteJoinError, match=msg):
        A.trips(vehicles, FLOW, horizon=200)


def test_window_and_route_tables_recombine_to_the_overall_mean():
    t = A.trips(VEH, FLOW, horizon=200, max_entry_delay_s=60)
    w = A.window_table(t, width=15, horizon=200)
    assert w["window"].tolist() == [0, 1] and w["vehicles"].tolist() == [2, 1]
    assert np.average(w["travel_s"], weights=w["vehicles"]) == pytest.approx(t["travel_s"].mean())
    r = A.route_table(t, "turn_group")
    assert np.average(r["travel_s"], weights=r["vehicles"]) == pytest.approx(t["travel_s"].mean())


def test_intersection_table_weights_queues_by_uneven_step_lengths():
    tr = trace([0, 5, 20], {"a": [(1, True, 2, 0), (1, False, 4, 0), (2, True, 0, 1)]})
    row = A.intersection_table(tr, horizon=30).iloc[0]
    assert row["wait_veh_s"] == 2 * 5 + 4 * 15 + 1 * 10
    assert row["mean_waiting_veh"] == pytest.approx(7 / 3)
    assert row["decisions"] == 2 and row["phase_changes"] == 1
    assert A.network_queue(tr) == pytest.approx(7 / 3 / 12)


def test_flip_flop_definition():
    assert A.flip_flops([1, 2, 1]) == 1
    assert A.flip_flops([1, 2, 2, 1]) == 0  # B held for two decisions
    assert A.flip_flops([1, 2, 3]) == 0  # does not return to A
    assert A.flip_flops([1, 2, 1, 2, 1]) == 3
    assert A.flip_flops([1, 1]) == 0


def test_approach_table_groups_three_lanes_and_finds_the_longest_queue():
    tr = trace([0, 5, 20], {"a": [(1, True, 2, 0), (1, True, 4, 0), (2, True, 0, 1)]})
    a = A.approach_table(tr, horizon=30).set_index("approach")
    assert a.loc["E", "wait_veh_s"] == 70 and a.loc["E", "longest_queue_s"] == 20
    assert a.loc["N", "wait_veh_s"] == 10 and a.loc["N", "longest_queue_s"] == 10
    assert a.loc["W", "wait_veh_s"] == 0


def test_seed_difference_marks_clear_differences():
    a = pd.DataFrame({"iid": ["x"] * 3 + ["y"] * 3, "seed": [0, 1, 2] * 2,
                      "v": [10.0, 11.0, 12.0, 5.0, 6.0, 4.0]})  # fmt: skip
    b = pd.DataFrame({"iid": ["x"] * 3 + ["y"] * 3, "seed": [0, 1, 2] * 2,
                      "v": [5.0, 6.1, 7.0, 5.0, 4.0, 6.0]})  # fmt: skip
    d = A.seed_difference(a, b, ["iid"], "v").set_index("iid")
    assert d.loc["x", "mean"] == pytest.approx(4.9667, abs=1e-3) and d.loc["x", "clear"]
    assert d.loc["y", "mean"] == pytest.approx(0.0) and not d.loc["y", "clear"]


def test_gate_effect_compares_inside_the_same_intersection_and_window():
    tr = trace(
        [0, 15, 30, 45, 60],
        {"a": [(1, True, 4, 0), (1, True, 2, 0), (1, True, 6, 0), (1, True, 6, 0),
               (1, True, 6, 0)]},
    )  # fmt: skip
    dec = pd.DataFrame(
        {"t": [0, 30], "iid": ["a", "a"], "fallback": [True, False],
         "greedy_phase": [1, 1], "mp_phase": [2, 2]}
    )  # fmt: skip
    d = A.after_decision_queue(dec, tr, horizon_s=15)
    assert d["waiting_now"].tolist() == [4, 6]
    assert d["waiting_after"].tolist() == [3.0, 6.0]
    g = A.gate_effect(d, window=600).iloc[0]
    assert g["delta_changed"] == -1.0 and g["delta_quiet"] == 0.0 and g["cells"] == 1


def test_pick_cases_follows_the_rule():
    um = pd.DataFrame({"iid": ["a", "a", "b"], "window": [0, 1, 0], "wait_veh_s": [10, 50, 30]})
    ub = pd.DataFrame({"iid": ["a", "a", "b"], "window": [0, 1, 0], "wait_veh_s": [40, 20, 30]})
    dec = pd.DataFrame(
        {"t": [10, 20, 320, 330, 340], "iid": ["b", "b", "a", "a", "a"],
         "fallback": [True, True, True, True, False],
         "greedy_phase": [1, 1, 1, 2, 1], "mp_phase": [2, 3, 2, 2, 2]}
    )  # fmt: skip
    c = A.pick_cases(um, ub, dec, width=300)
    assert (c["win"]["iid"], c["win"]["window"]) == ("a", 0)
    assert (c["loss"]["iid"], c["loss"]["window"]) == ("a", 1)
    assert (c["gate"]["iid"], c["gate"]["window"]) == ("b", 0)
    assert c["gate"]["gate_changed_decisions"] == 2
    assert A.median_seed({0: 3.0, 1: 1.0, 2: 2.0}) == 2
