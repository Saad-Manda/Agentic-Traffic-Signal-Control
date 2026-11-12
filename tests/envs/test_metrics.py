import pandas as pd
import pytest

from trafficrl.envs.metrics import EpisodeTracker


def test_vehicle_metrics_on_a_hand_fed_sequence():
    tr = EpisodeTracker(["a", "b"])
    seq = [{"v1": 5.0}, {"v1": 0.0, "v2": 0.05}, {"v2": 0.0}, {"v2": 3.0}, {}]
    for speeds in seq:
        tr.on_second(speeds)
    tr.on_step({"a": 2, "b": 0, "zzz": 99})  # lanes outside in_lanes ignored
    tr.on_step({"a": 4, "b": 2})
    tr.on_switch()
    m = tr.result(avg_travel_time_s=10.0, in_network=0)
    assert m["vehicles_entered"] == 2 and m["throughput_veh"] == 2
    assert m["avg_waiting_time_s"] == pytest.approx(1.5)  # v1 waited 1 s, v2 2 s
    assert m["max_waiting_time_s"] == 2
    assert m["avg_queue_veh"] == pytest.approx((1 + 3) / 2)
    assert m["phase_switches"] == 1 and m["avg_travel_time_s"] == 10.0


def test_light_tracker_and_bad_level():
    m = EpisodeTracker(["a"], "light").result(5.0, 3)
    assert m["throughput_veh"] is None and m["in_network_at_end"] == 3
    with pytest.raises(ValueError):
        EpisodeTracker(["a"], "medium")


def test_vehicle_frame_and_wait_percentiles():
    tr = EpisodeTracker(["a"])
    seq = [{"v1": 5.0}, {"v1": 0.0, "v2": 0.05}, {"v2": 0.0}, {"v2": 3.0}, {"v2": 3.0}]
    for speeds in seq:
        tr.on_second(speeds)
    f = tr.vehicle_frame().set_index("vehicle")
    assert f.loc["v1", "enter_t"] == 1 and f.loc["v1", "leave_t"] == 3
    assert f.loc["v1", "wait_s"] == 1
    assert f.loc["v2", "enter_t"] == 2 and pd.isna(f.loc["v2", "leave_t"])  # still inside
    assert f.loc["v2", "wait_s"] == 2
    m = tr.result(10.0, 1)
    assert m["max_waiting_time_s"] == 2 and m["throughput_veh"] == 1  # unchanged definitions
    assert m["wait_p95_s"] == pytest.approx(1.95) and m["wait_p99_s"] == pytest.approx(1.99)
    light = EpisodeTracker(["a"], "light")
    assert light.result(5.0, 0)["wait_p95_s"] is None
    with pytest.raises(RuntimeError):
        light.vehicle_frame()
