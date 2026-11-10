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
