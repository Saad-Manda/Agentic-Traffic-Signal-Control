import pytest

from tests.envs.test_traffic_env import toy_env
from trafficrl.agents import FixedTimeAgent, MaxPressureAgent


def test_max_pressure_picks_max_phase_and_breaks_ties_low():
    env, _ = toy_env()
    mp = MaxPressureAgent(env.features)
    assert mp.choose("C", {"in_1_1": 9, "in_3_1": 9}) == 2  # north/south straight pair
    assert mp.choose("C", {}) == 1  # all equal -> lowest id
    assert mp.choose("C", {"in_0_1": 5, "in_0_0": 5, "out_0_0": 1}) == 5  # E straight + left
    obs, info = env.reset()
    info["lane_count"] = {"in_1_1": 9, "in_3_1": 9}
    assert mp.act(obs, info) == {"C": 2}


def test_fixed_time_cycle_in_the_env():
    env, _ = toy_env(episode=35 * 9)
    ft = FixedTimeAgent(env.phase_ids, env.delta_t, green_seconds=30)
    obs, info = env.reset()
    greens = []
    done = False
    while not done:
        actions = ft.act(obs, info)
        greens.append((info["t"], actions["C"]))
        obs, _, done, info = env.step(actions)
    # 30 s green + 5 s yellow per phase: decide at 0, keep at 15, switch at 30, ...
    assert greens[:5] == [(0, 1), (15, 1), (30, 2), (50, 2), (65, 3)]
    order = [p for k, (_, p) in enumerate(greens) if k == 0 or p != greens[k - 1][1]]
    assert order == [1, 2, 3, 4, 5, 6, 7, 8, 1, 2]  # wraps 8 -> 1; 9 cycles of 35 s


def test_fixed_time_rejects_green_not_multiple_of_delta_t():
    with pytest.raises(ValueError, match="multiple"):
        FixedTimeAgent(tuple(range(1, 9)), 15, green_seconds=20)
