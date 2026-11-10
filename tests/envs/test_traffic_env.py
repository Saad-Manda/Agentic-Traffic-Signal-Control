import numpy as np
import pytest

from tests.data.test_splits import needs_raw
from tests.data.toy import toy_roadnet
from tests.envs.fake_engine import FakeEngine
from trafficrl.data.roadnet import YELLOW_PHASE, parse_roadnet
from trafficrl.data.splits import REPO_ROOT, SplitError
from trafficrl.envs.traffic_env import TrafficEnv, make_env
from trafficrl.utils.config import load_config

BASE = REPO_ROOT / "configs" / "base.yaml"


def toy_env(episode=120, track="full", **rob):
    cfg = load_config(BASE, [f"sim.episode_seconds={episode}"])
    cfg["robustness"] = {"noise_std": 0.0, "dropout_p": 0.0, "surge": 1.0, **rob}
    roadnet = parse_roadnet(toy_roadnet())
    engine = FakeEngine(roadnet)
    return TrafficEnv(cfg, 0, roadnet, lambda: engine, track=track), engine


def test_reset_everything_ready_and_obs_shape():
    env, engine = toy_env()
    obs, info = env.reset()
    assert info["t"] == 0 and info["ready"] == ["C"]
    assert set(obs) == {"C"} and obs["C"].shape == (44,) and obs["C"].dtype == np.float32
    assert env.obs_dim == 44 and env.phase_ids == tuple(range(1, 9))
    assert engine.seeds == [0]


def test_keep_phase_decides_every_delta_t_without_yellow():
    env, engine = toy_env()
    env.reset()
    _, rewards, done, info = env.step({"C": 3})
    assert info["t"] == 15 and info["ready"] == ["C"] and not done
    assert engine.calls == [(0, "C", 3)]
    assert info["green_elapsed"]["C"] == 15
    _, _, _, info = env.step({"C": 3})
    assert info["t"] == 30 and info["green_elapsed"]["C"] == 30
    tr = info["transitions"]
    assert len(tr) == 1 and tr[0].action == 3 and tr[0].duration_s == 15
    assert env.tracker.switches == 0


def test_change_inserts_yellow_on_top_of_delta_t():
    env, engine = toy_env()
    env.reset()
    env.step({"C": 1})
    _, _, _, info = env.step({"C": 2})  # decided at t = 15
    assert engine.calls[-2:] == [(15, "C", YELLOW_PHASE), (20, "C", 2)]
    assert info["t"] == 35 and info["green_elapsed"]["C"] == 15
    assert info["transitions"][0].duration_s == 20
    assert info["phase"]["C"] == 2
    assert env.tracker.switches == 1


def test_phase_one_hot_shows_pending_green_during_yellow():
    env, _ = toy_env()
    env.reset()
    env.step({"C": 1})
    env._apply("C", 6)  # white-box: start a yellow without simulating
    obs, _ = env._observe([], [])
    one_hot = obs["C"][env.features.slices()["phase"]]
    assert one_hot.argmax() == env.phase_ids.index(6)


def test_episode_end_closes_pending_transitions():
    env, _ = toy_env(episode=40)
    env.reset()
    env.step({"C": 1})  # t = 15
    env.step({"C": 2})  # yellow 15-20, green 20-35 -> t = 35
    _, rewards, done, info = env.step({"C": 2})  # would be ready at 50 -> cut at 40
    assert done and info["t"] == 40 and info["ready"] == []
    (tr,) = info["transitions"]
    assert tr.done and tr.duration_s == 5 and set(rewards) == {"C"}
    with pytest.raises(RuntimeError):
        env.step({"C": 1})


def test_reward_is_negative_abs_pressure_from_true_counts():
    env, engine = toy_env(noise_std=0.2, dropout_p=0.5)
    env.reset()
    for k in range(3):
        engine.counts[f"in_0_{k}"] = 10  # 30 in, 0 out -> P = 30
    _, rewards, _, info = env.step({"C": 1})
    assert rewards == {"C": -30.0}
    assert info["lane_count"] != engine.counts  # observation is perturbed


def test_invalid_or_missing_actions():
    env, _ = toy_env()
    env.reset()
    with pytest.raises(ValueError, match="not a green phase"):
        env.step({"C": 0})
    env.reset()
    with pytest.raises(KeyError):
        env.step({})


def test_reset_reuses_engine():
    env, engine = toy_env()
    env.reset()
    env.step({"C": 1})
    env.reset()
    assert engine.resets == 1 and env.t == 0 and engine.seeds == [0, 0]


def test_light_tracking_skips_vehicle_metrics():
    env, engine = toy_env(track="light")
    env.reset()
    engine.speeds = {"v": 0.0}
    while not env.step({"C": 1})[2]:
        pass
    m = env.metrics()
    assert m["throughput_veh"] is None and m["avg_travel_time_s"] == 123.0


@pytest.mark.parametrize("split", ["test", "heldout"])
def test_final_splits_need_allow_final(split):
    cfg = load_config(BASE)
    cfg["data"] = {"dataset": "jinan" if split == "test" else "hangzhou", "split": split}
    with pytest.raises(SplitError, match="final-evaluation split"):
        make_env(cfg, 0)


# --- real network (slow) -------------------------------------------------------------------


def _jinan(split="train", episode=300, **rob):
    cfg = load_config(BASE, [f"sim.episode_seconds={episode}", f"data.split={split}"])
    cfg["robustness"] = {"noise_std": 0.0, "dropout_p": 0.0, "surge": 1.0, **rob}
    return cfg


def _rollout(env, policy):
    obs, info = env.reset()
    log = []
    done = False
    while not done:
        obs, rewards, done, info = env.step(policy(info))
        log.append((info["t"], tuple(info["ready"]), tuple(sorted(rewards.items()))))
    return log, env.metrics()


def _cycle(info):
    return {iid: (info["phase"][iid] % 8) + 1 for iid in info["phase"]}


@needs_raw
@pytest.mark.slow
def test_jinan_env_topology_and_grid():
    pytest.importorskip("cityflow")
    env = make_env(_jinan(), 0)
    assert len(env.intersection_ids) == 12
    index, mask = env.neighbor_index()
    for k, iid in enumerate(env.intersection_ids):
        nbs = [env.intersection_ids[j] for j, m in zip(index[k], mask[k], strict=True) if m]
        assert nbs == env.neighbors(iid)
    assert 2 <= mask.sum(1).min() and mask.sum(1).max() <= 4
    log, m = _rollout(env, _cycle)
    assert all(t % 5 == 0 for t, _, _ in log)
    assert m["vehicles_entered"] > 0 and m["throughput_veh"] >= 0
    assert m["max_waiting_time_s"] >= m["avg_waiting_time_s"] >= 0


@needs_raw
@pytest.mark.slow
def test_jinan_env_is_deterministic_and_reset_matches_fresh():
    pytest.importorskip("cityflow")
    env = make_env(_jinan(noise_std=0.1, dropout_p=0.1), 3)
    a = _rollout(env, _cycle)
    b = _rollout(env, _cycle)  # same env after reset()
    c = _rollout(make_env(_jinan(noise_std=0.1, dropout_p=0.1), 3), _cycle)
    assert a == c
    assert a[1] == b[1]  # engine reset reproduces the traffic (perturbation stream continues)


@needs_raw
@pytest.mark.slow
def test_jinan_val_opens_and_surge_adds_vehicles():
    pytest.importorskip("cityflow")
    env = make_env(_jinan(split="val", surge=1.5), 0)
    n = len(
        __import__("json").loads(
            (
                REPO_ROOT / "data/raw/colight_src/data/Jinan/3_4/anon_3_4_jinan_real_2000.json"
            ).read_text()
        )
    )
    added = env.data_info["surge_vehicles_added"]
    assert 0.4 * n < added < 0.6 * n
    assert env.data_info["split"] == "val"
