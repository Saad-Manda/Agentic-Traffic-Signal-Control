"""Zero-shot code path for the held-out city (plan phase7): 16 intersections, shared network
loaded from a Jinan-shaped checkpoint.

No Hangzhou **flow** is opened here (protocol §1: the held-out flows are simulated only in
the final run). The test uses the Hangzhou *roadnet* — already read by the audit and
``test_roadnet`` — with a small synthetic flow of two-road routes built from its road links.
"""

import json
import shutil

import pytest

from tests.data.test_splits import needs_raw
from trafficrl.agents import make_agent
from trafficrl.data.roadnet import load_roadnet
from trafficrl.data.splits import REPO_ROOT, load_splits
from trafficrl.envs.engine import make_engine
from trafficrl.envs.traffic_env import TrafficEnv
from trafficrl.eval.runner import run_episode
from trafficrl.utils.config import load_config

pytestmark = [needs_raw, pytest.mark.slow]

VEHICLE = {"length": 5.0, "width": 2.0, "maxPosAcc": 2.0, "maxNegAcc": 4.5, "usualPosAcc": 2.0,
           "usualNegAcc": 4.5, "minGap": 2.5, "maxSpeed": 11.111, "headwayTime": 2.0}  # fmt: skip
SMALL = ["agents.dqn.hidden=[16]", "agents.idqn.hidden=[16]", "agents.ugmarl.heads=3",
         "agents.ugmarl.attn_dim=8", "agents.ugmarl.head_hidden=16",
         "sim.episode_seconds=120"]  # fmt: skip


def synthetic_env(dataset, tmp_path, cfg):
    """TrafficEnv on the dataset's roadnet with 60 made-up vehicles (never a real flow)."""
    src = load_splits()[dataset]["roadnet"].path
    d = tmp_path / dataset
    d.mkdir()
    roadnet_path = shutil.copy(src, d / "roadnet.json")
    roadnet = load_roadnet(roadnet_path)
    links = sorted(roadnet.road_links)[:: max(1, len(roadnet.road_links) // 60)][:60]
    flow = [
        {"vehicle": VEHICLE, "route": list(link), "interval": 1.0, "startTime": k, "endTime": k}
        for k, link in enumerate(links)
    ]
    (d / "flow.json").write_text(json.dumps(flow))
    return TrafficEnv(
        cfg, 0, roadnet, lambda: make_engine(d / "roadnet.json", d / "flow.json", 0, 1, d)
    )


@pytest.mark.parametrize("name", ["fixed_time", "max_pressure", "dqn", "ugmarl"])
def test_agents_run_on_the_hangzhou_roadnet(name, tmp_path):
    pytest.importorskip("cityflow")
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml", SMALL)
    env = synthetic_env("hangzhou", tmp_path, cfg)
    assert len(env.intersection_ids) == 16
    agent = make_agent(name, env, cfg)
    if name in ("dqn", "ugmarl"):  # weights from a Jinan-shaped (12-intersection) agent
        jinan = synthetic_env("jinan", tmp_path, cfg)
        assert len(jinan.intersection_ids) == 12
        ckpt = tmp_path / "jinan.pt"
        make_agent(name, jinan, cfg).save(ckpt)
        agent.load(ckpt)
    if name == "ugmarl":
        agent.tau = 0.0  # gate fires whenever σ > 0: exercises the Max-Pressure fallback too
    obs, info = env.reset()
    actions = agent.act(obs, info, explore=False)
    assert set(actions) == set(env.intersection_ids)
    assert all(a in env.phase_ids for a in actions.values())
    m = run_episode(env, agent)
    assert m["decisions"] >= 16 * 6 and m["vehicles_entered"] > 0
    if name == "ugmarl":
        assert m["fallback_rate"] > 0


def test_idqn_checkpoint_is_not_transferable(tmp_path):
    pytest.importorskip("cityflow")
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml", SMALL)
    ckpt = tmp_path / "jinan.pt"
    make_agent("idqn", synthetic_env("jinan", tmp_path, cfg), cfg).save(ckpt)
    agent = make_agent("idqn", synthetic_env("hangzhou", tmp_path, cfg), cfg)
    with pytest.raises(ValueError, match="different set of intersections"):
        agent.load(ckpt)
