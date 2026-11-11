"""DQN / IDQN agents (ROADMAP 5.2, 5.3): sharing vs independence, save / load, determinism."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.envs.test_traffic_env import toy_env
from trafficrl.agents import DQNAgent, IDQNAgent, make_agent
from trafficrl.agents.core import n_params
from trafficrl.data.splits import REPO_ROOT
from trafficrl.envs.traffic_env import Transition
from trafficrl.utils.config import load_config

FAST = {"learning_starts": 0, "train_every": 1, "batch_size": 4}


def fake_env(n=3):
    return SimpleNamespace(
        intersection_ids=[f"i{k}" for k in range(n)],
        obs_dim=44,
        phase_ids=tuple(range(1, 9)),
        delta_t=15,
    )


def obs_for(env, value=0.5):
    return {iid: np.full(44, value, np.float32) for iid in env.intersection_ids}


def test_dqn_shares_one_network_idqn_has_one_per_intersection():
    env = fake_env(3)
    dqn, idqn = DQNAgent(env), IDQNAgent(env)
    assert n_params(dqn.net) == 23_304 and n_params(idqn.net) == 3 * 23_304
    assert dqn.buffer.groups == 1 and idqn.buffer.groups == 3
    # shared net + same local state -> same action everywhere
    acts = dqn.act(obs_for(env), {})
    assert len(set(acts.values())) == 1 and set(acts) == set(env.intersection_ids)


def test_idqn_networks_are_independent():
    """A loss on intersection 0's output gives zero gradient to every other network."""
    env = fake_env(3)
    agent = IDQNAgent(env)
    x = torch.randn(3, 5, 44)
    agent.net(x)[0].sum().backward()
    for w in agent.net.weights:
        assert w.grad[0].abs().sum() > 0
        assert torch.count_nonzero(w.grad[1:]) == 0


def test_idqn_learns_per_intersection_from_its_own_buffer():
    env = fake_env(2)
    agent = IDQNAgent(env, FAST)
    agent.training = True
    o = obs_for(env)
    agent.observe([Transition("i0", o, 3, -50.0, o, False, 15) for _ in range(3)])
    agent.observe([Transition("i1", o, 5, -1.0, o, False, 20)])
    assert agent.buffer.size.tolist() == [3, 1]
    assert set(agent.buffer.action[1, :1].tolist()) == {4}  # phase 5 -> index 4
    assert agent.update() and agent.grad_steps == 4


def test_save_load_round_trip(tmp_path):
    env = fake_env(2)
    for cls in (DQNAgent, IDQNAgent):
        a, b = cls(env, seed=0), cls(env, seed=1)
        a.save(tmp_path / f"{cls.name}.pt")
        b.load(tmp_path / f"{cls.name}.pt")
        x = a.encoder.encode_all(obs_for(env, 0.3))
        assert torch.equal(a.q_values(x), b.q_values(x))
    with pytest.raises(ValueError, match="checkpoint is for"):
        IDQNAgent(env).load(tmp_path / "dqn.pt")


def test_same_seed_same_training():
    def run(seed):
        torch.manual_seed(seed)
        env = fake_env(2)
        agent = DQNAgent(env, FAST, seed=seed)
        agent.training = True
        agent.epsilon = 1.0
        rng = np.random.default_rng(0)
        for _ in range(10):
            o = {i: rng.random(44, dtype=np.float32) for i in env.intersection_ids}
            acts = agent.act(o, {}, explore=True)
            agent.observe([Transition(i, o, acts[i], -1.0, o, False, 15) for i in o])
            agent.update()
        return torch.cat([p.flatten() for p in agent.net.parameters()])

    assert torch.equal(run(0), run(0)) and not torch.equal(run(0), run(1))


def test_make_agent_reads_train_block_and_runs_on_toy_env():
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml", ["train.lr=0.001", "seed=3"])
    env, _ = toy_env()
    agent = make_agent("idqn", env, cfg)
    assert agent.params["lr"] == 0.001 and agent.params["hidden"] == [128, 128]
    assert agent.optim.param_groups[0]["lr"] == 0.001
