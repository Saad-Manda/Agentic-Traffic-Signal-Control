"""Q-learning core: replay, networks, schedule, Double-DQN SMDP target (plan phase5 D3–D5)."""

import numpy as np
import pytest
import torch

from trafficrl.agents.core import (
    LinearSchedule,
    LocalEncoder,
    MLPQNet,
    QLearner,
    ReplayBuffer,
    StackedMLPQNet,
    n_params,
)
from trafficrl.envs.traffic_env import Transition


def test_replay_wraps_and_samples_per_group():
    buf = ReplayBuffer(2, 3, (2,), np.random.default_rng(0))
    for k in range(5):
        buf.add(0, np.full(2, k), k % 8, -k, np.full(2, k + 1), 15, False)
    assert buf.size.tolist() == [3, 0] and len(buf) == 0 and buf.total == 3
    with pytest.raises(ValueError):
        buf.sample(4)
    buf.add(1, np.zeros(2), 1, -1.0, np.ones(2), 20, True)
    out = buf.sample(4)
    assert out["state"].shape == (2, 4, 2) and out["action"].shape == (2, 4)
    assert set(out["state"][0, :, 0].tolist()) <= {2.0, 3.0, 4.0}  # 0, 1 overwritten
    assert (out["duration"][1] == 20).all() and out["done"][1].all()


def test_replay_sampling_is_seeded():
    def batch(seed):
        buf = ReplayBuffer(1, 50, (1,), np.random.default_rng(seed))
        for k in range(50):
            buf.add(0, np.array([k]), 0, 0.0, np.array([k]), 15, False)
        return buf.sample(8)["state"]

    assert np.array_equal(batch(1), batch(1)) and not np.array_equal(batch(1), batch(2))


def test_mlp_shapes_and_param_count():
    net = MLPQNet(44, 8, (128, 128), heads=3)
    assert net(torch.zeros(2, 5, 44)).shape == (2, 5, 3, 8)
    assert n_params(MLPQNet(44, 8)) == 44 * 128 + 128 + 128 * 128 + 128 + 128 * 8 + 8  # 23 304


def test_stacked_equals_loop_of_mlps():
    torch.manual_seed(0)
    nets = [MLPQNet(44, 8) for _ in range(3)]
    stacked = StackedMLPQNet(3, 44, 8)
    stacked.load_from(nets)
    x = torch.randn(3, 7, 44)
    want = torch.stack([nets[g](x[g]) for g in range(3)])
    assert torch.allclose(stacked(x), want, atol=1e-6)
    assert n_params(stacked) == 3 * n_params(nets[0])
    with pytest.raises(ValueError):
        stacked(torch.zeros(2, 1, 44))


def test_linear_schedule():
    s = LinearSchedule(1.0, 0.05, 50)
    assert (
        s.value(0) == 1.0
        and s.value(25) == pytest.approx(0.525)
        and s.value(80) == pytest.approx(0.05)
    )


class ConstNet(torch.nn.Module):
    """Q(s, a) = s[0] * (a + 1) for every head — makes targets easy to compute by hand."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        a = torch.arange(1, 4, dtype=x.dtype)
        return (x[..., :1] * a * self.w).unsqueeze(-2)  # (G, B, 1, 3)


def make_learner(net=None, groups_shared=True, n=1, **params):
    ids = [f"i{k}" for k in range(n)]
    return QLearner(
        net or ConstNet(),
        LocalEncoder(ids, 1),
        ids,
        (1, 2, 3),
        15,
        shared=groups_shared,
        params={"gamma": 0.9, "reward_scale": 0.01, **params},
    )


def test_double_dqn_smdp_target_by_hand():
    lr = make_learner()
    # target net differs from online: online picks a*, target evaluates it
    with torch.no_grad():
        lr.target.w.fill_(0.5)
    batch = {
        "next_state": torch.tensor([[[2.0], [2.0]]]),
        "reward": torch.tensor([[-10.0, -10.0]]),
        "duration": torch.tensor([[15.0, 20.0]]),
    }
    y = lr.td_target(batch)  # online argmax = action index 2 (Q = 6); target Q = 3
    assert y.shape == (1, 2, 1)
    assert y[0, 0, 0] == pytest.approx(-0.1 + 0.9 * 3)
    assert y[0, 1, 0] == pytest.approx(-0.1 + 0.9 ** (20 / 15) * 3)


def tr(iid, s, a, r, s2, d=15, done=False):
    return Transition(
        iid, {iid: np.array([s], np.float32)}, a, r, {iid: np.array([s2], np.float32)}, done, d
    )


def test_observe_only_while_training_and_update_schedule():
    lr = make_learner(net=MLPQNet(1, 3, (4,)), learning_starts=2, train_every=2, batch_size=2)
    lr.observe([tr("i0", 1, 1, -1, 1)])
    assert lr.buffer.total == 0  # evaluation mode: nothing stored
    lr.training = True
    lr.observe([tr("i0", 1, 1, -1, 1) for _ in range(3)])  # 3rd one is past learning_starts
    assert lr.buffer.total == 3 and lr.update() == {}  # 1 pending < train_every
    lr.observe([tr("i0", 1, 2, -1, 1)])
    stats = lr.update()
    assert lr.grad_steps == 1 and set(stats) == {"loss", "q_mean", "td_abs", "grad_norm"}


def test_episode_end_still_bootstraps():
    lr = make_learner()
    batch = {
        "next_state": torch.tensor([[[1.0]]]),
        "reward": torch.tensor([[0.0]]),
        "duration": torch.tensor([[5.0]]),
    }
    assert lr.td_target(batch)[0, 0, 0] > 0  # done is never used to cut the target


def test_target_sync_and_epsilon_actions():
    lr = make_learner(
        net=MLPQNet(1, 3, (4,)), learning_starts=0, train_every=1, batch_size=1, target_sync=2
    )
    lr.training = True
    lr.observe([tr("i0", 1, 1, -1, 1), tr("i0", 1, 3, -1, 1)])
    lr.update()
    for a, b in zip(lr.net.parameters(), lr.target.parameters(), strict=True):
        assert torch.equal(a, b)
    lr.epsilon = 1.0
    seen = {lr.act({"i0": np.ones(1, np.float32)}, {}, explore=True)["i0"] for _ in range(200)}
    assert seen == {1, 2, 3}
    greedy = {lr.act({"i0": np.ones(1, np.float32)}, {})["i0"] for _ in range(10)}
    assert len(greedy) == 1
