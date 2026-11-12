"""Q-learning core: replay, networks, schedule, Double-DQN SMDP target (plan phase5 D3–D5)."""

import copy

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
from trafficrl.agents.core.qlearner import masked_ensemble_loss
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


# --- Phase 6 additions: replay extras, bootstrap masks, masked ensemble loss (plan D4) -----


def test_replay_extras_round_trip_and_wrap():
    buf = ReplayBuffer(1, 3, (1,), np.random.default_rng(0), extras={"mask": ((2,), np.float32)})
    for k in range(5):
        buf.add(0, np.array([k]), 0, 0.0, np.array([k]), 15, False, mask=np.array([k, -k]))
    assert buf.extras["mask"].shape == (1, 3, 2)
    out = buf.sample(16)
    # the mask travels with its transition, and transitions 0, 1 were overwritten
    assert np.array_equal(out["mask"][0, :, 0], out["state"][0, :, 0])
    assert set(out["state"][0, :, 0].tolist()) <= {2.0, 3.0, 4.0}
    with pytest.raises(ValueError, match="extras"):
        buf.add(0, np.zeros(1), 0, 0.0, np.zeros(1), 15, False)
    with pytest.raises(ValueError, match="extras"):
        ReplayBuffer(1, 3, (1,), np.random.default_rng(0)).add(
            0, np.zeros(1), 0, 0.0, np.zeros(1), 15, False, mask=np.ones(2)
        )


def test_replay_extras_do_not_change_the_sampling_stream():
    def batch(extras):
        buf = ReplayBuffer(1, 50, (1,), np.random.default_rng(7), extras=extras)
        for k in range(50):
            kw = {"mask": np.ones(3)} if extras else {}
            buf.add(0, np.array([k]), 0, 0.0, np.array([k]), 15, False, **kw)
        return buf.sample(8)["state"]

    assert np.array_equal(batch(None), batch({"mask": ((3,), np.float32)}))


def test_masked_loss_by_hand_including_an_empty_head():
    q = torch.tensor([[[1.0, 0.0, 5.0], [3.0, 0.0, 5.0]]])  # (G=1, B=2, K=3)
    y = torch.zeros(1, 2, 3)
    m = torch.tensor([[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])  # head 2: no transition
    # head 0: Huber(1) = 0.5 over 1 sample; head 1: 0 over 2; head 2: 0 / max(1, 0)
    assert masked_ensemble_loss(q, y, m).item() == pytest.approx((0.5 + 0.0 + 0.0) / 3)


def test_masked_loss_with_one_head_and_full_mask_is_the_phase5_loss():
    torch.manual_seed(0)
    q, y = torch.randn(1, 64, 1), torch.randn(1, 64, 1)
    want = torch.nn.functional.smooth_l1_loss(q, y)
    assert torch.allclose(masked_ensemble_loss(q, y, torch.ones_like(q)), want)


def _filled(learner, n=20):
    learner.training = True
    rng = np.random.default_rng(1)
    for _ in range(n):
        s, s2 = rng.random(), rng.random()
        learner.observe([tr("i0", s, int(rng.integers(1, 4)), -rng.random(), s2)])
    return learner


def test_single_head_learner_keeps_the_phase5_loss_and_stores_no_mask():
    """Regression for DQN / IDQN: no extras, and the loss equals smooth_l1 on the same batch."""
    torch.manual_seed(0)
    lr = _filled(make_learner(net=MLPQNet(1, 3, (4,)), learning_starts=100, batch_size=8))
    assert lr.heads == 1 and lr.buffer.extras == {}
    twin = copy.deepcopy(lr)  # same weights and same replay RNG state
    raw = twin.buffer.sample(8)
    batch = {k: torch.as_tensor(v) for k, v in raw.items()}
    q = twin.net(batch["state"])
    q_sa = q.gather(-1, batch["action"][..., None, None].expand(*q.shape[:-1], 1)).squeeze(-1)
    want = torch.nn.functional.smooth_l1_loss(q_sa, twin.td_target(batch))
    stats = lr._gradient_step()
    assert stats["loss"] == float(want.detach()) and "sigma_mean" not in stats


def test_ensemble_learner_stores_bernoulli_masks_and_reports_sigma():
    torch.manual_seed(0)
    lr = _filled(
        make_learner(net=MLPQNet(1, 3, (4,), heads=4), learning_starts=0, batch_size=8), n=400
    )
    masks = lr.buffer.extras["mask"][0, :400]
    assert masks.shape == (400, 4) and set(np.unique(masks).tolist()) <= {0.0, 1.0}
    assert abs(masks.mean() - 0.8) < 0.05  # mask_prob default
    stats = lr._gradient_step()
    assert stats["sigma_mean"] > 0


def test_mask_stream_is_seeded_and_separate():
    def masks(seed):
        lr = QLearner(
            MLPQNet(1, 3, (4,), heads=3), LocalEncoder(["i0"], 1), ["i0"], (1, 2, 3), 15,
            shared=True, seed=seed,
        )  # fmt: skip
        return _filled(lr).buffer.extras["mask"][0, :20].copy()

    assert np.array_equal(masks(0), masks(0)) and not np.array_equal(masks(0), masks(1))


def test_shared_checkpoint_loads_on_other_intersections_independent_does_not(tmp_path):
    a = make_learner(net=MLPQNet(1, 3, (4,)), n=2)
    a.save(tmp_path / "shared.pt")
    b = make_learner(net=MLPQNet(1, 3, (4,)), n=5)
    b.load(tmp_path / "shared.pt")
    x = np.ones((5, 1), np.float32)
    assert torch.equal(b.q_values(x)[0], a.q_values(x[:2])[0])
    ind = make_learner(net=StackedMLPQNet(2, 1, 3, (4,)), groups_shared=False, n=2)
    ind.save(tmp_path / "ind.pt")
    with pytest.raises(ValueError, match="intersections"):
        make_learner(net=StackedMLPQNet(3, 1, 3, (4,)), groups_shared=False, n=3).load(
            tmp_path / "ind.pt"
        )
