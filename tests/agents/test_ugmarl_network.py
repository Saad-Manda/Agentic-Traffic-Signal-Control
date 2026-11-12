"""UG-MARL neighbourhood encoder and network (plan phase6-ugmarl D2, D3)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trafficrl.agents.core import NeighborhoodEncoder, n_params
from trafficrl.agents.ugmarl.network import UGMARLNet, expected_params
from trafficrl.envs.traffic_env import Transition


def line_env():
    """a — b — c from west to east: a's E neighbour is b, b has E = c and W = a."""
    ids = ["a", "b", "c"]
    index = np.zeros((3, 4), np.int64)
    mask = np.zeros((3, 4), bool)
    index[0, 0], mask[0, 0] = 1, True  # a: E = b
    index[1, 0], mask[1, 0] = 2, True  # b: E = c
    index[1, 2], mask[1, 2] = 0, True  # b: W = a
    index[2, 2], mask[2, 2] = 1, True  # c: W = b
    return SimpleNamespace(
        intersection_ids=ids,
        obs_dim=44,
        phase_ids=tuple(range(1, 9)),
        delta_t=15,
        neighbor_index=lambda: (index, mask),
    )


def obs_of(env, scale=1.0):
    return {
        iid: np.full(44, (k + 1) * scale, np.float32) for k, iid in enumerate(env.intersection_ids)
    }


def test_encoder_shape_slots_and_boundary():
    env = line_env()
    enc = NeighborhoodEncoder.from_env(env)
    x = enc.encode_all(obs_of(env))
    assert x.shape == (3, 5, 45) and x.dtype == np.float32 and enc.state_shape == (5, 45)
    # b: self = 2, E (slot 1) = c = 3, N absent, W (slot 3) = a = 1, S absent
    assert x[1, 0, 0] == 2 and x[1, 1, 0] == 3 and x[1, 3, 0] == 1
    assert x[1, :, -1].tolist() == [1, 1, 0, 1, 0]
    assert not x[1, 2].any() and not x[1, 4].any()  # absent slot: all zeros incl. flag
    assert x[0, :, -1].tolist() == [1, 1, 0, 0, 0]  # a: only E


def test_encode_transition_equals_encode_all_row():
    env = line_env()
    enc = NeighborhoodEncoder.from_env(env)
    o1, o2 = obs_of(env), obs_of(env, 2.0)
    s, s2 = enc.encode_transition(Transition("b", o1, 3, -1.0, o2, False, 15))
    assert np.array_equal(s, enc.encode_all(o1)[1]) and np.array_equal(s2, enc.encode_all(o2)[1])


def batch(env, obs=None):
    return torch.as_tensor(NeighborhoodEncoder.from_env(env).encode_all(obs or obs_of(env)))


def test_network_shapes_and_attention_weights():
    env = line_env()
    torch.manual_seed(0)
    net = UGMARLNet(heads=3, attn_dim=16)
    x = batch(env).unsqueeze(0)  # (G=1, B=3, 5, 45)
    q, alpha = net(x, return_attention=True)
    assert q.shape == (1, 3, 3, 8) and alpha.shape == (1, 3, 5)
    assert torch.allclose(alpha.sum(-1), torch.ones(1, 3))
    absent = x[..., -1] < 0.5
    assert (alpha[absent] == 0).all()  # a corner never attends to a missing neighbour
    assert (alpha[0, 0, 2:] == 0).all() and alpha[0, 0, 1] > 0


def test_neighbour_changes_q_only_with_attention():
    env = line_env()
    torch.manual_seed(0)
    nets = {flag: UGMARLNet(heads=2, attn_dim=16, use_attention=flag) for flag in (True, False)}
    o = obs_of(env)
    o2 = {**o, "c": np.zeros(44, np.float32) + 0.3}  # change b's E neighbour only
    for flag, net in nets.items():
        qb, qb2 = net(batch(env, o))[1], net(batch(env, o2))[1]
        assert torch.equal(qb, qb2) != flag
    assert nets[False](batch(env), return_attention=True)[1] is None


def test_stacked_heads_equal_a_loop_of_separate_heads():
    torch.manual_seed(0)
    net = UGMARLNet(heads=4, attn_dim=8)
    z = torch.randn(6, 16)
    want = torch.stack(
        [torch.relu(z @ net.w1[k] + net.b1[k]) @ net.w2[k] + net.b2[k] for k in range(4)],
        dim=-2,
    )
    assert torch.allclose(net.heads_forward(z), want, atol=1e-6)
    assert not torch.equal(net.w1[0], net.w1[1])  # independent initialisation


def test_loss_on_one_head_reaches_only_that_head_and_the_encoder():
    env = line_env()
    torch.manual_seed(0)
    net = UGMARLNet(heads=3, attn_dim=8)
    net(batch(env).unsqueeze(0))[..., 1, :].sum().backward()
    for w in (net.w1, net.b1, net.w2, net.b2):
        assert w.grad[1].abs().sum() > 0
        assert torch.count_nonzero(w.grad[0]) == 0 and torch.count_nonzero(w.grad[2]) == 0
    assert net.embed.weight.grad.abs().sum() > 0 and net.key.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("k,d,att", [(5, 64, True), (3, 32, False), (10, 128, True), (1, 64, True)])
def test_parameter_count_formula(k, d, att):
    net = UGMARLNet(attn_dim=d, heads=k, use_attention=att)
    assert n_params(net) == expected_params(44, 8, d, 128, k, att)


def test_reference_size():
    assert expected_params(44, 8, 64, 128, 5, True) == 15_680 + 5 * 17_544  # plan D3: ≈ 105k
