"""UG-MARL agent: σ, gate, flags, explanations, checkpoints (plan phase6-ugmarl D5, D6)."""

import numpy as np
import pytest
import torch

from tests.agents.test_ugmarl_network import line_env, obs_of
from tests.envs.test_traffic_env import toy_env
from trafficrl.agents import UGMARLAgent, make_agent, model_label
from trafficrl.agents.baselines.max_pressure import MaxPressureAgent
from trafficrl.agents.core import n_params
from trafficrl.agents.ugmarl import expected_params, variant_name
from trafficrl.data.splits import REPO_ROOT
from trafficrl.envs.traffic_env import Transition
from trafficrl.utils.config import load_config

SMALL = {"heads": 3, "attn_dim": 8, "head_hidden": 16}


def agent_for(env, seed=0, **kw):
    return UGMARLAgent(env, {**SMALL, **kw}, seed=seed)


def play(env, agent):
    """One episode; returns the list of action dicts and the explanation dicts."""
    obs, info = env.reset()
    acts, expl, done = [], [], False
    while not done:
        a = agent.act(obs, info)
        acts.append(a)
        expl.append(agent.last_explanations())
        obs, _, done, info = env.step(a)
    return acts, expl


def busy_toy():
    env, engine = toy_env(episode=120)
    # unequal lane counts so Max-Pressure has a clear choice
    rng = np.random.default_rng(0)
    engine.counts = {ln: int(rng.integers(0, 10)) for ln in engine.lanes}
    return env


def test_variant_names_and_model_label():
    assert variant_name({}) == "ugmarl"
    assert variant_name({"use_attention": False}) == "ugmarl-noattn"
    assert variant_name({"use_ensemble": False, "use_gate": True}) == "ugmarl-noens"
    assert variant_name({"use_gate": False}) == "ugmarl-nogate"
    cfg = load_config(REPO_ROOT / "configs" / "ugmarl_noattn.yaml")
    assert model_label("ugmarl", cfg) == "ugmarl-noattn" and model_label("dqn", cfg) == "dqn"


def test_flags_build_the_right_network():
    env = line_env()
    full, noattn, noens = (
        agent_for(env),
        agent_for(env, use_attention=False),
        agent_for(env, use_ensemble=False),
    )
    assert full.heads == 3 and noens.heads == 1 and noattn.heads == 3
    assert n_params(full.net) == expected_params(44, 8, 8, 16, 3, True)
    assert n_params(noattn.net) == expected_params(44, 8, 8, 16, 3, False)
    assert full.buffer.extras and not noens.buffer.extras  # masks only with an ensemble
    assert not noens.gate_requested and full.gate_requested and not full.gate_active


def test_tau_none_is_ungated_and_explanations_have_every_key():
    env = line_env()
    agent = agent_for(env)
    acts = agent.act(obs_of(env), {})
    e = agent.last_explanations()
    assert set(e) == {"fallback", "sigma", "tau", "greedy_phase", "mp_phase", "attention", "q_mean"}
    assert e["tau"] is None and not any(e["fallback"].values())
    assert acts == e["greedy_phase"]
    assert e["attention"]["b"].shape == (5,) and e["q_mean"]["a"].shape == (8,)
    assert all(s > 0 for s in e["sigma"].values())  # independent heads disagree at init


def test_sigma_is_zero_and_never_fires_without_ensemble():
    env = busy_toy()
    agent = UGMARLAgent(env, {**SMALL, "use_ensemble": False, "gate": {"tau": -1.0}})
    _, expl = play(env, agent)
    assert all(s == 0.0 for e in expl for s in e["sigma"].values())
    assert not any(f for e in expl for f in e["fallback"].values())


def test_tau_minus_one_always_fires_and_equals_max_pressure():
    env = busy_toy()
    gated, _ = play(env, UGMARLAgent(env, {**SMALL, "gate": {"tau": -1.0}}))
    mp, _ = play(env, MaxPressureAgent(env.features))
    assert gated == mp


def test_tau_infinite_never_fires_and_equals_gate_off():
    env = busy_toy()
    torch.manual_seed(4)  # weights come from the global torch RNG (seed_everything in scripts)
    a = play(env, UGMARLAgent(env, {**SMALL, "gate": {"tau": float("inf")}}))
    torch.manual_seed(4)
    b = play(env, UGMARLAgent(env, {**SMALL, "use_gate": False, "gate": {"tau": -1.0}}))
    assert a[0] == b[0] and not any(f for e in a[1] for f in e["fallback"].values())


def test_gate_compares_sigma_with_tau_strictly():
    env = busy_toy()
    agent = UGMARLAgent(env, SMALL)
    obs, info = env.reset()
    agent.act(obs, info)
    e = agent.last_explanations()
    s = e["sigma"]["C"]
    agent.tau = s  # σ ≤ τ -> learned action
    assert agent.act(obs, info)["C"] == e["greedy_phase"]["C"]
    agent.tau = s - 1e-6  # σ > τ -> Max-Pressure
    assert agent.act(obs, info)["C"] == MaxPressureAgent(env.features).choose(
        "C", info["lane_count"]
    )
    assert agent.last_explanations()["fallback"]["C"]


def test_exploration_never_gates():
    env = busy_toy()
    agent = UGMARLAgent(env, {**SMALL, "gate": {"tau": -1.0}})
    agent.epsilon = 0.0
    obs, info = env.reset()
    greedy = agent.act(obs, info, explore=True)
    assert agent.last_explanations() == {}
    agent.tau = float("inf")
    assert greedy == agent.act(obs, info)


def test_save_load_round_trip_and_architecture_check(tmp_path):
    env = line_env()
    a = agent_for(env, seed=0)
    a.save(tmp_path / "a.pt")
    b = agent_for(env, seed=1, use_gate=False)  # use_gate may differ: that is −gate
    b.load(tmp_path / "a.pt")
    x = a.encoder.encode_all(obs_of(env))
    assert torch.equal(a.forward_all(x)[0], b.forward_all(x)[0])
    for kw in ({"heads": 5}, {"attn_dim": 16}, {"use_attention": False}, {"use_ensemble": False}):
        with pytest.raises(ValueError, match="architecture"):
            agent_for(env, **kw).load(tmp_path / "a.pt")
    with pytest.raises(ValueError, match="checkpoint is for"):
        make_agent("dqn", env, {}).load(tmp_path / "a.pt")


def test_shared_agent_loads_on_a_network_of_another_size(tmp_path):
    big = line_env()
    agent_for(big).save(tmp_path / "a.pt")
    small, _ = toy_env()
    UGMARLAgent(small, SMALL).load(tmp_path / "a.pt")  # 3 intersections -> 1


def test_same_seed_same_training_different_seed_differs():
    def run(seed):
        torch.manual_seed(seed)
        env = line_env()
        agent = agent_for(env, seed=seed, learning_starts=0, train_every=1, batch_size=4)
        agent.training, agent.epsilon = True, 1.0
        rng = np.random.default_rng(0)
        for _ in range(10):
            o = {i: rng.random(44, dtype=np.float32) for i in env.intersection_ids}
            acts = agent.act(o, {}, explore=True)
            agent.observe([Transition(i, o, acts[i], -1.0, o, False, 15) for i in o])
            agent.update()
        return torch.cat([p.flatten() for p in agent.net.parameters()])

    assert torch.equal(run(0), run(0)) and not torch.equal(run(0), run(1))


def test_make_agent_reads_ugmarl_block():
    cfg = load_config(REPO_ROOT / "configs" / "ugmarl.yaml", ["agents.ugmarl.heads=3", "seed=2"])
    env, _ = toy_env()
    agent = make_agent("ugmarl", env, cfg)
    assert agent.heads == 3 and agent.params["lr"] == 1e-4 and agent.tau is None
