"""Attention faithfulness functions on a tiny UG-MARL network (plan phase8-analysis A6)."""

import numpy as np
import pandas as pd
import pytest
import torch

from trafficrl.agents.ugmarl.network import UGMARLNet
from trafficrl.eval import faithfulness as Fa

OBS = 6


@pytest.fixture
def net():
    torch.manual_seed(0)
    return UGMARLNet(obs_dim=OBS, n_actions=4, attn_dim=8, head_hidden=8, heads=3)


def inputs(n=12, seed=0):
    """Rows 0..n-1 alternate between two intersections; group 'a' has E, N, W neighbours,
    group 'b' only E (one neighbour)."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, 5, OBS + 1)).astype(np.float32)
    groups = np.array(["a", "b"] * (n // 2))
    present = np.array([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]], dtype=np.float32)
    x[:, :, -1] = present[np.arange(n) % 2]
    x[:, :, :-1] *= x[:, :, -1:]
    return x, groups


def test_replacing_a_slot_with_itself_changes_nothing(net):
    x, _ = inputs()
    base = Fa.outputs(net, x)
    e = Fa.effect(net, base, Fa.replace_slot(x, 2, np.arange(len(x))))
    assert np.allclose(e["abs_dmu"], 0) and e["flip"].sum() == 0 and np.allclose(e["dsigma"], 0)


def test_knock_out_sets_the_attention_weight_of_that_slot_to_zero(net):
    x, _ = inputs()
    alpha = Fa.outputs(net, Fa.knock_out(x, 1))[3]
    assert np.allclose(alpha[:, 1], 0) and np.allclose(alpha.sum(axis=1), 1, atol=1e-5)


def test_donors_come_from_the_same_intersection_and_are_seeded():
    groups = np.array(["a", "b"] * 6 + ["c"])
    d = Fa.donors(groups, 4, np.random.default_rng(3))
    for r in range(12):
        assert (groups[d[r]] == groups[r]).all() and (d[r] != r).all()
    assert (d[12] == -1).all()  # a single decision has no donor
    assert (d == Fa.donors(groups, 4, np.random.default_rng(3))).all()
    assert (d != Fa.donors(groups, 4, np.random.default_rng(4))).any()


def test_only_present_slots_are_perturbed_and_ranked_among_neighbours(net):
    x, groups = inputs()
    eff = Fa.slot_effects(net, x, groups, draws=3, rng=np.random.default_rng(0))
    rows_b = eff[eff["decision"] % 2 == 1]
    assert set(rows_b["slot"]) == {0, 1}  # group b: self and E only
    assert set(eff.loc[eff["perturbation"] == "knock-out", "slot"]) == {1, 2, 3}
    ranked = Fa.rank_slots(eff)
    assert (ranked["decision"] % 2 == 0).all()  # one neighbour -> skipped
    rep = ranked[ranked["perturbation"] == "replacement"]
    for _, g in rep.groupby("decision"):
        hi, lo = g[g["rank"] == "hi"].iloc[0], g[g["rank"] == "lo"].iloc[0]
        assert hi["alpha"] >= lo["alpha"] and hi["slot"] > 0 and lo["slot"] > 0
        assert (g["rank"] == "self").sum() == 1
    s = Fa.seed_summary(eff)
    assert set(s["slot"]) == {"hi", "lo", "self"} and s["spearman"].between(-1, 1).all()


def test_verdict_needs_a_paired_difference_whose_ci_excludes_zero():
    def frame(hi, lo):
        rows = []
        for s, (h, lw) in enumerate(zip(hi, lo, strict=True)):
            for slot, v in (("hi", h), ("lo", lw), ("self", 1.0)):
                rows.append({"seed": s, "perturbation": "replacement", "slot": slot, "abs_dmu": v})
        return pd.DataFrame(rows)

    yes = Fa.verdict(frame([0.30, 0.31, 0.29], [0.10, 0.12, 0.11]))
    assert yes["consistent"] and yes["text"].startswith("consistent")
    no = Fa.verdict(frame([0.30, 0.10, 0.20], [0.10, 0.30, 0.21]))
    assert not no["consistent"] and "do not rank" in no["text"]
    reverse = Fa.verdict(frame([0.10, 0.12, 0.11], [0.30, 0.31, 0.29]))
    assert not reverse["consistent"]
