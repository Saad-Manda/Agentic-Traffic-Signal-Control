"""Phase 7 verdict rules (fixed before the test flow was opened; ROADMAP Phase 7 set-up)."""

import pandas as pd
import pytest

from trafficrl.eval import hypotheses as hyp

UG = [300.0, 301.0, 302.0, 303.0, 304.0]  # mean 302, 95 % CI about [300.0, 304.0]
MP = [326.0] * 5  # deterministic baseline: its CI is its value


def test_ci_rule_and_compare():
    assert hyp.ci_below(UG, MP) and not hyp.ci_below(MP, UG)
    assert hyp.compare(UG, MP) == "better" and hyp.compare(MP, UG) == "worse"
    assert hyp.compare(UG, [301.0, 302.0, 303.0, 304.0, 305.0]) == "level"  # CIs overlap
    assert not hyp.ci_below(UG, [303.9] * 5)  # UG upper end 303.96 is not below 303.9


def test_h1_needs_both_baselines():
    idqn = [310.0, 312.0, 314.0, 316.0, 318.0]  # CI about [310.1, 317.9]
    r = hyp.h1(UG, idqn, MP)
    assert r == {"below_max_pressure": True, "below_idqn": True, "verdict": "supported"}
    r = hyp.h1(UG, [303.0, 304.0, 305.0, 306.0, 307.0], MP)  # overlaps UG's CI
    assert r["below_max_pressure"] and not r["below_idqn"] and r["verdict"] == "not supported"
    assert hyp.h1(UG, idqn, [303.0] * 5)["verdict"] == "not supported"


def test_h3():
    assert hyp.h3(UG, [310.0, 311.0, 312.0, 313.0, 314.0])["verdict"] == "supported"
    assert hyp.h3(UG, [300.5, 301.5, 302.5, 303.5, 304.5])["verdict"] == "not supported"


def side(wc, wn, tc, tn):
    return {"wait_clean": wc, "wait_noise": wn, "tt_clean": tc, "tt_noise": tn}


def test_h2_three_levels_and_paired_differences():
    no_gate = side([700, 700], [3000, 3400], [300.0, 302.0], [900.0, 920.0])
    full = hyp.h2(side([650, 690], [1200, 1300], [301.0, 303.0], [420.0, 430.0]), no_gate)
    assert full["verdict"] == "supported"
    assert full["increase_gate"] == pytest.approx(123.0)
    assert full["increase_no_gate"] == pytest.approx(609.0)
    assert full["paired_wait_noise"]["mean"] == pytest.approx(-1950.0)
    assert full["paired_increase"]["mean"] == pytest.approx(-486.0)

    partly = hyp.h2(side([760, 760], [1200, 1300], [301.0, 303.0], [420.0, 430.0]), no_gate)
    assert not partly["i_max_wait_clean_lower"] and partly["verdict"] == "partly supported"

    no = hyp.h2(side([650, 690], [3300, 3500], [301.0, 303.0], [420.0, 430.0]), no_gate)
    assert not no["ii_max_wait_noise_lower"] and no["verdict"] == "not supported"
    no = hyp.h2(side([650, 690], [1200, 1300], [301.0, 303.0], [1000.0, 1100.0]), no_gate)
    assert not no["iii_degradation_smaller"] and no["verdict"] == "not supported"
    with pytest.raises(ValueError):
        hyp.paired([1.0], [1.0, 2.0])


def test_minimum_outcome():
    assert hyp.minimum_outcome([700, 800], [1334, 1334])["met"]
    assert hyp.minimum_outcome([1334, 1334], [1334, 1334])["met"]  # "no worse" includes equal
    assert not hyp.minimum_outcome([1400, 1400], [1334, 1334])["met"]


def test_regret_against_the_reference_agents_only():
    means = pd.DataFrame(
        [("dqn", "clean", 300.0), ("mp", "clean", 326.0), ("ug", "clean", 302.0),
         ("abl", "clean", 299.0),  # not a reference model: may have negative regret
         ("dqn", "noise", 855.0), ("mp", "noise", 415.0), ("ug", "noise", 423.0),
         ("abl", "noise", 909.0)],
        columns=["model", "condition", "mean"],
    )  # fmt: skip
    r = hyp.regret_table(means, ["dqn", "mp", "ug"])
    get = lambda m, c: r[(r.model == m) & (r.condition == c)]["regret"].item()  # noqa: E731
    assert get("dqn", "clean") == 0 and get("ug", "clean") == 2 and get("abl", "clean") == -1
    assert get("mp", "noise") == 0 and get("dqn", "noise") == 440
    s = hyp.regret_summary(r).set_index("model")
    assert s.loc["ug", "max_regret"] == 8 and s.loc["ug", "worst_condition"] == "noise"
    assert s.loc["dqn", "max_regret"] == 440 and s.loc["mp", "mean_regret"] == 13
