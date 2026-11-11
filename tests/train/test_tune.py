"""Tuning helpers (ROADMAP 5.5): search file, sampler, selection rule."""

import pandas as pd
import pytest

from trafficrl.data.splits import REPO_ROOT
from trafficrl.train.tune import load_search, overrides, sample_trials, select_best

SEARCH = REPO_ROOT / "configs" / "search" / "rl_baselines.yaml"


def test_search_file_is_complete():
    spec = load_search(SEARCH)
    assert spec["trials"] == 8 and spec["seed"] == 0
    assert spec["space"]["train.lr"] == [1e-4, 3e-4, 1e-3]  # floats, not strings


def test_sampler_deterministic_distinct_and_in_space():
    spec = load_search(SEARCH)
    a = sample_trials(spec["space"], 8, spec["sampler_seed"])
    assert a == sample_trials(spec["space"], 8, spec["sampler_seed"])
    assert len({tuple(t.values()) for t in a}) == 8
    for t in a:
        assert all(t[k] in spec["space"][k] for k in spec["space"])
    with pytest.raises(ValueError):
        sample_trials(spec["space"], 37, 0)


def test_overrides_format():
    assert overrides({"train.lr": 0.001, "train.batch_size": 64}) == [
        "train.lr=0.001",
        "train.batch_size=64",
    ]


def test_selection_rule_prefers_lower_lr_within_tolerance():
    df = pd.DataFrame(
        {
            "run_name": ["a", "b", "c", "d"],
            "train.lr": [1e-3, 3e-4, 1e-4, 1e-4],
            "best_val_avg_travel_time_s": [300.0, 300.5, 305.0, None],
        }
    )
    assert select_best(df, "best_val_avg_travel_time_s", 1.0)["run_name"] == "b"
    assert select_best(df, "best_val_avg_travel_time_s", 0.0)["run_name"] == "a"
