"""scripts/tau_sweep.py guards and the pre-fixed reading (plan phase8-analysis, A2)."""

import importlib.util

import pandas as pd
import pytest

from trafficrl.data.splits import REPO_ROOT
from trafficrl.utils.config import read_yaml

SCRIPT = REPO_ROOT / "scripts" / "tau_sweep.py"
SPEC = REPO_ROOT / "configs" / "search" / "tau_sensitivity.yaml"


@pytest.fixture(scope="module")
def ts():
    spec = importlib.util.spec_from_file_location("tau_sweep", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_spec_is_valid_and_validation_only(ts):
    spec = read_yaml(SPEC)
    ts.check_spec(spec)
    assert spec["split"] == "val" and spec["p_star"] == 80


@pytest.mark.parametrize("split", ["train", "test", "heldout"])
def test_refuses_every_split_but_val(ts, split):
    with pytest.raises(ts.SweepError, match="validation split only"):
        ts.check_spec({**read_yaml(SPEC), "split": split})


def test_refuses_a_percentile_of_the_calibration_grid(ts):
    with pytest.raises(ts.SweepError, match="already in the calibration"):
        ts.check_spec({**read_yaml(SPEC), "percentiles": [70, 80]})


def test_thresholds_are_percentiles_per_seed(ts):
    sigma = pd.DataFrame({"seed": [0] * 5 + [1] * 5, "sigma": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]})
    t = ts.thresholds(sigma, [0, 1], [50.0])
    assert t[0][50.0] == 3 and t[1][50.0] == 30
    with pytest.raises(ts.SweepError, match="seed 2"):
        ts.thresholds(sigma, [2], [50.0])


def test_jobs_run_on_validation_with_a_threshold_file_outside_the_training_runs(ts, tmp_path):
    f = ts.tau_file(tmp_path, 60.0)
    assert f.name == "p60_s{seed}.json" and f.parent.name == "taus"
    argv = ts.eval_argv("cfg.yaml", "ckpt", [0, 1], tmp_path, f, {"robustness.noise_std": 0.1}, "n")
    assert argv[argv.index("--split") + 1] == "val" and "--final" not in argv
    assert f"agents.ugmarl.gate.file={f}" in argv and "robustness.noise_std=0.1" in argv
    with pytest.raises(ts.SweepError, match="training run"):
        ts.tau_file(REPO_ROOT / "results" / "train" / "ugmarl_final_s0", 60.0)


RULE = {"insensitive_within_s": 5.0, "tie_tolerance_s": 1.0}


def test_reading_follows_the_rule_fixed_before_the_runs(ts):
    flat = pd.Series({50.0: 344.0, 60.0: 343.5, 70.0: 343.9, 80.0: 343.7, 90.0: 343.7, 99.0: 348.0})
    assert ts.reading(flat, 80.0, RULE).startswith("insensitive")
    better = pd.Series({50.0: 340.0, 70.0: 342.0, 80.0: 343.7, 90.0: 343.7})
    r = ts.reading(better, 80.0, RULE)
    assert "insensitive" in r and "lower end was binding" in r and "p = 50" in r
    worse = pd.Series({50.0: 360.0, 70.0: 350.0, 80.0: 343.7, 90.0: 343.7})
    assert ts.reading(worse, 80.0, RULE) == (
        "lower thresholds score worse: the clean-data cost of more fallback"
    )
