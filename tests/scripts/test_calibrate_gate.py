"""scripts/calibrate_gate.py guards (plan phase6-ugmarl D9); the τ / p logic is in test_gate."""

import importlib.util
import subprocess
import sys

import pytest

from trafficrl.data.splits import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "calibrate_gate.py"
ARGS = ["--search", "configs/search/ugmarl_gate.yaml", "--config", "configs/eval_ugmarl.yaml"]


def run(*extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *ARGS, *extra], capture_output=True, text=True, cwd=REPO_ROOT
    )


@pytest.mark.parametrize("split", ["train", "test", "heldout"])
def test_refuses_every_split_but_val(split):
    r = run(f"data.split={split}")
    assert r.returncode == 2 and "validation split only" in r.stderr


@pytest.mark.parametrize("flag", ["use_gate", "use_ensemble"])
def test_refuses_variants_without_a_gate(flag):
    r = run(f"agents.ugmarl.{flag}=false")
    assert r.returncode == 2 and "no gate" in r.stderr


def test_missing_checkpoints_refused():
    r = run("--tag", "no_such_tag")
    assert r.returncode == 2 and "missing checkpoint" in r.stderr


def test_condition_names():
    spec = importlib.util.spec_from_file_location("cg", SCRIPT)
    cg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cg)
    assert cg.cond_name({}) == "clean"
    assert cg.cond_name({"robustness.noise_std": 0.1}) == "noise0.1"
    assert cg.cond_name({"robustness.dropout_p": 0.2}) == "drop0.2"
