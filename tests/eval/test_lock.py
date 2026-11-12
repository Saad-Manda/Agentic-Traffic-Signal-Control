"""Lock manifest (plan phase7 P8): hashes of the files a final evaluation reads."""

import json
import subprocess
import sys

import pytest

from trafficrl.data.splits import REPO_ROOT
from trafficrl.eval.lock import build_manifest, uncovered, verify_manifest

SCRIPT = REPO_ROOT / "scripts" / "lock_models.py"


def make_runs(train):
    for model in ("dqn", "ugmarl"):
        d = train / f"{model}_final_s0" / "checkpoints"
        d.mkdir(parents=True)
        (d / "best.pt").write_bytes(model.encode())
        (d.parent / "config.yaml").write_text("seed: 0\n")
        (d.parent / "summary.json").write_text(
            json.dumps({"best_val_avg_travel_time_s": 300.0, "best_episode": 5, "params": 7})
        )
    (train / "ugmarl_final_s0" / "gate.json").write_text(json.dumps({"tau": 0.5, "p": 80.0}))


def test_build_and_verify(tmp_path):
    make_runs(tmp_path)
    m = build_manifest(tmp_path, ["dqn", "ugmarl"], [0])
    assert set(m["runs"]) == {"dqn_final_s0", "ugmarl_final_s0"}
    ug = m["runs"]["ugmarl_final_s0"]
    assert set(ug["files"]) == {"checkpoints/best.pt", "config.yaml", "gate.json"}
    assert ug["tau"] == 0.5 and ug["p"] == 80.0 and ug["best_val_avg_travel_time_s"] == 300.0
    assert "gate.json" not in m["runs"]["dqn_final_s0"]["files"]
    assert verify_manifest(m, tmp_path) == []
    assert uncovered(m, ["dqn_final_s0", "idqn_final_s0"]) == ["idqn_final_s0"]

    (tmp_path / "ugmarl_final_s0" / "gate.json").write_text(json.dumps({"tau": 0.6}))
    (tmp_path / "dqn_final_s0" / "checkpoints" / "best.pt").unlink()
    assert verify_manifest(m, tmp_path) == [
        "dqn_final_s0/checkpoints/best.pt: missing",
        "ugmarl_final_s0/gate.json: changed since the lock",
    ]


def test_missing_gate_refused(tmp_path):
    make_runs(tmp_path)
    (tmp_path / "ugmarl_final_s0" / "gate.json").unlink()
    with pytest.raises(FileNotFoundError, match="gate.json"):
        build_manifest(tmp_path, ["ugmarl"], [0])


def test_script_write_then_verify(tmp_path):
    make_runs(tmp_path)
    base = [sys.executable, str(SCRIPT), "--manifest", str(tmp_path / "m.json"), "--train-dir",
            str(tmp_path), "--models", "dqn", "ugmarl", "--seeds", "0"]  # fmt: skip
    kw = {"capture_output": True, "text": True, "cwd": REPO_ROOT}
    assert subprocess.run([*base, "--verify"], **kw).returncode == 2  # nothing written yet
    assert subprocess.run([*base, "--write"], **kw).returncode == 0
    assert subprocess.run([*base, "--verify"], **kw).returncode == 0
    (tmp_path / "dqn_final_s0" / "config.yaml").write_text("seed: 1\n")
    r = subprocess.run([*base, "--verify"], **kw)
    assert r.returncode == 2 and "changed since the lock" in r.stderr
