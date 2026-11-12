"""scripts/robustness.py: job construction and guards (plan val-robustness-references)."""

import importlib.util
import json
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from trafficrl.data.splits import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "robustness.py"
SPEC = {
    "study": "t",
    "dataset": "jinan",
    "split": "val",
    "seeds": [0, 1],
    "tag": "final",
    "models": {
        "max_pressure": {"config": "configs/eval_baselines.yaml"},
        "dqn": {"config": "configs/eval_rl_baselines.yaml"},
        "ugmarl": {"config": "configs/eval_ugmarl.yaml"},
        "ugmarl-nogate": {
            "config": "configs/eval_ugmarl.yaml",
            "agent": "ugmarl",
            "train_model": "ugmarl",
            "overrides": ["agents.ugmarl.use_gate=false"],
        },
    },
    "conditions": [{}, {"robustness.noise_std": 0.1}],
}


@pytest.fixture(scope="module")
def rb():
    spec = importlib.util.spec_from_file_location("rb", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_spec(tmp_path, **changes):
    f = tmp_path / "spec.yaml"
    f.write_text(yaml.safe_dump({**SPEC, **changes}))
    return f


def run(spec, tmp_path, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--spec", str(spec), "--results-dir", str(tmp_path / "r"),
         "--train-dir", str(tmp_path / "train"), "--out", str(tmp_path / "table"), *extra],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )  # fmt: skip


def test_jobs(rb, tmp_path):
    jobs = {(m, c): (d, a) for m, c, d, a in rb.build_jobs(SPEC, tmp_path, tmp_path / "tr", False)}
    assert list(jobs) == [
        (m, c) for m in SPEC["models"] for c in ("clean", "noise0.1")
    ]  # table order
    assert len({d for d, _ in jobs.values()}) == 8  # one results dir per job

    _, mp = jobs[("max_pressure", "noise0.1")]
    assert "--checkpoint" not in mp and "robustness.noise_std=0.1" in mp
    assert mp[mp.index("--seeds") + 1 : mp.index("--seeds") + 3] == ["0", "1"]
    assert "--final" not in mp and mp[mp.index("--split") + 1] == "val"

    _, dqn = jobs[("dqn", "clean")]
    assert dqn[dqn.index("--checkpoint") + 1].endswith("dqn_final_s{seed}/checkpoints/best.pt")
    assert not any(a.startswith("robustness.") for a in dqn)

    d, ng = jobs[("ugmarl-nogate", "noise0.1")]
    assert ng[ng.index("--agent") + 1] == "ugmarl"
    assert ng[ng.index("--checkpoint") + 1].endswith("ugmarl_final_s{seed}/checkpoints/best.pt")
    assert "agents.ugmarl.use_gate=false" in ng and ng[-1] == f"output.results_dir={d}"
    assert "ugmarl-nogate, noise0.1" in ng[ng.index("--note") + 1]

    _, full = jobs[("ugmarl", "clean")]
    assert "agents.ugmarl.use_gate=false" not in full
    assert "--final" in rb.build_jobs(SPEC, tmp_path, tmp_path / "tr", True)[0][3]


def test_missing_inputs_lists_checkpoints_and_gate(rb, tmp_path):
    train = tmp_path / "train"
    for model in ("dqn", "ugmarl"):
        for s in (0, 1):
            d = train / f"{model}_final_s{s}" / "checkpoints"
            d.mkdir(parents=True)
            (d / "best.pt").touch()
    (train / "ugmarl_final_s0" / "gate.json").write_text("{}")
    missing = rb.missing_inputs(SPEC, train)  # gated ugmarl s1 has no gate.json; -gate is fine
    assert missing == [str(train / "ugmarl_final_s1" / "gate.json")]


@pytest.mark.parametrize("split", ["test", "heldout"])
def test_final_splits_refused(tmp_path, split):
    r = run(write_spec(tmp_path, split=split), tmp_path)
    assert r.returncode == 2 and "final-evaluation only" in r.stderr


def test_missing_checkpoint_refused(tmp_path):
    r = run(write_spec(tmp_path), tmp_path)
    assert r.returncode == 2 and "dqn_final_s0" in r.stderr


def test_finished_job_is_found_and_table_only_needs_all_runs(rb, tmp_path):
    spec = write_spec(tmp_path, models={"max_pressure": SPEC["models"]["max_pressure"]})
    job = tmp_path / "r" / "t" / "eval" / "max_pressure__clean"
    assert rb.finished_run(job) is None
    run_dir = job / "max_pressure_jinan-val_clean_x"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        {"seed": [0, 1], "avg_travel_time_s": [326.0, 326.0], "max_waiting_time_s": [10, 20],
         "fallback_rate": [0.0, 0.0]}
    ).to_csv(run_dir / "metrics.csv", index=False)  # fmt: skip
    assert rb.finished_run(job) == run_dir
    r = run(spec, tmp_path, "--table-only")
    assert r.returncode == 1 and "max_pressure__noise0.1" in r.stderr


def test_named_conditions_and_logging_flags(rb, tmp_path):
    spec = {
        **SPEC,
        "log_trace": True,
        "models": {
            "max_pressure": SPEC["models"]["max_pressure"],
            "ugmarl": {"config": "configs/eval_ugmarl.yaml", "log_decisions": True},
        },
        "conditions": [{"name": "real", "data.flow_index": 0}, {"name": "b", "data.flow_index": 1}],
    }
    jobs = {(m, c): (d, a) for m, c, d, a in rb.build_jobs(spec, tmp_path, tmp_path / "tr", True)}
    assert [c for _, c in jobs] == ["real", "b", "real", "b"]
    d, mp = jobs[("max_pressure", "b")]
    assert d.name == "max_pressure__b" and "data.flow_index=1" in mp
    assert not any(a.startswith("name=") for a in mp)
    assert "--log-trace" in mp and "--log-decisions" not in mp
    assert "--log-decisions" in jobs[("ugmarl", "real")][1]
    assert rb.training_runs(spec) == ["ugmarl_final_s0", "ugmarl_final_s1"]
    assert rb.training_runs(SPEC) == [
        "dqn_final_s0", "dqn_final_s1", "ugmarl_final_s0", "ugmarl_final_s1"
    ]  # fmt: skip


def _train_runs(tmp_path):
    from trafficrl.eval.lock import build_manifest

    train = tmp_path / "train"
    for s in (0, 1):
        d = train / f"dqn_final_s{s}" / "checkpoints"
        d.mkdir(parents=True)
        (d / "best.pt").write_bytes(b"w%d" % s)
        (d.parent / "config.yaml").write_text("seed: 0\n")
    return train, build_manifest(train, ["dqn"], [0, 1])


def test_final_needs_matching_manifest(tmp_path):
    train, manifest = _train_runs(tmp_path)
    spec = write_spec(tmp_path, split="test", models={"dqn": SPEC["models"]["dqn"]})
    mf = tmp_path / "manifest.json"
    r = run(spec, tmp_path, "--final", "--manifest", str(mf))
    assert r.returncode == 2 and "lock manifest" in r.stderr  # no manifest
    mf.write_text(json.dumps(manifest))
    (train / "dqn_final_s1" / "checkpoints" / "best.pt").write_bytes(b"changed")
    r = run(spec, tmp_path, "--final", "--manifest", str(mf))
    assert r.returncode == 2 and "dqn_final_s1/checkpoints/best.pt: changed" in r.stderr
    mf.write_text(json.dumps({**manifest, "runs": {}}))  # run not covered by the lock
    r = run(spec, tmp_path, "--final", "--manifest", str(mf))
    assert r.returncode == 2 and "not in the lock manifest" in r.stderr


def test_unfinished_final_job_stops(tmp_path):
    train, manifest = _train_runs(tmp_path)
    spec = write_spec(tmp_path, split="test", models={"dqn": SPEC["models"]["dqn"]})
    mf = tmp_path / "manifest.json"
    mf.write_text(json.dumps(manifest))
    (tmp_path / "r" / "t" / "eval" / "dqn__clean" / "dqn_jinan-test_clean_x").mkdir(parents=True)
    r = run(spec, tmp_path, "--final", "--manifest", str(mf))
    assert r.returncode == 2 and "unfinished final runs ['dqn__clean']" in r.stderr
