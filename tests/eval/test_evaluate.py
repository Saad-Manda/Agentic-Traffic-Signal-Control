import csv
import subprocess
import sys

import pandas as pd
import pytest

from tests.data.test_splits import needs_raw
from tests.envs.test_traffic_env import toy_env
from trafficrl.agents import make_agent
from trafficrl.data.splits import REPO_ROOT
from trafficrl.eval.runner import condition_label, evaluate, run_episode, summary_table
from trafficrl.utils.config import load_config

CFG = REPO_ROOT / "configs" / "eval_baselines.yaml"
SCRIPT = REPO_ROOT / "scripts" / "evaluate.py"


def test_run_episode_adds_agent_metrics():
    env, engine = toy_env(episode=60)
    engine.speeds = {"v": 0.0}
    m = run_episode(env, make_agent("max_pressure", env, {}))
    assert m["decisions"] == 4 and m["fallback_rate"] == 0.0  # t = 0, 15, 30, 45
    assert m["decision_latency_ms"] >= 0 and m["avg_travel_time_s"] == 123.0


def test_condition_label():
    assert condition_label({}) == "clean"
    rob = {"noise_std": 0.1, "dropout_p": 0.2, "surge": 1.5}
    assert condition_label({"robustness": rob}) == "noise0.1-drop0.2-surge1.5"


def test_summary_marks_deterministic_metrics():
    df = pd.DataFrame({"seed": [0, 1, 2], "a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]})
    s = summary_table(df).set_index("metric")
    assert s.loc["a", "deterministic"] and not s.loc["b", "deterministic"]
    assert "seed" not in s.index


@pytest.mark.parametrize("split", ["test", "heldout"])
def test_final_splits_refused_without_flag(split, tmp_path):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--agent", "max_pressure", "--split", split],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert r.returncode == 2 and "final-evaluation only" in r.stderr
    cfg = load_config(CFG)
    cfg["data"]["split"] = split
    with pytest.raises(PermissionError):
        evaluate(cfg, "max_pressure", [0], log_path=tmp_path / "log.csv")


@needs_raw
@pytest.mark.slow
def test_evaluate_writes_outputs_and_log_rows(tmp_path):
    pytest.importorskip("cityflow")
    cfg = load_config(CFG, ["sim.episode_seconds=120", f"output.results_dir={tmp_path}"])
    log = tmp_path / "log.csv"
    out, metrics = evaluate(cfg, "fixed_time", [0, 1], log_path=log)
    assert (out / "metrics.csv").exists() and (out / "summary.csv").exists()
    assert (out / "env.json").exists() and (out / "config.yaml").exists()
    assert len(metrics) == 2 and metrics["avg_travel_time_s"].nunique() == 1  # deterministic
    rows = list(csv.DictReader(log.open()))
    assert [r["seed"] for r in rows] == ["0", "1"]
    assert all(r["split"] == "val" and r["validation_metric"] for r in rows)
