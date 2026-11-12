"""scripts/final_tables.py on fake finished studies (plan phase7): tables and verdicts."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from trafficrl.data.splits import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "final_tables.py"
SEEDS = [0, 1, 2]
CFG = {"config": "configs/eval_ugmarl.yaml"}
MODELS = {
    "fixed_time": {}, "max_pressure": {}, "dqn": {}, "idqn": {}, "ugmarl": CFG,
    "ugmarl-nogate": {**CFG, "agent": "ugmarl", "train_model": "ugmarl"},
    "ugmarl-noattn": {**CFG, "agent": "ugmarl"}, "ugmarl-noens": {**CFG, "agent": "ugmarl"},
}  # fmt: skip
# travel time (clean, noise0.2) and max wait (clean, noise0.2) per model
NUMBERS = {
    "fixed_time": (435, 435, 1124, 1124), "max_pressure": (326, 415, 1334, 1277),
    "dqn": (300, 855, 640, 3400), "idqn": (313, 767, 720, 2900),
    "ugmarl": (302, 423, 760, 1240), "ugmarl-nogate": (301.5, 909, 690, 3300),
    "ugmarl-noattn": (302, 425, 650, 1250), "ugmarl-noens": (304, 900, 710, 3300),
}  # fmt: skip


def write_run(study, model, cond, tt, wait, fallback=0.0, decisions=False):
    d = study / "eval" / f"{model}__{cond}" / "run"
    d.mkdir(parents=True)
    jitter = np.array([-1.0, 0.0, 1.0]) * (0 if model in ("fixed_time", "max_pressure") else 1)
    pd.DataFrame(
        {"seed": SEEDS, "avg_travel_time_s": tt + jitter, "avg_waiting_time_s": 50 + jitter,
         "avg_queue_veh": 0.5, "throughput_veh": 4000, "max_waiting_time_s": wait + 10 * jitter,
         "wait_p95_s": 200.0, "wait_p99_s": 300.0, "fallback_rate": fallback,
         "decision_latency_ms": 1.0, "decision_latency_ms_p95": 2.0}
    ).to_csv(d / "metrics.csv", index=False)  # fmt: skip
    if decisions:
        for s in SEEDS:
            pd.DataFrame(
                {"t": [0, 0, 700, 700], "iid": ["a", "b", "a", "b"], "sigma": [1.0, 3.0, 1.0, 3.0],
                 "tau": 2.0, "fallback": [False, True, False, True],
                 "greedy_phase": [1, 2, 3, 4], "mp_phase": [1, 3, 3, 4], "phase_taken": 1,
                 "attn_self": 0.6, "attn_E": 0.2, "attn_N": 0.2, "attn_W": 0.0, "attn_S": 0.0,
                 "q_gap": 0.1, "in_veh_observed": 5.0}
            ).to_csv(d / f"decisions_s{s}.csv", index=False)  # fmt: skip


def test_tables_and_verdicts(tmp_path):
    results, train, out = tmp_path / "final", tmp_path / "train", tmp_path / "docs"
    spec = {"study": "ft", "dataset": "jinan", "split": "test", "seeds": SEEDS, "tag": "final",
            "models": MODELS, "conditions": [{}, {"robustness.noise_std": 0.2}]}  # fmt: skip
    hspec = {**spec, "study": "fh", "dataset": "hangzhou", "split": "heldout",
             "models": {k: v for k, v in MODELS.items() if k != "idqn"},
             "conditions": [{"name": "real", "data.flow_index": 0},
                            {"name": "real_5734", "data.flow_index": 1}]}  # fmt: skip
    (tmp_path / "t.yaml").write_text(yaml.safe_dump(spec))
    (tmp_path / "h.yaml").write_text(yaml.safe_dump(hspec))
    for m, (tc, tn, wc, wn) in NUMBERS.items():
        gated = m in ("ugmarl", "ugmarl-noattn")
        write_run(results / "ft", m, "clean", tc, wc, 0.2 * gated, decisions=m == "ugmarl")
        write_run(results / "ft", m, "noise0.2", tn, wn, 0.9 * gated, decisions=m == "ugmarl")
        if m != "idqn":
            for flow in ("real", "real_5734"):
                write_run(results / "fh", m, flow, tc + 100, wc, 0.5 * gated)
    for m in ("dqn", "idqn", "ugmarl", "ugmarl-noattn", "ugmarl-noens"):
        for s in SEEDS:
            d = train / f"{m}_final_s{s}"
            d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps({"training_time_s": 600, "params": 1234}))
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--study", "all", "--test-spec", str(tmp_path / "t.yaml"),
         "--heldout-spec", str(tmp_path / "h.yaml"), "--results-dir", str(results),
         "--train-dir", str(train), "--out-dir", str(out),
         "--val-table", str(tmp_path / "none.csv")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    for name in ("test_main", "test_ablation", "test_robustness", "test_efficiency",
                 "test_safety_explain", "heldout_generalisation"):  # fmt: skip
        assert (out / f"{name}.md").exists() and (out / f"{name}.csv").exists()

    assert "H1: supported" in r.stdout and "H3: not supported" in r.stdout
    assert "H2: partly supported" in r.stdout  # clean max wait 760 > 690, noise results better
    assert "minimum_outcome: True" in r.stdout and "vs_dqn: level" in r.stdout
    assert "smallest_worst_case_regret: ugmarl" in r.stdout

    main = (out / "test_main.md").read_text()
    assert "**H1 (performance): supported.**" in main and "| 1,234 |" in main
    assert "Final evaluation: every episode was run once" in main
    rob = pd.read_csv(out / "test_robustness.csv").set_index(["model", "condition"])
    assert rob.loc[("ugmarl", "noise0.2"), "regret"] == 8  # 423 − Max-Pressure 415
    assert rob.loc[("dqn", "clean"), "regret"] == 0
    held = (out / "heldout_generalisation.md").read_text()
    assert "| IDQN | not transferable |" in held and "Jinan test (clean)" in held
    safe = pd.read_csv(out / "test_safety_explain.csv").set_index(["model", "condition"])
    row = safe.loc[("ugmarl", "clean")]
    assert row["fallback_rate"] == 0.5 and row["differs_given_fired"] == 0.5
    assert row["policy_differs_from_mp"] == 0.25 and row["sigma_over_tau"] == 0.5
    assert "0–10" in (out / "test_safety_explain.md").read_text()


def test_missing_job_is_an_error(tmp_path):
    spec = {"study": "ft", "dataset": "jinan", "split": "test", "seeds": SEEDS, "tag": "final",
            "models": {"dqn": {}}, "conditions": [{}]}  # fmt: skip
    (tmp_path / "t.yaml").write_text(yaml.safe_dump(spec))
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--study", "test", "--test-spec", str(tmp_path / "t.yaml"),
         "--results-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )  # fmt: skip
    assert r.returncode == 1 and "dqn__clean" in r.stderr
