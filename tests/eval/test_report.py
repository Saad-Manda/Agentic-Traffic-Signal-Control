"""Report helpers (plan D10): val curves over seeds, formatting, run lookup."""

import json

import pandas as pd
import pytest

from trafficrl.eval.report import fmt, latest_eval_run, training_summaries, val_curves
from trafficrl.utils.stats import summarise


def write_run(root, name, att):
    d = root / name
    d.mkdir(parents=True)
    pd.DataFrame({"episode": [5, 10], "avg_travel_time_s": att}).to_csv(
        d / "val_log.csv", index=False
    )
    (d / "summary.json").write_text(json.dumps({"params": 7, "training_time_s": 60.0}))


def test_val_curves_mean_and_ci(tmp_path):
    write_run(tmp_path, "dqn_final_s0", [400.0, 300.0])
    write_run(tmp_path, "dqn_final_s1", [500.0, 310.0])
    c = val_curves(tmp_path, "dqn", "final", [0, 1, 2]).set_index("episode")
    assert c.loc[5, "mean"] == 450.0 and c.loc[5, "n"] == 2
    ref = summarise([400.0, 500.0])
    assert c.loc[5, "ci95_low"] == pytest.approx(ref["ci95_low"])
    assert len(training_summaries(tmp_path, "dqn", "final", [0, 1, 2])) == 2
    with pytest.raises(FileNotFoundError):
        val_curves(tmp_path, "idqn", "final", [0])


def test_fmt_and_latest_run(tmp_path):
    assert fmt(summarise([3.0, 3.0])) == "3.0 (deterministic)"
    assert fmt(summarise([1.0, 3.0])).startswith("2.0 ± 1.4 [")
    for stamp in ("20261110-1", "20261111-1"):
        (tmp_path / f"dqn_jinan-val_clean_{stamp}").mkdir()
    (tmp_path / "dqn_jinan-val_noise0.1_20261112-1").mkdir()
    assert latest_eval_run(tmp_path, "dqn", "jinan", "val").name.endswith("20261111-1")
