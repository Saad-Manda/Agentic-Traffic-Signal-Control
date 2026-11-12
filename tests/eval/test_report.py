"""Report helpers (plan D10): val curves over seeds, formatting, run lookup."""

import json

import pandas as pd
import pytest

from trafficrl.eval.report import (
    condition_table,
    fmt,
    latest_eval_run,
    training_summaries,
    val_curves,
)
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


def test_condition_table(tmp_path):
    def run(name, att, wait, fb):
        d = tmp_path / name
        d.mkdir()
        pd.DataFrame(
            {"seed": range(len(att)), "avg_travel_time_s": att, "max_waiting_time_s": wait,
             "fallback_rate": fb}
        ).to_csv(d / "metrics.csv", index=False)  # fmt: skip
        return d

    t = condition_table(
        {
            ("max_pressure", "clean"): run("a", [326.0, 326.0], [100, 100], [0.0, 0.0]),
            ("ugmarl", "noise0.2"): run("b", [420.0, 430.0], [1200, 1300], [0.9, 1.0]),
        }
    ).set_index(["model", "condition"])
    mp, ug = t.loc[("max_pressure", "clean")], t.loc[("ugmarl", "noise0.2")]
    assert mp["deterministic"] and mp["avg_travel_time_std"] == 0.0 and mp["n_seeds"] == 2
    ref = summarise([420.0, 430.0])
    assert ug["avg_travel_time_mean"] == 425.0 and not ug["deterministic"]
    assert ug["avg_travel_time_ci95_low"] == pytest.approx(ref["ci95_low"])
    assert ug["max_waiting_time_mean"] == 1250.0
    assert ug["fallback_rate_mean"] == pytest.approx(0.95)
