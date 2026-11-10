import csv
import json

from trafficrl.utils.runlog import COLUMNS, append_experiment_row, env_info, save_env_info


def _row(path, **kw):
    args = dict(
        run_id="r1",
        split="train",
        seed=0,
        model="fixed_time",
        hyperparameters={"green_seconds": 30},
        metric_name="avg_travel_time_s",
        metric_value=500.0,
        runtime_s=1.234,
        path=path,
    )
    args.update(kw)
    return append_experiment_row(**args)


def test_header_written_once_and_rows_appended(tmp_path):
    p = tmp_path / "log.csv"
    _row(p)
    _row(p, run_id="r2", seed=1, validation_metric=321.5, notes="a, b")
    lines = p.read_text().splitlines()
    assert lines[0] == ",".join(COLUMNS)
    assert len(lines) == 3
    rows = list(csv.DictReader(p.open()))
    assert [r["run_id"] for r in rows] == ["r1", "r2"]
    assert rows[0]["validation_metric"] == "" and rows[1]["validation_metric"] == "321.5"
    assert json.loads(rows[0]["hyperparameters"]) == {"green_seconds": 30}
    assert rows[1]["notes"] == "a, b"
    assert rows[0]["runtime_s"] == "1.23"
    assert rows[0]["git_commit"]


def test_env_info_keys(tmp_path):
    info = env_info()
    for key in ("platform", "python", "torch", "cuda", "gpu", "cpu", "cpu_count", "git_commit"):
        assert key in info
    p = save_env_info(tmp_path / "run" / "env.json")
    assert json.loads(p.read_text())["python"] == info["python"]
