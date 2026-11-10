import json
import re

import pandas as pd
import pytest

from tests.data.test_splits import needs_raw
from tests.data.toy import toy_roadnet
from trafficrl.data.roadnet import load_roadnet, parse_roadnet
from trafficrl.data.splits import REPO_ROOT, SplitError, load_splits
from trafficrl.data.trace import (
    CONTROLLERS,
    collect_trace,
    make_chooser,
    make_engine,
    train_flow,
)
from trafficrl.envs.features import FeatureBuilder
from trafficrl.utils.config import load_config

EDA_CONFIG = REPO_ROOT / "configs" / "eda.yaml"
NOTEBOOK = REPO_ROOT / "notebooks" / "01_eda_train.ipynb"

# --- leakage guards ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dataset", "split"),
    [
        ("jinan", "val"),
        ("jinan", "test"),
        ("jinan", "heldout"),
        ("hangzhou", "train"),
        ("hangzhou", "heldout"),
        ("nowhere", "train"),
        (None, None),
    ],
)
def test_traces_refuse_everything_but_the_train_split(dataset, split):
    with pytest.raises(SplitError, match="train split only"):
        train_flow({"data": {"dataset": dataset, "split": split}})


def test_eda_config_points_at_the_train_split():
    cfg = load_config(EDA_CONFIG)
    assert cfg["data"] == {"dataset": "jinan", "split": "train"}
    assert set(cfg["eda"]["controllers"]) == set(CONTROLLERS)


def test_audit_script_plots_train_only():
    src = (REPO_ROOT / "scripts" / "audit_data.py").read_text()
    assert re.search(r'^PLOT_SPLITS = \("train",\)', src, flags=re.M)


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="EDA notebook not written yet")
def test_notebook_reads_only_the_train_split():
    """The EDA notebook must not name a val / test / held-out file or split."""
    nb = json.loads(NOTEBOOK.read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    splits = load_splits()
    forbidden = [
        f.path.name
        for name, ds in splits.items()
        for split, files in ds["splits"].items()
        if split != "train"
        for f in files
    ] + [ds["roadnet"].path.name for ds in splits.values() if ds["role"] != "primary"]
    for name in forbidden:
        assert name not in code, f"notebook references {name}"
    assert not re.search(r"hangzhou", code, flags=re.I)
    for split in ("val", "test", "heldout"):
        assert not re.search(rf"""["']{split}["']""", code), f"notebook names split {split!r}"
    assert "train_flow(" in code  # data is obtained through the guarded loader


# --- controllers ---------------------------------------------------------------------------


def test_choosers_on_the_toy_roadnet():
    fb = FeatureBuilder(parse_roadnet(toy_roadnet()))
    ft = make_chooser("fixed_time", fb, seed=0)
    assert ft("C", None, {}) == 1
    assert [ft("C", k, {}) for k in (1, 7, 8)] == [2, 8, 1]

    mp = make_chooser("max_pressure", fb, seed=0)
    assert mp("C", 1, {"in_1_1": 9, "in_3_1": 9}) == 2  # north/south straight pair
    assert mp("C", 5, {}) == 1  # all pressures equal -> lowest phase id

    r1, r2 = make_chooser("random", fb, seed=3), make_chooser("random", fb, seed=3)
    seq = [r1("C", 1, {}) for _ in range(50)]
    assert seq == [r2("C", 1, {}) for _ in range(50)]
    assert set(seq) <= set(fb.phase_ids) and len(set(seq)) > 3

    with pytest.raises(ValueError, match="unknown controller"):
        make_chooser("dqn", fb, seed=0)


# --- simulator (slow) ----------------------------------------------------------------------


def _short_trace(tmp_path, controller, seed):
    cfg = load_config(EDA_CONFIG)
    roadnet_path, flow_path = train_flow(cfg)
    roadnet = load_roadnet(roadnet_path)
    fb = FeatureBuilder(roadnet)
    tmp_path.mkdir(parents=True, exist_ok=True)
    engine = make_engine(roadnet_path, flow_path, seed, 1, tmp_path)
    start = [v["startTime"] for v in json.loads(flow_path.read_text())]
    return collect_trace(
        engine,
        roadnet,
        fb,
        make_chooser(controller, fb, seed),
        green_seconds=10,
        yellow_seconds=5,
        episode_seconds=300,
        record_every=10,
        start_times=start,
    )


@needs_raw
@pytest.mark.slow
@pytest.mark.parametrize("controller", ["max_pressure", "random"])
def test_trace_is_deterministic_and_consistent(tmp_path, controller):
    pytest.importorskip("cityflow")
    steps_a, veh_a, sum_a = _short_trace(tmp_path / "a", controller, seed=1)
    steps_b, veh_b, sum_b = _short_trace(tmp_path / "b", controller, seed=1)
    pd.testing.assert_frame_equal(steps_a, steps_b)
    pd.testing.assert_frame_equal(veh_a, veh_b)
    sum_a.pop("wall_s"), sum_a.pop("steps_per_s"), sum_b.pop("wall_s"), sum_b.pop("steps_per_s")
    assert sum_a == sum_b

    assert len(steps_a) == 30 * 12  # 300 s / 10 s x 12 intersections
    assert steps_a["phase"].between(1, 8).all()
    q = steps_a.filter(regex=r"^queue_").to_numpy()
    c = steps_a.filter(regex=r"^count_").to_numpy()
    assert (q <= c).all() and (q >= 0).all()  # waiting vehicles are a subset of all vehicles

    done = veh_a.dropna(subset=["exit_time"])
    assert len(done) == sum_a["completed"] > 0
    assert (done["travel_time"] > 0).all()
    assert (done["enter_time"] >= done["start_time"]).all()
    entered = veh_a["enter_time"].notna().sum()
    assert entered - len(done) == sum_a["in_network_at_end"]
