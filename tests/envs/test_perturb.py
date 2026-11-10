import json

import numpy as np
import pytest

from trafficrl.envs.perturb import SensorModel, surge_flow

CAP = {f"l{k}": 40 for k in range(2000)}
COUNT = {ln: 20 for ln in CAP}
WAIT = {ln: 10 for ln in CAP}


def test_inactive_model_is_identity():
    c, w = SensorModel(CAP).observe(COUNT, WAIT)
    assert c == COUNT and w == WAIT


def test_noise_stays_in_capacity_bounds_and_is_seeded():
    a = SensorModel(CAP, noise_std=0.5, rng=np.random.default_rng(1)).observe(COUNT, WAIT)
    b = SensorModel(CAP, noise_std=0.5, rng=np.random.default_rng(1)).observe(COUNT, WAIT)
    assert a == b
    vals = np.array(list(a[0].values()))
    assert vals.min() >= 0 and vals.max() <= 40 and vals.std() > 5
    small = SensorModel(CAP, noise_std=0.1, rng=np.random.default_rng(2)).observe(COUNT, WAIT)
    scaled = (np.array(list(small[0].values())) - 20) / 40
    assert scaled.std() == pytest.approx(0.1, rel=0.1)


def test_dropout_rate_and_dead_detector_loses_both_readings():
    c, w = SensorModel(CAP, dropout_p=0.2, rng=np.random.default_rng(0)).observe(COUNT, WAIT)
    dead = [ln for ln in CAP if c[ln] == 0]
    assert len(dead) / len(CAP) == pytest.approx(0.2, abs=0.03)
    assert all(w[ln] == 0 for ln in dead)
    c1, _ = SensorModel(CAP, dropout_p=1.0, rng=np.random.default_rng(0)).observe(COUNT, WAIT)
    assert set(c1.values()) == {0.0}


def test_bad_arguments():
    with pytest.raises(ValueError):
        SensorModel(CAP, noise_std=-1)
    with pytest.raises(ValueError):
        SensorModel(CAP, dropout_p=1.5)
    with pytest.raises(ValueError, match="rng"):
        SensorModel(CAP, noise_std=0.1)


def test_surge_duplicates_vehicles(tmp_path):
    flows = [
        {"vehicle": {}, "route": [f"r{k}", "x"], "interval": 1.0, "startTime": k, "endTime": k}
        for k in range(4000)
    ]
    src = tmp_path / "src"
    src.mkdir()
    (src / "flow.json").write_text(json.dumps(flows))
    (src / "roadnet.json").write_text("{}")
    rn, fl, added = surge_flow(
        src / "roadnet.json", src / "flow.json", tmp_path / "out", 1.5, np.random.default_rng(0)
    )
    out = json.loads(fl.read_text())
    assert rn.read_text() == "{}" and rn.parent == fl.parent
    assert len(out) == 4000 + added and added == pytest.approx(2000, rel=0.05)
    dups = [(a, b) for a, b in zip(out, out[1:], strict=False) if a["route"] == b["route"]]
    assert len(dups) == added
    assert all(b["startTime"] == a["startTime"] + 1 == b["endTime"] for a, b in dups)
    with pytest.raises(ValueError):
        surge_flow(src / "roadnet.json", src / "flow.json", tmp_path, 3.0, np.random.default_rng())
