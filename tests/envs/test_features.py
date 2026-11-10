import numpy as np
import pytest

from tests.data.test_splits import needs_raw
from tests.data.toy import toy_roadnet
from trafficrl.data.roadnet import load_roadnet, parse_roadnet
from trafficrl.data.splits import REPO_ROOT, load_splits
from trafficrl.envs.features import FeatureBuilder, FeatureSpec
from trafficrl.utils.config import load_config

RN = parse_roadnet(toy_roadnet())  # every lane holds 40 vehicles (300 m / 7.5 m)


def test_default_observation_layout_and_values():
    fb = FeatureBuilder(RN)
    assert fb.dim == 44
    assert fb.slices() == {
        "queue": slice(0, 12),
        "count": slice(12, 24),
        "pressure": slice(24, 36),
        "phase": slice(36, 44),
    }
    count = {"in_0_1": 20, "in_0_0": 10, "out_0_0": 4, "out_0_1": 4, "out_0_2": 4}
    waiting = {"in_0_1": 8}
    obs = fb.observation("C", count, waiting, green_phase=3)
    assert obs.dtype == np.float32 and obs.shape == (44,)

    queue, cnt, pressure, phase = (obs[s] for s in fb.slices().values())
    # canonical lane order: in_0_0, in_0_1, in_0_2, in_1_0, ...
    np.testing.assert_allclose(queue[:3], [0, 8 / 40, 0])
    np.testing.assert_allclose(cnt[:3], [10 / 40, 20 / 40, 0])
    assert queue[3:].sum() == 0 and cnt[3:].sum() == 0

    c = RN.intersections["C"]
    straight = next(
        i for i, m in enumerate(c.movements) if (m.start_road, m.kind) == ("in_0", "go_straight")
    )
    left = next(
        i for i, m in enumerate(c.movements) if (m.start_road, m.kind) == ("in_0", "turn_left")
    )
    assert pressure[straight] == pytest.approx(20 / 40 - 4 / 40)  # into out_0 (4 per lane)
    assert pressure[left] == pytest.approx(10 / 40)  # into the empty out_1
    np.testing.assert_array_equal(phase, [0, 0, 1, 0, 0, 0, 0, 0])


def test_values_are_bounded_and_clipped():
    fb = FeatureBuilder(RN)
    rng = np.random.default_rng(0)
    lanes = list(fb.capacity)
    for _ in range(20):
        count = {ln: int(rng.integers(0, 80)) for ln in lanes}  # up to 2x capacity
        waiting = {ln: int(rng.integers(0, count[ln] + 1)) for ln in lanes}
        obs = fb.observation("C", count, waiting, green_phase=1)
        s = fb.slices()
        assert obs[s["queue"]].min() >= 0 and obs[s["queue"]].max() <= 1
        assert obs[s["count"]].min() >= 0 and obs[s["count"]].max() <= 1
        assert obs[s["pressure"]].min() >= -1 and obs[s["pressure"]].max() <= 1
        assert obs[s["phase"]].sum() == 1

    unclipped = FeatureBuilder(RN, FeatureSpec(clip=False))
    assert unclipped.observation("C", {"in_0_0": 80}, {}, 1)[12] == pytest.approx(2.0)
    assert fb.observation("C", {"in_0_0": 80}, {}, 1)[12] == 1.0


def test_group_selection_changes_dim_and_order():
    fb = FeatureBuilder(RN, FeatureSpec(groups=("phase", "queue")))
    assert fb.dim == 20
    obs = fb.observation("C", {}, {"in_0_0": 40}, green_phase=8)
    assert obs[7] == 1 and obs[8] == 1.0


def test_invalid_spec_and_phase():
    with pytest.raises(ValueError, match="unknown feature group"):
        FeatureSpec(groups=("queue", "speed"))
    with pytest.raises(ValueError, match="unique"):
        FeatureSpec(groups=("queue", "queue"))
    with pytest.raises(ValueError, match="normalisation"):
        FeatureSpec(normalisation="zscore")
    with pytest.raises(ValueError, match="not a green phase"):
        FeatureBuilder(RN).observation("C", {}, {}, green_phase=0)  # 0 = transition


def test_raw_pressures():
    fb = FeatureBuilder(RN)
    c = RN.intersections["C"]
    # 30 vehicles waiting to go straight east, 6 already downstream on out_0
    count = {"in_0_1": 30, "out_0_0": 2, "out_0_1": 2, "out_0_2": 2, "in_1_1": 5}
    pp = fb.phase_pressure("C", count)
    assert set(pp) == set(c.green_phases)
    best = max(pp, key=pp.get)
    served = {(c.movements[i].start_road, c.movements[i].kind) for i in c.green_phases[best]}
    assert ("in_0", "go_straight") in served
    assert fb.intersection_pressure("C", count) == 30 + 5 - 6


def test_observations_for_all_intersections():
    fb = FeatureBuilder(RN)
    obs = fb.observations({}, {}, {"C": 2})
    assert list(obs) == ["C"] and obs["C"].sum() == 1  # only the phase bit is set


def test_builder_needs_no_flow_file(tmp_path):
    """Leakage guard: features depend on the roadnet only — no flow file exists here."""
    import json

    p = tmp_path / "roadnet.json"
    p.write_text(json.dumps(toy_roadnet()))
    assert list(tmp_path.iterdir()) == [p]
    fb = FeatureBuilder(load_roadnet(p))
    assert fb.dim == 44 and set(fb.capacity.values()) == {40}


def test_spec_from_base_config():
    spec = FeatureSpec.from_config(load_config(REPO_ROOT / "configs" / "base.yaml"))
    assert spec == FeatureSpec()


@needs_raw
@pytest.mark.parametrize("dataset", ["jinan", "hangzhou"])
def test_real_roadnets_give_the_same_observation_size(dataset):
    rn = load_roadnet(load_splits()[dataset]["roadnet"].path)
    fb = FeatureBuilder(rn)
    assert fb.dim == 44
    obs = fb.observations({}, {}, dict.fromkeys(rn.intersections, 1))
    assert len(obs) == len(rn.intersections)
