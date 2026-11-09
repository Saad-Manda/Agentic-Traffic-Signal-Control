from pathlib import Path

import pytest

from trafficrl.utils.config import (
    ConfigError,
    apply_overrides,
    deep_merge,
    load_config,
    save_config,
)

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_deep_merge_nested_and_no_mutation():
    base = {"a": 1, "b": {"x": 1, "y": 2}, "l": [1, 2]}
    out = deep_merge(base, {"b": {"y": 3}, "l": [9]})
    assert out == {"a": 1, "b": {"x": 1, "y": 3}, "l": [9]}
    assert base["b"]["y"] == 2


def test_base_inheritance(tmp_path):
    _write(tmp_path / "base.yaml", "seed: 0\nsim: {threads: 1, episode_seconds: 3600}\n")
    exp = _write(tmp_path / "exp.yaml", "base: base.yaml\nsim: {episode_seconds: 1800}\n")
    cfg = load_config(exp)
    assert cfg == {"seed": 0, "sim": {"threads": 1, "episode_seconds": 1800}}
    assert "base" not in cfg


def test_multiple_bases_merge_in_order(tmp_path):
    _write(tmp_path / "a.yaml", "seed: 0\nx: a\n")
    _write(tmp_path / "b.yaml", "x: b\ny: b\n")
    exp = _write(tmp_path / "exp.yaml", "base: [a.yaml, b.yaml]\n")
    assert load_config(exp) == {"seed": 0, "x": "b", "y": "b"}


def test_cyclic_base_rejected(tmp_path):
    _write(tmp_path / "a.yaml", "base: b.yaml\nseed: 0\n")
    _write(tmp_path / "b.yaml", "base: a.yaml\n")
    with pytest.raises(ConfigError, match="cyclic"):
        load_config(tmp_path / "a.yaml")


def test_overrides_parse_yaml_types():
    cfg = apply_overrides(
        {"seed": 0, "train": {"lr": 1e-3}},
        ["train.lr=5e-4", "train.gamma=0.99", "flag=true", "new.list=[1, 2]", "x=null"],
    )
    assert cfg["train"] == {"lr": 5e-4, "gamma": 0.99}
    assert cfg["flag"] is True
    assert cfg["new"] == {"list": [1, 2]}
    assert cfg["x"] is None


@pytest.mark.parametrize("bad", ["noequals", "=3", "seed.x=1"])
def test_bad_overrides(bad):
    with pytest.raises(ConfigError):
        apply_overrides({"seed": 0}, [bad])


@pytest.mark.parametrize("text", ["sim: {}\n", "seed: -1\n", "seed: 1.5\n", "seed: true\n"])
def test_seed_required(tmp_path, text):
    with pytest.raises(ConfigError, match="seed"):
        load_config(_write(tmp_path / "c.yaml", text))


def test_override_can_fix_seed(tmp_path):
    cfg = load_config(_write(tmp_path / "c.yaml", "a: 1\n"), ["seed=3"])
    assert cfg["seed"] == 3


def test_save_roundtrip(tmp_path):
    cfg = {"seed": 1, "sim": {"decision_interval": 10}, "z": [1, 2]}
    out = save_config(cfg, tmp_path / "run" / "config.yaml")
    assert load_config(out) == cfg
    assert out.read_text().startswith("seed:")  # key order preserved


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_repo_base_config_loads():
    cfg = load_config(REPO_CONFIGS / "base.yaml")
    assert cfg["seed"] == 0 and cfg["sim"]["threads"] == 1


def test_scientific_notation_is_float(tmp_path):
    cfg = load_config(_write(tmp_path / "c.yaml", "seed: 0\nlr: 1e-3\nbig: 5E+4\nv: '1e-3'\n"))
    assert cfg["lr"] == 1e-3 and isinstance(cfg["lr"], float)
    assert cfg["big"] == 5e4
    assert cfg["v"] == "1e-3"  # quoted stays a string
