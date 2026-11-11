"""YAML config loading (ROADMAP 0.3).

One YAML file per experiment drives every run. Features kept deliberately small:

* ``base:`` — a path (or list of paths), relative to the file, of configs to inherit from.
  Bases are merged in order, then the file itself on top (nested dicts merge recursively;
  lists and scalars are replaced).
* CLI-style overrides ``"train.lr=0.0005"`` — values are parsed as YAML, so numbers, bools,
  lists and ``null`` work; missing intermediate keys are created.
* A top-level integer ``seed`` is required, so no run is ever unseeded.

``save_config`` writes the fully resolved config next to the run's results.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

BASE_KEY = "base"


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads ``1e-3`` / ``5E+4`` as floats.

    Plain PyYAML follows YAML 1.1, where a float needs a '.', so ``lr: 1e-3`` would load
    as the *string* "1e-3" — a silent hyperparameter bug.
    """


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


def _parse(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)  # noqa: S506 — _Loader is a SafeLoader


class ConfigError(ValueError):
    """Raised for malformed configs, cyclic ``base`` chains or bad overrides."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict: ``override`` merged recursively onto ``base``."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    data = _parse(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read one YAML mapping with the project loader (no ``base:`` resolution, no checks)."""
    return _read_yaml(Path(path))


def _load_with_bases(path: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in chain:
        cycle = " -> ".join(str(p) for p in (*chain, path))
        raise ConfigError(f"cyclic base configs: {cycle}")
    raw = _read_yaml(path)
    bases = raw.pop(BASE_KEY, None) or []
    if isinstance(bases, str):
        bases = [bases]
    merged: dict[str, Any] = {}
    for b in bases:
        merged = deep_merge(merged, _load_with_bases(path.parent / b, (*chain, path)))
    return deep_merge(merged, raw)


def apply_overrides(cfg: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``"dotted.key=value"`` overrides; values are parsed as YAML."""
    out = copy.deepcopy(dict(cfg))
    for item in overrides:
        key, sep, raw_value = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"override must look like 'a.b=value', got {item!r}")
        parts = key.strip().split(".")
        node = out
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigError(f"override {item!r}: '{part}' is not a mapping")
            node = child
        node[parts[-1]] = _parse(raw_value)
    return out


def validate(cfg: Mapping[str, Any]) -> None:
    """Minimal schema checks shared by every experiment."""
    seed = cfg.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError(f"config needs a non-negative integer 'seed', got {seed!r}")


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load a YAML config, resolve its ``base`` chain, apply overrides and validate."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cfg = apply_overrides(_load_with_bases(path, ()), overrides)
    validate(cfg)
    return cfg


def save_config(cfg: Mapping[str, Any], path: str | Path) -> Path:
    """Write the resolved config as YAML (key order preserved); creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)
    return path
