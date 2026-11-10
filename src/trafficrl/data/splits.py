"""Train / val / test / held-out split of the traffic datasets (ROADMAP 1.4, 3.1–3.2).

The split is defined in ``data/processed/splits.json``: whole CityFlow flow files are
assigned to splits, each pinned by SHA-256. Every consumer (env, training, evaluation)
must get its flow files through :func:`flow_files` so that

* training code can only ever reach the ``train`` split (and ``val`` for selection),
* a silently replaced or edited file is detected (hash mismatch), and
* the held-out city is reachable only through the explicit ``heldout`` split.

CityFlow assigns vehicle IDs per file (``flow_<i>_<j>``), so "vehicle-set overlap" is
checked as *file* overlap: a file (by path or content hash) may belong to one split only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLITS = REPO_ROOT / "data" / "processed" / "splits.json"

SPLIT_NAMES = ("train", "val", "test", "heldout")
PRIMARY_SPLITS = ("train", "val", "test")


class SplitError(ValueError):
    """Malformed split file, leakage between splits, or a file failing its hash check."""


@dataclass(frozen=True)
class DataFile:
    path: Path
    sha256: str


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _entry(raw: dict, root: Path) -> DataFile:
    return DataFile(path=root / raw["path"], sha256=raw["sha256"])


def load_splits(path: Path = DEFAULT_SPLITS, root: Path = REPO_ROOT) -> dict:
    """Load and structurally validate the split file (no file I/O on the datasets)."""
    spec = json.loads(Path(path).read_text())
    datasets = spec.get("datasets")
    if not datasets:
        raise SplitError("splits file has no 'datasets'")
    out: dict = {}
    for name, ds in datasets.items():
        role = ds.get("role")
        splits = ds.get("splits", {})
        if role == "primary":
            expected = set(PRIMARY_SPLITS)
        elif role == "heldout":
            expected = {"heldout"}
        else:
            raise SplitError(f"{name}: role must be 'primary' or 'heldout', got {role!r}")
        if set(splits) != expected:
            raise SplitError(
                f"{name}: role {role} needs splits {sorted(expected)}, got {sorted(splits)}"
            )
        out[name] = {
            "role": role,
            "roadnet": _entry(ds["roadnet"], root),
            "splits": {s: [_entry(e, root) for e in files] for s, files in splits.items()},
        }
        for s, files in out[name]["splits"].items():
            if not files:
                raise SplitError(f"{name}/{s}: empty split")
    check_disjoint(out)
    return out


def check_disjoint(splits: dict) -> None:
    """Raise if any flow file (by path or by hash) appears in more than one split."""
    seen_path: dict[Path, str] = {}
    seen_hash: dict[str, str] = {}
    for name, ds in splits.items():
        for split, files in ds["splits"].items():
            where = f"{name}/{split}"
            for f in files:
                for key, seen in ((f.path.resolve(), seen_path), (f.sha256, seen_hash)):
                    if key in seen and seen[key] != where:
                        raise SplitError(f"leakage: {f.path.name} is in {seen[key]} and {where}")
                    seen[key] = where


def verify_hashes(splits: dict) -> None:
    """Raise if any roadnet or flow file is missing or differs from its pinned hash."""
    for name, ds in splits.items():
        files = [ds["roadnet"]] + [f for fs in ds["splits"].values() for f in fs]
        for f in files:
            if not f.path.is_file():
                raise SplitError(f"{name}: missing {f.path} — see README 'Data'")
            actual = sha256_of(f.path)
            if actual != f.sha256:
                raise SplitError(
                    f"{name}: hash mismatch for {f.path.name} "
                    f"(expected {f.sha256[:12]}…, got {actual[:12]}…)"
                )


def flow_files(
    dataset: str, split: str, splits: dict | None = None, verify: bool = True
) -> tuple[Path, list[Path]]:
    """Return ``(roadnet, [flow files])`` for one split of one dataset."""
    if split not in SPLIT_NAMES:
        raise SplitError(f"unknown split {split!r}; one of {SPLIT_NAMES}")
    splits = splits if splits is not None else load_splits()
    if dataset not in splits:
        raise SplitError(f"unknown dataset {dataset!r}; one of {sorted(splits)}")
    ds = splits[dataset]
    if split not in ds["splits"]:
        raise SplitError(
            f"{dataset} ({ds['role']}) has no {split!r} split; available: {sorted(ds['splits'])}"
        )
    files = ds["splits"][split]
    if verify:
        verify_hashes({dataset: {"roadnet": ds["roadnet"], "splits": {split: files}}})
    return ds["roadnet"].path, [f.path for f in files]
