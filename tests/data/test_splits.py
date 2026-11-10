import json
from pathlib import Path

import pytest

from trafficrl.data.splits import (
    DEFAULT_SPLITS,
    SplitError,
    flow_files,
    load_splits,
    sha256_of,
    verify_hashes,
)

RAW_PRESENT = Path(DEFAULT_SPLITS).parents[1].joinpath("raw", "colight_src", "data").is_dir()
needs_raw = pytest.mark.skipif(not RAW_PRESENT, reason="raw datasets not downloaded")


def _spec(tmp_path: Path, datasets: dict) -> Path:
    p = tmp_path / "splits.json"
    p.write_text(json.dumps({"datasets": datasets}))
    return p


def _f(name: str, h: str) -> dict:
    return {"path": name, "sha256": h}


def _primary(train, val, test):
    return {
        "role": "primary",
        "roadnet": _f("rn.json", "r"),
        "splits": {"train": train, "val": val, "test": test},
    }


# --- the committed split file -------------------------------------------------------------


def test_repo_split_structure():
    s = load_splits()
    assert {n: d["role"] for n, d in s.items()} == {"jinan": "primary", "hangzhou": "heldout"}
    jinan = s["jinan"]["splits"]
    assert [len(jinan[k]) for k in ("train", "val", "test")] == [1, 1, 1]
    assert len(s["hangzhou"]["splits"]["heldout"]) == 3


def test_heldout_city_never_in_train_val_test():
    s = load_splits()
    hz = {f.sha256 for f in s["hangzhou"]["splits"]["heldout"]}
    primary = {f.sha256 for k in ("train", "val", "test") for f in s["jinan"]["splits"][k]}
    assert hz.isdisjoint(primary)
    with pytest.raises(SplitError):
        flow_files("hangzhou", "train", splits=s, verify=False)


@needs_raw
def test_repo_files_match_pinned_hashes():
    verify_hashes(load_splits())


@needs_raw
def test_flow_files_returns_existing_paths():
    roadnet, flows = flow_files("jinan", "train")
    assert roadnet.name == "roadnet_3_4.json"
    assert [f.name for f in flows] == ["anon_3_4_jinan_real.json"]


# --- leakage / validation logic on synthetic specs ---------------------------------------


def test_same_file_in_two_splits_is_leakage(tmp_path):
    p = _spec(
        tmp_path, {"a": _primary([_f("x.json", "1")], [_f("x.json", "1")], [_f("z.json", "3")])}
    )
    with pytest.raises(SplitError, match="leakage"):
        load_splits(p, root=tmp_path)


def test_identical_content_under_new_name_is_leakage(tmp_path):
    p = _spec(
        tmp_path, {"a": _primary([_f("x.json", "1")], [_f("copy.json", "1")], [_f("z.json", "3")])}
    )
    with pytest.raises(SplitError, match="leakage"):
        load_splits(p, root=tmp_path)


def test_cross_dataset_leakage(tmp_path):
    p = _spec(
        tmp_path,
        {
            "a": _primary([_f("x.json", "1")], [_f("y.json", "2")], [_f("z.json", "3")]),
            "b": {
                "role": "heldout",
                "roadnet": _f("rn2.json", "r2"),
                "splits": {"heldout": [_f("x2.json", "1")]},
            },
        },
    )
    with pytest.raises(SplitError, match="leakage"):
        load_splits(p, root=tmp_path)


@pytest.mark.parametrize(
    "splits",
    [
        {"train": [_f("x", "1")], "val": [_f("y", "2")]},  # missing test
        {"train": [_f("x", "1")], "val": [_f("y", "2")], "test": []},  # empty split
    ],
)
def test_malformed_primary(tmp_path, splits):
    p = _spec(tmp_path, {"a": {"role": "primary", "roadnet": _f("rn", "r"), "splits": splits}})
    with pytest.raises(SplitError):
        load_splits(p, root=tmp_path)


def test_hash_mismatch_detected(tmp_path):
    (tmp_path / "rn.json").write_text("{}")
    (tmp_path / "x.json").write_text("[]")
    good = sha256_of(tmp_path / "x.json")
    p = _spec(
        tmp_path,
        {
            "a": {
                "role": "primary",
                "roadnet": _f("rn.json", sha256_of(tmp_path / "rn.json")),
                "splits": {
                    "train": [_f("x.json", good)],
                    "val": [_f("y.json", "2")],
                    "test": [_f("z.json", "3")],
                },
            }
        },
    )
    s = load_splits(p, root=tmp_path)
    flow_files("a", "train", splits=s)  # passes
    (tmp_path / "x.json").write_text("[1]")
    with pytest.raises(SplitError, match="hash mismatch"):
        flow_files("a", "train", splits=s)
