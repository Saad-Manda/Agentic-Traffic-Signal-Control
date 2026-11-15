"""The poster is a pure reader of ``docs/results`` and builds to one 16:9 slide (ROADMAP 9.2)."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from trafficrl.eval import poster as P

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "docs" / "results"
CONTENT = REPO_ROOT / "poster" / "content.yaml"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def res() -> P.Results:
    return P.load(RESULTS)


def strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for v in node:
            yield from strings(v)
    elif isinstance(node, dict):
        for v in node.values():
            yield from strings(v)


def test_numbers_equal_the_result_files(res):
    nums = P.numbers(res)
    rob = pd.read_csv(RESULTS / "test_robustness.csv").set_index(["model", "condition"])
    for model, cond in [("ugmarl", "clean"), ("dqn", "noise0.2"), ("max_pressure", "surge1.5")]:
        mean = rob.loc[(model, cond), "avg_travel_time_s_mean"]
        assert nums[f"tt@{model}@{cond}"] == f"{mean:.1f}"
    worst = rob.loc["ugmarl", "regret"].max()
    assert nums["regret@ugmarl@worst"] == f"{worst:.1f}"
    assert nums["params@ugmarl"] == "157,720"
    assert nums["fewer@ugmarl-noattn"] == "63"
    low, high = nums["regret@baselines@range"].split("–")
    assert float(low) < float(high)
    assert not any("$" in v or "\\" in v for v in nums.values())


def test_fill_replaces_placeholders_and_rejects_unknown_keys(res):
    nums = P.numbers(res)
    assert P.fill("a {tt@ugmarl@clean} s", nums) == f"a {nums['tt@ugmarl@clean']} s"
    assert P.fill_tree({"k": ["{params@dqn}"]}, nums) == {"k": [nums["params@dqn"]]}
    with pytest.raises(KeyError):
        P.fill("{tt@nobody@clean}", nums)


def test_results_table_cells_and_best_values(res):
    t = P.results_table(res)
    assert [r[0] for r in t.rows][: t.first_ablation] == [
        "Fixed-Time", "Max-Pressure", "DQN (shared)", "IDQN", "UG-MARL (proposed)"]  # fmt: skip
    assert all(len(r) == len(t.head) for r in t.rows)
    rob = pd.read_csv(RESULTS / "test_robustness.csv").set_index(["model", "condition"])
    row = t.rows[t.proposed]
    mean = rob.loc[("ugmarl", "clean"), "avg_travel_time_s_mean"]
    std = rob.loc[("ugmarl", "clean"), "avg_travel_time_s_std"]
    assert row[1] == f"{mean:.1f} ± {std:.1f}"
    agents = ["fixed_time", "max_pressure", "dqn", "idqn", "ugmarl"]
    clean = [rob.loc[(m, "clean"), "avg_travel_time_s_mean"] for m in agents]
    assert (clean.index(min(clean)), 1) in t.bold
    assert all(i < t.first_ablation for i, _ in t.bold)  # ablations are never the "best"
    assert len(t.bold) == 5


def test_content_placeholders_resolve_and_voice_is_impersonal(res):
    raw = yaml.safe_load(CONTENT.read_text())
    filled = P.fill_tree(raw, P.numbers(res))
    for text in strings(filled):
        assert not P.PLACEHOLDER.search(text), text
        assert "{" not in text and "}" not in text, text
        assert not re.search(r"\b(we|our|ours|us|I|my)\b", text), text


def test_build_writes_one_16_9_slide_with_everything_inside(tmp_path, res):
    from pptx import Presentation

    out = load_script("build_poster").build(CONTENT, RESULTS, tmp_path)
    prs = Presentation(out)
    assert len(prs.slides) == 1
    assert prs.slide_width / 914400 == pytest.approx(13.333, abs=0.01)
    assert prs.slide_height / 914400 == pytest.approx(7.5, abs=0.01)
    shapes = list(prs.slides[0].shapes)
    for s in shapes:
        assert s.left >= 0 and s.top >= 0, s.name
        assert s.left + s.width <= prs.slide_width + 1, s.name
        assert s.top + s.height <= prs.slide_height + 1, s.name
    text = "\n".join(s.text_frame.text for s in shapes if s.has_text_frame)
    assert "{" not in text
    assert P.numbers(res)["tt@ugmarl@clean"] in text
    assert any(s.has_table for s in shapes if hasattr(s, "has_table"))
    assert (tmp_path / "figures" / "robustness.png").exists()
    assert not any(s.shape_type == 13 and "logo" in s.name.lower() for s in shapes)


def test_cli_reports_a_missing_results_directory(tmp_path):
    script = load_script("build_poster")
    assert script.main(["--results-dir", str(tmp_path / "none"), "--out", str(tmp_path)]) == 1
