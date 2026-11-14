"""The report sources under ``report/`` are consistent and up to date (ROADMAP 9.1)."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "report"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_generated_tables_and_macros_are_up_to_date():
    files = load("report_tables").build(REPO_ROOT / "docs" / "results")
    assert set(files) >= {"tables/main.tex", "tables/robustness.tex", "generated/numbers.tex"}
    for rel, text in files.items():
        assert (REPORT / rel).read_text() == text, f"{rel} is stale: run scripts/report_tables.py"


def test_report_tables_cli_writes_files(tmp_path):
    assert load("report_tables").main(["--out", str(tmp_path)]) == 0
    assert (tmp_path / "tables" / "main.tex").exists()
    assert (tmp_path / "generated" / "numbers.tex").exists()
    assert load("report_tables").main(["--results-dir", str(tmp_path / "none")]) == 1


def test_labels_citations_and_wording():
    assert load("check_report").static_problems(REPORT) == []


def test_chapters_follow_the_prescribed_structure():
    expected = {
        "01-introduction": ["Background and Motivation", "Problem Statement", "Objectives",
                            "Scope of the Work", "Contributions"],
        "02-literature": ["Existing Approaches", "Research Gaps"],
        "03-methodology": ["Dataset Description", "Data Preprocessing",
                           "Exploratory Data Analysis (EDA)", "Proposed Model Architecture",
                           "Algorithm", "Experimental Setup"],
        "04-results": ["Performance Evaluation Metrics", "Experimental Results",
                       "Comparison with Existing Models", "Graphical Analysis",
                       "Discussion of Findings"],
        "05-conclusion": ["Key Findings", "Limitations", "Future Research Directions"],
    }  # fmt: skip
    for name, sections in expected.items():
        text = (REPORT / "chapters" / f"{name}.tex").read_text()
        assert text.count(r"\chapter{") == 1
        assert re.findall(r"\\section\{([^}]+)\}", text) == sections
    main = (REPORT / "main.tex").read_text()
    order = [main.index(f"chapters/{n}") for n in ("00-abstract", *expected)]
    assert order == sorted(order)
    for needed in (r"\tableofcontents", r"\listoffigures", r"\listoftables", r"\bibliography{"):
        assert needed in main


def test_checker_finds_planted_problems(tmp_path):
    check = load("check_report")
    (tmp_path / "chapters").mkdir()
    (tmp_path / "main.tex").write_text("[Author Name]")
    (tmp_path / "references.bib").write_text("@article{known,\n title={x}}\n")
    (tmp_path / "chapters" / "01.tex").write_text(
        r"\label{fig:unused} \cref{tab:missing} \citep{known,unknown} a novel idea % TODO hidden"
    )
    problems = check.static_problems(tmp_path)
    assert "label never referenced: fig:unused" in problems
    assert "reference to unknown label: tab:missing" in problems
    assert "citation not in references.bib: unknown" in problems
    assert any("novel" in p for p in problems)
    assert not any("TODO" in p for p in problems)  # comments are ignored
    assert check.placeholders(tmp_path) == ["[Author Name]"]
    assert check.build_problems(tmp_path)[1] is None  # no build: reported, not a crash
