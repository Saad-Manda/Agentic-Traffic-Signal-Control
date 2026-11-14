"""LaTeX table / macro formatting for the report (ROADMAP 9.1)."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from trafficrl.eval import latex

RESULTS = Path(__file__).resolve().parents[2] / "docs" / "results"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{name}.csv", dtype={"params": str})


def test_mean_std_and_ci_formatting():
    assert latex.fmt_mean_std(295.44, 1.06) == r"295.4 $\pm$ 1.1"
    assert latex.fmt_mean_std(896.2, 168.9, 0) == r"896 $\pm$ 169"
    assert latex.fmt_mean_std(320.77, 0.0) == "320.8"  # deterministic: no ±
    assert latex.fmt_mean_std(320.77, math.nan) == "320.8"
    assert latex.fmt_ci(294.13, 296.76) == "[294.1, 296.8]"
    assert latex.fmt_ci(320.8, 320.8) == "--"


def test_number_and_sign_formatting():
    assert latex.fmt_num(157720, 0) == r"157\,720"
    assert latex.fmt_num(4996, 0) == "4996"
    assert latex.fmt_signed(-0.44) == "$-$0.4"
    assert latex.fmt_signed(1.9) == "$+$1.9"
    assert latex.fmt_signed(-0.01) == "$+$0.0"  # rounds to zero: no "minus zero"


def test_escape():
    assert latex.escape("a_b 5% & c") == r"a\_b 5\% \& c"
    assert latex.escape("UG-MARL −gate ± ×") == r"UG-MARL $-$gate $\pm$ $\times$"


def test_best_index_direction_and_nan():
    assert latex.best_index([3.0, 1.0, 2.0]) == 1
    assert latex.best_index([3.0, 1.0, 2.0], lower_is_better=False) == 0
    assert latex.best_index([math.nan, 5.0, 4.0]) == 2


def test_tabular_checks_row_width_and_rules():
    out = latex.tabular("lr", ["a", "b"], [["x", "1"], ["y", "2"]], rules_after=[0])
    lines = out.splitlines()
    assert lines[0] == r"\begin{tabular}{lr}" and lines[-1] == r"\end{tabular}"
    assert out.count(r"\midrule") == 2
    with pytest.raises(ValueError):
        latex.tabular("lr", ["a", "b"], [["only one"]])


def test_main_table_values_and_bold():
    out = latex.table_main(read("test_main"))
    assert out.count(r"\begin{tabular}") == out.count(r"\end{tabular}") == 1
    assert r"\textbf{292.2 $\pm$ 0.6}" in out  # DQN: lowest test travel time
    row = next(line for line in out.splitlines() if line.startswith("UG-MARL (proposed)"))
    assert r"295.4 $\pm$ 1.1" in row and "[294.1, 296.8]" in row and r"157\,720" in row
    mp = next(line for line in out.splitlines() if line.startswith("Max-Pressure"))
    assert "320.8 & --" in mp and r"\pm" not in mp  # deterministic


def test_robustness_table_bolds_best_of_the_five_agents():
    out = latex.table_robustness(read("test_robustness"))
    ug = next(line for line in out.splitlines() if line.startswith("UG-MARL (proposed)"))
    assert ug.rstrip(" \\").endswith(r"\textbf{6.8}")  # smallest worst-case regret
    gate_off = next(line for line in out.splitlines() if line.startswith("UG-MARL $-$gate"))
    assert r"\textbf" not in gate_off  # ablation rows are never the "best agent"
    assert r"\textbf{413.7 $\pm$ 4.0}" in out  # Max-Pressure at noise 0.2


def test_ablation_and_heldout_tables():
    eff = pd.read_csv(RESULTS / "test_efficiency.csv", dtype=str)
    params = dict(zip(eff["model"], eff["params"], strict=True))
    abl = latex.table_ablation(read("test_ablation"), params)
    assert "UG-MARL (full) & 295.4" in abl and "$-$1.2" in abl and r"59\,032" in abl
    held = latex.table_heldout(read("heldout_generalisation"), read("heldout_generalisation_all"))
    assert r"\multicolumn{7}{c}{not transferable" in held
    assert r"\textbf{403.8 $\pm$ 1.2}" in held  # −attention: lowest on Hangzhou


def test_macros_match_the_result_files():
    main, rob = read("test_main"), read("test_robustness")
    held, held_all = read("heldout_generalisation"), read("heldout_generalisation_all")
    m = latex.number_macros(main, rob, held, held_all)
    assert m["tt@ugmarl@clean"] == "295.4"
    assert m["tt@dqn@clean"] == "292.2"
    assert m["regret@ugmarl@clean"] == "3.2"
    assert m["regret@ugmarl@worst"] == "6.8"
    assert m["regret@dqn@worst"] == "414.6"
    assert m["maxw@ugmarl@surge1.5"] == "2162" and m["maxw@ugmarl-nogate@surge1.5"] == "1430"
    assert m["fb@ugmarl@noise0.2"] == "96"
    assert m["dtt@ugmarl-nogate@noise0.2"] == "575.2"
    assert m["tt@ugmarl@hz-all"] == "410.0"
    row = rob[(rob["model"] == "idqn") & (rob["condition"] == "noise0.2")].iloc[0]
    assert m["tt@idqn@noise0.2"] == f"{row['avg_travel_time_s_mean']:.1f}"
    text = latex.render_macros(m)
    assert r"\@namedef{res@tt@ugmarl@clean}{295.4}" in text
    assert text.count(r"\@namedef") == len(m)
