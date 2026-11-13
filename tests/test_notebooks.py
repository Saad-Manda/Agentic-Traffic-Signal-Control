"""The Phase 8 notebook is a front end only (Decision Log 13 Nov, plan phase8-analysis A12).

Static checks of the notebook file — it is executed by ``jupyter nbconvert`` (it needs
``results/final``), not by pytest.
"""

import json
import re

from trafficrl.data.splits import REPO_ROOT

NOTEBOOK = REPO_ROOT / "notebooks" / "02_analysis.ipynb"


def cells(kind):
    nb = json.loads(NOTEBOOK.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == kind]


def test_no_logic_lives_in_the_notebook():
    for src in cells("code"):
        assert not re.search(r"^\s*(def|class)\s", src, re.M), src[:80]
        assert "lambda" not in src, src[:80]


def test_it_never_simulates_or_runs_scripts():
    code = "\n".join(cells("code"))
    for banned in ("make_env", "make_agent", "cityflow", "subprocess", "scripts.", "evaluate("):
        assert banned not in code, banned
    assert "trafficrl.eval" in code


def test_every_figure_is_followed_by_an_interpretation():
    nb = json.loads(NOTEBOOK.read_text())["cells"]
    for i, c in enumerate(nb):
        src = "".join(c["source"])
        if c["cell_type"] == "code" and "F.fig_" in src:
            nxt = nb[i + 1]
            assert nxt["cell_type"] == "markdown", src[:60]
            assert "Reading" in "".join(nxt["source"]), src[:60]
