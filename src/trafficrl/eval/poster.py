"""Numbers, table cells and figures for the poster (ROADMAP 9.2, plan ``poster``).

A pure reader of the seed-level statistics in ``docs/results/*.csv`` (written by
``scripts/final_tables.py``): nothing is simulated, selected or recomputed, and the numbers
are formatted by the same functions as the report (:mod:`trafficrl.eval.latex`), so the
poster and the report cannot disagree. The slide itself is laid out by
``scripts/build_poster.py``; the prose is in ``poster/content.yaml``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from trafficrl.eval import latex
from trafficrl.eval.latex import AGENTS, ATT, CONDITIONS, FB, MAXW, Stats
from trafficrl.utils.plotting import (
    CONTROLLER_COLOR,
    GRID,
    INK,
    INK_SECONDARY,
    apply_style,
)

PLACEHOLDER = re.compile(r"\{([a-z]+@[^{}\s]+)\}")
NAMES = {
    "fixed_time": "Fixed-Time",
    "max_pressure": "Max-Pressure",
    "dqn": "DQN (shared)",
    "idqn": "IDQN",
    "ugmarl": "UG-MARL (proposed)",
    "ugmarl-nogate": "UG-MARL −gate",
    "ugmarl-noattn": "UG-MARL −attention",
    "ugmarl-noens": "UG-MARL −ensemble",
}
CONDITION_NAMES = {
    "clean": "clean",
    "drop0.1": "dropout 10 %",
    "drop0.2": "dropout 20 %",
    "noise0.1": "noise 0.1",
    "noise0.2": "noise 0.2",
    "surge1.5": "surge ×1.5",
}
ABLATIONS = ("ugmarl-nogate", "ugmarl-noattn", "ugmarl-noens")


@dataclass
class Results:
    """The result frames the poster reads (all from ``docs/results``)."""

    main: pd.DataFrame
    rob: pd.DataFrame
    held: pd.DataFrame
    held_all: pd.DataFrame
    params: dict[str, str] = field(default_factory=dict)


def load(results_dir: str | Path) -> Results:
    d = Path(results_dir)

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(d / f"{name}.csv", dtype={"params": str})

    eff = pd.read_csv(d / "test_efficiency.csv", dtype=str)
    return Results(
        main=read("test_main"),
        rob=read("test_robustness"),
        held=read("heldout_generalisation"),
        held_all=read("heldout_generalisation_all"),
        params=dict(zip(eff["model"], eff["params"], strict=True)),
    )


def plain(text: str) -> str:
    """LaTeX fragments of the report formatters -> plain text for a slide."""
    for a, b in (("$\\pm$", "±"), ("$-$", "−"), ("$+$", "+"), ("\\,", ","), ("--", "–")):
        text = text.replace(a, b)
    return text


def numbers(res: Results) -> dict[str, str]:
    """``{"metric@model@condition": "value"}``: the report's number macros as plain text,
    plus ``params@<model>`` and ``fewer@<model>`` (percent fewer parameters than UG-MARL)."""
    out = {
        k: plain(v)
        for k, v in latex.number_macros(res.main, res.rob, res.held, res.held_all).items()
    }
    out["regret@baselines@range"] = regret_range(res)
    full = int(res.params["ugmarl"].replace(",", ""))
    for m, p in res.params.items():
        out[f"params@{m}"] = p
        n = int(p.replace(",", ""))
        if 0 < n < full:
            out[f"fewer@{m}"] = f"{100 * (1 - n / full):.0f}"
    return out


def fill(text: str, nums: Mapping[str, str]) -> str:
    """Replace every ``{metric@model@condition}``; an unknown key is an error."""

    def sub(m: re.Match) -> str:
        if m.group(1) not in nums:
            raise KeyError(f"unknown poster number {{{m.group(1)}}}")
        return nums[m.group(1)]

    return PLACEHOLDER.sub(sub, text)


def fill_tree(node, nums: Mapping[str, str]):
    """:func:`fill` on every string of a nested dict / list (the content file)."""
    if isinstance(node, str):
        return fill(node, nums)
    if isinstance(node, list):
        return [fill_tree(v, nums) for v in node]
    if isinstance(node, dict):
        return {k: fill_tree(v, nums) for k, v in node.items()}
    return node


@dataclass
class Table:
    head: list[str]
    rows: list[list[str]]
    bold: set[tuple[int, int]]  # (row, column) of the best of the five agents
    proposed: int  # row of the proposed model
    first_ablation: int  # first row below the five agents


def results_table(res: Results) -> Table:
    """Five agents and the three ablations on the Jinan test flow: travel time (mean ± std
    over seeds) clean / noise 0.2 / surge, maximum wait, largest regret, parameters."""
    st = Stats(res.rob)
    models = [*AGENTS, *ABLATIONS]
    conds = ("clean", "noise0.2", "surge1.5")
    rows, worst = [], []
    for m in models:
        regret = max(st.get(m, c, "regret") for c in CONDITIONS)
        worst.append(regret)
        rows.append([
            NAMES[m],
            *(plain(st.pm(m, c, ATT)) for c in conds),
            plain(st.pm(m, "clean", MAXW, 0)),
            latex.fmt_num(regret),
            res.params[m],
        ])  # fmt: skip
    n = len(AGENTS)
    bold = set()
    columns = [[st.mean(m, c, ATT) for m in AGENTS] for c in conds]
    columns += [[st.mean(m, "clean", MAXW) for m in AGENTS], worst[:n]]
    for j, vals in enumerate(columns):
        bold.add((latex.best_index(vals), j + 1))
    head = ["Model", "Clean", "Noise 0.2", "Surge ×1.5", "Max wait", "Regret", "Params"]
    return Table(head, rows, bold, models.index("ugmarl"), n)


# --- figure ------------------------------------------------------------------------------

POSTER_RC = {
    "font.family": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "standard",
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
}


def _short(model: str) -> str:
    return NAMES[model].replace(" (proposed)", "").replace(" (shared)", "")


def fig_robustness(res: Results, out: str | Path | None = None, size=(4.83, 1.62)) -> plt.Figure:
    """Poster figure: (a) travel time against detector noise, (b) largest regret over the
    six test conditions, (c) share of UG-MARL decisions handed to Max-Pressure."""
    apply_style()
    st = Stats(res.rob)
    with mpl.rc_context(POSTER_RC):
        fig, (a, b, c) = plt.subplots(1, 3, figsize=size, gridspec_kw={"width_ratios": [2.6, 2, 2]})
        xs, conds = [0, 0.1, 0.2], ["clean", "noise0.1", "noise0.2"]
        for m in [*AGENTS[:4], "ugmarl-nogate", "ugmarl"]:
            a.plot(xs, [st.mean(m, k, ATT) for k in conds], color=CONTROLLER_COLOR[m],
                   marker="o", markersize=2.5, linestyle="--" if m == "ugmarl-nogate" else "-",
                   linewidth=2.0 if m == "ugmarl" else 1.1, label=_short(m),
                   zorder=4 if m == "ugmarl" else 3)  # fmt: skip
            a.fill_between(xs, [st.get(m, k, f"{ATT}_ci95_low") for k in conds],
                           [st.get(m, k, f"{ATT}_ci95_high") for k in conds],
                           color=CONTROLLER_COLOR[m], alpha=0.15, linewidth=0)  # fmt: skip
        a.set_xticks(xs)
        a.set_ylim(250, 1400)
        a.set_yticks([400, 600, 800, 1000, 1200])
        a.set_xlabel("Detector noise (std)")
        a.set_ylabel("Average travel time (s)")
        a.set_title("(a) Detector noise")
        a.legend(ncol=1, loc="upper left", handlelength=1.6, borderaxespad=0.1,
                 labelspacing=0.2)  # fmt: skip

        models = list(AGENTS)
        worst = [max(st.get(m, k, "regret") for k in CONDITIONS) for m in models]
        y = range(len(models))
        b.barh(y, worst, height=0.62, color=[CONTROLLER_COLOR[m] for m in models])
        for k, v in enumerate(worst):
            b.text(v * 1.15, k, f"{v:.1f}", va="center", fontsize=6.5, color=INK)
        b.set_yticks(y, [_short(m) for m in models])
        b.invert_yaxis()
        b.set_xscale("log")
        b.set_xlim(1, 3000)
        b.set_xlabel("Largest regret (s, log scale)")
        b.set_title("(b) Worst-case regret")
        b.grid(axis="y", visible=False)

        names = list(CONDITION_NAMES)
        share = [100 * st.mean("ugmarl", k, FB) for k in names]
        c.barh(range(len(names)), share, height=0.62, color=CONTROLLER_COLOR["ugmarl"])
        for k, v in enumerate(share):
            c.text(v + 2, k, f"{v:.0f}", va="center", fontsize=6.5, color=INK)
        c.set_yticks(range(len(names)), [CONDITION_NAMES[k] for k in names])
        c.invert_yaxis()
        c.set_xlim(0, 118)
        c.set_xticks([0, 50, 100])
        c.set_xlabel("Fallback (%)")
        c.set_title("(c) Gate firing")
        c.grid(axis="y", visible=False)
        for ax in (a, b, c):
            ax.tick_params(colors=INK_SECONDARY)
            ax.grid(color=GRID)
        fig.tight_layout(pad=0.3, w_pad=0.9)
        if out is not None:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out)
    return fig


def regret_range(res: Results, models=AGENTS[:4]) -> str:
    """Smallest–largest worst-case regret of the four baselines, e.g. ``62.2–414.6``."""
    st = Stats(res.rob)
    worst = [max(st.get(m, k, "regret") for k in CONDITIONS) for m in models]
    if any(math.isnan(v) for v in worst):
        raise ValueError("missing regret")
    return f"{latex.fmt_num(min(worst))}–{latex.fmt_num(max(worst))}"
