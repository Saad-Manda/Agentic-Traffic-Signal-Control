"""LaTeX tables and number macros for the report (ROADMAP 9.1, plan phase9-report R5).

Pure formatting of the seed-level statistics that ``scripts/final_tables.py`` wrote to
``docs/results/*.csv``: nothing is simulated, selected or recomputed here, so the report,
the poster and ``docs/results/`` cannot disagree. Every function returns a ``tabular``
string (the float, caption and label live in the chapter file).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import pandas as pd

ATT = "avg_travel_time_s"
WAIT = "avg_waiting_time_s"
QUEUE = "avg_queue_veh"
THR = "throughput_veh"
MAXW = "max_waiting_time_s"
FB = "fallback_rate"

LABELS = {
    "fixed_time": "Fixed-Time",
    "max_pressure": "Max-Pressure",
    "dqn": "DQN (shared)",
    "idqn": "IDQN",
    "ugmarl": "UG-MARL (proposed)",
    "ugmarl-nogate": "UG-MARL $-$gate",
    "ugmarl-noattn": "UG-MARL $-$attention",
    "ugmarl-noens": "UG-MARL $-$ensemble",
}
AGENTS = ("fixed_time", "max_pressure", "dqn", "idqn", "ugmarl")
VARIANTS = ("ugmarl", "ugmarl-nogate", "ugmarl-noattn", "ugmarl-noens")
CONDITIONS = {
    "clean": "Clean",
    "noise0.1": "Noise 0.1",
    "noise0.2": "Noise 0.2",
    "drop0.1": "Dropout 10\\,\\%",
    "drop0.2": "Dropout 20\\,\\%",
    "surge1.5": "Surge $\\times$1.5",
}
HELDOUT_FLOWS = {"real": "Flow A", "real_5734": "Flow B", "real_5816": "Flow C"}

_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}", "−": "$-$", "±": r"$\pm$", "×": r"$\times$"}  # fmt: skip


def escape(text: str) -> str:
    """Plain text -> LaTeX text (special characters and the Unicode signs the tables use)."""
    return "".join(_ESCAPES.get(ch, ch) for ch in str(text))


def fmt_num(value: float, digits: int = 1) -> str:
    """Fixed-point number; integers get a thin-space thousands separator from 10 000 up."""
    if digits == 0 and abs(value) >= 10000:
        return f"{value:,.0f}".replace(",", r"\,")
    return f"{value:.{digits}f}"


def fmt_signed(value: float, digits: int = 1) -> str:
    """Signed difference with a typographic minus: ``$+$1.9`` / ``$-$0.4``."""
    r = round(value, digits)
    return ("$-$" if r < 0 else "$+$") + f"{abs(r):.{digits}f}"


def fmt_mean_std(mean: float, std: float, digits: int = 1) -> str:
    """``mean $\\pm$ std``; a deterministic value (std 0 or NaN) is printed alone."""
    if std is None or math.isnan(std) or std == 0:
        return fmt_num(mean, digits)
    return f"{fmt_num(mean, digits)} $\\pm$ {fmt_num(std, digits)}"


def fmt_ci(low: float, high: float, digits: int = 1) -> str:
    """``[low, high]``; ``--`` when the interval is a single point (deterministic)."""
    if low == high or math.isnan(low) or math.isnan(high):
        return "--"
    return f"[{fmt_num(low, digits)}, {fmt_num(high, digits)}]"


def bold(text: str) -> str:
    return f"\\textbf{{{text}}}"


def best_index(values: Sequence[float], lower_is_better: bool = True) -> int:
    """Index of the best (unrounded) value; NaN entries are skipped, ties go to the first."""
    sign = 1 if lower_is_better else -1
    xs = [math.inf if v is None or math.isnan(v) else sign * v for v in values]
    return min(range(len(xs)), key=xs.__getitem__)


def tabular(spec: str, head: Sequence[str], rows: Iterable[Sequence[str] | str],
            rules_after: Iterable[int] = ()) -> str:  # fmt: skip
    """``booktabs`` tabular; ``rules_after`` = row indices followed by a ``\\midrule``.

    A row given as one string is written as it is (e.g. a ``\\multicolumn`` line).
    """
    rows = [r if isinstance(r, str) else list(r) for r in rows]
    for r in rows:
        if not isinstance(r, str) and len(r) != len(head):
            raise ValueError(f"row has {len(r)} cells, header has {len(head)}: {r}")
    rules = set(rules_after)
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    for i, r in enumerate(rows):
        out.append((r if isinstance(r, str) else " & ".join(r)) + " \\\\")
        if i in rules:
            out.append("\\midrule")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


class Stats:
    """Lookup in a long statistics frame: one row per (model, condition)."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self._rows = {(r["model"], r["condition"]): r for _, r in frame.iterrows()}

    def has(self, model: str, cond: str) -> bool:
        return (model, cond) in self._rows

    def get(self, model: str, cond: str, column: str) -> float:
        try:
            return float(self._rows[(model, cond)][column])
        except KeyError as e:
            raise KeyError(f"no value for {model} / {cond} / {column}") from e

    def mean(self, model: str, cond: str, metric: str) -> float:
        return self.get(model, cond, f"{metric}_mean")

    def pm(self, model: str, cond: str, metric: str, digits: int = 1) -> str:
        return fmt_mean_std(self.mean(model, cond, metric),
                            self.get(model, cond, f"{metric}_std"), digits)  # fmt: skip

    def ci(self, model: str, cond: str, metric: str, digits: int = 1) -> str:
        return fmt_ci(self.get(model, cond, f"{metric}_ci95_low"),
                      self.get(model, cond, f"{metric}_ci95_high"), digits)  # fmt: skip


def _column(rows: list[list[str]], col: int, values: Sequence[float], lower: bool = True) -> None:
    """Bold the best cell of column ``col`` (``values`` = one number per row, NaN = skip)."""
    rows[best_index(values, lower)][col] = bold(rows[best_index(values, lower)][col])


def table_main(main: pd.DataFrame) -> str:
    """7.1: validation / test travel time, secondary metrics, training time, parameters."""
    st = Stats(main)
    extra = main.set_index("model")
    rows = []
    for m in AGENTS:
        rows.append([
            LABELS[m],
            escape(extra.loc[m, "validation"]),
            st.pm(m, "clean", ATT),
            st.ci(m, "clean", ATT),
            fmt_num(st.mean(m, "clean", WAIT)),
            fmt_num(st.mean(m, "clean", QUEUE), 2),
            fmt_num(st.mean(m, "clean", THR), 0),
            st.pm(m, "clean", MAXW, 0),
            escape(extra.loc[m, "training_time_min"]).replace("—", "--"),
            escape(extra.loc[m, "params"]).replace(",", r"\,"),
        ])  # fmt: skip
    _column(rows, 2, [st.mean(m, "clean", ATT) for m in AGENTS])
    head = ["Model", "Val.\\ (s)", "Test (s)", "95\\,\\% CI", "Wait (s)", "Queue", "Thr.",
            "Max wait (s)", "Train", "Params"]  # fmt: skip
    return tabular("lrrcrrrrrr", head, rows, rules_after=[1, 3])


def table_ablation(abl: pd.DataFrame, params: Mapping[str, str]) -> str:
    """7.2: full model vs −gate / −attention / −ensemble (clean and noise 0.2)."""
    st = Stats(abl)
    full = st.mean("ugmarl", "clean", ATT)
    rows = []
    for m in VARIANTS:
        d = st.mean(m, "clean", ATT) - full
        rows.append([
            LABELS[m].replace(" (proposed)", " (full)"),
            st.pm(m, "clean", ATT),
            "--" if m == "ugmarl" else fmt_signed(d),
            st.pm(m, "noise0.2", ATT),
            st.pm(m, "surge1.5", ATT),
            st.pm(m, "clean", MAXW, 0),
            st.pm(m, "noise0.2", MAXW, 0),
            fmt_num(100 * st.mean(m, "clean", FB), 0),
            escape(params[m]).replace(",", r"\,"),
        ])  # fmt: skip
    head = ["Variant", "Clean (s)", "$\\Delta$ (s)", "Noise 0.2 (s)", "Surge (s)",
            "Max wait, clean", "Max wait, noise 0.2", "FB (\\%)", "Params"]  # fmt: skip
    return tabular("lrrrrrrrr", head, rows)


def table_robustness(rob: pd.DataFrame) -> str:
    """7.3: travel time per condition (mean ± std) and the largest regret of each model."""
    st = Stats(rob)
    models = [*AGENTS, "ugmarl-nogate"]
    rows = []
    for m in models:
        regrets = [st.get(m, c, "regret") for c in CONDITIONS]
        rows.append([LABELS[m], *(st.pm(m, c, ATT) for c in CONDITIONS),
                     fmt_num(max(regrets))])  # fmt: skip
    for j, c in enumerate(CONDITIONS):  # best of the five agents, as in the regret definition
        vals = [st.mean(m, c, ATT) if m in AGENTS else math.nan for m in models]
        _column(rows, j + 1, vals)
    worst = [max(st.get(m, c, "regret") for c in CONDITIONS) if m in AGENTS else math.nan
             for m in models]  # fmt: skip
    _column(rows, len(CONDITIONS) + 1, worst)
    head = ["Model", *CONDITIONS.values(), "Regret"]
    return tabular("l" + "r" * (len(CONDITIONS) + 1), head, rows, rules_after=[4])


def table_heldout(held: pd.DataFrame, held_all: pd.DataFrame) -> str:
    """7.4: zero-shot travel time on the three Hangzhou flows, and over all flows."""
    st, sa = Stats(held), Stats(held_all)
    models = [m for m in LABELS if sa.has(m, "all")]
    rows = []
    for m in models:
        rows.append([
            LABELS[m],
            *(st.pm(m, f, ATT) for f in HELDOUT_FLOWS),
            sa.pm(m, "all", ATT),
            sa.ci(m, "all", ATT),
            sa.pm(m, "all", MAXW, 0),
            fmt_num(100 * sa.mean(m, "all", FB), 0),
        ])  # fmt: skip
    _column(rows, len(HELDOUT_FLOWS) + 1, [sa.mean(m, "all", ATT) for m in models])
    at = next(i for i, m in enumerate(models) if m == "dqn") + 1
    width = len(HELDOUT_FLOWS) + 4
    rows.insert(at, f"IDQN & \\multicolumn{{{width}}}{{c}}{{not transferable "
                    "(one network per Jinan intersection)}")  # fmt: skip
    head = ["Model", *HELDOUT_FLOWS.values(), "All flows (s)", "95\\,\\% CI", "Max wait (s)",
            "FB (\\%)"]  # fmt: skip
    return tabular("l" + "r" * (width - 3) + "crr", head, rows, rules_after=[1, at])


def _mean_std_text(cell: str) -> str:
    """``0.812 ± 0.067 [0.729, 0.896]`` -> ``0.812 $\\pm$ 0.067``."""
    return escape(str(cell).split("[")[0].strip())


def table_efficiency(eff: pd.DataFrame) -> str:
    """7.5: decision latency, training time, parameters, checkpoint size."""
    e = eff.set_index("model")
    rows = []
    for m in LABELS:
        r = e.loc[m]
        rows.append([
            LABELS[m],
            _mean_std_text(r["latency_ms"]),
            escape(r["latency_p95_ms"]),
            escape(r["latency_heldout_ms"]).replace("—", "--"),
            escape(r["training_time_min"]).replace("—", "--"),
            escape(r["params"]).replace(",", r"\,"),
            escape(r["checkpoint_mb"]).replace("—", "--"),
        ])  # fmt: skip
    head = ["Model", "Latency (ms)", "p95 (ms)", "Hangzhou (ms)", "Train (min)", "Params",
            "File (MB)"]  # fmt: skip
    return tabular("lrrrrrr", head, rows, rules_after=[4])


# --- number macros ---------------------------------------------------------------------


def number_macros(main: pd.DataFrame, rob: pd.DataFrame, held: pd.DataFrame,
                  held_all: pd.DataFrame) -> dict[str, str]:  # fmt: skip
    """``{"metric@model@condition": "value"}`` for every number the report text may quote.

    Metrics: ``tt`` travel time, ``ttsd`` its std, ``ttlo`` / ``tthi`` its 95 % CI, ``dtt``
    increase over the clean flow, ``wait``, ``maxw``, ``pnn`` 99th-percentile wait, ``thr``,
    ``fb`` fallback rate in percent, ``regret``; ``queue`` on the clean flow only. Conditions:
    the six test conditions, ``hz-all`` and ``hz-<flow>`` for Hangzhou, ``worst`` / ``mean``
    for the regret over the six conditions.
    """
    out: dict[str, str] = {}
    st = Stats(rob)
    for m, c in st._rows:
        tt = st.mean(m, c, ATT)
        out[f"tt@{m}@{c}"] = fmt_num(tt)
        out[f"ttsd@{m}@{c}"] = fmt_num(st.get(m, c, f"{ATT}_std"))
        out[f"ttlo@{m}@{c}"] = fmt_num(st.get(m, c, f"{ATT}_ci95_low"))
        out[f"tthi@{m}@{c}"] = fmt_num(st.get(m, c, f"{ATT}_ci95_high"))
        out[f"dtt@{m}@{c}"] = fmt_num(tt - st.mean(m, "clean", ATT))
        out[f"wait@{m}@{c}"] = fmt_num(st.mean(m, c, WAIT))
        out[f"maxw@{m}@{c}"] = fmt_num(st.mean(m, c, MAXW), 0)
        out[f"pnn@{m}@{c}"] = fmt_num(st.mean(m, c, "wait_p99_s"), 0)
        out[f"thr@{m}@{c}"] = fmt_num(st.mean(m, c, THR), 0)
        out[f"fb@{m}@{c}"] = fmt_num(100 * st.mean(m, c, FB), 0)
        out[f"regret@{m}@{c}"] = fmt_num(st.get(m, c, "regret"))
    for m in dict.fromkeys(rob["model"]):
        regrets = [st.get(m, c, "regret") for c in CONDITIONS]
        out[f"regret@{m}@worst"] = fmt_num(max(regrets))
        out[f"regret@{m}@mean"] = fmt_num(sum(regrets) / len(regrets))
    sm = Stats(main)
    for m in AGENTS:
        out[f"queue@{m}@clean"] = fmt_num(sm.mean(m, "clean", QUEUE), 2)
    for frame, name in ((held, None), (held_all, "all")):
        sh = Stats(frame)
        for m, c in sh._rows:
            cond = f"hz-{name or c}"
            out[f"tt@{m}@{cond}"] = fmt_num(sh.mean(m, c, ATT))
            out[f"ttsd@{m}@{cond}"] = fmt_num(sh.get(m, c, f"{ATT}_std"))
            out[f"maxw@{m}@{cond}"] = fmt_num(sh.mean(m, c, MAXW), 0)
            out[f"thr@{m}@{cond}"] = fmt_num(sh.mean(m, c, THR), 0)
            out[f"fb@{m}@{cond}"] = fmt_num(100 * sh.mean(m, c, FB), 0)
    return out


def render_macros(macros: Mapping[str, str]) -> str:
    """``\\res{metric}{model}{condition}`` with a compile error for an unknown key."""
    lines = [
        "% Generated by scripts/report_tables.py from docs/results/*.csv -- do not edit.",
        "\\makeatletter",
        "\\newcommand{\\res}[3]{\\ifcsname res@#1@#2@#3\\endcsname"
        "\\csname res@#1@#2@#3\\endcsname\\else"
        "\\PackageError{report}{unknown result #1/#2/#3}{}\\fi}",
    ]
    for key in sorted(macros):
        lines.append(f"\\@namedef{{res@{key}}}{{{macros[key]}}}")
    lines.append("\\makeatother")
    return "\n".join(lines) + "\n"
