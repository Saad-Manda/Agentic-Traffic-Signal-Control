"""Phase 8 figures (ROADMAP 8.1, 8.2, 8.4; plan phase8-analysis).

One function per figure. Each takes already-computed frames (or a
:class:`~trafficrl.eval.analysis.Study`), returns the matplotlib figure and — when ``out`` is
given — writes ``<name>.png``, ``<name>.pdf`` and ``<name>.md`` (caption, values, pointer to
the interpretation). Pure readers: no simulation, nothing selected. Called from
``notebooks/02_analysis.ipynb``; style and colours come from :mod:`trafficrl.utils.plotting`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from trafficrl.eval import analysis as A
from trafficrl.eval.hypotheses import regret_summary, regret_table
from trafficrl.utils.plotting import (
    BLUE,
    BLUE_RAMP,
    CONTROLLER_COLOR,
    CONTROLLER_LABEL,
    INK,
    INK_SECONDARY,
    MUTED,
    ORANGE,
    SURFACE,
    VIOLET,
    apply_style,
)

INTERPRETATION = "`notebooks/02_analysis.ipynb` and the report (Ch. 4)"
CONDITION_LABEL = {
    "clean": "clean",
    "noise0.1": "noise 0.1",
    "noise0.2": "noise 0.2",
    "drop0.1": "dropout 10 %",
    "drop0.2": "dropout 20 %",
    "surge1.5": "surge ×1.5",
}
DIVERGING = LinearSegmentedColormap.from_list("ugmarl_div", [BLUE, SURFACE, ORANGE])
SEQUENTIAL = LinearSegmentedColormap.from_list("ugmarl_seq", [SURFACE, *BLUE_RAMP])
_IID = re.compile(r"_(\d+)_(\d+)$")


def label(model: str) -> str:
    return CONTROLLER_LABEL.get(model, model)


def short(model: str) -> str:
    return label(model).replace(" (proposed)", "").replace(" (shared)", "")


def cond(c: str) -> str:
    return CONDITION_LABEL.get(c, c)


def finish(
    fig: plt.Figure, out: str | Path | None, name: str, title: str, caption: str, values: str
) -> plt.Figure:
    """Save PNG + PDF + the caption file (when ``out`` is given) and return the figure."""
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.png")
        fig.savefig(out / f"{name}.pdf")
        (out / f"{name}.md").write_text(
            f"# Figure — {title}\n\n![{name}]({name}.png)\n\n"
            f"**Caption.** {caption}\n\n**Values.** {values}\n\n"
            f"**Interpretation.** See {INTERPRETATION}. Attention weights, σ and gate "
            "statistics are associations, not causes.\n"
        )
    return fig


def grid_xy(iid: str) -> tuple[int, int]:
    m = _IID.search(iid)
    if not m:
        raise ValueError(f"cannot place {iid!r} on the grid")
    return int(m.group(1)), int(m.group(2))


def grid_panel(
    ax: plt.Axes,
    values: Mapping[str, float],
    cmap,
    vmin: float,
    vmax: float,
    fmt: str = "{:.0f}",
    marks: Mapping[str, bool] | None = None,
):
    """Draw one value per intersection at its place in the network (x → east, y → north).
    ``marks`` adds a ``*`` to cells whose difference is clear (CI excludes 0)."""
    xs, ys = zip(*(grid_xy(i) for i in values), strict=True)
    nx, ny = max(xs), max(ys)
    grid = np.full((ny, nx), np.nan)
    for iid, v in values.items():
        x, y = grid_xy(iid)
        grid[y - 1, x - 1] = v
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="equal")
    for iid, v in values.items():
        x, y = grid_xy(iid)
        star = "*" if marks and marks.get(iid) else ""
        dark = cmap is SEQUENTIAL and (v - vmin) / max(vmax - vmin, 1e-12) > 0.6
        ax.text(x - 1, y - 1, fmt.format(v) + star, ha="center", va="center", fontsize=8,
                color=SURFACE if dark else INK)  # fmt: skip
    ax.set_xticks(range(nx), [str(i + 1) for i in range(nx)])
    ax.set_yticks(range(ny), [str(i + 1) for i in range(ny)])
    ax.set_xlabel("column (west → east)")
    ax.set_ylabel("row (south → north)")
    ax.grid(False)
    return im


def _ci_points(ax, x, stats: dict, color, seeds=None, marker="o", width=0.0) -> None:
    half_lo, half_hi = stats["mean"] - stats["ci95_low"], stats["ci95_high"] - stats["mean"]
    if not stats["deterministic"]:
        ax.errorbar(x, stats["mean"], yerr=[[half_lo], [half_hi]], color=INK_SECONDARY,
                    capsize=3, linewidth=1.2, zorder=3)  # fmt: skip
    ax.plot([x], [stats["mean"]], marker=marker, color=color, markersize=7, zorder=4)
    if seeds is not None and not stats["deterministic"]:
        jitter = np.linspace(-0.12, 0.12, len(seeds)) + width
        ax.scatter(x + jitter, seeds, s=10, color=color, alpha=0.45, zorder=2, linewidths=0)


# --- 8.1 comparison -------------------------------------------------------------------------


def fig_main_comparison(study: A.Study, agents: Sequence[str], out=None) -> plt.Figure:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    panels = [("avg_travel_time_s", "Average travel time (s)"),
              ("max_waiting_time_s", "Maximum per-vehicle wait (s)")]  # fmt: skip
    vals = []
    for ax, (metric, ylabel) in zip(axes, panels, strict=True):
        for k, m in enumerate(agents):
            s = study.stats(m, "clean", metric)
            ax.bar(k, s["mean"], color=CONTROLLER_COLOR[m], width=0.62, alpha=0.9)
            v = study.values(m, "clean", metric)
            if not s["deterministic"]:
                err = [[s["mean"] - s["ci95_low"]], [s["ci95_high"] - s["mean"]]]
                ax.errorbar(k, s["mean"], yerr=err, color=INK, capsize=4, linewidth=1.3)
                ax.scatter(k + np.linspace(-0.18, 0.18, len(v)), v, s=12, color=INK, zorder=3)
            ax.text(k, s["mean"] * 1.02 if s["deterministic"] else s["ci95_high"] * 1.02,
                    f"{s['mean']:.1f}" if metric.startswith("avg") else f"{s['mean']:.0f}",
                    ha="center", va="bottom", fontsize=9)  # fmt: skip
            vals.append(f"{short(m)} {metric.split('_')[0]} {s['mean']:.1f}")
        ax.set_xticks(range(len(agents)), [short(m) for m in agents], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
        ax.grid(axis="x", visible=False)
    axes[0].set_title("(a) Travel time — lower is better", fontsize=10)
    axes[1].set_title("(b) Longest wait of any vehicle", fontsize=10)
    fig.suptitle("Jinan test flow, clean detectors, 5 seeds", fontweight="bold")
    fig.tight_layout()
    return finish(
        fig, out, "main_comparison", "main comparison on the Jinan test flow",
        "Five agents on the clean Jinan test flow (`real_2500`, 3600 s, Δt = 15 s). Bars: mean "
        "over seeds 0–4; error bars: 95 % CI (Student t); dots: seeds. Fixed-Time and "
        "Max-Pressure are deterministic (no error bar). (a) average travel time, (b) maximum "
        "per-vehicle wait inside the network.",
        "; ".join(vals) + ". Source: `results/final/final_test` (= `docs/results/test_main.csv`).",
    )  # fmt: skip


def fig_ablation(study: A.Study, variants: Sequence[str], conditions: Sequence[str], out=None):
    apply_style()
    metrics = [("avg_travel_time_s", "Average travel time (s)"),
               ("max_waiting_time_s", "Maximum per-vehicle wait (s)")]  # fmt: skip
    fig, axes = plt.subplots(len(metrics), len(conditions), figsize=(10.5, 6.2), squeeze=False)
    vals = []
    for r, (metric, ylabel) in enumerate(metrics):
        for c, cn in enumerate(conditions):
            ax = axes[r][c]
            for k, m in enumerate(variants):
                s = study.stats(m, cn, metric)
                _ci_points(ax, k, s, CONTROLLER_COLOR[m], study.values(m, cn, metric),
                           marker="D" if m == "ugmarl-nogate" else "o")  # fmt: skip
                vals.append(f"{short(m)} / {cond(cn)} / {metric.split('_')[0]}: {s['mean']:.1f}")
            ax.set_xticks(range(len(variants)),
                          [short(m).replace("UG-MARL ", "") for m in variants])  # fmt: skip
            ax.set_xlim(-0.6, len(variants) - 0.4)
            ax.set_ylabel(ylabel if c == 0 else "")
            ax.set_title(f"{cond(cn)}", fontsize=10)
            ax.grid(axis="x", visible=False)
    fig.suptitle("Ablation on the Jinan test flow (mean, 95 % CI, seeds)", fontweight="bold")
    fig.tight_layout()
    return finish(
        fig, out, "ablation", "ablation of gate, attention and ensemble",
        "Full UG-MARL and its three ablations on the Jinan test flow, clean and with detector "
        "noise 0.2. Points: mean over seeds 0–4; bars: 95 % CI; small dots: seeds. −gate uses "
        "the same checkpoints as the full model (diamond); −attention and −ensemble are "
        "trained separately. The y axes do not start at 0.",
        "; ".join(vals) + ". Source: `docs/results/test_ablation.csv`.",
    )  # fmt: skip


def fig_wait_tail(
    study: A.Study, models: Sequence[str], conditions: Sequence[str], out=None
) -> plt.Figure:
    apply_style()
    fig, axes = plt.subplots(1, len(conditions), figsize=(10.5, 4.0), sharey=True)
    vals = []
    for ax, cn in zip(np.atleast_1d(axes), conditions, strict=True):
        for m in models:
            w = np.sort(np.concatenate(
                [study.log(m, cn, "vehicles", s)["wait_s"].to_numpy() for s in study.seeds]
            ))  # fmt: skip
            share = 1.0 - np.arange(1, len(w) + 1) / len(w)
            ax.step(w, np.maximum(share, 1e-5), where="post", color=CONTROLLER_COLOR[m],
                    linestyle="--" if m == "ugmarl-nogate" else "-",
                    linewidth=2.2 if m == "ugmarl" else 1.5, label=short(m))  # fmt: skip
            vals.append(f"{short(m)} / {cond(cn)}: p99 {np.percentile(w, 99):.0f} s, "
                        f"max {w.max():.0f} s")  # fmt: skip
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1)
        ax.set_xlabel("Waiting time inside the network, x (s)")
        ax.set_title(cond(cn), fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("Share of vehicles waiting longer than x")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Tail of the per-vehicle waiting time, Jinan test flow", fontweight="bold")
    fig.tight_layout()
    return finish(
        fig, out, "wait_tail", "tail of the per-vehicle waiting time",
        "Share of vehicles whose total waiting time (seconds at speed < 0.1 m/s inside the "
        "network) exceeds x; the five seeds are pooled (log y axis). Waiting outside the "
        "network — vehicles that cannot enter a full entry road — is not part of this metric.",
        "; ".join(vals) + ". Source: `vehicles_s*.csv` of `results/final/final_test`.",
    )  # fmt: skip


def fig_generalisation(study: A.Study, models: Sequence[str], out=None) -> plt.Figure:
    apply_style()
    flows = study.conditions
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    width = 0.8 / len(models)
    vals = []
    for k, m in enumerate(models):
        for j, f in enumerate(flows):
            s = study.stats(m, f, "avg_travel_time_s")
            x = j - 0.4 + (k + 0.5) * width
            ax.bar(x, s["mean"], width=width * 0.9, color=CONTROLLER_COLOR[m],
                   hatch="//" if m == "ugmarl-nogate" else None, edgecolor=SURFACE,
                   label=short(m) if j == 0 else None)  # fmt: skip
            if not s["deterministic"]:
                err = [[s["mean"] - s["ci95_low"]], [s["ci95_high"] - s["mean"]]]
                ax.errorbar(x, s["mean"], yerr=err, color=INK, capsize=2, linewidth=1)
            vals.append(f"{short(m)} / {f}: {s['mean']:.1f}")
    ax.set_xticks(range(len(flows)), [f"Hangzhou `{f}`".replace("`", "") for f in flows])
    ax.set_ylabel("Average travel time (s)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("Zero-shot on the held-out city (trained on Jinan only)", fontsize=10)
    fig.tight_layout()
    return finish(
        fig, out, "generalisation", "zero-shot generalisation to Hangzhou",
        "Average travel time on the three Hangzhou flow files (4 × 4 network), models trained "
        "and calibrated on Jinan only (τ from Jinan validation). Bars: mean over seeds 0–4; "
        "error bars: 95 % CI. IDQN is not transferable (one network per Jinan intersection).",
        "; ".join(vals) + ". Source: `docs/results/heldout_generalisation.csv`.",
    )  # fmt: skip


def fig_seed_spread(study: A.Study, models: Sequence[str], out=None) -> plt.Figure:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    panels = [("avg_travel_time_s", "Average travel time (s)"),
              ("max_waiting_time_s", "Maximum per-vehicle wait (s)")]  # fmt: skip
    vals = []
    for ax, (metric, ylabel) in zip(axes, panels, strict=True):
        for k, m in enumerate(models):
            v = study.values(m, "clean", metric)
            ax.scatter([k] * len(v), v, color=CONTROLLER_COLOR[m], s=28, zorder=3,
                       marker="D" if m == "ugmarl-nogate" else "o")  # fmt: skip
            for s, x in zip(study.seeds, v, strict=True):
                ax.annotate(str(s), (k, x), xytext=(6, -3), textcoords="offset points",
                            fontsize=7, color=MUTED)  # fmt: skip
            vals.append(f"{short(m)} {metric.split('_')[0]} range {max(v) - min(v):.1f}")
        names = [short(m).replace("UG-MARL ", "") for m in models]
        ax.set_xticks(range(len(models)), names, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Spread over the five training seeds, clean Jinan test flow", fontweight="bold")
    fig.tight_layout()
    return finish(
        fig, out, "seed_spread", "spread over training seeds",
        "Every seed's result on the clean Jinan test flow for the learned models (the number "
        "next to a point is the seed; seed s = the model trained with seed s). The y axes do "
        "not start at 0.",
        "; ".join(vals) + ".",
    )  # fmt: skip


# --- 8.2 robustness, regret, gate, attention ------------------------------------------------


def fig_robustness(study: A.Study, models: Sequence[str], out=None) -> plt.Figure:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), gridspec_kw={"width_ratios": [3, 3, 2]})
    series = [
        (["clean", "noise0.1", "noise0.2"], [0, 0.1, 0.2],
         "Detector noise (std, share of capacity)"),
        (["clean", "drop0.1", "drop0.2"], [0, 10, 20], "Detector dropout (%)"),
    ]  # fmt: skip
    vals = []
    for ax, (conds, xs, xlabel) in zip(axes[:2], series, strict=True):
        for m in models:
            st = [study.stats(m, c, "avg_travel_time_s") for c in conds]
            mean = [s["mean"] for s in st]
            ax.plot(xs, mean, color=CONTROLLER_COLOR[m], marker="o", markersize=4,
                    linestyle="--" if m == "ugmarl-nogate" else "-",
                    linewidth=2.4 if m == "ugmarl" else 1.5, label=short(m))  # fmt: skip
            ax.fill_between(xs, [s["ci95_low"] for s in st], [s["ci95_high"] for s in st],
                            color=CONTROLLER_COLOR[m], alpha=0.15, linewidth=0)  # fmt: skip
            vals.append(f"{short(m)}: " + " → ".join(f"{v:.0f}" for v in mean))
        ax.set_xticks(xs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Average travel time (s)")
    axes[0].set_title("(a) noise", fontsize=10)
    axes[1].set_title("(b) dropout", fontsize=10)
    axes[0].legend(fontsize=8)
    ax = axes[2]
    for k, m in enumerate(models):
        s = study.stats(m, "surge1.5", "avg_travel_time_s")
        _ci_points(ax, k, s, CONTROLLER_COLOR[m], study.values(m, "surge1.5", "avg_travel_time_s"),
                   marker="D" if m == "ugmarl-nogate" else "o")  # fmt: skip
        vals.append(f"{short(m)} surge {s['mean']:.0f}")
    ax.set_xticks(range(len(models)), [short(m).replace("UG-MARL ", "") for m in models],
                  rotation=35, ha="right", fontsize=8)  # fmt: skip
    ax.set_title("(c) demand surge ×1.5", fontsize=10)
    ax.set_ylabel("Average travel time (s)")
    ax.grid(axis="x", visible=False)
    fig.suptitle("Robustness on the Jinan test flow (mean and 95 % CI over 5 seeds)",
                 fontweight="bold")  # fmt: skip
    fig.tight_layout()
    return finish(
        fig, out, "robustness_degradation", "robustness to detector noise, dropout and surge",
        "Average travel time on the Jinan test flow as detector noise (a) or dropout (b) "
        "grows, and under a demand surge of ×1.5 (c). Lines / points: mean over seeds 0–4; "
        "bands / bars: 95 % CI. −gate (dashed) is the same checkpoints as UG-MARL with the "
        "gate switched off. Noise and dropout change what the controllers observe, not the "
        "traffic; Fixed-Time does not use detectors.",
        "; ".join(vals) + ". Source: `docs/results/test_robustness.csv`.",
    )  # fmt: skip


def regret_frames(study: A.Study, models: Sequence[str], reference: Sequence[str]):
    means = pd.DataFrame(
        [
            {"model": m, "condition": c, "mean": study.stats(m, c, "avg_travel_time_s")["mean"]}
            for m in models
            for c in study.conditions
        ]
    )
    reg = regret_table(means, reference)
    return reg, regret_summary(reg)


def fig_regret(study: A.Study, models: Sequence[str], reference: Sequence[str], out=None):
    apply_style()
    reg, summ = regret_frames(study, models, reference)
    conds = study.conditions
    mat = reg.pivot(index="model", columns="condition", values="regret").loc[models, conds]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [3, 2]})
    im = ax1.imshow(np.log10(1 + mat.to_numpy()), cmap=SEQUENTIAL, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iloc[i, j]
            ax1.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                     color=SURFACE if v > 150 else INK)  # fmt: skip
    ax1.set_xticks(range(len(conds)), [cond(c) for c in conds], rotation=20, ha="right")
    ax1.set_yticks(range(len(models)), [short(m) for m in models])
    ax1.grid(False)
    ax1.set_title("(a) Regret per condition (s)", fontsize=10)
    fig.colorbar(im, ax=ax1, label="log10(1 + regret)", shrink=0.8)
    summ = summ.set_index("model").loc[models]
    y = np.arange(len(models))
    ax2.barh(y - 0.2, summ["max_regret"], height=0.38, color=[CONTROLLER_COLOR[m] for m in models])
    ax2.barh(y + 0.2, summ["mean_regret"], height=0.38, alpha=0.45,
             color=[CONTROLLER_COLOR[m] for m in models])  # fmt: skip
    for k, m in enumerate(models):
        ax2.text(summ.loc[m, "max_regret"], k - 0.2, f" {summ.loc[m, 'max_regret']:.1f}",
                 va="center", fontsize=8)  # fmt: skip
        ax2.text(summ.loc[m, "mean_regret"], k + 0.2, f" {summ.loc[m, 'mean_regret']:.1f}",
                 va="center", fontsize=8, color=INK_SECONDARY)  # fmt: skip
    ax2.set_yticks(y, [short(m) for m in models])
    ax2.invert_yaxis()
    ax2.set_xscale("symlog", linthresh=10)
    ax2.set_xlabel("Regret (s): largest (solid) and mean (light) over the conditions")
    ax2.set_title("(b) Worst-case and mean regret", fontsize=10)
    ax2.grid(axis="y", visible=False)
    fig.suptitle("Regret against the best of the five agents in each condition",
                 fontweight="bold")  # fmt: skip
    fig.tight_layout()
    return finish(
        fig, out, "regret", "regret against the best agent per condition",
        "Regret = a model's mean travel time − the lowest mean among the five agents "
        f"({', '.join(short(m) for m in reference)}) in the same condition, Jinan test flow, "
        "seeds 0–4. (a) per condition (colour on a log scale); (b) each model's largest and "
        "mean regret over the six conditions (symmetric-log x axis). Descriptive statistic "
        "fixed before the test runs.",
        "; ".join(f"{short(m)}: max {summ.loc[m, 'max_regret']:.1f} s "
                  f"({cond(summ.loc[m, 'worst_condition'])}), "
                  f"mean {summ.loc[m, 'mean_regret']:.1f} s" for m in models) + ".",
    )  # fmt: skip


def fig_gate_firing(
    study: A.Study, model: str, conditions: Sequence[str], grid_conditions: Sequence[str],
    window_s: int = 300, out=None,
) -> plt.Figure:  # fmt: skip
    apply_style()
    fig = plt.figure(figsize=(12.0, 7.4))
    gs = fig.add_gridspec(2, 2 * len(grid_conditions), height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, : len(grid_conditions)])
    ax2 = fig.add_subplot(gs[0, len(grid_conditions) :])
    ramp = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(conditions)))
    dec = {c: pd.concat([study.log(model, c, "decisions", s) for s in study.seeds])
           for c in conditions}  # fmt: skip
    data = [np.log10(d["sigma"] / d["tau"]).to_numpy() for d in dec.values()]
    bp = ax1.boxplot(data, showfliers=False, patch_artist=True, widths=0.55)
    for patch, col in zip(bp["boxes"], ramp, strict=True):
        patch.set(facecolor=col, edgecolor=INK_SECONDARY, alpha=0.75)
    for med in bp["medians"]:
        med.set(color=INK, linewidth=1.8)
    ax1.axhline(0, color=INK, linestyle=":", linewidth=1.2)
    ax1.text(0.99, 0.02, "the gate fires above the dotted line (σ > τ)", fontsize=8,
             color=INK_SECONDARY, transform=ax1.transAxes, ha="right")  # fmt: skip
    ax1.set_xticks(range(1, len(conditions) + 1), [cond(c) for c in conditions], rotation=15)
    ax1.set_ylabel("log10(σ / τ)")
    ax1.set_title("(a) Uncertainty relative to the threshold", fontsize=10)
    ax1.grid(axis="x", visible=False)
    vals = []
    for c, col in zip(conditions, ramp, strict=True):
        d = dec[c]
        r = d.groupby(d["t"] // window_s)["fallback"].mean()
        ax2.plot((r.index + 0.5) * window_s / 60, r.to_numpy(), color=col, marker="o",
                 markersize=3, label=cond(c))  # fmt: skip
        vals.append(f"{cond(c)}: fallback {d['fallback'].mean():.3f}")
    ax2.set_xlabel("Time in the episode (min)")
    ax2.set_ylabel("Share of decisions handed to Max-Pressure")
    ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=8, ncol=2)
    ax2.set_title(f"(b) Fallback rate per {window_s // 60}-minute window", fontsize=10)
    for k, c in enumerate(grid_conditions):
        ax = fig.add_subplot(gs[1, 2 * k : 2 * k + 2])
        rate = dec[c].groupby("iid")["fallback"].mean().to_dict()
        im = grid_panel(ax, rate, SEQUENTIAL, 0, 1, "{:.2f}")
        ax.set_title(f"(c{k + 1}) Fallback rate per intersection — {cond(c)}", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.75, label="fallback rate")
    fig.suptitle(f"When the gate fires — {short(model)}, Jinan test flow, 5 seeds pooled",
                 fontweight="bold")  # fmt: skip
    fig.tight_layout()
    return finish(
        fig, out, "gate_firing", "when and where the uncertainty gate fires",
        "(a) Ensemble disagreement σ of every decision divided by that model's threshold τ "
        "(boxes: quartiles; whiskers 1.5 IQR; outliers hidden). (b) Share of decisions handed "
        f"to Max-Pressure per {window_s // 60}-minute window. (c) The same share per "
        "intersection, drawn at the intersection's place in the 3 × 4 network. Seeds 0–4 "
        "pooled; τ per seed-model from Jinan validation (p* = 80).",
        "; ".join(vals)
        + ". Source: `decisions_s*.csv` (= `docs/results/test_safety_explain.csv`).",
    )  # fmt: skip


def attention_frame(study: A.Study, model: str, condition: str) -> pd.DataFrame:
    """Mean attention weight per intersection and slot; an absent neighbour stays NaN."""
    d = pd.concat([study.log(model, condition, "decisions", s) for s in study.seeds])
    cols = ["attn_self", "attn_E", "attn_N", "attn_W", "attn_S"]
    return d.groupby("iid")[cols].mean().replace(0.0, np.nan)


def fig_attention_map(study: A.Study, model: str, condition: str = "clean", out=None):
    apply_style()
    att = attention_frame(study, model, condition)
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    offsets = {"E": (1, 0), "N": (0, 1), "W": (-1, 0), "S": (0, -1)}
    top = float(np.nanmax(att[["attn_E", "attn_N", "attn_W", "attn_S"]].to_numpy()))
    for iid, row in att.iterrows():
        x, y = grid_xy(iid)
        ax.scatter([x], [y], s=2600 * row["attn_self"], color=VIOLET, alpha=0.25, zorder=2)
        ax.text(x, y, f"{row['attn_self']:.2f}", ha="center", va="center", fontsize=9, zorder=4)
        for slot, (dx, dy) in offsets.items():
            w = row[f"attn_{slot}"]
            if np.isnan(w):
                continue
            ax.annotate("", xy=(x + 0.14 * dx, y + 0.14 * dy),
                        xytext=(x + 0.46 * dx, y + 0.46 * dy),
                        arrowprops={"arrowstyle": "-|>", "color": SEQUENTIAL(0.25 + 0.75 * w / top),
                                    "linewidth": 1 + 5 * w / top}, zorder=3)  # fmt: skip
            ax.text(x + 0.33 * dx + 0.09 * abs(dy), y + 0.33 * dy + 0.09 * abs(dx), f"{w:.2f}",
                    fontsize=7, color=INK_SECONDARY, ha="center", va="center")  # fmt: skip
    xs, ys = zip(*(grid_xy(i) for i in att.index), strict=True)
    ax.set_xlim(0.4, max(xs) + 0.6)
    ax.set_ylim(0.4, max(ys) + 0.6)
    ax.set_xticks(range(1, max(xs) + 1))
    ax.set_yticks(range(1, max(ys) + 1))
    ax.set_xlabel("column (west → east)")
    ax.set_ylabel("row (south → north)")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title(f"Mean attention weights — {short(model)}, {cond(condition)} test flow",
                 fontsize=10)  # fmt: skip
    fig.tight_layout()
    mean = att.mean()
    return finish(
        fig, out, "attention_map", "attention weights on the network",
        "Mean attention weight of every intersection on itself (number in the circle; circle "
        "area ∝ weight) and on each existing neighbour (arrow from the neighbour, width and "
        "shade ∝ weight), over all decisions of the five seed-models on the clean Jinan test "
        "flow. The weights show where the encoder looks; they are not a measure of a "
        "neighbour's influence on the chosen phase (see the faithfulness check).",
        "Mean over intersections: " + ", ".join(
            f"{c.split('_')[1]} {mean[c]:.3f}" for c in att.columns) + ".",
    )  # fmt: skip


# --- 8.4 error analysis ----------------------------------------------------------------------


def fig_error_intersections(
    comps: Mapping[tuple[str, str], pd.DataFrame], model: str, out=None, name="error_intersections"
) -> plt.Figure:
    """``comps[(baseline, condition)]`` = :func:`analysis.comparison` on ``intersections``."""
    apply_style()
    baselines = list(dict.fromkeys(b for b, _ in comps))
    conds = list(dict.fromkeys(c for _, c in comps))
    size = (5.6 * len(baselines), 3.9 * len(conds))
    fig, axes = plt.subplots(len(conds), len(baselines), figsize=size, squeeze=False)
    vals = []
    for r, c in enumerate(conds):
        for k, b in enumerate(baselines):
            d = comps[(b, c)].set_index("iid")
            rel = 100 * d["mean"] / d["wait_veh_s_b"]
            lim = max(5.0, float(np.abs(rel).max()))
            im = grid_panel(axes[r][k], rel.to_dict(), DIVERGING, -lim, lim, "{:+.0f}",
                            d["clear"].to_dict())  # fmt: skip
            axes[r][k].set_title(f"{short(model)} − {short(b)}, {cond(c)}", fontsize=10)
            fig.colorbar(im, ax=axes[r][k], shrink=0.75, label="difference (%)")
            tot = 100 * d["mean"].sum() / d["wait_veh_s_b"].sum()
            vals.append(f"vs {short(b)} / {cond(c)}: network {tot:+.1f} %, "
                        f"{int(d['clear'].sum())} of {len(d)} intersections clear")  # fmt: skip
    fig.suptitle("Where the waiting happens: waiting-vehicle-seconds per intersection, "
                 "difference to each baseline", fontweight="bold")  # fmt: skip
    fig.tight_layout()
    return finish(
        fig, out, name, "difference in waiting per intersection",
        f"Waiting-vehicle-seconds (Σ over the episode of waiting vehicles on the incoming "
        f"lanes × seconds) of {short(model)} minus the baseline's, in percent of the "
        "baseline, per intersection at its place in the network; blue = fewer waiting "
        "vehicles than the baseline, orange = more. Mean over seeds 0–4; * = the 95 % CI of "
        "the difference excludes 0. Jinan test flow, read from the saved traces.",
        "; ".join(vals) + ".",
    )  # fmt: skip


def fig_error_windows(
    comps: Mapping[tuple[str, str], pd.DataFrame], model: str, window_s: int, out=None
) -> plt.Figure:
    """``comps[(baseline, condition)]`` = :func:`analysis.comparison` on ``windows``."""
    apply_style()
    conds = list(dict.fromkeys(c for _, c in comps))
    fig, axes = plt.subplots(1, len(conds), figsize=(5.4 * len(conds), 4.0), squeeze=False)
    vals = []
    for ax, c in zip(axes[0], conds, strict=True):
        for b, cc in comps:
            if cc != c:
                continue
            d = comps[(b, c)].sort_values("window")
            x = (d["window"] + 0.5) * window_s / 60
            style = "--" if b == "ugmarl-nogate" else "-"
            ax.plot(x, d["mean"], color=CONTROLLER_COLOR[b], marker="o", linestyle=style,
                    label=f"− {short(b)}")  # fmt: skip
            ax.fill_between(x, d["ci95_low"], d["ci95_high"], color=CONTROLLER_COLOR[b],
                            alpha=0.15, linewidth=0)  # fmt: skip
            vals.append(f"vs {short(b)} / {cond(c)}: " + ", ".join(f"{v:+.1f}" for v in d["mean"]))
        ax.axhline(0, color=INK, linewidth=1)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xlabel("Scheduled start of the trip (min)")
        ax.set_ylabel(f"Travel time, {short(model)} − baseline (s)")
        ax.set_title(cond(c), fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("When the difference arises: travel time by start window", fontweight="bold")
    fig.tight_layout()
    return finish(
        fig, out, "error_windows", "difference in travel time by start window",
        f"Mean travel time of the trips scheduled to start in each {window_s // 60}-minute "
        f"window, {short(model)} minus each baseline (below 0 = {short(model)} faster). Mean "
        "over seeds 0–4 with 95 % CI band; symmetric-log y axis (linear within ±10 s). Travel "
        "time counts from the scheduled start, so time blocked outside the network and trips "
        "unfinished at 3600 s are included, as in the simulator's average.",
        "; ".join(vals) + ".",
    )  # fmt: skip


def fig_error_routes(
    comps: Mapping[tuple[str, str], pd.DataFrame], model: str, by: str, out=None
) -> plt.Figure:
    """``comps[(baseline, condition)]`` = :func:`analysis.comparison` on a route table."""
    apply_style()
    conds = list(dict.fromkeys(c for _, c in comps))
    fig, axes = plt.subplots(1, len(conds), figsize=(5.4 * len(conds), 4.0), squeeze=False)
    vals = []
    for ax, c in zip(axes[0], conds, strict=True):
        bs = [b for b, cc in comps if cc == c]
        for k, b in enumerate(bs):
            d = comps[(b, c)].sort_values(by)
            x = np.arange(len(d)) + (k - (len(bs) - 1) / 2) * 0.22
            ax.errorbar(x, d["mean"], yerr=[d["mean"] - d["ci95_low"], d["ci95_high"] - d["mean"]],
                        fmt="D" if b == "ugmarl-nogate" else "o", color=CONTROLLER_COLOR[b],
                        capsize=3, label=f"− {short(b)}")  # fmt: skip
            ax.set_xticks(range(len(d)), d[by])
            vals.append(f"vs {short(b)} / {cond(c)}: " + ", ".join(
                f"{g} {v:+.1f}" for g, v in zip(d[by], d["mean"], strict=True)))  # fmt: skip
        ax.axhline(0, color=INK, linewidth=1)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xlabel("Intersections on the route" if by == "length_group" else "Route type")
        ax.set_ylabel(f"Travel time, {short(model)} − baseline (s)")
        ax.set_title(cond(c), fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=8)
    fig.suptitle("Which trips differ: travel time by route group", fontweight="bold")
    fig.tight_layout()
    return finish(
        fig, out, "error_routes", "difference in travel time by route group",
        f"Mean travel time per route group, {short(model)} minus each baseline (below 0 = "
        f"{short(model)} faster); mean over seeds 0–4, 95 % CI; symmetric-log y axis. Routes "
        "come from the test flow file, joined to the saved per-vehicle logs by flow index "
        "(read-only, Decision Log 13 Nov).",
        "; ".join(vals) + ".",
    )  # fmt: skip


def fig_case(
    kind: str, unit: Mapping, series: Mapping[str, pd.DataFrame], decisions: pd.DataFrame,
    width: int, seed: int, condition: str, out=None,
) -> plt.Figure:  # fmt: skip
    """One representative case: waiting vehicles of the model and the baselines at one
    intersection in one window, the model's phases, and σ against τ with gate firings."""
    apply_style()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10.0, 7.2), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1.3, 2]})  # fmt: skip
    for m, s in series.items():
        ax1.step(s["t"] / 60, s["waiting"], where="post", color=CONTROLLER_COLOR[m],
                 linewidth=2.2 if m == "ugmarl" else 1.4, label=short(m))  # fmt: skip
    ax1.set_ylabel("Waiting vehicles on the\nincoming lanes")
    ax1.legend(fontsize=8, ncol=len(series))
    model = next(iter(series))
    s = series[model]
    ax2.step(s["t"] / 60, s["phase"], where="post", color=CONTROLLER_COLOR[model])
    ax2.set_ylabel("Green phase\n(1–8)")
    ax2.set_yticks(range(1, 9, 2))
    d = decisions[(decisions["iid"] == unit["iid"]) & (decisions["t"] // width == unit["window"])]
    ax3.plot(d["t"] / 60, d["sigma"], color=VIOLET, marker="o", markersize=3, label="σ(s)")
    if d["tau"].notna().any():
        ax3.axhline(float(d["tau"].iloc[0]), color=INK, linestyle=":", label="τ")
    fired = d[d["fallback"].astype(bool)]
    changed = fired[fired["greedy_phase"] != fired["mp_phase"]]
    ax3.scatter(fired["t"] / 60, fired["sigma"], s=42, facecolors="none", edgecolors=ORANGE,
                label="gate fired", zorder=4)  # fmt: skip
    ax3.scatter(changed["t"] / 60, changed["sigma"], s=42, color=ORANGE,
                label="fired and changed the phase", zorder=5)  # fmt: skip
    ax3.set_ylabel("Ensemble std σ")
    ax3.set_xlabel("Time in the episode (min)")
    ax3.legend(fontsize=8, ncol=4)
    title = {"win": "largest gain", "loss": "largest loss", "gate": "most gate interventions"}[kind]
    fig.suptitle(f"Case “{kind}” ({title}): {unit['iid']}, minutes "
                 f"{unit['window'] * width // 60}–{(unit['window'] + 1) * width // 60}, "
                 f"{cond(condition)} test flow, seed {seed}", fontweight="bold")  # fmt: skip
    fig.tight_layout()
    detail = ", ".join(f"{k} = {v:.0f}" if isinstance(v, float) else f"{k} = {v}"
                       for k, v in unit.items() if k not in ("iid", "window"))  # fmt: skip
    return finish(
        fig, out, f"case_{kind}", f"representative case — {title}",
        f"One intersection and one {width // 60}-minute window of the {cond(condition)} Jinan "
        f"test flow, seed {seed} (the UG-MARL seed with the median clean travel time; "
        "baselines of the same seed number, which are separate episodes with their own "
        "traffic). Top: waiting vehicles on the incoming lanes. Middle: UG-MARL's green "
        "phase. Bottom: σ of each UG-MARL decision against τ; open circles = gate fired, "
        "filled = the fallback changed the phase. The case was picked by the rule fixed in "
        "the plan (A7), not by eye; it illustrates, the tables carry the evidence.",
        f"{unit['iid']}, window {unit['window']}; {detail}; decisions {len(d)}, gate fired "
        f"{len(fired)}, changed the phase {len(changed)}.",
    )  # fmt: skip


# --- 8.4 τ sensitivity, 8.2 attention faithfulness -----------------------------------------


def fig_tau_sensitivity(table: pd.DataFrame, p_star: float, out=None) -> plt.Figure:
    """``table``: rows ``p`` (number, ``ungated`` or ``max_pressure``), ``condition``,
    ``avg_travel_time_mean``, CI columns, ``fallback_rate_mean``
    (``docs/results/tau_sensitivity.csv``)."""
    apply_style()
    t = table.copy()
    num = t[pd.to_numeric(t["p"], errors="coerce").notna()].copy()
    num["p"] = num["p"].astype(float)
    conds = list(dict.fromkeys(t["condition"]))
    ramp = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(conds)))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.0, 4.2))
    ps = sorted(num["p"].unique())
    xpos = {p: i + 1 for i, p in enumerate(ps)}
    xpos.update({"max_pressure": 0, "ungated": len(ps) + 1})
    ticks = ["MP\n(τ = 0)", *[f"{p:g}" for p in ps], "no gate\n(τ = ∞)"]
    for c, col in zip(conds, ramp, strict=True):
        g = t[t["condition"] == c]
        x = [xpos[float(p)] if str(p).replace(".", "").isdigit() else xpos[p] for p in g["p"]]
        o = np.argsort(x)
        x = np.asarray(x)[o]
        ax1.plot(x, g["avg_travel_time_mean"].to_numpy()[o], color=col, marker="o", label=cond(c))
        ax2.plot(x, g["fallback_rate_mean"].to_numpy()[o], color=col, marker="o")
    score = t.groupby("p", sort=False)["avg_travel_time_mean"].mean()
    xs = [xpos[float(p)] if str(p).replace(".", "").isdigit() else xpos[p] for p in score.index]
    o = np.argsort(xs)
    sx, sy = np.asarray(xs)[o], score.to_numpy()[o]
    gated = sx != xpos["ungated"]  # the ungated score is far off the scale of the others
    ax3.plot(sx[gated], sy[gated], color=INK, marker="o")
    for x, v in zip(sx[gated], sy[gated], strict=True):
        ax3.annotate(f"{v:.1f}", (x, v), xytext=(0, 7), textcoords="offset points",
                     ha="center", fontsize=8)  # fmt: skip
    if (~gated).any():
        ax3.annotate(f"{sy[~gated][0]:.1f} ↑\n(off scale)", (xpos["ungated"], sy[gated].max()),
                     ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)  # fmt: skip
    ax3.set_ylim(sy[gated].min() - 2, sy[gated].max() + 4)
    for ax in (ax1, ax2, ax3):
        ax.axvline(xpos[float(p_star)], color=VIOLET, linestyle=":", linewidth=1.4)
        ax.set_xticks(range(len(ticks)), ticks, fontsize=8)
        ax.set_xlabel("Percentile p of clean validation σ used as τ")
    ax1.set_yscale("log")
    ax1.set_ylabel("Average travel time (s, log scale)")
    ax1.legend(fontsize=8)
    ax1.set_title("(a) Travel time per condition", fontsize=10)
    ax2.set_ylabel("Share of decisions handed to Max-Pressure")
    ax2.set_title("(b) Fallback rate", fontsize=10)
    ax3.set_ylabel("Mean over the five conditions (s)")
    ax3.set_title("(c) Calibration score", fontsize=10)
    fig.suptitle("Sensitivity to the gate threshold on the validation flow "
                 f"(dotted: p* = {p_star:g})", fontweight="bold")  # fmt: skip
    fig.tight_layout()
    return finish(
        fig, out, "tau_sensitivity", "sensitivity to the gate threshold τ (validation)",
        "UG-MARL on the Jinan validation flow with τ set to the p-th percentile of each "
        "seed-model's clean validation σ, from Max-Pressure (the gate always fires) to the "
        "ungated policy; mean over seeds 0–4. (a) travel time per detector condition, (b) "
        "fallback rate, (c) the calibration score (mean travel time over the five "
        "conditions). p = 50, 60, 70 were added after the final evaluation as a sensitivity "
        "check; nothing was re-selected (p* = 80).",
        "Score: " + ", ".join(f"p {p} = {v:.1f} s" for p, v in score.items()) + ".",
    )  # fmt: skip


def fig_faithfulness(per_seed: pd.DataFrame, out=None) -> plt.Figure:
    """``per_seed``: rows ``seed, perturbation, slot`` (hi / lo / self) with
    ``abs_dmu``, ``flip`` (docs/results/attention_faithfulness.csv)."""
    apply_style()
    found = set(per_seed["perturbation"])
    perts = [p for p in ("replacement", "knock-out") if p in found]
    fig, axes = plt.subplots(len(perts), 2, figsize=(10.5, 3.8 * len(perts)), squeeze=False)
    names = {"hi": "highest-α\nneighbour", "lo": "lowest-α\nneighbour", "self": "own\nobservation"}
    vals = []
    for r, p in enumerate(perts):
        d = per_seed[per_seed["perturbation"] == p]
        slots = [s for s in ("hi", "lo", "self") if s in set(d["slot"])]
        for ax, (col, ylabel) in zip(
            axes[r],
            [("abs_dmu", "Mean |Δμ| at the chosen action\n(scaled reward units)"),
             ("flip", "Share of decisions whose\ngreedy action changes")],
            strict=True,
        ):  # fmt: skip
            piv = d.pivot(index="seed", columns="slot", values=col)[slots]
            for s, row in piv.iterrows():
                ax.plot(range(len(slots)), row.to_numpy(), color=BLUE_RAMP[2], marker="o",
                        linewidth=1, markersize=4)  # fmt: skip
                ax.annotate(str(s), (len(slots) - 1, row.iloc[-1]), xytext=(5, -3),
                            textcoords="offset points", fontsize=7, color=MUTED)  # fmt: skip
            ax.plot(range(len(slots)), piv.mean().to_numpy(), color=BLUE, marker="o", linewidth=2.5)
            ax.set_xticks(range(len(slots)), [names[s] for s in slots])
            ax.set_ylabel(ylabel)
            ax.set_yscale("log")
            ax.set_title(f"{p}", fontsize=10)
            ax.grid(axis="x", visible=False)
            vals.append(f"{p} / {col}: " + ", ".join(
                f"{s} {piv[s].mean():.4f}" for s in slots))  # fmt: skip
    fig.suptitle("Attention faithfulness check on the validation flow "
                 "(thin: seed-models, thick: mean)", fontweight="bold")  # fmt: skip
    fig.tight_layout()
    return finish(
        fig, out, "attention_faithfulness", "attention faithfulness check (validation)",
        "Change in the network's output when one input slot of a decision is perturbed: the "
        "neighbour with the highest attention weight, the one with the lowest, and — as an "
        "upper reference — the intersection's own observation. Replacement = the slot's 44 "
        "values are swapped for the same slot's values from another decision of the same "
        "intersection in the same episode (8 draws); knock-out = the slot is masked as "
        "absent. One clean validation episode per seed-model, decisions with ≥ 2 neighbours; "
        "log y axes. An association check on validation data.",
        "; ".join(vals) + ".",
    )  # fmt: skip
