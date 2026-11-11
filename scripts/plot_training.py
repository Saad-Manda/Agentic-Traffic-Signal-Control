"""Learning curves of the RL baselines on the validation flow (plan D10).

    python scripts/plot_training.py [--agents dqn idqn] [--tag final] [--seeds 0 1 2 3 4]

Plots the greedy validation average travel time (s) against the training episode: mean over
seeds with a 95 % CI band (Student t), Fixed-Time and Max-Pressure validation values as dashed
reference lines. Writes ``docs/figures/training/learning_curves.png`` + ``.md`` (caption).
"""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt

from trafficrl.data.splits import REPO_ROOT
from trafficrl.eval.report import latest_eval_run, metric_stats, val_curves
from trafficrl.utils.plotting import (
    CONTROLLER_COLOR,
    CONTROLLER_LABEL,
    INK_SECONDARY,
    apply_style,
    save_figure,
)

OUT = REPO_ROOT / "docs" / "figures" / "training" / "learning_curves.png"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents", nargs="+", default=["dqn", "idqn"])
    ap.add_argument("--tag", default="final")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args(argv)

    apply_style()
    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1, 1.25]}
    )
    curves = {
        a: val_curves(REPO_ROOT / "results" / "train", a, args.tag, args.seeds) for a in args.agents
    }
    rules = {}
    for rule in ("fixed_time", "max_pressure"):
        run = latest_eval_run(REPO_ROOT / "results" / "eval", rule, "jinan", "val")
        rules[rule] = metric_stats(run / "metrics.csv", "avg_travel_time_s")["mean"]
    episodes_max = max(int(c["episode"].max()) for c in curves.values())
    summary = []
    for agent, c in curves.items():
        color = CONTROLLER_COLOR[agent]
        for ax in (ax_full, ax_zoom):
            ax.fill_between(
                c["episode"], c["ci95_low"], c["ci95_high"], color=color, alpha=0.18, linewidth=0
            )
            ax.plot(
                c["episode"],
                c["mean"],
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
                label=f"{CONTROLLER_LABEL[agent]} (n = {int(c['n'].iloc[-1])} seeds)",
            )
        last = c.iloc[-1]
        ax_zoom.annotate(
            CONTROLLER_LABEL[agent],
            (last["episode"], last["mean"]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
            fontsize=9,
        )
        best = c.loc[c["mean"].idxmin()]
        summary.append(
            f"{CONTROLLER_LABEL[agent]}: best mean {best['mean']:.1f} s at episode "
            f"{int(best['episode'])}, last {last['mean']:.1f} s"
        )
    for rule, v in rules.items():
        for ax in (ax_full, ax_zoom):
            ax.axhline(
                v,
                color=CONTROLLER_COLOR[rule],
                linestyle="--",
                linewidth=1.5,
                label=f"{CONTROLLER_LABEL[rule]} {v:.1f} s" if ax is ax_full else None,
            )
        summary.append(f"{CONTROLLER_LABEL[rule]}: {v:.1f} s (deterministic)")

    # zoom: episodes >= 30, y range from the learned curves and Max-Pressure only
    z0 = 30
    lo = min(c.loc[c["episode"] >= z0, "ci95_low"].min() for c in curves.values())
    hi = max(c.loc[c["episode"] >= z0, "ci95_high"].max() for c in curves.values())
    lo, hi = min(lo, rules["max_pressure"]), max(hi, rules["max_pressure"])
    pad = 0.08 * (hi - lo)
    ax_zoom.set_xlim(z0 - 2, episodes_max * 1.18)  # room for the direct labels
    ax_zoom.set_ylim(lo - pad, hi + pad)
    ax_zoom.annotate(
        CONTROLLER_LABEL["max_pressure"],
        (episodes_max, rules["max_pressure"]),
        xytext=(6, 3),
        textcoords="offset points",
        va="bottom",
        color=INK_SECONDARY,
        fontsize=9,
    )
    ax_full.set_xlim(0, episodes_max + 3)
    ax_full.legend(loc="upper right", fontsize=8, frameon=False)
    ax_full.set_title("All validation episodes", fontsize=10)
    ax_zoom.set_title(f"Zoom: episodes ≥ {z0} (y axis cut)", fontsize=10)
    for ax in (ax_full, ax_zoom):
        ax.set_xlabel("Training episode (1 episode = 3600 s of Jinan train flow)")
    ax_full.set_ylabel("Validation avg travel time (s)")
    fig.suptitle("RL baselines: greedy validation travel time during training", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, OUT)
    caption = (
        "# Figure — RL-baseline learning curves (dev, WSL)\n\n"
        f"![learning curves]({OUT.name})\n\n"
        "**Caption.** Average travel time (s, lower is better) of the greedy policy on the "
        "Jinan validation flow (`real_2000`), evaluated every 5 training episodes. Line = mean "
        f"over training seeds {args.seeds}; band = 95 % CI (Student t). Dashed lines: "
        "Fixed-Time and Max-Pressure on the same flow (deterministic). Left: all evaluations; "
        "right: the same curves from episode 30 with the y axis cut to the learned agents' "
        "range. Generated by "
        "`scripts/plot_training.py` from `results/train/<agent>_"
        + args.tag
        + "_s<seed>/val_log.csv`.\n\n"
        "**Values.** " + "; ".join(summary) + ".\n\n"
        "**Interpretation.** See worklog/016 (Result) for the reading of this figure.\n"
    )
    OUT.with_suffix(".md").write_text(caption)
    print(caption)
    return 0


if __name__ == "__main__":
    sys.exit(main())
