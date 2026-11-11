"""Shared matplotlib style for every project figure (audit, EDA, results).

One place for colours and axis styling so figures in the report and slides look like one
set. Colours are the first slots of a categorical palette checked for colour-blind
separation; each controller keeps its colour in every figure (colour follows the entity).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
# sequential ramp (one hue, light -> dark) for magnitudes
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]

# Results figures use slots 1-4 in this fixed order (validated: adjacent-pair CVD and
# normal-vision separation pass; aqua / yellow are < 3:1 on the surface, so these series are
# always direct-labelled). "random" appears only in EDA figures, never next to DQN.
CONTROLLER_COLOR = {
    "fixed_time": BLUE,
    "max_pressure": ORANGE,
    "dqn": AQUA,
    "idqn": YELLOW,
    "random": AQUA,
}
CONTROLLER_LABEL = {
    "fixed_time": "Fixed-Time",
    "max_pressure": "Max-Pressure",
    "dqn": "DQN (shared)",
    "idqn": "IDQN",
    "random": "Random phase",
}


def apply_style() -> None:
    """Set the project-wide matplotlib defaults (call once per script / notebook)."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 1.8,
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE, AQUA, YELLOW]),
        }
    )


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    """Save a figure as PNG (parent directories created) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path
