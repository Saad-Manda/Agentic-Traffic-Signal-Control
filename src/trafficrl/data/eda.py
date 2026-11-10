"""Aggregations for the exploratory data analysis (ROADMAP 3.3).

Demand-side descriptions of a flow file (who enters where, which movements are used) and
loaders for the simulator traces written by ``scripts/collect_traces.py``.

**Train split only.** These functions describe demand distributions, so they are called
only on the training flow file (``database.md`` rule 9). The audit script and the EDA
notebook enforce this; nothing here is used on validation, test or held-out flows.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from trafficrl.data.roadnet import DIRECTIONS, Roadnet
from trafficrl.envs.features import FeatureBuilder


def entry_demand(flows: Sequence[dict]) -> pd.Series:
    """Vehicles per entry road (first road of the route), sorted descending."""
    s = pd.Series(Counter(v["route"][0] for v in flows), name="vehicles")
    return s.sort_values(ascending=False)


def movement_demand(flows: Sequence[dict], roadnet: Roadnet) -> pd.DataFrame:
    """Vehicles per movement at every signalised intersection.

    One row per (intersection, movement) including movements nobody uses (``vehicles`` = 0).
    Columns: intersection, approach (direction of travel of the incoming road), kind,
    start_road, end_road, vehicles.
    """
    used: Counter[tuple[str, str]] = Counter()
    for v in flows:
        route = v["route"]
        used.update(zip(route, route[1:], strict=False))
    rows = []
    for inter in roadnet.intersections.values():
        for m in inter.movements:
            rows.append(
                {
                    "intersection": inter.id,
                    "approach": DIRECTIONS[inter.in_roads.index(m.start_road)],
                    "kind": m.kind,
                    "start_road": m.start_road,
                    "end_road": m.end_road,
                    "vehicles": used[(m.start_road, m.end_road)],
                }
            )
    return pd.DataFrame(rows)


def turning_ratios(movements: pd.DataFrame) -> pd.DataFrame:
    """Share of left / through / right per approach direction (rows sum to 1)."""
    t = movements.pivot_table(
        index="approach", columns="kind", values="vehicles", aggfunc="sum", fill_value=0
    )
    t = t.reindex(index=list(DIRECTIONS), columns=["turn_left", "go_straight", "turn_right"])
    return t.div(t.sum(axis=1), axis=0)


def intersection_load(movements: pd.DataFrame) -> pd.Series:
    """Vehicles passing through each signalised intersection over the episode."""
    return movements.groupby("intersection")["vehicles"].sum()


def route_lengths(flows: Sequence[dict]) -> pd.Series:
    return pd.Series([len(v["route"]) for v in flows], name="roads_per_route")


def signalised_crossings(flows: Sequence[dict], roadnet: Roadnet) -> pd.Series:
    """Number of signalised intersections each vehicle crosses."""
    return pd.Series(
        [sum(roadnet.road_end[r] in roadnet.intersections for r in v["route"][:-1]) for v in flows],
        name="signalised_intersections",
    )


# --- simulator traces (results/eda/<run>/) -------------------------------------------------


def load_run(run_dir: str | Path) -> dict:
    """Load one trace directory: ``summary`` (dict), ``steps`` and ``vehicles`` (DataFrames)."""
    run_dir = Path(run_dir)
    return {
        "summary": json.loads((run_dir / "summary.json").read_text()),
        "steps": pd.read_csv(run_dir / "steps.csv.gz"),
        "vehicles": pd.read_csv(run_dir / "vehicles.csv.gz"),
    }


def load_summaries(root: str | Path) -> pd.DataFrame:
    """One row per trace run under ``root`` (from each ``summary.json``)."""
    rows = [json.loads(p.read_text()) for p in sorted(Path(root).glob("*/summary.json"))]
    return pd.DataFrame(rows)


def feature_arrays(steps: pd.DataFrame, features: FeatureBuilder) -> dict[str, np.ndarray]:
    """Per-lane and per-movement values of a trace, raw and capacity-normalised.

    Returns flat arrays over all recorded (time, intersection) rows:
    ``count`` / ``queue`` (vehicles per incoming lane), ``count_norm`` / ``queue_norm``
    (divided by lane capacity, **not clipped**, so values above 1 show where clipping
    would act) and ``pressure`` (normalised movement pressure, as in the observation).
    """
    out: dict[str, list[np.ndarray]] = {
        k: [] for k in ("count", "queue", "count_norm", "queue_norm", "pressure")
    }
    cap = features.capacity
    for iid, rows in steps.groupby("intersection", sort=False):
        inter = features.roadnet.intersections[iid]
        count = rows.filter(regex=r"^count_").to_numpy(dtype=float)
        queue = rows.filter(regex=r"^queue_").to_numpy(dtype=float)
        outflow = rows.filter(regex=r"^out_").to_numpy(dtype=float)
        cap_in = np.array([cap[ln] for ln in inter.in_lanes])
        cap_out = np.array([cap[ln] for ln in inter.out_lanes])
        x_in = np.minimum(count / cap_in, 1.0)
        x_out = np.minimum(outflow / cap_out, 1.0)
        pressure = np.stack(
            [
                x_in[:, [inter.in_lanes.index(ln) for ln in m.in_lanes]].mean(axis=1)
                - x_out[:, [inter.out_lanes.index(ln) for ln in m.out_lanes]].mean(axis=1)
                for m in inter.movements
            ],
            axis=1,
        )
        out["count"].append(count.ravel())
        out["queue"].append(queue.ravel())
        out["count_norm"].append((count / cap_in).ravel())
        out["queue_norm"].append((queue / cap_in).ravel())
        out["pressure"].append(pressure.ravel())
    return {k: np.concatenate(v) for k, v in out.items()}


def network_series(steps: pd.DataFrame) -> pd.DataFrame:
    """Network totals per recorded time: ``queue`` (waiting vehicles on incoming lanes),
    ``vehicles`` (all vehicles on incoming lanes) and ``abs_pressure`` (Σ_i |P_i|)."""
    df = pd.DataFrame(
        {
            "t": steps["t"],
            "queue": steps.filter(regex=r"^queue_").sum(axis=1),
            "vehicles": steps.filter(regex=r"^count_").sum(axis=1),
            "abs_pressure": steps["pressure"].abs(),
        }
    )
    return df.groupby("t").sum()
