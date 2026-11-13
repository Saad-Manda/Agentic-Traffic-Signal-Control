"""Phase 8 error analysis (ROADMAP 8.4; plan phase8-analysis).

Pure functions on the per-episode logs that the Phase 7 pass saved under ``results/final``:

* ``vehicles_s<seed>.csv`` — ``vehicle, enter_t, leave_t, wait_s``;
* ``trace_s<seed>.csv``    — one row per env step × intersection: ``t, iid, phase, in_yellow,
  ready, q_0…q_11`` (true waiting vehicles per incoming lane), ``n_0…n_11`` (true counts);
* ``decisions_s<seed>.csv`` — UG-MARL variants: σ, τ, gate, phases, attention weights.

Nothing here runs a simulation, and nothing is fitted or selected. Travel time exists per
vehicle only, so per-intersection results are queues and waiting-vehicle-seconds; travel time
is broken down by entry-time window and by route group.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from trafficrl.eval.report import finished_run
from trafficrl.utils.stats import summarise

HORIZON_S = 3600
APPROACHES = ("E", "N", "W", "S")  # canonical order of the incoming roads (FeatureBuilder)
LANES_PER_APPROACH = 3
_VEHICLE_ID = re.compile(r"^flow_(\d+)_(\d+)$")


# --- reading the saved logs ----------------------------------------------------------------


def job_run(study_dir: str | Path, model: str, condition: str) -> Path:
    """Run directory of one finished ``<model>__<condition>`` job of a study."""
    run = finished_run(Path(study_dir) / "eval" / f"{model}__{condition}")
    if run is None:
        raise FileNotFoundError(f"no finished run for {model}__{condition} in {study_dir}")
    return run


def read_log(run: str | Path, kind: str, seed: int) -> pd.DataFrame:
    """One per-seed log (``kind`` = ``trace``, ``vehicles`` or ``decisions``)."""
    f = Path(run) / f"{kind}_s{seed}.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} does not exist (was the run logged with this option?)")
    return pd.read_csv(f)


def per_seed(
    study_dir: str | Path,
    model: str,
    condition: str,
    kind: str,
    seeds: Sequence[int],
    fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    """Apply ``fn`` to each seed's log and stack the results with ``model, condition, seed``."""
    run = job_run(study_dir, model, condition)
    frames = []
    for s in seeds:
        out = fn(read_log(run, kind, s)).copy()
        out.insert(0, "seed", s)
        out.insert(0, "condition", condition)
        out.insert(0, "model", model)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


# --- vehicles: travel time by entry window ---------------------------------------------------


def vehicle_times(vehicles: pd.DataFrame, horizon: int = HORIZON_S) -> pd.DataFrame:
    """Adds ``travel_s`` and ``completed`` from the per-vehicle log alone (time inside the
    network; a vehicle still inside at the end is counted up to the horizon).

    This is **not** the simulator's average travel time when the network is congested: a
    vehicle that cannot enter waits outside the network, and that time (and vehicles that
    never get in) is missing from the log. Use :func:`trips` with the flow file for the
    protocol's definition; this function is the fallback where no route join exists (surge).
    ``leave_t`` is the first second a vehicle is no longer seen, hence the −1."""
    out = vehicles.copy()
    out["completed"] = out["leave_t"].notna()
    out["travel_s"] = (out["leave_t"] - 1).fillna(horizon) - out["enter_t"]
    return out


def window_table(trips_: pd.DataFrame, width: int = 600, horizon: int = HORIZON_S) -> pd.DataFrame:
    """Per window of the scheduled start time (``start_time``; ``enter_t`` for frames from
    :func:`vehicle_times`): vehicles, mean travel time, mean wait, share completed."""
    v = trips_.copy()
    t0 = v["start_time"] if "start_time" in v else v["enter_t"]
    v["window"] = (np.minimum(t0, horizon - 1) // width).astype(int)
    out = (
        v.groupby("window")
        .agg(
            vehicles=("travel_s", "size"),
            travel_s=("travel_s", "mean"),
            wait_s=("wait_s", "mean"),
            completed=("completed", "mean"),
        )
        .reset_index()
    )
    out.insert(1, "start_min", out["window"] * width // 60)
    return out


# --- traces: queues per intersection / approach, phase stability ----------------------------


def _lane_columns(trace: pd.DataFrame, prefix: str = "q_") -> list[str]:
    cols = [c for c in trace.columns if c.startswith(prefix) and c[len(prefix) :].isdigit()]
    return sorted(cols, key=lambda c: int(c[len(prefix) :]))


def step_lengths(trace: pd.DataFrame, horizon: int = HORIZON_S) -> pd.Series:
    """Seconds each row's state is held: time to the next env step (the last one runs to the
    horizon). Indexed like ``trace``."""
    times = np.sort(trace["t"].unique())
    nxt = dict(zip(times, np.append(times[1:], horizon), strict=True))
    return (trace["t"].map(nxt) - trace["t"]).clip(lower=0)


def flip_flops(phases: Sequence[int]) -> int:
    """Number of A → B → A patterns where B was held for exactly one decision."""
    p = np.asarray(phases)
    if len(p) < 3:
        return 0
    starts = np.flatnonzero(np.diff(p) != 0) + 1
    bounds = np.concatenate([[0], starts, [len(p)]])
    values = p[bounds[:-1]]
    lengths = np.diff(bounds)
    inner = (lengths[1:-1] == 1) & (values[:-2] == values[2:])
    return int(inner.sum())


def intersection_table(trace: pd.DataFrame, horizon: int = HORIZON_S) -> pd.DataFrame:
    """Per intersection: mean waiting vehicles (over env steps), waiting-vehicle-seconds
    (Σ waiting vehicles × seconds held), decisions, phase changes and flip-flops."""
    q = _lane_columns(trace)
    t = trace[["t", "iid", "phase", "ready"]].copy()
    t["waiting"] = trace[q].sum(axis=1)
    t["wait_veh_s"] = t["waiting"] * step_lengths(trace, horizon)
    rows = []
    for iid, g in t.groupby("iid", sort=True):
        seq = g.loc[g["ready"].astype(bool), "phase"].to_numpy()
        rows.append(
            {
                "iid": iid,
                "mean_waiting_veh": float(g["waiting"].mean()),
                "wait_veh_s": float(g["wait_veh_s"].sum()),
                "decisions": int(len(seq)),
                "phase_changes": int((np.diff(seq) != 0).sum()) if len(seq) > 1 else 0,
                "flip_flops": flip_flops(seq),
            }
        )
    return pd.DataFrame(rows)


def network_queue(trace: pd.DataFrame) -> float:
    """Mean waiting vehicles per incoming lane over env steps — the protocol's average queue
    (``avg_queue_veh`` in ``metrics.csv``); used to check the traces against the metrics."""
    q = _lane_columns(trace)
    per_step = trace.groupby("t")[q].sum().sum(axis=1) / (trace["iid"].nunique() * len(q))
    return float(per_step.mean())


def approach_table(trace: pd.DataFrame, horizon: int = HORIZON_S) -> pd.DataFrame:
    """Waiting-vehicle-seconds and the longest stretch with a standing queue, per intersection
    and approach (E, N, W, S = direction of the incoming road; three lanes each)."""
    q = _lane_columns(trace)
    dt = step_lengths(trace, horizon)
    rows = []
    for k, name in enumerate(APPROACHES):
        cols = q[k * LANES_PER_APPROACH : (k + 1) * LANES_PER_APPROACH]
        if not cols:
            continue
        waiting = trace[cols].sum(axis=1)
        d = pd.DataFrame({"iid": trace["iid"], "waiting": waiting, "dt": dt})
        for iid, g in d.groupby("iid", sort=True):
            busy = (g["waiting"] > 0).to_numpy()
            run = best = 0.0
            for b, step in zip(busy, g["dt"].to_numpy(), strict=True):
                run = run + step if b else 0.0
                best = max(best, run)
            rows.append(
                {
                    "iid": iid,
                    "approach": name,
                    "wait_veh_s": float((g["waiting"] * g["dt"]).sum()),
                    "longest_queue_s": float(best),
                }
            )
    return pd.DataFrame(rows)


def unit_table(trace: pd.DataFrame, width: int = 300, horizon: int = HORIZON_S) -> pd.DataFrame:
    """Waiting-vehicle-seconds per (intersection, time window) — the unit of the case rule."""
    q = _lane_columns(trace)
    d = pd.DataFrame(
        {
            "iid": trace["iid"],
            "window": (np.minimum(trace["t"], horizon - 1) // width).astype(int),
            "wait_veh_s": trace[q].sum(axis=1) * step_lengths(trace, horizon),
        }
    )
    return d.groupby(["iid", "window"], as_index=False)["wait_veh_s"].sum()


# --- routes ----------------------------------------------------------------------------------


class RouteJoinError(ValueError):
    """Vehicle IDs could not be matched to flow entries consistently."""


def load_flow(path: str | Path) -> list[dict]:
    """Flow entries of one flow file (read-only; Decision Log 13 Nov, plan A3)."""
    return json.loads(Path(path).read_text())


def route_groups(flow: Sequence[Mapping]) -> pd.DataFrame:
    """Per flow entry: number of intersections passed, number of turns, start time.

    Road ids are ``road_<x>_<y>_<d>`` with ``d`` the direction of travel; a turn is a change of
    ``d`` between consecutive roads of the route."""
    rows = []
    for i, f in enumerate(flow):
        route = list(f["route"])
        heading = [r.rsplit("_", 1)[-1] for r in route]
        turns = sum(a != b for a, b in zip(heading, heading[1:], strict=False))
        rows.append(
            {
                "flow": i,
                "intersections": len(route) - 1,
                "turns": turns,
                "start_time": float(f["startTime"]),
            }
        )
    out = pd.DataFrame(rows)
    out["length_group"] = np.where(
        out["intersections"] >= 4, "4+", out["intersections"].astype(str)
    )
    out["turn_group"] = np.where(out["turns"] == 0, "straight", "turning")
    return out


def trips(
    vehicles: pd.DataFrame,
    flow: Sequence[Mapping],
    horizon: int = HORIZON_S,
    max_entry_delay_s: float = 5.0,
) -> pd.DataFrame:
    """One row per **flow entry** (= scheduled vehicle): route groups, ``entered``,
    ``completed``, ``entry_delay_s`` (seconds blocked outside the network), ``wait_s`` (inside
    the network only) and ``travel_s`` = (leave or horizon) − scheduled start — the
    simulator's definition, which also counts vehicles that never got in.

    Vehicle ``flow_<i>_0`` is flow entry i (every entry of these files is one vehicle). The
    join is checked: every id must parse and point at an existing entry, each entry holds one
    vehicle, no vehicle is seen before its start time, and the median delay between start and
    first sighting is small — otherwise :class:`RouteJoinError`."""
    m = vehicles["vehicle"].astype(str).str.extract(_VEHICLE_ID)
    if m[0].isna().any():
        raise RouteJoinError("vehicle ids are not all of the form flow_<i>_<j>")
    idx = m[0].astype(int).to_numpy()
    if (m[1].astype(int) != 0).any() or len(set(idx)) != len(idx):
        raise RouteJoinError("a flow entry holds more than one vehicle")
    groups = route_groups(flow)
    if idx.max() >= len(groups):
        raise RouteJoinError("a vehicle id points past the end of the flow file")
    seen = vehicles.assign(flow=idx)
    out = groups.merge(seen, on="flow", how="left")
    out["entered"] = out["enter_t"].notna()
    out["completed"] = out["leave_t"].notna()
    out["wait_s"] = out["wait_s"].fillna(0)
    out["entry_delay_s"] = (out["enter_t"] - 1).fillna(horizon) - out["start_time"]
    out["travel_s"] = (out["leave_t"] - 1).fillna(horizon) - out["start_time"]
    delay = out.loc[out["entered"], "entry_delay_s"]
    if (delay < 0).any():
        raise RouteJoinError("a vehicle was seen before its flow entry's start time")
    if float(delay.median()) > max_entry_delay_s:
        raise RouteJoinError(f"median entry delay {delay.median():.1f} s: ids do not match")
    return out


def route_table(trips_: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per route group (``by`` = ``length_group`` or ``turn_group``): vehicles, mean travel
    time, mean and 95th-percentile wait. ``trips_`` comes from :func:`trips`."""
    return (
        trips_.groupby(by)
        .agg(
            vehicles=("travel_s", "size"),
            travel_s=("travel_s", "mean"),
            wait_s=("wait_s", "mean"),
            wait_p95_s=("wait_s", lambda x: float(np.percentile(x, 95))),
        )
        .reset_index()
    )


def entry_summary(trips_: pd.DataFrame) -> pd.DataFrame:
    """Vehicles that never entered, and time blocked outside the network (one row)."""
    d = trips_["entry_delay_s"]
    return pd.DataFrame(
        [
            {
                "scheduled": len(trips_),
                "never_entered": int((~trips_["entered"]).sum()),
                "entry_delay_mean_s": float(d.mean()),
                "entry_delay_max_s": float(d.max()),
                "blocked_over_60s": int((d > 60).sum()),
                "travel_s": float(trips_["travel_s"].mean()),
            }
        ]
    )


# --- statistics over seeds -------------------------------------------------------------------


def seed_stats(df: pd.DataFrame, keys: Sequence[str], value: str) -> pd.DataFrame:
    """mean, std, 95 % CI and n of ``value`` over seeds, per ``keys``."""
    rows = []
    for k, g in df.groupby(list(keys), sort=False):
        k = k if isinstance(k, tuple) else (k,)
        s = summarise(g[value].tolist())
        rows.append({**dict(zip(keys, k, strict=True)), **s})
    return pd.DataFrame(rows)


def seed_difference(
    a: pd.DataFrame, b: pd.DataFrame, keys: Sequence[str], value: str
) -> pd.DataFrame:
    """``a − b`` per ``keys``: differences are taken seed by seed (same seed number) and
    summarised (mean, std, 95 % CI). ``clear`` = the CI excludes 0.

    For the gate vs −gate this is a true pairing (same checkpoints). For two different models
    the seed numbers only index independent runs; the interval is still a valid t-interval of
    the mean difference."""
    on = [*keys, "seed"]
    m = a[[*on, value]].merge(b[[*on, value]], on=on, suffixes=("_a", "_b"))
    m["diff"] = m[f"{value}_a"] - m[f"{value}_b"]
    out = seed_stats(m, keys, "diff")
    means = m.groupby(list(keys), sort=False)[[f"{value}_a", f"{value}_b"]].mean().reset_index()
    out = out.merge(means, on=list(keys))
    out["clear"] = (out["ci95_low"] > 0) | (out["ci95_high"] < 0)
    return out


# --- gate: what follows a firing ------------------------------------------------------------


def after_decision_queue(
    decisions: pd.DataFrame, trace: pd.DataFrame, horizon_s: int = 60
) -> pd.DataFrame:
    """For every logged decision: waiting vehicles at that intersection at the decision and
    their mean over the following ``horizon_s`` seconds (from the trace)."""
    q = _lane_columns(trace)
    tr = trace[["t", "iid"]].copy()
    tr["waiting"] = trace[q].sum(axis=1)
    out = []
    for iid, d in decisions.groupby("iid", sort=True):
        g = tr[tr["iid"] == iid].sort_values("t")
        times, waiting = g["t"].to_numpy(), g["waiting"].to_numpy(float)
        csum = np.concatenate([[0.0], np.cumsum(waiting)])
        t0 = d["t"].to_numpy()
        lo = np.searchsorted(times, t0, side="left")
        hi = np.searchsorted(times, t0 + horizon_s, side="right")
        n = np.maximum(hi - lo, 1)
        now = waiting[np.minimum(lo, len(waiting) - 1)]
        res = d.copy()
        res["waiting_now"] = now
        res["waiting_after"] = (csum[hi] - csum[lo]) / n
        out.append(res)
    return pd.concat(out, ignore_index=True)


def gate_effect(decisions: pd.DataFrame, window: int = 600) -> pd.DataFrame:
    """Change in waiting vehicles after a decision (``waiting_after − waiting_now``) for
    decisions where the gate changed the action vs decisions where it did not fire, matched
    on intersection and time window (only cells holding both kinds are used). An association:
    the gate fires in different states than it stays quiet in."""
    d = decisions.copy()
    d["changed"] = d["fallback"].astype(bool) & (d["greedy_phase"] != d["mp_phase"])
    d["quiet"] = ~d["fallback"].astype(bool)
    d["delta"] = d["waiting_after"] - d["waiting_now"]
    d["window"] = d["t"] // window
    rows = []
    for _, g in d.groupby(["iid", "window"]):
        a, b = g[g["changed"]], g[g["quiet"]]
        if len(a) and len(b):
            rows.append(
                {
                    "n_changed": len(a),
                    "n_quiet": len(b),
                    "delta_changed": a["delta"].mean(),
                    "delta_quiet": b["delta"].mean(),
                    "now_changed": a["waiting_now"].mean(),
                    "now_quiet": b["waiting_now"].mean(),
                }
            )
    cells = pd.DataFrame(rows)
    if cells.empty:
        return pd.DataFrame([{"cells": 0}])
    w = cells["n_changed"]
    return pd.DataFrame(
        [
            {
                "cells": len(cells),
                "decisions_changed": int(cells["n_changed"].sum()),
                "delta_changed": float(np.average(cells["delta_changed"], weights=w)),
                "delta_quiet": float(np.average(cells["delta_quiet"], weights=w)),
                "waiting_now_changed": float(np.average(cells["now_changed"], weights=w)),
                "waiting_now_quiet": float(np.average(cells["now_quiet"], weights=w)),
            }
        ]
    )


# --- representative cases (rule fixed in the plan, A7) --------------------------------------


def median_seed(values: Mapping[int, float]) -> int:
    """Seed whose value is the median of the given per-seed values (odd count)."""
    order = sorted(values, key=lambda s: (values[s], s))
    return order[len(order) // 2]


def pick_cases(
    units_model: pd.DataFrame, units_baseline: pd.DataFrame, decisions: pd.DataFrame, width: int
) -> dict[str, dict]:
    """The three cases, all on one seed's clean episode.

    * ``win``  — (intersection, window) with the most negative waiting-vehicle-seconds
      difference model − baseline;
    * ``loss`` — the most positive difference;
    * ``gate`` — the unit with the most decisions where the gate fired and changed the action.
    Ties go to the earlier window, then the intersection id."""
    m = units_model.merge(units_baseline, on=["iid", "window"], suffixes=("_model", "_base"))
    m["diff"] = m["wait_veh_s_model"] - m["wait_veh_s_base"]
    m = m.sort_values(["window", "iid"]).reset_index(drop=True)
    win, loss = m.loc[m["diff"].idxmin()], m.loc[m["diff"].idxmax()]
    d = decisions.copy()
    d["window"] = d["t"] // width
    d["changed"] = d["fallback"].astype(bool) & (d["greedy_phase"] != d["mp_phase"])
    g = d.groupby(["window", "iid"], as_index=False)["changed"].sum()
    g = g.sort_values(["window", "iid"]).reset_index(drop=True)
    gate = g.loc[g["changed"].idxmax()]

    def unit(row, **extra) -> dict:
        return {"iid": str(row["iid"]), "window": int(row["window"]), **extra}

    return {
        "win": unit(win, diff_wait_veh_s=float(win["diff"])),
        "loss": unit(loss, diff_wait_veh_s=float(loss["diff"])),
        "gate": unit(gate, gate_changed_decisions=int(gate["changed"])),
    }


def case_series(trace: pd.DataFrame, iid: str, window: int, width: int) -> pd.DataFrame:
    """Rows of one intersection inside one window, with waiting vehicles per approach."""
    q = _lane_columns(trace)
    g = trace[(trace["iid"] == iid) & (trace["t"] // width == window)].copy()
    for k, name in enumerate(APPROACHES):
        cols = q[k * LANES_PER_APPROACH : (k + 1) * LANES_PER_APPROACH]
        g[f"wait_{name}"] = g[cols].sum(axis=1)
    g["waiting"] = g[q].sum(axis=1)
    return g[["t", "phase", "in_yellow", "ready", "waiting", *(f"wait_{a}" for a in APPROACHES)]]


# --- study context and the 8.4 breakdowns ---------------------------------------------------


class Study:
    """Per-seed metrics and lazily read logs of one finished model × condition study."""

    def __init__(self, spec: Mapping, study_dir: str | Path) -> None:
        from trafficrl.eval.report import load_study

        self.spec, self.dir = dict(spec), Path(study_dir)
        self.metrics = load_study(self.spec, self.dir)
        self.seeds = [int(s) for s in self.spec["seeds"]]
        self.models = list(self.spec["models"])
        self.conditions = list(dict.fromkeys(self.metrics["condition"]))
        self._logs: dict[tuple, pd.DataFrame] = {}

    def run(self, model: str, condition: str) -> Path:
        return job_run(self.dir, model, condition)

    def log(self, model: str, condition: str, kind: str, seed: int) -> pd.DataFrame:
        key = (model, condition, kind, seed)
        if key not in self._logs:
            self._logs[key] = read_log(self.run(model, condition), kind, seed)
        return self._logs[key]

    def values(self, model: str, condition: str, metric: str) -> list[float]:
        m = self.metrics
        sel = m[(m["model"] == model) & (m["condition"] == condition)].sort_values("seed")
        return sel[metric].astype(float).tolist()

    def stats(self, model: str, condition: str, metric: str) -> dict:
        return summarise(self.values(model, condition, metric))

    def each_seed(self, model: str, condition: str, kind: str, fn) -> pd.DataFrame:
        frames = []
        for s in self.seeds:
            out = fn(self.log(model, condition, kind, s)).copy()
            out.insert(0, "seed", s)
            out.insert(0, "condition", condition)
            out.insert(0, "model", model)
            frames.append(out)
        return pd.concat(frames, ignore_index=True)


def breakdowns(
    study: Study,
    flow: Sequence[Mapping],
    models: Sequence[str],
    conditions: Sequence[str],
    window_s: int = 600,
) -> dict[str, pd.DataFrame]:
    """Per-seed breakdown frames for ``models`` × ``conditions`` of one study (same flow
    file): ``intersections``, ``approaches``, ``windows``, ``length`` / ``turn`` (route
    groups), ``entry`` (vehicles blocked outside the network) and ``check`` (travel time and
    queue recomputed from the logs next to the values in ``metrics.csv``)."""
    out: dict[str, list] = {
        k: [] for k in ("intersections", "approaches", "windows", "length", "turn", "entry")
    }
    check = []
    for m in models:
        for c in conditions:
            out["intersections"].append(study.each_seed(m, c, "trace", intersection_table))
            out["approaches"].append(study.each_seed(m, c, "trace", approach_table))

            def tr(v):
                return trips(v, flow)

            out["windows"].append(
                study.each_seed(m, c, "vehicles", lambda v: window_table(tr(v), window_s))
            )
            out["length"].append(
                study.each_seed(m, c, "vehicles", lambda v: route_table(tr(v), "length_group"))
            )
            out["turn"].append(
                study.each_seed(m, c, "vehicles", lambda v: route_table(tr(v), "turn_group"))
            )
            out["entry"].append(study.each_seed(m, c, "vehicles", lambda v: entry_summary(tr(v))))
            for s, att, q in zip(
                study.seeds,
                study.values(m, c, "avg_travel_time_s"),
                study.values(m, c, "avg_queue_veh"),
                strict=True,
            ):
                check.append(
                    {
                        "model": m,
                        "condition": c,
                        "seed": s,
                        "travel_metric": att,
                        "travel_logs": float(tr(study.log(m, c, "vehicles", s))["travel_s"].mean()),
                        "queue_metric": q,
                        "queue_logs": network_queue(study.log(m, c, "trace", s)),
                    }
                )
    frames = {k: pd.concat(v, ignore_index=True) for k, v in out.items()}
    frames["check"] = pd.DataFrame(check)
    return frames


def comparison(
    frames: Mapping[str, pd.DataFrame],
    table: str,
    keys: Sequence[str],
    value: str,
    model: str,
    baseline: str,
    condition: str,
) -> pd.DataFrame:
    """``model − baseline`` for one breakdown table and condition (see :func:`seed_difference`)."""
    f = frames[table]
    f = f[f["condition"] == condition]
    d = seed_difference(f[f["model"] == model], f[f["model"] == baseline], keys, value)
    d.insert(0, "condition", condition)
    d.insert(0, "baseline", baseline)
    d.insert(0, "model", model)
    return d


def stability_table(study: Study, models: Sequence[str], conditions: Sequence[str]) -> pd.DataFrame:
    """Per model × condition, mean over seeds: decisions, phase changes and flip-flops per
    episode, and their shares of all decisions (from the traces)."""
    rows = []
    for m in models:
        for c in conditions:
            t = study.each_seed(m, c, "trace", intersection_table)
            per = t.groupby("seed")[["decisions", "phase_changes", "flip_flops"]].sum()
            rows.append(
                {
                    "model": m,
                    "condition": c,
                    "decisions": per["decisions"].mean(),
                    "phase_changes": per["phase_changes"].mean(),
                    "flip_flops": per["flip_flops"].mean(),
                    "change_share": (per["phase_changes"] / per["decisions"]).mean(),
                    "flip_flop_share": (per["flip_flops"] / per["decisions"]).mean(),
                    "flip_flop_share_std": (per["flip_flops"] / per["decisions"]).std(ddof=1),
                }
            )
    return pd.DataFrame(rows)


def seed_spread(study: Study, models: Sequence[str], condition: str, metric: str) -> pd.DataFrame:
    """Per model: every seed's value plus min, max and range over seeds."""
    rows = []
    for m in models:
        v = study.values(m, condition, metric)
        rows.append(
            {
                "model": m,
                **{f"s{s}": x for s, x in zip(study.seeds, v, strict=True)},
                "min": min(v),
                "max": max(v),
                "range": max(v) - min(v),
            }
        )
    return pd.DataFrame(rows)
