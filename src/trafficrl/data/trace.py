"""Train-split simulator traces for the EDA (ROADMAP 3.3).

A flow file holds only demand. Queues, pressure and travel times exist only once a
controller operates the signals, so the EDA runs three simple rule controllers on the
**training** flow and records what the simulator shows:

* ``fixed_time``   — cycle through the green phases (the "target" stand-in of ROADMAP 3.3)
* ``max_pressure`` — serve the phase with the largest pressure (well-controlled regime)
* ``random``       — uniform-random green phase (the regime an exploring agent starts in)

These rules exist for data description only. The baselines that are *evaluated* live in
``trafficrl.agents.baselines`` (Phase 4) behind the shared ``Agent`` interface.

Leakage guard: :func:`train_flow` is the only way this module obtains a flow file, and it
refuses every dataset / split except the primary dataset's ``train`` split.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from trafficrl.data.roadnet import YELLOW_PHASE, Roadnet
from trafficrl.data.splits import SplitError, flow_files, load_splits
from trafficrl.envs.features import FeatureBuilder
from trafficrl.utils.seed import make_rng, seed_cityflow

CONTROLLERS = ("fixed_time", "max_pressure", "random")
# chooser(intersection id, current green phase or None, lane vehicle counts) -> green phase
Chooser = Callable[[str, int | None, Mapping[str, int]], int]


def train_flow(cfg: Mapping) -> tuple[Path, Path]:
    """``(roadnet, flow file)`` of the training split named in ``cfg['data']``.

    Raises :class:`SplitError` unless the config asks for the ``train`` split of the
    primary dataset — EDA never touches validation, test or held-out flows.
    """
    data = cfg.get("data", {})
    dataset, split = data.get("dataset"), data.get("split")
    splits = load_splits()
    if split != "train" or dataset not in splits or splits[dataset]["role"] != "primary":
        raise SplitError(
            f"EDA traces use the primary dataset's train split only, got {dataset!r}/{split!r}"
        )
    roadnet, flows = flow_files(dataset, "train", splits=splits)
    if len(flows) != 1:
        raise SplitError(f"expected one training flow file, got {len(flows)}")
    return roadnet, flows[0]


def make_engine(roadnet_path: Path, flow_path: Path, seed: int, threads: int, workdir: Path):
    """Create a seeded ``cityflow.Engine`` for one roadnet / flow pair (no replay output)."""
    import cityflow  # imported here so the rest of the package works without the simulator

    roadnet_path, flow_path = Path(roadnet_path).resolve(), Path(flow_path).resolve()
    if roadnet_path.parent != flow_path.parent:
        raise ValueError("roadnet and flow file must be in the same directory")
    cfg = {
        "interval": 1.0,
        "seed": seed,
        "dir": str(roadnet_path.parent) + "/",
        "roadnetFile": roadnet_path.name,
        "flowFile": flow_path.name,
        "rlTrafficLight": True,
        "laneChange": False,
        "saveReplay": False,
        "roadnetLogFile": "",
        "replayLogFile": "",
    }
    cfg_path = Path(workdir) / "cityflow_config.json"
    cfg_path.write_text(json.dumps(cfg))
    engine = cityflow.Engine(str(cfg_path), thread_num=threads)
    seed_cityflow(engine, seed)
    return engine


def make_chooser(name: str, features: FeatureBuilder, seed: int) -> Chooser:
    """Phase-selection rule for one of :data:`CONTROLLERS`."""
    phases = list(features.phase_ids)
    if name == "fixed_time":

        def choose(iid, current, lane_count):
            return (
                phases[0] if current is None else phases[(phases.index(current) + 1) % len(phases)]
            )

    elif name == "max_pressure":

        def choose(iid, current, lane_count):
            pressure = features.phase_pressure(iid, lane_count)
            return max(phases, key=pressure.__getitem__)  # ties: lowest phase id

    elif name == "random":
        rng = make_rng(seed)

        def choose(iid, current, lane_count):
            return int(rng.choice(phases))

    else:
        raise ValueError(f"unknown controller {name!r}; one of {CONTROLLERS}")
    return choose


def collect_trace(
    engine,
    roadnet: Roadnet,
    features: FeatureBuilder,
    choose: Chooser,
    *,
    green_seconds: int,
    yellow_seconds: int,
    episode_seconds: int,
    record_every: int,
    start_times: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run one episode and record lane states and vehicle trips.

    Each intersection keeps its phase for ``green_seconds``, then asks ``choose`` for the
    next one; a change inserts ``yellow_seconds`` of the transition phase.

    Returns ``(steps, vehicles, summary)``:

    * ``steps`` — one row per intersection every ``record_every`` s: time, phase, whether it
      is in transition, vehicles and queue on the 12 incoming lanes (canonical order),
      vehicles on the 12 outgoing lanes, raw intersection pressure.
    * ``vehicles`` — one row per vehicle of the flow file: planned start, time it entered
      the network, time it left (NaN if still inside or never inserted), travel time
      (exit − planned start, the definition CityFlow uses for its average).
    * ``summary`` — episode-level numbers.
    """
    ids = list(roadnet.intersections)
    state: dict[str, list] = {}  # iid -> [green, seconds left, pending green or None]
    counts = engine.get_lane_vehicle_count()
    for iid in ids:
        g = choose(iid, None, counts)
        engine.set_tl_phase(iid, g)
        state[iid] = [g, green_seconds, None]

    n_veh = len(start_times)
    enter = np.full(n_veh, np.nan)
    leave = np.full(n_veh, np.nan)
    inside: set[str] = set()
    rows = []
    queue_sum = 0.0
    switches = 0

    t0 = time.perf_counter()
    for t in range(episode_seconds):
        counts = None
        for iid, s in state.items():
            s[1] -= 1
            if s[1] > 0:
                continue
            if s[2] is not None:  # transition finished -> start the pending green
                s[0], s[2], s[1] = s[2], None, green_seconds
                engine.set_tl_phase(iid, s[0])
                continue
            if counts is None:
                counts = engine.get_lane_vehicle_count()
            nxt = choose(iid, s[0], counts)
            if nxt == s[0] or yellow_seconds == 0:
                s[0], s[1] = nxt, green_seconds
                engine.set_tl_phase(iid, nxt)
            else:
                switches += 1
                s[2], s[1] = nxt, yellow_seconds
                engine.set_tl_phase(iid, YELLOW_PHASE)
        engine.next_step()
        now = t + 1

        waiting = engine.get_lane_waiting_vehicle_count()
        queue_sum += sum(waiting.values())

        current = set(engine.get_vehicles(include_waiting=False))
        for vid in current - inside:
            enter[int(vid.split("_")[1])] = now
        for vid in inside - current:
            leave[int(vid.split("_")[1])] = now
        inside = current

        if now % record_every == 0:
            lane_count = engine.get_lane_vehicle_count()
            for iid in ids:
                inter = roadnet.intersections[iid]
                green, _, pending = state[iid]
                rows.append(
                    [now, iid, pending if pending is not None else green, int(pending is not None)]
                    + [lane_count[ln] for ln in inter.in_lanes]
                    + [waiting[ln] for ln in inter.in_lanes]
                    + [lane_count[ln] for ln in inter.out_lanes]
                    + [features.intersection_pressure(iid, lane_count)]
                )
    wall = time.perf_counter() - t0

    n = len(roadnet.intersections[ids[0]].in_lanes)
    columns = (
        ["t", "intersection", "phase", "in_transition"]
        + [f"count_{k}" for k in range(n)]
        + [f"queue_{k}" for k in range(n)]
        + [f"out_{k}" for k in range(n)]
        + ["pressure"]
    )
    steps = pd.DataFrame(rows, columns=columns)
    start = np.asarray(start_times, dtype=float)
    vehicles = pd.DataFrame(
        {
            "vehicle": np.arange(n_veh),
            "start_time": start,
            "enter_time": enter,
            "exit_time": leave,
            "travel_time": leave - start,
        }
    )
    done = vehicles["exit_time"].notna()
    summary = {
        "episode_seconds": episode_seconds,
        "green_seconds": green_seconds,
        "yellow_seconds": yellow_seconds,
        "vehicles": n_veh,
        "avg_travel_time_s": round(engine.get_average_travel_time(), 2),
        "completed": int(done.sum()),
        "in_network_at_end": int(engine.get_vehicle_count()),
        "never_entered": int(vehicles["enter_time"].isna().sum()),
        "completed_avg_travel_time_s": round(float(vehicles.loc[done, "travel_time"].mean()), 2),
        "mean_total_queue": round(queue_sum / episode_seconds, 2),
        "phase_switches": switches,
        "wall_s": round(wall, 2),
        "steps_per_s": round(episode_seconds / wall, 1),
    }
    return steps, vehicles, summary
