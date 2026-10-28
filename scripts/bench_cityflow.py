"""Smoke test + timing for the CityFlow install (ROADMAP 0.1b).

Runs one episode of Fixed-Time and Max-Pressure on a CityFlow roadnet/flow pair and
reports wall-clock time, simulator steps/s, average travel time and throughput.

Standalone on purpose (stdlib + cityflow only): it validates the install before the
`src/trafficrl` package exists. The real baselines will live in
`src/trafficrl/agents/baselines/`.

    python scripts/bench_cityflow.py \
        --data-dir data/raw/colight_src/data/Jinan/3_4 \
        --roadnet roadnet_3_4.json --flow anon_3_4_jinan_real.json
"""

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path

import cityflow

YELLOW_PHASE = 0  # CityFlow/CoLight convention: phase 0 = transition (right turns only)


def load_intersections(roadnet_path):
    """Return {intersection_id: [green_phase -> list of (in_lanes, out_lanes)]}."""
    roadnet = json.loads(Path(roadnet_path).read_text())
    out = {}
    for inter in roadnet["intersections"]:
        if inter["virtual"]:
            continue
        movements = []
        for link in inter["roadLinks"]:
            in_lanes = {f"{link['startRoad']}_{ll['startLaneIndex']}" for ll in link["laneLinks"]}
            out_lanes = {f"{link['endRoad']}_{ll['endLaneIndex']}" for ll in link["laneLinks"]}
            movements.append((sorted(in_lanes), sorted(out_lanes)))
        phases = inter["trafficLight"]["lightphases"]
        # skip the yellow phase; keep the index so we can set it on the engine
        out[inter["id"]] = {
            k: [movements[i] for i in p["availableRoadLinks"]]
            for k, p in enumerate(phases)
            if k != YELLOW_PHASE
        }
    return out


def make_engine(data_dir, roadnet, flow, seed, threads, workdir):
    cfg = {
        "interval": 1.0,
        "seed": seed,
        "dir": str(Path(data_dir).resolve()) + "/",
        "roadnetFile": roadnet,
        "flowFile": flow,
        "rlTrafficLight": True,
        "laneChange": False,
        "saveReplay": False,
        "roadnetLogFile": "",
        "replayLogFile": "",
    }
    cfg_path = Path(workdir) / "cityflow_config.json"
    cfg_path.write_text(json.dumps(cfg))
    return cityflow.Engine(str(cfg_path), thread_num=threads)


class FixedTime:
    """Cycle through the green phases in order, `green` s each, `yellow` s between."""

    def __init__(self, phases, green=30):
        self.phases = {i: sorted(p) for i, p in phases.items()}
        self.green = green

    def choose(self, inter_id, current, engine):
        order = self.phases[inter_id]
        if current is None:
            return order[0]
        return order[(order.index(current) + 1) % len(order)]


class MaxPressure:
    """Pick the green phase with the largest pressure, sum over its movements of
    (vehicles on incoming lanes - vehicles on outgoing lanes) (Varaiya 2013)."""

    def __init__(self, phases, green=10):
        self.phases = phases
        self.green = green

    def choose(self, inter_id, current, engine):
        counts = engine.get_lane_vehicle_count()

        def pressure(movs):
            return sum(
                sum(counts.get(lane, 0) for lane in ins) - sum(counts.get(lane, 0) for lane in outs)
                for ins, outs in movs
            )

        return max(self.phases[inter_id], key=lambda k: pressure(self.phases[inter_id][k]))


def run_episode(engine, controller, steps, yellow):
    """Per-intersection state machine: green for `controller.green` s, then decide; a
    phase change inserts `yellow` s of the transition phase."""
    state = {}  # inter_id -> [current_green, seconds_left, pending_green or None]
    for inter_id in controller.phases:
        g = controller.choose(inter_id, None, engine)
        engine.set_tl_phase(inter_id, g)
        state[inter_id] = [g, controller.green, None]

    queue_sum = 0.0
    t0 = time.perf_counter()
    for _ in range(steps):
        for inter_id, s in state.items():
            s[1] -= 1
            if s[1] > 0:
                continue
            if s[2] is not None:  # yellow finished -> start pending green
                s[0], s[2], s[1] = s[2], None, controller.green
                engine.set_tl_phase(inter_id, s[0])
                continue
            nxt = controller.choose(inter_id, s[0], engine)
            if nxt == s[0] or yellow == 0:
                s[0], s[1] = nxt, controller.green
                engine.set_tl_phase(inter_id, nxt)
            else:
                s[2], s[1] = nxt, yellow
                engine.set_tl_phase(inter_id, YELLOW_PHASE)
        engine.next_step()
        queue_sum += sum(engine.get_lane_waiting_vehicle_count().values())
    wall = time.perf_counter() - t0
    return {
        "wall_s": round(wall, 2),
        "steps_per_s": round(steps / wall, 1),
        "avg_travel_time_s": round(engine.get_average_travel_time(), 2),
        "vehicles_in_network_at_end": engine.get_vehicle_count(),
        "mean_total_queue": round(queue_sum / steps, 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--roadnet", required=True)
    ap.add_argument("--flow", required=True)
    ap.add_argument("--steps", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--yellow", type=int, default=5)
    ap.add_argument("--fixed-green", type=int, default=30)
    ap.add_argument("--mp-green", type=int, default=10)
    ap.add_argument("--out", help="optional JSON file for the results")
    args = ap.parse_args()

    phases = load_intersections(Path(args.data_dir) / args.roadnet)
    flow = json.loads((Path(args.data_dir) / args.flow).read_text())
    controllers = {
        "fixed_time": FixedTime(phases, green=args.fixed_green),
        "max_pressure": MaxPressure(phases, green=args.mp_green),
    }

    results = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "roadnet": args.roadnet,
        "flow": args.flow,
        "n_intersections": len(phases),
        "n_vehicles": len(flow),
        "args": vars(args),
        "runs": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, ctrl in controllers.items():
            engine = make_engine(args.data_dir, args.roadnet, args.flow, args.seed, args.threads, tmp)
            results["runs"][name] = run_episode(engine, ctrl, args.steps, args.yellow)
            print(name, results["runs"][name], flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
