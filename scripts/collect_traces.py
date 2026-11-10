"""Collect train-split simulator traces for the EDA (ROADMAP 3.3).

Runs the rule controllers in ``configs/eda.yaml`` (Fixed-Time, Max-Pressure, random phase;
each for every seed) plus a decision-interval check for Max-Pressure, on the **training
flow only**. Any other dataset / split in the config is rejected.

    python scripts/collect_traces.py --config configs/eda.yaml

Per run, in ``results/eda/<run>/``: ``steps.csv.gz``, ``vehicles.csv.gz``, ``summary.json``,
resolved ``config.yaml``, ``env.json``. Each run appends one row to
``docs/experiment_log.csv``. These are descriptive rollouts, not baseline results.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from trafficrl.data.roadnet import load_roadnet
from trafficrl.data.splits import REPO_ROOT
from trafficrl.data.trace import collect_trace, make_chooser, make_engine, train_flow
from trafficrl.envs.features import FeatureBuilder, FeatureSpec
from trafficrl.utils.config import load_config, save_config
from trafficrl.utils.runlog import append_experiment_row, save_env_info
from trafficrl.utils.seed import seed_everything


def plan_runs(cfg: dict) -> list[dict]:
    """Expand the ``eda`` block into a list of runs (name, controller, interval, seed)."""
    eda = cfg["eda"]
    runs, seen = [], set()

    def add(controller: str, interval: int, seed: int, purpose: str) -> None:
        if (controller, interval, seed) in seen:
            return
        seen.add((controller, interval, seed))
        runs.append(
            {
                "name": f"{controller}_dt{interval}_seed{seed}",
                "controller": controller,
                "interval": interval,
                "seed": seed,
                "purpose": purpose,
            }
        )

    for controller, params in eda["controllers"].items():
        interval = params.get("green_seconds", params.get("decision_interval"))
        for seed in eda["seeds"]:
            add(controller, interval, seed, "EDA trace")
    sweep = eda.get("dt_sweep")
    if sweep:
        for interval in sweep["values"]:
            add(sweep["controller"], interval, cfg["seed"], "EDA decision-interval check")
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "eda.yaml")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. sim.episode_seconds=600")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    roadnet_path, flow_path = train_flow(cfg)  # raises unless primary dataset / train split
    roadnet = load_roadnet(roadnet_path)
    features = FeatureBuilder(roadnet, FeatureSpec.from_config(cfg))
    start_times = [v["startTime"] for v in json.loads(flow_path.read_text())]
    sim = cfg["sim"]
    out_root = REPO_ROOT / cfg["output"]["results_dir"]

    for run in plan_runs(cfg):
        seed_everything(run["seed"], cfg.get("deterministic", True))
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(roadnet_path, flow_path, run["seed"], sim["threads"], Path(tmp))
            steps, vehicles, summary = collect_trace(
                engine,
                roadnet,
                features,
                make_chooser(run["controller"], features, run["seed"]),
                green_seconds=run["interval"],
                yellow_seconds=sim["yellow_seconds"],
                episode_seconds=sim["episode_seconds"],
                record_every=sim["decision_interval"],
                start_times=start_times,
            )
        runtime = time.perf_counter() - t0

        out = out_root / run["name"]
        out.mkdir(parents=True, exist_ok=True)
        steps.to_csv(out / "steps.csv.gz", index=False)
        vehicles.to_csv(out / "vehicles.csv.gz", index=False)
        summary = {
            "run": run["name"],
            "controller": run["controller"],
            "interval_s": run["interval"],
            "seed": run["seed"],
            "dataset": cfg["data"]["dataset"],
            "split": cfg["data"]["split"],
            "flow_file": flow_path.name,
            **summary,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        save_config({**cfg, "seed": run["seed"], "run": run}, out / "config.yaml")
        save_env_info(out / "env.json")
        append_experiment_row(
            run_id=f"eda-{run['name']}",
            split=cfg["data"]["split"],
            seed=run["seed"],
            model=f"{run['controller']} (rule, {run['purpose']})",
            hyperparameters={
                "interval_s": run["interval"],
                "yellow_s": sim["yellow_seconds"],
                "episode_s": sim["episode_seconds"],
            },
            metric_name="avg_travel_time_s",
            metric_value=summary["avg_travel_time_s"],
            runtime_s=runtime,
            notes="descriptive rollout on the train flow; not a baseline result",
        )
        print(
            f"{run['name']:28s} travel {summary['avg_travel_time_s']:7.1f} s  "
            f"queue {summary['mean_total_queue']:7.1f}  done {summary['completed']:5d}  "
            f"{summary['wall_s']:5.1f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
