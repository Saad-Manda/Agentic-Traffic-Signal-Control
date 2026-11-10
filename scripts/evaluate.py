"""Evaluate one agent on one split over several seeds (ROADMAP 4.6).

    python scripts/evaluate.py --config configs/eval_baselines.yaml --agent max_pressure \
        --split val [--seeds 0 1 2 3 4] [--checkpoint PATH] [--final] [overrides ...]

Writes ``results/eval/<run_id>/{metrics,summary}.csv`` (+ config, env.json, data.json) and
appends one row per seed to ``docs/experiment_log.csv``. ``--split test|heldout`` is refused
unless ``--final`` is given — final evaluation happens once, in Phase 7, after every choice
has been made on the validation flow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trafficrl.agents import AGENTS
from trafficrl.data.splits import REPO_ROOT, SPLIT_NAMES
from trafficrl.envs.traffic_env import FINAL_SPLITS
from trafficrl.eval.runner import evaluate, summary_table
from trafficrl.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "eval_baselines.yaml")
    ap.add_argument("--agent", required=True, choices=AGENTS)
    ap.add_argument("--split", choices=SPLIT_NAMES, help="overrides data.split")
    ap.add_argument("--dataset", help="overrides data.dataset")
    ap.add_argument("--seeds", type=int, nargs="+", help="default: eval.seeds from the config")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--track", choices=("full", "light"))
    ap.add_argument("--final", action="store_true", help="allow test / held-out (Phase 7 only)")
    ap.add_argument("--note", default="", help="free text for the experiment log")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. robustness.noise_std=0.1")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    if args.split:
        cfg["data"]["split"] = args.split
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    split = cfg["data"]["split"]
    if split in FINAL_SPLITS and not args.final:
        print(
            f"error: split {split!r} is final-evaluation only; it is run once in Phase 7 "
            "with --final, after model selection on the validation flow",
            file=sys.stderr,
        )
        return 2
    if args.final:
        print(f"WARNING: final evaluation on {cfg['data']['dataset']}/{split}", file=sys.stderr)

    seeds = args.seeds or cfg.get("eval", {}).get("seeds", [cfg["seed"]])
    track = args.track or cfg.get("eval", {}).get("track", "full")
    out, metrics = evaluate(
        cfg,
        args.agent,
        seeds,
        final=args.final,
        checkpoint=args.checkpoint,
        track=track,
        note=args.note,
    )
    summ = summary_table(metrics).set_index("metric")
    for name in (
        "avg_travel_time_s",
        "avg_waiting_time_s",
        "avg_queue_veh",
        "throughput_veh",
        "max_waiting_time_s",
        "decision_latency_ms",
    ):
        if name in summ.index:
            r = summ.loc[name]
            ci = (
                "deterministic"
                if r["deterministic"]
                else (f"95% CI [{r['ci95_low']:.2f}, {r['ci95_high']:.2f}]")
            )
            print(f"{name:22s} {r['mean']:10.2f} ± {r['std']:.2f}  {ci}  (n={int(r['n'])})")
    print(f"results: {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
