"""Train one learned agent on the train flow, select its checkpoint on validation (ROADMAP 5.4).

    python scripts/train.py --config configs/dqn.yaml [--run-name NAME] [overrides ...]

Writes ``results/train/<run>/`` (train / val logs, best + last checkpoints, TensorBoard,
config, env.json, summary.json) and one row in ``docs/experiment_log.csv``. Only
``data.split=train`` is accepted (exit 2 otherwise); test / held-out flows are never opened.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trafficrl.data.splits import REPO_ROOT
from trafficrl.train.loop import TrainSplitError, train
from trafficrl.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-name", help="results sub-directory (default: <agent>_s<seed>_<time>)")
    ap.add_argument("--note", default="", help="free text for the experiment log")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. train.lr=1e-3 seed=1")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    try:
        out = train(cfg, run_name=args.run_name, note=args.note, verbose=not args.quiet)
    except TrainSplitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"results: {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
