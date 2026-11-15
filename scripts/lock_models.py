"""Write or verify the SHA-256 lock of the final models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from trafficrl.data.splits import REPO_ROOT
from trafficrl.eval.lock import MODELS, build_manifest, verify_manifest

MANIFEST = REPO_ROOT / "models" / "manifest.json"


def git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
    )
    return out.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify", action="store_true")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--train-dir", type=Path, default=REPO_ROOT / "results" / "train")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", default="final")
    args = ap.parse_args(argv)

    if args.write:
        try:
            m = build_manifest(args.train_dir, args.models, args.seeds, args.tag)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        m = {
            "_comment": "Models locked on validation before the final evaluation (ROADMAP "
            "Phase 7). Written by scripts/lock_models.py --write; verified by "
            "scripts/robustness.py --final. Paths are relative to results/train/.",
            "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            **m,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(m, indent=2) + "\n")
        print(f"wrote {args.manifest} ({len(m['runs'])} runs)")
        return 0

    if not args.manifest.is_file():
        print(f"error: {args.manifest} does not exist (run --write first)", file=sys.stderr)
        return 2
    problems = verify_manifest(json.loads(args.manifest.read_text()), args.train_dir)
    if problems:
        print("error: lock manifest does not match:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    print(f"ok: {args.manifest} matches {args.train_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
