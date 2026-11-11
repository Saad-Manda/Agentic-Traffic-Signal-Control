"""Random-search tuning on validation, or one config over several seeds (ROADMAP 5.5).

    # study: N trials from the committed search space, training seed from the search file
    python scripts/tune.py --search configs/search/rl_baselines.yaml --agent dqn [--workers 4]
    # chosen config on seeds 0-4
    python scripts/tune.py --config configs/dqn.yaml --seeds 0 1 2 3 4 [--workers 5]

Each run is a ``scripts/train.py`` process (train flow, checkpoint on val, own experiment-log
row). Writes ``results/tune/<study>/{trials.csv, best.json, configs/, logs/}`` or
``results/tune/<agent>_seeds/seeds.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trafficrl.agents import LEARNED_AGENTS
from trafficrl.data.splits import REPO_ROOT
from trafficrl.train.tune import (
    collect,
    load_search,
    overrides,
    run_parallel,
    sample_trials,
    select_best,
    train_argv,
)
from trafficrl.utils.config import load_config, save_config

TUNE_DIR = REPO_ROOT / "results" / "tune"


def study(args) -> int:
    spec = load_search(args.search)
    cfg_path = args.config or REPO_ROOT / "configs" / f"{args.agent}.yaml"
    n = args.trials or int(spec["trials"])
    trials = sample_trials(spec["space"], n, int(spec["sampler_seed"]))
    name = f"{spec['study']}_{args.agent}"
    out = TUNE_DIR / name
    runs, jobs = {}, []
    for k, params in enumerate(trials):
        run = f"{args.agent}_{spec['study']}_t{k}_s{spec['seed']}"
        cfg = load_config(
            cfg_path, [*overrides(params), f"seed={spec['seed']}", f"train.agent={args.agent}"]
        )
        resolved = save_config(cfg, out / "configs" / f"{run}.yaml")  # written before launch
        runs[run] = {"trial": k, **params}
        jobs.append((run, train_argv(resolved, run, args.overrides, f"tuning {name} trial {k}")))
    codes = run_parallel(jobs, args.workers, out / "logs")
    results_dir = REPO_ROOT / load_config(cfg_path, args.overrides)["output"]["results_dir"]
    df = collect(results_dir, runs)
    df["exit_code"] = [codes.get(r) for r in df["run_name"]]
    df.to_csv(out / "trials.csv", index=False)
    sel = spec["selection"]
    best = select_best(df, sel["metric"], float(sel["tie_tolerance_s"]))
    best_params = {k: getattr(best[k], "item", lambda v=best[k]: v)() for k in spec["space"]}
    (out / "best.json").write_text(
        json.dumps(
            {
                "run_name": best["run_name"],
                "params": best_params,
                sel["metric"]: float(best[sel["metric"]]),
                "search": str(args.search),
            },
            indent=2,
            default=float,
        )
        + "\n"
    )
    print(df.drop(columns=["params"]).to_string(index=False))
    print(f"best: {best['run_name']} {best_params} -> {best[sel['metric']]:.2f} s")
    print(f"results: {out.relative_to(REPO_ROOT)}")
    return 0 if all(c == 0 for c in codes.values()) else 1


def seeds(args) -> int:
    cfg = load_config(args.config, args.overrides)
    agent = cfg["train"]["agent"]
    out = TUNE_DIR / f"{agent}_{args.tag}_seeds"
    runs, jobs = {}, []
    for s in args.seeds:
        run = f"{agent}_{args.tag}_s{s}"
        runs[run] = {"seed": s}
        jobs.append(
            (
                run,
                train_argv(
                    args.config, run, [*args.overrides, f"seed={s}"], f"{args.tag} config, seed {s}"
                ),
            )
        )
    codes = run_parallel(jobs, args.workers, out / "logs")
    df = collect(REPO_ROOT / cfg["output"]["results_dir"], runs)
    df["exit_code"] = [codes.get(r) for r in df["run_name"]]
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "seeds.csv", index=False)
    print(df.to_string(index=False))
    print(f"results: {out.relative_to(REPO_ROOT)}")
    return 0 if all(c == 0 for c in codes.values()) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--search", type=Path, help="search-space file (study mode)")
    ap.add_argument("--agent", choices=LEARNED_AGENTS, help="study mode")
    ap.add_argument("--config", type=Path, help="agent config (seeds mode; study: base config)")
    ap.add_argument("--seeds", type=int, nargs="+", help="seeds mode")
    ap.add_argument("--tag", default="final", help="seeds mode: run-name tag")
    ap.add_argument("--trials", type=int, help="study: override the number of trials")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("overrides", nargs="*", help="extra config overrides for every run")
    args = ap.parse_args(argv)
    if args.search and args.agent:
        return study(args)
    if args.config and args.seeds:
        return seeds(args)
    ap.error("use --search + --agent (study) or --config + --seeds (multi-seed)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
