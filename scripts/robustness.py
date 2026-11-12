"""Model × condition evaluation table (ROADMAP 6.8 on validation; 7.3 on test with --final).

    python scripts/robustness.py --spec configs/robustness_val.yaml [--workers 5] \
        [--out docs/results/val_robustness] [--table-only] [--final]

The spec lists models, seeds and observation / demand conditions. Each (model, condition) is
one ``scripts/evaluate.py`` process over all seeds, written to
``<results-dir>/<study>/eval/<model>__<condition>/`` (one experiment-log row per episode).
Learned models load ``results/train/<train_model>_<tag>_s{seed}/checkpoints/best.pt``; gated
UG-MARL reads τ from the ``gate.json`` next to each checkpoint. A finished job is skipped on a
re-run. The table (``<out>.{csv,md}``) is built from the finished runs.

**Nothing is selected here.** ``split: test | heldout`` is refused unless ``--final``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from trafficrl.agents import LEARNED_AGENTS
from trafficrl.data.splits import REPO_ROOT
from trafficrl.envs.traffic_env import FINAL_SPLITS
from trafficrl.eval.report import condition_table, fmt
from trafficrl.eval.runner import condition_label
from trafficrl.train.tune import run_parallel
from trafficrl.utils.config import apply_overrides, read_yaml
from trafficrl.utils.plotting import CONTROLLER_LABEL

EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate.py"
RESULTS = REPO_ROOT / "results" / "robustness"
DOCS = REPO_ROOT / "docs" / "results"


def cond_overrides(cond: dict) -> list[str]:
    return [f"{k}={v}" for k, v in cond.items()]


def cond_name(cond: dict) -> str:
    return condition_label(apply_overrides({"robustness": {}}, cond_overrides(cond)))


def checkpoint_template(model: str, m: dict, tag: str, train_dir: Path) -> str | None:
    """``…/<train_model>_<tag>_s{seed}/checkpoints/best.pt`` for learned agents, else None."""
    if m.get("agent", model) not in LEARNED_AGENTS:
        return None
    return str(train_dir / f"{m.get('train_model', model)}_{tag}_s{{seed}}/checkpoints/best.pt")


def finished_run(job_dir: Path) -> Path | None:
    runs = sorted(p for p in job_dir.glob("*") if (p / "metrics.csv").exists())
    return runs[-1] if runs else None


def build_jobs(spec: dict, study_dir: Path, train_dir: Path, final: bool):
    """``[(model, condition, job_dir, argv)]`` in table order."""
    jobs = []
    seeds = [str(int(s)) for s in spec["seeds"]]
    for model, m in spec["models"].items():
        m = m or {}
        ckpt = checkpoint_template(model, m, spec.get("tag", "final"), train_dir)
        for cond in spec["conditions"]:
            cond = dict(cond or {})
            name = cond_name(cond)
            job_dir = study_dir / "eval" / f"{model}__{name}"
            argv = [
                sys.executable, str(EVAL_SCRIPT), "--config", str(REPO_ROOT / m["config"]),
                "--agent", m.get("agent", model), "--split", spec["split"],
                "--dataset", spec["dataset"], "--seeds", *seeds,
                "--note", f"{spec['study']}: {model}, {name}",
            ]  # fmt: skip
            if ckpt:
                argv += ["--checkpoint", ckpt]
            if final:
                argv.append("--final")
            argv += [*m.get("overrides", []), *cond_overrides(cond)]
            argv.append(f"output.results_dir={job_dir}")
            jobs.append((model, name, job_dir, argv))
    return jobs


def missing_inputs(spec: dict, train_dir: Path) -> list[str]:
    """Checkpoints (and gate.json for gated UG-MARL) that the spec needs but are absent."""
    missing = []
    for model, m in spec["models"].items():
        m = m or {}
        tpl = checkpoint_template(model, m, spec.get("tag", "final"), train_dir)
        if tpl is None:
            continue
        gated = m.get("agent", model) == "ugmarl" and not any(
            o.replace(" ", "").lower() in ("agents.ugmarl.use_gate=false",
                                           "agents.ugmarl.use_ensemble=false")
            for o in m.get("overrides", [])
        )  # fmt: skip
        for s in spec["seeds"]:
            ckpt = Path(tpl.replace("{seed}", str(s)))
            if not ckpt.exists():
                missing.append(str(ckpt))
            elif gated and not (ckpt.parent.parent / "gate.json").exists():
                missing.append(str(ckpt.parent.parent / "gate.json"))
    return missing


def label(model: str) -> str:
    return CONTROLLER_LABEL.get(model, model)


def write_md(path: Path, spec: dict, table: pd.DataFrame, spec_path: Path) -> None:
    models = list(dict.fromkeys(table["model"]))
    conds = list(dict.fromkeys(table["condition"]))
    cell = table.set_index(["model", "condition"])

    def stats(model, cond):
        r = cell.loc[(model, cond)]
        return {
            "mean": r["avg_travel_time_mean"], "std": r["avg_travel_time_std"],
            "ci95_low": r["avg_travel_time_ci95_low"],
            "ci95_high": r["avg_travel_time_ci95_high"], "deterministic": r["deterministic"],
        }  # fmt: skip

    def grid(title, fn):
        out = [f"## {title}", "", "| Model | " + " | ".join(conds) + " |",
               "|---|" + "---|" * len(conds)]  # fmt: skip
        for m in models:
            vals = [fn(m, c) if (m, c) in cell.index else "—" for c in conds]
            out.append(f"| {label(m)} | " + " | ".join(vals) + " |")
        return [*out, ""]

    final = spec["split"] in FINAL_SPLITS
    try:
        shown = spec_path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = spec_path
    lines = [
        f"# Robustness — {spec['dataset']} `{spec['split']}`, model × condition",
        "",
        f"Generated by `scripts/robustness.py --spec {shown}`. "
        f"Dataset {spec['dataset']}, split `{spec['split']}`, 3600 s, Δt = 15 s, seeds "
        f"{', '.join(str(s) for s in spec['seeds'])} (a learned model's seed s uses the "
        "checkpoint trained with seed s; gated UG-MARL uses τ from its `gate.json`). "
        "Perturbations act on what every controller observes (protocol §6): noise = Gaussian "
        "on capacity-scaled lane values, drop = lane reading set to 0. Travel time: mean ± std "
        "[95 % CI, Student t] over seeds; rule-based controllers are deterministic on the clean "
        "flow only (the perturbation is drawn from the seed). **Evaluation only — no "
        "checkpoint, hyperparameter or τ is selected from this table.**"
        + ("" if final else " Validation was used for checkpoint, (K, d) and p selection, so "
           "learned-model scores are optimistic; the reported robustness result is ROADMAP 7.3 "
           "on the test flow."),
        "",
    ]  # fmt: skip
    lines += grid("Average travel time (s)", lambda m, c: fmt(stats(m, c)))
    lines += grid(
        "Maximum per-vehicle wait (s), mean over seeds",
        lambda m, c: f"{cell.loc[(m, c), 'max_waiting_time_mean']:.0f}",
    )
    lines += grid(
        "Fallback rate (share of decisions taken by Max-Pressure)",
        lambda m, c: f"{cell.loc[(m, c), 'fallback_rate_mean']:.3f}",
    )
    if "clean" in conds and len(conds) > 1:
        others = [c for c in conds if c != "clean"]
        lines += [
            "## Travel-time increase over the clean flow (s, difference of means)",
            "",
            "| Model | " + " | ".join(others) + " |",
            "|---|" + "---|" * len(others),
        ]
        for m in models:
            if (m, "clean") not in cell.index:
                continue
            base = cell.loc[(m, "clean"), "avg_travel_time_mean"]
            vals = [
                f"{cell.loc[(m, c), 'avg_travel_time_mean'] - base:+.1f}"
                if (m, c) in cell.index else "—"
                for c in others
            ]  # fmt: skip
            lines.append(f"| {label(m)} | " + " | ".join(vals) + " |")
        lines.append("")
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", help="table path without suffix (default docs/results/<study>)")
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    ap.add_argument("--train-dir", type=Path, default=REPO_ROOT / "results" / "train")
    ap.add_argument("--table-only", action="store_true", help="run nothing, rebuild the table")
    ap.add_argument("--final", action="store_true", help="allow test / held-out (Phase 7 only)")
    args = ap.parse_args(argv)

    spec = read_yaml(args.spec)
    if spec["split"] in FINAL_SPLITS and not args.final:
        print(
            f"error: split {spec['split']!r} is final-evaluation only (Phase 7, --final)",
            file=sys.stderr,
        )
        return 2
    missing = missing_inputs(spec, args.train_dir)
    if missing:
        print("error: missing " + ", ".join(missing), file=sys.stderr)
        return 2
    study_dir = args.results_dir / spec["study"]
    jobs = build_jobs(spec, study_dir, args.train_dir, args.final)

    if not args.table_only:
        todo = [(f"{m}__{c}", a) for m, c, d, a in jobs if finished_run(d) is None]
        print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already finished")
        codes = run_parallel(todo, args.workers, study_dir / "logs")
        failed = {k: v for k, v in codes.items() if v}
        if failed:
            print(f"error: jobs failed {failed} (logs in {study_dir / 'logs'})", file=sys.stderr)
            return 1

    runs = {(m, c): finished_run(d) for m, c, d, _ in jobs}
    absent = [f"{m}__{c}" for (m, c), r in runs.items() if r is None]
    if absent:
        print(f"error: no finished run for {absent}", file=sys.stderr)
        return 1
    table = condition_table(runs)
    out = Path(args.out) if args.out else DOCS / spec["study"]
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out.with_suffix(".csv"), index=False)
    write_md(out.with_suffix(".md"), spec, table, args.spec)
    print(table.to_string(index=False))
    print(f"wrote {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
