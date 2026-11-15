"""Calibrate the UG-MARL gate threshold τ on validation (ROADMAP 6.3 / 6.7b; plan D9).

    python scripts/calibrate_gate.py --search configs/search/ugmarl_gate.yaml \
        --config configs/eval_ugmarl.yaml --tag final [--workers 4] [overrides ...]
    # −attention models: add  agents.ugmarl.use_attention=false

For the seed-models ``results/train/<variant>_<tag>_s<seed>/checkpoints/best.pt``, Jinan
**validation** only:

1. **Reference pass** — ungated greedy episode per model × condition (clean + the search
   file's observation perturbations), decisions logged. The clean decisions give σ;
   τ_{p,seed} = p-th percentile of that model's σ. The perturbed ones are the −gate reference
   and the data of the σ check in ``notebooks/04_gate_attention.ipynb``.
2. **Sweep** — gated episode per model × p × condition.
3. **Select** p* = lowest mean travel time over seeds × conditions (ties within the tolerance
   -> smaller p) and write ``gate.json`` into every model's run directory, plus
   ``docs/results/gate_calibration[_<variant>].{csv,md}``.

Each (condition, p) job is one ``scripts/evaluate.py`` process over all seeds (one experiment-
log row per episode), τ passed per seed through ``agents.ugmarl.gate.file``. Any split other
than ``val`` is refused (exit 2); ``--final`` is never passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from trafficrl.agents import model_label
from trafficrl.agents.ugmarl.gate import (
    percentile_thresholds,
    save_gate,
    select_percentile,
)
from trafficrl.data.splits import REPO_ROOT
from trafficrl.eval.runner import condition_label
from trafficrl.train.tune import run_parallel
from trafficrl.utils.config import apply_overrides, load_config, read_yaml
from trafficrl.utils.runlog import git_commit
from trafficrl.utils.stats import summarise

EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate.py"
TUNE_DIR = REPO_ROOT / "results" / "tune"
DOCS = REPO_ROOT / "docs" / "results"
VAL = "val"


def cond_overrides(cond: dict) -> list[str]:
    return [f"{k}={v}" for k, v in cond.items()]


def cond_name(cond: dict) -> str:
    return condition_label(apply_overrides({"robustness": {}}, cond_overrides(cond)))


def eval_argv(config, ckpt, seeds, results_dir, extra, note, log_decisions=False):
    argv = [
        sys.executable, str(EVAL_SCRIPT), "--config", str(config), "--agent", "ugmarl",
        "--split", VAL, "--seeds", *map(str, seeds), "--checkpoint", ckpt, "--note", note,
    ]  # fmt: skip
    if log_decisions:
        argv.append("--log-decisions")
    return [*argv, *extra, f"output.results_dir={results_dir}"]


def only_run(d: Path) -> Path:
    runs = sorted(p for p in d.iterdir() if p.is_dir())
    if len(runs) != 1:
        raise RuntimeError(f"expected one evaluation run in {d}, found {len(runs)}")
    return runs[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--search", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--tag", default="final", help="training runs <variant>_<tag>_s<seed>")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("overrides", nargs="*", help="e.g. agents.ugmarl.use_attention=false")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    if cfg["data"]["split"] != VAL:
        print("error: gate calibration uses the validation split only", file=sys.stderr)
        return 2
    spec = read_yaml(args.search)
    variant = model_label("ugmarl", cfg)
    if variant.endswith("-nogate") or variant.endswith("-noens"):
        print(f"error: {variant} has no gate to calibrate", file=sys.stderr)
        return 2
    seeds = [int(s) for s in spec["seeds"]]
    ps = [float(p) for p in spec["percentiles"]]
    conds = [dict(c or {}) for c in spec["conditions"]]
    if cond_name(conds[0]) != "clean":
        print("error: the first condition must be the clean flow (σ pass)", file=sys.stderr)
        return 2
    train_dir = REPO_ROOT / "results" / "train"
    ckpt_tpl = str(train_dir / f"{variant}_{args.tag}_s{{seed}}" / "checkpoints" / "best.pt")
    for s in seeds:
        if not Path(ckpt_tpl.replace("{seed}", str(s))).exists():
            print(
                f"error: missing checkpoint {ckpt_tpl.replace('{seed}', str(s))}", file=sys.stderr
            )
            return 2
    out = TUNE_DIR / f"{spec['study']}_{variant}_{args.tag}"
    if out.exists():
        print(f"error: {out} exists (calibration runs once per model set)", file=sys.stderr)
        return 2
    base = [*args.overrides]
    study = f"{spec['study']} {variant}_{args.tag}"

    # 1. reference pass: ungated, decisions logged (clean = σ pass)
    jobs = []
    for c in conds:
        name = f"ref_{cond_name(c)}"
        jobs.append((name, eval_argv(
            args.config, ckpt_tpl, seeds, out / "eval" / name,
            [*base, "agents.ugmarl.use_gate=false", *cond_overrides(c)],
            f"{study}: reference (ungated), {cond_name(c)}", log_decisions=True,
        )))  # fmt: skip
    codes = run_parallel(jobs, args.workers, out / "logs")
    if any(codes.values()):
        print(f"error: reference pass failed {codes}", file=sys.stderr)
        return 1

    clean_run = only_run(out / "eval" / f"ref_{cond_name(conds[0])}")
    sig_rows, taus = [], {}
    for s in seeds:
        d = pd.read_csv(clean_run / f"decisions_s{s}.csv")
        sig_rows.append(d.assign(seed=s))
        taus[s] = percentile_thresholds(d["sigma"].to_numpy(), ps)
        for p in ps:
            save_gate(out / "taus" / f"p{p:g}_s{s}.json", {"p": p, "tau": taus[s][p]})
    pd.concat(sig_rows).to_csv(out / "sigma_val.csv", index=False)

    # 2. sweep: gated, every p x condition
    jobs = []
    for p in ps:
        for c in conds:
            name = f"p{p:g}_{cond_name(c)}"
            gate_file = out / "taus" / f"p{p:g}_s{{seed}}.json"
            jobs.append((name, eval_argv(
                args.config, ckpt_tpl, seeds, out / "eval" / name,
                [*base, "agents.ugmarl.use_gate=true", f"agents.ugmarl.gate.file={gate_file}",
                 *cond_overrides(c)],
                f"{study}: p={p:g}, {cond_name(c)}",
            )))  # fmt: skip
    codes = run_parallel(jobs, args.workers, out / "logs")
    if any(codes.values()):
        print(f"error: sweep failed {codes}", file=sys.stderr)
        return 1

    # 3. table + selection
    rows = []
    for p in [None, *ps]:
        for c in conds:
            name = f"ref_{cond_name(c)}" if p is None else f"p{p:g}_{cond_name(c)}"
            m = pd.read_csv(only_run(out / "eval" / name) / "metrics.csv")
            att = summarise(m["avg_travel_time_s"].tolist())
            rows.append({
                "p": "ungated" if p is None else p,
                "condition": cond_name(c),
                "avg_travel_time_mean": att["mean"],
                "avg_travel_time_std": att["std"],
                "avg_travel_time_ci95_low": att["ci95_low"],
                "avg_travel_time_ci95_high": att["ci95_high"],
                "fallback_rate_mean": m["fallback_rate"].mean(),
                "max_waiting_time_mean": m["max_waiting_time_s"].mean(),
                "n_seeds": len(m),
            })  # fmt: skip
    table = pd.DataFrame(rows)
    gated = table[table["p"] != "ungated"]
    scores = (
        gated.groupby("p")["avg_travel_time_mean"].mean().rename("score").reset_index()
    )  # equal weights: every condition has the same seeds
    tol = float(spec["selection"]["tie_tolerance_s"])
    p_star = select_percentile(scores, tol)
    ref_score = float(table[table["p"] == "ungated"]["avg_travel_time_mean"].mean())

    commit = git_commit()
    for s in seeds:
        d = sig_rows[seeds.index(s)]
        save_gate(
            train_dir / f"{variant}_{args.tag}_s{s}" / "gate.json",
            {
                "p": p_star,
                "tau": taus[s][p_star],
                "sigma_percentiles": {f"{p:g}": taus[s][p] for p in ps},
                "n_decisions": int(len(d)),
                "sigma_mean": float(d["sigma"].mean()),
                "study": str(args.search),
                "calibration_dir": str(out.relative_to(REPO_ROOT)),
                "variant": variant,
                "git_commit": commit,
            },
        )
    suffix = "" if variant == "ugmarl" else "_" + variant.split("-", 1)[1]
    DOCS.mkdir(parents=True, exist_ok=True)
    table.to_csv(DOCS / f"gate_calibration{suffix}.csv", index=False)
    scores.assign(selected=scores["p"] == p_star).to_csv(out / "scores.csv", index=False)
    write_md(DOCS / f"gate_calibration{suffix}.md", variant, args, table, scores, p_star,
             ref_score, taus, seeds, tol)  # fmt: skip
    (out / "selection.json").write_text(json.dumps(
        {"p_star": p_star, "scores": dict(zip(scores["p"], scores["score"], strict=True)),
         "ungated_score": ref_score, "taus": {str(s): taus[s] for s in seeds}},
        indent=2, default=float) + "\n")  # fmt: skip
    print(scores.to_string(index=False))
    print(f"ungated reference score {ref_score:.2f} s; selected p* = {p_star:g}")
    print(f"results: {out.relative_to(REPO_ROOT)}")
    return 0


def write_md(path, variant, args, table, scores, p_star, ref_score, taus, seeds, tol) -> None:
    lines = [
        f"# Gate calibration — {variant} (validation)",
        "",
        f"Generated by `scripts/calibrate_gate.py --search {args.search} --config {args.config}"
        f" --tag {args.tag}`{' ' + ' '.join(args.overrides) if args.overrides else ''}."
        " Jinan `real_2000` (validation flow), 3600 s, seeds "
        f"{', '.join(map(str, seeds))} (seed s = model trained with seed s). τ_p = p-th"
        " percentile of the model's σ over every decision of one clean, ungated validation"
        " episode. Travel time: mean ± std [95 % CI, Student t] over seeds. Native Ubuntu."
        " Validation is used for checkpoint, (K, d) and p, so these numbers are optimistic.",
        "",
        "## Selection",
        "",
        f"Score = mean travel time over seeds × conditions (equal weights); ties within {tol:g} s"
        " → smaller p.",
        "",
        "| p | Score (s) |",
        "|---|---|",
        *(
            f"| {r.p:g}{' **(selected)**' if r.p == p_star else ''} | {r.score:.2f} |"
            for r in scores.itertuples()
        ),
        f"| ungated (reference, not a candidate) | {ref_score:.2f} |",
        "",
        "## τ per seed-model",
        "",
        "| Seed | " + " | ".join(f"τ (p = {p:g})" for p in taus[seeds[0]]) + " |",
        "|---|" + "---|" * len(taus[seeds[0]]),
        *(f"| {s} | " + " | ".join(f"{v:.4f}" for v in taus[s].values()) + " |" for s in seeds),
        "",
        "## Per condition",
        "",
        "| p | Condition | Avg travel time (s) | Fallback rate | Max per-vehicle wait (s) |",
        "|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        p = r.p if isinstance(r.p, str) else f"{r.p:g}"
        lines.append(
            f"| {p} | {r.condition} | {r.avg_travel_time_mean:.1f} ± {r.avg_travel_time_std:.1f}"
            f" [{r.avg_travel_time_ci95_low:.1f}, {r.avg_travel_time_ci95_high:.1f}] |"
            f" {r.fallback_rate_mean:.3f} | {r.max_waiting_time_mean:.0f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
