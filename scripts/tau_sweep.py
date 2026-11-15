"""Sensitivity of UG-MARL to the gate threshold τ on the validation flow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from trafficrl.agents.ugmarl.gate import gate_file_for, load_gate, percentile_thresholds, save_gate
from trafficrl.data.splits import REPO_ROOT
from trafficrl.eval.report import condition_name
from trafficrl.train.tune import run_parallel
from trafficrl.utils.config import read_yaml
from trafficrl.utils.stats import summarise

EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate.py"
OUT_ROOT = REPO_ROOT / "results" / "analysis"
DOCS = REPO_ROOT / "docs" / "results"
VAL = "val"


class SweepError(ValueError):
    pass


def thresholds(sigma: pd.DataFrame, seeds: list[int], percentiles: list[float]) -> dict:
    """``{seed: {p: τ}}`` from the calibration's σ sample (columns ``seed``, ``sigma``)."""
    out = {}
    for s in seeds:
        v = sigma.loc[sigma["seed"] == s, "sigma"].to_numpy()
        if len(v) == 0:
            raise SweepError(f"no σ values for seed {s} in the σ source")
        out[s] = percentile_thresholds(v, percentiles)
    return out


def check_spec(spec: dict) -> None:
    if spec.get("split", VAL) != VAL:
        raise SweepError("the τ sweep runs on the validation split only")
    known = set(pd.read_csv(REPO_ROOT / spec["existing"]["calibration"])["p"].astype(str))
    clash = [p for p in spec["percentiles"] if f"{float(p)}" in known or f"{p}" in known]
    if clash:
        raise SweepError(f"percentiles {clash} are already in the calibration table")
    if condition_name(dict(spec["conditions"][0] or {})) != "clean":
        raise SweepError("the first condition must be the clean flow")


def tau_file(out: Path, p: float) -> Path:
    """Threshold file template (``{seed}`` filled by evaluate.py); never a ``gate.json``."""
    f = out / "taus" / f"p{p:g}_s{{seed}}.json"
    if f.name == "gate.json" or (REPO_ROOT / "results" / "train") in f.parents:
        raise SweepError("the sweep must not write into a training run directory")
    return f


def eval_argv(config, ckpt, seeds, results_dir, gate_file, cond: dict, note: str) -> list[str]:
    return [
        sys.executable,
        str(EVAL_SCRIPT),
        "--config",
        str(config),
        "--agent",
        "ugmarl",
        "--split",
        VAL,
        "--seeds",
        *map(str, seeds),
        "--checkpoint",
        ckpt,
        "--note",
        note,
        "agents.ugmarl.use_gate=true",
        f"agents.ugmarl.gate.file={gate_file}",
        *(f"{k}={v}" for k, v in cond.items()),
        f"output.results_dir={results_dir}",
    ]


def only_run(d: Path) -> Path:
    runs = sorted(p for p in d.iterdir() if p.is_dir())
    if len(runs) != 1:
        raise RuntimeError(f"expected one evaluation run in {d}, found {len(runs)}")
    return runs[0]


def reading(scores: pd.Series, p_star: float, rule: dict) -> str:
    """The verdict fixed before the runs. ``scores``: index p (numbers only), value = mean
    travel time over seeds × conditions."""
    star = float(scores[p_star])
    lower = scores[[p for p in scores.index if p < p_star]]
    band = scores[[p for p in scores.index if 50 <= p <= 90]]
    parts = []
    if (band - star).abs().max() <= rule["insensitive_within_s"]:
        parts.append(
            f"insensitive: every p in 50–90 scores within {rule['insensitive_within_s']:g} s "
            f"of p* = {p_star:g}"
        )
    best = lower.idxmin() if len(lower) else None
    if best is not None and star - lower[best] > rule["tie_tolerance_s"]:
        parts.append(
            f"a lower threshold scores better (p = {best:g}: {lower[best]:.2f} s vs "
            f"{star:.2f} s at p* = {p_star:g}) — the calibration grid's lower end was binding; "
            "p* stays as locked and the test flow is not re-run"
        )
    elif len(lower) and (lower - star).min() > rule["tie_tolerance_s"]:
        parts.append("lower thresholds score worse: the clean-data cost of more fallback")
    return "; ".join(parts) or "no p below p* differs from it by more than the tie tolerance"


def build_table(spec: dict, new_rows: pd.DataFrame) -> pd.DataFrame:
    cal = pd.read_csv(REPO_ROOT / spec["existing"]["calibration"]).assign(
        source=spec["existing"]["calibration"]
    )
    mp = pd.read_csv(REPO_ROOT / spec["existing"]["max_pressure"])
    mp = mp[mp["model"] == "max_pressure"].drop(columns=["model", "deterministic"])
    mp = mp.assign(p="max_pressure", source=spec["existing"]["max_pressure"])
    mp["fallback_rate_mean"] = 1.0
    cols = list(cal.columns)
    ungated = cal["p"].astype(str) == "ungated"
    return pd.concat([mp[cols], new_rows[cols], cal[~ungated], cal[ungated]], ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--search", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "eval_ugmarl.yaml")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--tables-only",
        action="store_true",
        help="rebuild the table from the finished runs (no simulation)",
    )
    args = ap.parse_args(argv)

    spec = read_yaml(args.search)
    try:
        check_spec(spec)
        seeds = [int(s) for s in spec["seeds"]]
        ps = [float(p) for p in spec["percentiles"]]
        conds = [dict(c or {}) for c in spec["conditions"]]
        variant = spec["variant"]
        train_dir = REPO_ROOT / "results" / "train"
        ckpt_tpl = str(train_dir / f"{variant}_{args.tag}_s{{seed}}" / "checkpoints" / "best.pt")
        sigma = pd.read_csv(REPO_ROOT / spec["sigma_source"])
        cp = float(spec["check_percentile"])
        ref = thresholds(sigma, seeds, [cp])
        for s in seeds:
            locked = load_gate(gate_file_for(Path(ckpt_tpl.replace("{seed}", str(s)))))["tau"]
            if abs(ref[s][cp] - locked) > 1e-12:
                raise SweepError(
                    f"seed {s}: p = {cp:g} of the σ source gives {ref[s][cp]!r}, "
                    f"gate.json holds {locked!r} — not the calibration's σ sample"
                )
        out = OUT_ROOT / spec["study"]
        if out.exists() != args.tables_only:
            raise SweepError(
                f"{out} does not exist: run the sweep first"
                if args.tables_only
                else f"{out} exists (the sweep runs once; --tables-only rebuilds the table)"
            )
        taus = thresholds(sigma, seeds, ps)
        files = {p: tau_file(out, p) for p in ps}
    except SweepError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"check passed: p = {cp:g} reproduces τ of every gate.json")

    jobs = []
    if not args.tables_only:
        for s in seeds:
            for p in ps:
                save_gate(
                    Path(str(files[p]).replace("{seed}", str(s))), {"p": p, "tau": taus[s][p]}
                )
    for p in ps if not args.tables_only else []:
        for c in conds:
            name = f"p{p:g}_{condition_name(c)}"
            note = f"{spec['study']}: p={p:g}, {condition_name(c)}"
            jobs.append(
                (
                    name,
                    eval_argv(args.config, ckpt_tpl, seeds, out / "eval" / name, files[p], c, note),
                )
            )
    codes = run_parallel(jobs, args.workers, out / "logs")
    if any(codes.values()):
        print(f"error: jobs failed {codes}", file=sys.stderr)
        return 1

    rows = []
    for p in ps:
        for c in conds:
            m = pd.read_csv(only_run(out / "eval" / f"p{p:g}_{condition_name(c)}") / "metrics.csv")
            att = summarise(m["avg_travel_time_s"].tolist())
            rows.append(
                {
                    "p": p,
                    "condition": condition_name(c),
                    "avg_travel_time_mean": att["mean"],
                    "avg_travel_time_std": att["std"],
                    "avg_travel_time_ci95_low": att["ci95_low"],
                    "avg_travel_time_ci95_high": att["ci95_high"],
                    "fallback_rate_mean": m["fallback_rate"].mean(),
                    "max_waiting_time_mean": m["max_waiting_time_s"].mean(),
                    "n_seeds": len(m),
                    "source": f"results/analysis/{spec['study']}",
                }
            )
    table = build_table(spec, pd.DataFrame(rows))
    DOCS.mkdir(parents=True, exist_ok=True)
    table.to_csv(DOCS / "tau_sensitivity.csv", index=False)
    write_md(DOCS / "tau_sensitivity.md", spec, table, taus, seeds, ps, args)
    print((DOCS / "tau_sensitivity.md").read_text())
    return 0


def write_md(path, spec, table, taus, seeds, ps, args) -> None:
    p_star = float(spec["p_star"])
    conds = list(dict.fromkeys(table["condition"]))
    score = table.groupby("p", sort=False)["avg_travel_time_mean"].mean()
    numeric = pd.Series(
        {float(p): v for p, v in score.items() if str(p) not in ("ungated", "max_pressure")}
    ).sort_index()
    verdict = reading(numeric, p_star, spec["reading"])

    def name(p) -> str:
        if str(p) == "max_pressure":
            return "Max-Pressure (τ = 0)"
        if str(p) == "ungated":
            return "ungated (τ = ∞)"
        new = " (new)" if float(p) in ps else ""
        return f"{float(p):g}{' **(p\\*, locked)**' if float(p) == p_star else new}"

    lines = [
        "# Sensitivity to the gate threshold τ — ugmarl (validation)",
        "",
        f"Generated by `scripts/tau_sweep.py --search {args.search} --tag {args.tag}`. Jinan "
        f"`real_2000` (validation flow), 3600 s, seeds {', '.join(map(str, seeds))}. τ_p = p-th "
        "percentile of each seed-model's σ on its clean, ungated validation episode (the "
        "calibration's σ sample). **Sensitivity check made after the final evaluation: nothing "
        "is selected, p\\* = 80 and every `gate.json` are unchanged, the test flow is not "
        f"re-run.** New rows: p = {', '.join(f'{p:g}' for p in ps)}; the other rows are copied "
        "from `gate_calibration.csv` and `val_robustness.csv`. Validation was used for the "
        "checkpoint, (K, d) and p\\*, so these numbers are optimistic.",
        "",
        "## Score (mean travel time over seeds × the five conditions, s)",
        "",
        "| p | Score (s) | Difference to p\\* (s) | Mean fallback rate |",
        "|---|---|---|---|",
    ]
    fb = table.groupby("p", sort=False)["fallback_rate_mean"].mean()
    star = float(numeric[p_star])
    for p, v in score.items():
        lines.append(f"| {name(p)} | {v:.2f} | {v - star:+.2f} | {fb[p]:.3f} |")
    lines += ["", f"**Reading (rule fixed before the runs):** {verdict}.", ""]
    lines += ["## Travel time per condition (s, mean ± std over seeds)", ""]
    lines += ["| p | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for p in score.index:
        g = table[table["p"].astype(str) == str(p)].set_index("condition")
        cells = [
            f"{g.loc[c, 'avg_travel_time_mean']:.1f} ± {g.loc[c, 'avg_travel_time_std']:.1f}"
            for c in conds
        ]
        lines.append(f"| {name(p)} | " + " | ".join(cells) + " |")
    lines += ["", "## Fallback rate and maximum per-vehicle wait (s)", ""]
    lines += ["| p | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for p in score.index:
        g = table[table["p"].astype(str) == str(p)].set_index("condition")
        cells = [
            f"{g.loc[c, 'fallback_rate_mean']:.2f} / {g.loc[c, 'max_waiting_time_mean']:.0f}"
            for c in conds
        ]
        lines.append(f"| {name(p)} | " + " | ".join(cells) + " |")
    lines += ["", "## New thresholds per seed-model", ""]
    lines += ["| Seed | " + " | ".join(f"τ (p = {p:g})" for p in ps) + " |"]
    lines += ["|---|" + "---|" * len(ps)]
    for s in seeds:
        lines.append(f"| {s} | " + " | ".join(f"{taus[s][p]:.4f}" for p in ps) + " |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
