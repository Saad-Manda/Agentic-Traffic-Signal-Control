"""Attention faithfulness check on validation (ROADMAP 8.2, gap analysis A8; plan A6).

    python scripts/attention_faithfulness.py [--tag final] [--seeds 0 1 2 3 4] [--draws 8]

For each seed-model ``results/train/ugmarl_<tag>_s<seed>``: one clean Jinan **validation**
episode with UG-MARL as deployed (gate on, τ from ``gate.json``); the encoder input of every
decision is recorded. Offline, one slot of each input is then perturbed
(:mod:`trafficrl.eval.faithfulness`) — the neighbour with the highest attention weight, the
one with the lowest, and the own observation — and the change in the network's output is
measured. No closed-loop re-simulation, nothing fitted or selected; any split other than
``val`` is refused. The episode must reproduce the validation travel time of the Phase 6
table, otherwise the script stops.

Output: ``docs/results/attention_faithfulness.{csv,md}`` (per seed and summary with the
reading fixed in the ROADMAP Decision Log, 13 Nov), raw per-decision effects under
``results/analysis/attention_faithfulness/``; one experiment-log row per seed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trafficrl.agents import make_agent
from trafficrl.data.splits import REPO_ROOT
from trafficrl.envs.traffic_env import make_env
from trafficrl.eval import faithfulness as Fa
from trafficrl.eval.report import fmt
from trafficrl.eval.runner import PRIMARY_METRIC, resolve_gate
from trafficrl.utils.config import load_config
from trafficrl.utils.runlog import append_experiment_row, save_env_info
from trafficrl.utils.seed import seed_everything
from trafficrl.utils.stats import summarise

CONFIG = REPO_ROOT / "configs" / "eval_ugmarl.yaml"
TRAIN = REPO_ROOT / "results" / "train"
OUT = REPO_ROOT / "results" / "analysis" / "attention_faithfulness"
DOCS = REPO_ROOT / "docs" / "results"
REFERENCE = REPO_ROOT / "docs" / "results" / "val_phase6.csv"
VAL = "val"


def record_episode(cfg, seed: int, ckpt: Path):
    """Play one gated greedy episode; return the encoder inputs of every decision
    ``(N, 5, obs_dim + 1)``, the intersection of each row, and the episode metrics."""
    seed_everything(seed, cfg.get("deterministic", True))
    env = make_env({**cfg, "seed": seed}, seed, track="full")
    agent = make_agent("ugmarl", env, {**cfg, "seed": seed})
    agent.load(ckpt)
    agent.tau, _ = resolve_gate({**cfg, "seed": seed}, ckpt, seed)
    agent.training = False
    pos = {iid: k for k, iid in enumerate(agent.intersection_ids)}
    states, iids = [], []
    obs, info = env.reset()
    done = False
    while not done:
        actions = agent.act(obs, info)
        x = agent.encoder.encode_all(obs)
        for iid in info["ready"]:
            states.append(x[pos[iid]])
            iids.append(iid)
        obs, _, done, info = env.step(actions)
    return agent, np.stack(states), np.asarray(iids), env.metrics()


def reference_travel_times(seeds: list[int]) -> dict[int, float]:
    """Per-seed validation travel time of gated UG-MARL from the Phase 6 evaluation run."""
    ref = pd.read_csv(REFERENCE).set_index("model")
    run = REPO_ROOT / "results" / "eval" / ref.loc["ugmarl", "eval_run"] / "metrics.csv"
    m = pd.read_csv(run).set_index("seed")[PRIMARY_METRIC]
    return {s: float(m[s]) for s in seeds}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--draws", type=int, default=8, help="replacement donors per decision")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    if cfg["data"]["split"] != VAL:
        print("error: the faithfulness check runs on the validation split only", file=sys.stderr)
        return 2
    rob = cfg.get("robustness", {}) or {}
    if rob.get("noise_std", 0) or rob.get("dropout_p", 0) or rob.get("surge", 1.0) != 1.0:
        print("error: the faithfulness check uses the clean validation flow", file=sys.stderr)
        return 2
    if OUT.exists():
        print(f"error: {OUT} exists (the check runs once)", file=sys.stderr)
        return 2
    try:
        expected = reference_travel_times(args.seeds)
    except (FileNotFoundError, KeyError) as e:
        print(f"error: no Phase 6 validation reference ({e})", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True)
    save_env_info(OUT / "env.json")

    per_seed = []
    for s in args.seeds:
        t0 = time.perf_counter()
        ckpt = TRAIN / f"ugmarl_{args.tag}_s{s}" / "checkpoints" / "best.pt"
        agent, x, iids, m = record_episode(cfg, s, ckpt)
        att = float(m[PRIMARY_METRIC])
        if abs(att - expected[s]) > 1e-3:
            print(f"error: seed {s}: travel time {att} != Phase 6 value {expected[s]}",
                  file=sys.stderr)  # fmt: skip
            return 1
        eff = Fa.slot_effects(agent.net, x, iids, args.draws, np.random.default_rng(s))
        eff.insert(0, "iid", iids[eff["decision"].to_numpy()])
        eff.insert(0, "seed", s)
        eff.to_csv(OUT / f"effects_s{s}.csv", index=False)
        summ = Fa.seed_summary(eff)
        summ.insert(0, "seed", s)
        summ["travel_time_s"] = att
        per_seed.append(summ)
        hi = summ[(summ["perturbation"] == "replacement") & (summ["slot"] == "hi")].iloc[0]
        append_experiment_row(
            run_id=f"attention_faithfulness_ugmarl_{args.tag}_s{s}",
            split=VAL,
            seed=s,
            model="ugmarl",
            hyperparameters={"checkpoint": str(ckpt.relative_to(REPO_ROOT)), "gate": "on",
                             "draws": args.draws, "decisions": int(len(x))},
            metric_name="abs_dmu_highest_alpha_neighbour_replaced",
            metric_value=float(hi["abs_dmu"]),
            validation_metric=att,
            runtime_s=time.perf_counter() - t0,
            notes="attention_faithfulness: seed " + str(s) + " (plan phase8 A6); clean val, "
            "offline slot perturbation, nothing selected",
        )  # fmt: skip
        print(f"seed {s}: {len(x)} decisions, travel time {att:.2f} s (reproduced)")
    table = pd.concat(per_seed, ignore_index=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    table.to_csv(DOCS / "attention_faithfulness.csv", index=False)
    write_md(DOCS / "attention_faithfulness.md", table, args)
    print((DOCS / "attention_faithfulness.md").read_text())
    return 0


def write_md(path: Path, table: pd.DataFrame, args) -> None:
    names = {"hi": "highest-α neighbour", "lo": "lowest-α neighbour", "self": "own observation"}
    lines = [
        "# Attention faithfulness check — ugmarl (validation)",
        "",
        f"Generated by `scripts/attention_faithfulness.py --tag {args.tag}`. Jinan `real_2000` "
        f"(validation flow), clean detectors, seeds {', '.join(map(str, args.seeds))} (seed s = "
        "model trained with seed s), UG-MARL as deployed (gate on); each episode reproduces "
        "the Phase 6 validation travel time. For every decision with at least two neighbours, "
        "one input slot is perturbed and the network is run again (attention recomputed): "
        f"**replacement** = the slot's 44 values are swapped for the same slot's values from "
        f"another decision of the same intersection in the same episode (mean of {args.draws} "
        "draws); **knock-out** = the slot is masked as absent. μ is the ensemble-mean Q in "
        "scaled reward units. An association check on validation data, not a causal claim; "
        "nothing is fitted or selected.",
        "",
    ]
    for p in Fa.PERTURBATIONS:
        v = Fa.verdict(table, p)
        d = table[table["perturbation"] == p]
        lines += [f"## {p.capitalize()}", ""]
        lines += [
            "| Slot perturbed | Mean α | Mean \\|Δμ\\| at the chosen action | Max over actions "
            "of \\|Δμ\\| | Greedy action changes | Mean Δσ |",
            "|---|---|---|---|---|---|",
        ]
        for slot in ("hi", "lo", "self"):
            g = d[d["slot"] == slot]
            if g.empty:
                continue
            lines.append(
                f"| {names[slot]} | {g['alpha'].mean():.3f} | "
                f"{fmt(summarise(g['abs_dmu'].tolist()), 4)} | "
                f"{fmt(summarise(g['max_abs_dmu'].tolist()), 4)} | "
                f"{fmt(summarise(g['flip'].tolist()), 3)} | {g['dsigma'].mean():+.5f} |"
            )
        rho = d.drop_duplicates("seed")["spearman"]
        lines += [
            "",
            f"Paired difference highest-α − lowest-α, mean \\|Δμ\\| over seeds: "
            f"{fmt(v['diff'], 4)}. Spearman ρ(α, \\|Δμ\\|) over all neighbour slots, per seed: "
            f"{', '.join(f'{r:.2f}' for r in rho)} (mean {rho.mean():.2f}). Decisions per seed: "
            f"{', '.join(str(int(n)) for n in d[d['slot'] == 'hi']['n'])}.",
            "",
            f"**Reading{' (rule fixed before the runs)' if p == 'replacement' else ''}:** "
            f"{v['text']}"
            + ("." if p == "replacement" else " (second variant, reported for completeness)."),
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
