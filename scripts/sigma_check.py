"""σ validity check for the UG-MARL gate signal, validation only (plan phase6-ugmarl D10).

    python scripts/sigma_check.py --tag final [overrides, e.g. agents.ugmarl.use_attention=false]

Two questions, both on the Jinan **validation** flow with the selected seed-models:

1. **Does σ rise when the inputs degrade?** σ of every decision of the ungated reference
   episodes of ``scripts/calibrate_gate.py`` (clean, noise 0.1 / 0.2, dropout 10 / 20 %) —
   no new simulation. Pooled over seeds as σ / (median clean σ of the same model), because
   σ's scale differs between models.
2. **Is σ larger where the learned values are wrong?** One clean, greedy, ungated validation
   episode per model collects transitions (``agent.training = False``: nothing enters
   replay). For each: |δ| = |μ(s, a) − (c·r + γ^(d/Δt) μ_target(s′, argmax_a′ μ(s′, a′)))|,
   the TD error of the ensemble-mean Q, and Spearman's ρ between σ(s) and |δ|.

Writes ``docs/results/sigma_validity.csv`` (+ ``_td.csv``) and
``docs/figures/ugmarl/sigma_validity.{png,md}``. These are associations, not causes: σ is
an ensemble disagreement, not a calibrated error probability.
"""

from __future__ import annotations

import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from trafficrl.agents import make_agent, model_label
from trafficrl.agents.ugmarl.gate import load_gate, sigma_from_q
from trafficrl.data.splits import REPO_ROOT
from trafficrl.envs.traffic_env import make_env
from trafficrl.eval.runner import PRIMARY_METRIC
from trafficrl.utils.config import load_config
from trafficrl.utils.plotting import BLUE, BLUE_RAMP, INK_SECONDARY, apply_style, save_figure
from trafficrl.utils.runlog import append_experiment_row, save_env_info
from trafficrl.utils.seed import seed_everything

CONFIG = REPO_ROOT / "configs" / "eval_ugmarl.yaml"
TUNE = REPO_ROOT / "results" / "tune"
TRAIN = REPO_ROOT / "results" / "train"
DOCS = REPO_ROOT / "docs"
CONDITIONS = ["clean", "noise0.1", "noise0.2", "drop0.1", "drop0.2"]
CONDITION_LABEL = {
    "clean": "clean",
    "noise0.1": "noise 0.1",
    "noise0.2": "noise 0.2",
    "drop0.1": "dropout 10 %",
    "drop0.2": "dropout 20 %",
}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's ρ = Pearson correlation of the ranks (average ranks for ties)."""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def td_episode(cfg, seed: int, ckpt) -> pd.DataFrame:
    """σ(s) and |TD error of μ| for every transition of one clean greedy val episode."""
    seed_everything(seed, cfg.get("deterministic", True))
    env = make_env({**cfg, "seed": seed}, seed, track="light")
    agent = make_agent("ugmarl", env, {**cfg, "seed": seed})
    agent.load(ckpt)
    agent.tau, agent.training = None, False  # ungated; nothing is stored
    p = agent.params
    obs, info = env.reset()
    rows, done = [], False
    while not done:
        actions = agent.act(obs, info)
        obs, _, done, info = env.step(actions)
        for tr in info["transitions"]:
            s, s2 = agent.encoder.encode_transition(tr)
            x = torch.as_tensor(np.stack([s, s2])).unsqueeze(0)
            with torch.no_grad():
                q = agent.net(x)[0]  # (2, K, A)
                qt = agent.target(x)[0]
            mu, a_next, sigma = sigma_from_q(q)
            a = agent.phase_ids.index(int(tr.action))
            boot = qt[1].mean(dim=0)[a_next[1]]
            y = float(p["reward_scale"]) * tr.reward + float(p["gamma"]) ** (
                tr.duration_s / agent.delta_t
            ) * float(boot)
            rows.append(
                {
                    "seed": seed,
                    "t": info["t"],
                    "iid": tr.iid,
                    "sigma": float(sigma[0]),
                    "td_abs": abs(float(mu[0, a]) - y),
                }
            )
    m = env.metrics()
    return pd.DataFrame(rows), m[PRIMARY_METRIC]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="final")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(CONFIG, args.overrides)
    if cfg["data"]["split"] != "val":
        print("error: the σ check uses the validation split only", file=sys.stderr)
        return 2
    variant = model_label("ugmarl", cfg)
    calib = TUNE / f"ugmarl_gate_{variant}_{args.tag}" / "eval"
    if not calib.exists():
        print(f"error: run scripts/calibrate_gate.py first ({calib} missing)", file=sys.stderr)
        return 2

    # 1. σ under degraded inputs (reference decision logs, ungated)
    frames = []
    for cond in CONDITIONS:
        run = next(p for p in (calib / f"ref_{cond}").iterdir() if p.is_dir())
        for s in args.seeds:
            d = pd.read_csv(run / f"decisions_s{s}.csv")
            frames.append(d[["sigma", "fallback"]].assign(seed=s, condition=cond))
    dec = pd.concat(frames, ignore_index=True)
    base = dec[dec["condition"] == "clean"].groupby("seed")["sigma"].median()
    dec["sigma_rel"] = dec["sigma"] / dec["seed"].map(base)
    taus = {
        s: load_gate(TRAIN / f"{variant}_{args.tag}_s{s}" / "gate.json")["tau"] for s in args.seeds
    }
    dec["above_tau"] = dec["sigma"] > dec["seed"].map(taus)
    rows = []
    for cond in CONDITIONS:
        g = dec[dec["condition"] == cond]
        per_seed = g.groupby("seed")
        rows.append(
            {
                "condition": cond,
                "median_sigma_rel": float(g["sigma_rel"].median()),
                "mean_sigma_rel": float(g["sigma_rel"].mean()),
                "p90_sigma_rel": float(g["sigma_rel"].quantile(0.9)),
                "share_above_tau_mean": float(per_seed["above_tau"].mean().mean()),
                "share_above_tau_std": float(per_seed["above_tau"].mean().std(ddof=1)),
                "median_sigma_rel_by_seed": " ".join(
                    f"{v:.3f}" for v in per_seed["sigma_rel"].median()
                ),
                "n_decisions": len(g),
            }
        )
    table = pd.DataFrame(rows)

    # 2. σ vs |TD error| on clean validation transitions (one new episode per model)
    save_env_info(REPO_ROOT / "results" / "tune" / f"sigma_check_{variant}_{args.tag}_env.json")
    tds, rhos = [], []
    for s in args.seeds:
        ckpt = TRAIN / f"{variant}_{args.tag}_s{s}" / "checkpoints" / "best.pt"
        t0 = time.perf_counter()
        td, att = td_episode(cfg, s, ckpt)
        rho = spearman(td["sigma"].to_numpy(), td["td_abs"].to_numpy())
        tds.append(td)
        rhos.append(rho)
        append_experiment_row(
            run_id=f"sigma_check_{variant}_{args.tag}_s{s}",
            split="val",
            seed=s,
            model=variant,
            hyperparameters={"checkpoint": str(ckpt.relative_to(REPO_ROOT)), "gate": "off"},
            metric_name="spearman_sigma_vs_abs_td",
            metric_value=rho,
            validation_metric=att,
            runtime_s=time.perf_counter() - t0,
            notes=f"σ validity check (plan phase6 D10); ungated clean val, {len(td)} transitions",
        )
    td_all = pd.concat(tds, ignore_index=True)
    td_all["decile"] = td_all.groupby("seed")["sigma"].transform(
        lambda v: pd.qcut(v.rank(method="first"), 10, labels=False) + 1
    )
    dec_tab = td_all.groupby(["seed", "decile"])["td_abs"].mean().unstack("seed")
    (DOCS / "results").mkdir(parents=True, exist_ok=True)
    table.assign(variant=variant).to_csv(DOCS / "results" / "sigma_validity.csv", index=False)
    rho_tab = pd.DataFrame({"seed": args.seeds, "spearman_sigma_vs_abs_td": rhos})
    rho_tab.assign(n=[len(t) for t in tds]).to_csv(
        DOCS / "results" / "sigma_validity_td.csv", index=False
    )

    # figure
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    data = [dec.loc[dec["condition"] == c, "sigma_rel"].to_numpy() for c in CONDITIONS]
    bp = ax1.boxplot(data, showfliers=False, patch_artist=True, widths=0.55)
    for patch in bp["boxes"]:
        patch.set(facecolor=BLUE_RAMP[1], edgecolor=INK_SECONDARY)
    for med in bp["medians"]:
        med.set(color=INK_SECONDARY, linewidth=2)
    ax1.set_xticks(range(1, len(CONDITIONS) + 1), [CONDITION_LABEL[c] for c in CONDITIONS])
    ax1.axhline(1.0, color=INK_SECONDARY, linestyle=":", linewidth=1)
    ax1.set_ylabel("σ / median clean σ of the same model (–)")
    ax1.set_xlabel("Observation condition (validation flow)")
    ax1.set_title("(a) σ under degraded detectors", fontsize=10)
    x = dec_tab.index.to_numpy()
    for s in dec_tab.columns:
        ax2.plot(x, dec_tab[s], color=BLUE_RAMP[2], linewidth=1, alpha=0.7)
    ax2.plot(x, dec_tab.mean(axis=1), color=BLUE, linewidth=2.5, marker="o", label="mean of seeds")
    ax2.set_xlabel("σ decile within the model (1 = lowest σ)")
    ax2.set_ylabel("Mean |TD error of μ| (scaled reward units)")
    ax2.set_title("(b) TD error by σ decile, clean validation", fontsize=10)
    ax2.legend(frameon=False, fontsize=8)
    fig.suptitle(f"σ validity check — {variant}, seeds {args.seeds}", fontweight="bold")
    fig.tight_layout()
    out = DOCS / "figures" / "ugmarl" / "sigma_validity.png"
    save_figure(fig, out)

    rel = dict(zip(table["condition"], table["median_sigma_rel"], strict=True))
    share = dict(zip(table["condition"], table["share_above_tau_mean"], strict=True))
    ratio = dec_tab.mean(axis=1)
    rho_m, rho_s = float(np.mean(rhos)), float(np.std(rhos, ddof=1)) if len(rhos) > 1 else 0.0
    caption = (
        f"# Figure — σ validity check ({variant}, validation)\n\n"
        f"![sigma validity]({out.name})\n\n"
        "**Caption.** (a) Distribution of σ (ensemble standard deviation of Q at the greedy "
        "action) over every decision of one ungated greedy episode per model on the Jinan "
        "validation flow, under clean and degraded detectors; σ is divided by the median clean "
        "σ of the same model so the five seed-models can be pooled (dotted line = 1). Boxes: "
        "quartiles, whiskers 1.5 IQR, outliers hidden. (b) Mean |TD error of the ensemble-mean "
        "Q| per σ decile on the transitions of one clean, ungated validation episode per model "
        "(thin lines: seeds; thick: mean). Generated by `scripts/sigma_check.py --tag "
        f"{args.tag}`; native Ubuntu.\n\n"
        "**Values.** Median σ / clean median: "
        + "; ".join(f"{CONDITION_LABEL[c]} {rel[c]:.2f}" for c in CONDITIONS)
        + ". Share of decisions with σ > τ (p* threshold, ungated trajectories): "
        + "; ".join(f"{CONDITION_LABEL[c]} {share[c]:.3f}" for c in CONDITIONS)
        + f". Spearman ρ(σ, |TD|) per seed: {', '.join(f'{r:.2f}' for r in rhos)} "
        f"(mean {rho_m:.2f} ± {rho_s:.2f}); mean |TD| in the top σ decile / bottom decile = "
        f"{ratio.iloc[-1] / ratio.iloc[0]:.2f}.\n\n"
        "**Interpretation.** See worklog/017 (Result). These are associations on validation "
        "data, not causes; σ is a disagreement between heads that share an encoder, not a "
        "calibrated probability of a wrong action.\n"
    )
    out.with_suffix(".md").write_text(caption)
    print(table.to_string(index=False))
    print(rho_tab.to_string(index=False))
    print(caption)
    return 0


if __name__ == "__main__":
    sys.exit(main())
