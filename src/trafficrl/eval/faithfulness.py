"""Attention faithfulness check (ROADMAP 8.2, gap analysis A8; plan phase8-analysis A6).

Question: does a neighbour with a larger attention weight α change the network's output more
when its observation is perturbed? Offline counterfactual on recorded encoder inputs
``x`` of shape ``(N, 5, obs_dim + 1)`` (slot 0 = self, 1–4 = neighbours E, N, W, S; last
column = presence flag), one row per decision of one intersection:

* **replacement** — the ``obs_dim`` values of one slot are swapped for the same slot's values
  from another decision of the same intersection (so the input stays in-distribution);
* **knock-out** — the slot is masked as absent (all zeros, presence flag 0).

The full network is run again, so the attention weights are recomputed. Outcomes per
decision: |Δμ| at the original greedy action, max over actions of |Δμ|, whether the greedy
action changes, and Δσ. This is an association check on validation data — not a causal
statement about the deployed, closed-loop system. Nothing is fitted or selected.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from trafficrl.agents.ugmarl.gate import sigma_from_q
from trafficrl.eval.hypotheses import paired

SLOT_NAMES = ("self", "E", "N", "W", "S")
PERTURBATIONS = ("replacement", "knock-out")
OUTCOMES = ("abs_dmu", "max_abs_dmu", "flip", "dsigma")


def outputs(net, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``μ (N, A)``, greedy action ``(N,)``, ``σ (N,)`` and ``α (N, 5)`` for inputs ``x``."""
    with torch.no_grad():
        q, alpha = net(torch.as_tensor(x).unsqueeze(0), return_attention=True)
    mu, a_star, sigma = sigma_from_q(q[0])
    if alpha is None:
        raise ValueError("the faithfulness check needs a network with attention")
    return mu.numpy(), a_star.numpy(), sigma.numpy(), alpha[0].numpy()


def donors(groups: Sequence, draws: int, rng: np.random.Generator) -> np.ndarray:
    """``(N, draws)`` row indices: for each row, other rows of the same group (intersection),
    drawn with replacement. A group with a single row has no donor (``-1``)."""
    groups = np.asarray(groups)
    out = np.full((len(groups), draws), -1, dtype=np.int64)
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        if len(rows) < 2:
            continue
        pick = rng.integers(0, len(rows) - 1, size=(len(rows), draws))
        pick += pick >= np.arange(len(rows))[:, None]  # skip the row itself
        out[rows] = rows[pick]
    return out


def replace_slot(x: np.ndarray, slot: int, donor: np.ndarray) -> np.ndarray:
    """Copy of ``x`` whose ``slot`` observation comes from row ``donor`` (presence kept)."""
    out = x.copy()
    out[:, slot, :-1] = x[donor, slot, :-1]
    return out


def knock_out(x: np.ndarray, slot: int) -> np.ndarray:
    """Copy of ``x`` with ``slot`` made absent (zeros, presence flag 0)."""
    out = x.copy()
    out[:, slot, :] = 0.0
    return out


def effect(net, base: tuple, x_pert: np.ndarray) -> dict[str, np.ndarray]:
    mu, a_star, sigma, _ = base
    mu2, a2, sigma2, _ = outputs(net, x_pert)
    d = np.abs(mu2 - mu)
    rows = np.arange(len(mu))
    return {
        "abs_dmu": d[rows, a_star],
        "max_abs_dmu": d.max(axis=1),
        "flip": (a2 != a_star).astype(float),
        "dsigma": sigma2 - sigma,
    }


def slot_effects(
    net, x: np.ndarray, groups: Sequence, draws: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Long frame, one row per (decision, perturbation, slot): the four outcomes (replacement:
    mean over ``draws`` donors), ``alpha`` of that slot in the unperturbed input and
    ``present``. Self is only replaced (it cannot be absent)."""
    base = outputs(net, x)
    alpha = base[3]
    present = x[:, :, -1] > 0.5
    don = donors(groups, draws, rng)
    usable = (don >= 0).all(axis=1)
    frames = []
    for slot in range(5):
        acc = {k: np.zeros(len(x)) for k in OUTCOMES}
        for r in range(draws):
            e = effect(net, base, replace_slot(x, slot, np.where(usable, don[:, r], 0)))
            for k in OUTCOMES:
                acc[k] += e[k] / draws
        frames.append(_frame("replacement", slot, acc, alpha, present & usable[:, None]))
        if slot > 0:
            frames.append(_frame("knock-out", slot, effect(net, base, knock_out(x, slot)),
                                 alpha, present))  # fmt: skip
    out = pd.concat(frames, ignore_index=True)
    return out[out["present"]].drop(columns="present").reset_index(drop=True)


def _frame(perturbation, slot, eff, alpha, present) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision": np.arange(len(alpha)),
            "perturbation": perturbation,
            "slot": slot,
            "alpha": alpha[:, slot],
            "present": present[:, slot],
            **eff,
        }
    )


def rank_slots(effects: pd.DataFrame) -> pd.DataFrame:
    """Keep, per decision and perturbation, the neighbour with the highest α (``hi``), the one
    with the lowest (``lo``) and the own observation (``self``). Decisions with fewer than two
    present neighbours are dropped."""
    nb = effects[effects["slot"] > 0]
    count = nb.groupby(["perturbation", "decision"])["slot"].transform("size")
    nb = nb[count >= 2]
    key = ["perturbation", "decision"]
    hi = nb.loc[nb.groupby(key)["alpha"].idxmax()].assign(rank="hi")
    lo = nb.loc[nb.groupby(key)["alpha"].idxmin()].assign(rank="lo")
    own = effects[effects["slot"] == 0]
    own = own[own["decision"].isin(set(hi["decision"]))].assign(rank="self")
    return pd.concat([hi, lo, own], ignore_index=True)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def seed_summary(effects: pd.DataFrame) -> pd.DataFrame:
    """Per perturbation and rank (hi / lo / self): mean outcomes, mean α and n decisions; plus
    Spearman ρ between α and |Δμ| over all neighbour slots (column ``spearman``)."""
    ranked = rank_slots(effects)
    out = (
        ranked.groupby(["perturbation", "rank"], sort=False)
        .agg(n=("decision", "size"), alpha=("alpha", "mean"), **{k: (k, "mean") for k in OUTCOMES})
        .reset_index()
        .rename(columns={"rank": "slot"})
    )
    nb = effects[effects["slot"] > 0]
    rho = {p: spearman(g["alpha"].to_numpy(), g["abs_dmu"].to_numpy())
           for p, g in nb.groupby("perturbation")}  # fmt: skip
    out["spearman"] = out["perturbation"].map(rho)
    return out


def verdict(per_seed: pd.DataFrame, perturbation: str = "replacement") -> dict:
    """The reading fixed before the runs: ``consistent`` iff the per-seed paired difference of
    mean |Δμ| (highest-α − lowest-α neighbour) is > 0 with a 95 % CI that excludes 0."""
    d = per_seed[per_seed["perturbation"] == perturbation]
    piv = d.pivot(index="seed", columns="slot", values="abs_dmu").sort_index()
    diff = paired(piv["hi"].tolist(), piv["lo"].tolist())
    ok = bool(diff["mean"] > 0 and diff["ci95_low"] > 0)
    return {
        "perturbation": perturbation,
        "hi_mean": float(piv["hi"].mean()),
        "lo_mean": float(piv["lo"].mean()),
        "self_mean": float(piv["self"].mean()) if "self" in piv else float("nan"),
        "diff": diff,
        "consistent": ok,
        "text": "consistent with the weights" if ok
        else "the attention weights do not rank the neighbours' influence on Q",
    }  # fmt: skip
