"""Hypothesis readings of Phase 7, fixed before the test flow was opened.

Every function takes per-seed values (lower is better for travel time and waiting time) and
returns plain dicts, so the verdicts in ``scripts/final_tables.py`` are computed, not judged
by eye. 95 % CI = Student t over seeds (:func:`trafficrl.utils.stats.summarise`); a
deterministic baseline's CI is its value.

* **H1** — UG-MARL's CI upper end below Max-Pressure's value and below IDQN's CI lower end.
* **H3** — UG-MARL's CI upper end below −attention's CI lower end.
* **H2** — gate vs −gate (same checkpoints, so seeds are paired): (i) mean max wait on the
  clean flow, (ii) mean max wait at noise 0.2, (iii) travel-time increase clean → noise 0.2;
  *supported* = all three favour the gate, *partly supported* = (ii) and (iii) only.
* **Minimum outcome** — UG-MARL's mean max wait ≤ Max-Pressure's.
* **Regret** — a model's mean travel time minus the lowest mean among the reference agents
  in the same condition (descriptive, no threshold).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from trafficrl.utils.stats import summarise

SUPPORTED, PARTLY, NOT_SUPPORTED = "supported", "partly supported", "not supported"


def ci_below(a: Sequence[float], b: Sequence[float]) -> bool:
    """True iff the whole 95 % CI of ``a`` lies below the whole 95 % CI of ``b``."""
    return bool(summarise(a)["ci95_high"] < summarise(b)["ci95_low"])


def compare(a: Sequence[float], b: Sequence[float]) -> str:
    """``a`` against ``b`` by CI overlap: ``better`` (lower), ``worse`` or ``level``."""
    if ci_below(a, b):
        return "better"
    if ci_below(b, a):
        return "worse"
    return "level"


def paired(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Summary of the per-seed differences ``a − b`` (same seeds, same order)."""
    if len(a) != len(b):
        raise ValueError(f"paired comparison needs equal lengths, got {len(a)} and {len(b)}")
    return summarise([float(x) - float(y) for x, y in zip(a, b, strict=True)])


def h1(ugmarl: Sequence[float], idqn: Sequence[float], max_pressure: Sequence[float]) -> dict:
    vs_mp, vs_idqn = ci_below(ugmarl, max_pressure), ci_below(ugmarl, idqn)
    return {
        "below_max_pressure": vs_mp,
        "below_idqn": vs_idqn,
        "verdict": SUPPORTED if vs_mp and vs_idqn else NOT_SUPPORTED,
    }


def h3(ugmarl: Sequence[float], no_attention: Sequence[float]) -> dict:
    ok = ci_below(ugmarl, no_attention)
    return {"below_no_attention": ok, "verdict": SUPPORTED if ok else NOT_SUPPORTED}


def _mean(x: Sequence[float]) -> float:
    return float(sum(float(v) for v in x) / len(x))


def h2(gate: Mapping[str, Sequence[float]], no_gate: Mapping[str, Sequence[float]]) -> dict:
    """``gate`` / ``no_gate``: per-seed ``wait_clean``, ``wait_noise``, ``tt_clean``,
    ``tt_noise`` (max per-vehicle wait and average travel time, clean flow and noise 0.2)."""
    i = _mean(gate["wait_clean"]) < _mean(no_gate["wait_clean"])
    ii = _mean(gate["wait_noise"]) < _mean(no_gate["wait_noise"])
    inc_gate = [n - c for n, c in zip(gate["tt_noise"], gate["tt_clean"], strict=True)]
    inc_no = [n - c for n, c in zip(no_gate["tt_noise"], no_gate["tt_clean"], strict=True)]
    iii = _mean(inc_gate) < _mean(inc_no)
    if i and ii and iii:
        verdict = SUPPORTED
    elif ii and iii:
        verdict = PARTLY
    else:
        verdict = NOT_SUPPORTED
    return {
        "i_max_wait_clean_lower": i,
        "ii_max_wait_noise_lower": ii,
        "iii_degradation_smaller": iii,
        "verdict": verdict,
        "increase_gate": _mean(inc_gate),
        "increase_no_gate": _mean(inc_no),
        "paired_wait_clean": paired(gate["wait_clean"], no_gate["wait_clean"]),
        "paired_wait_noise": paired(gate["wait_noise"], no_gate["wait_noise"]),
        "paired_increase": paired(inc_gate, inc_no),
    }


def minimum_outcome(ugmarl_wait: Sequence[float], max_pressure_wait: Sequence[float]) -> dict:
    u, m = _mean(ugmarl_wait), _mean(max_pressure_wait)
    return {"ugmarl": u, "max_pressure": m, "met": u <= m}


def regret_table(means: pd.DataFrame, reference: Sequence[str]) -> pd.DataFrame:
    """``means``: columns ``model``, ``condition``, ``mean``. Adds ``regret`` = mean − the
    lowest mean among the ``reference`` models in that condition; one row per (model,
    condition). Use :func:`regret_summary` for the largest / mean regret per model."""
    best = means[means["model"].isin(list(reference))].groupby("condition")["mean"].min()
    out = means.copy()
    out["regret"] = out["mean"] - out["condition"].map(best)
    return out


def regret_summary(regret: pd.DataFrame) -> pd.DataFrame:
    """Per model: largest regret, the condition where it occurs, and the mean regret."""
    rows = []
    for model, g in regret.groupby("model", sort=False):
        worst = g.loc[g["regret"].idxmax()]
        rows.append(
            {
                "model": model,
                "max_regret": float(worst["regret"]),
                "worst_condition": worst["condition"],
                "mean_regret": float(g["regret"].mean()),
            }
        )
    return pd.DataFrame(rows)
