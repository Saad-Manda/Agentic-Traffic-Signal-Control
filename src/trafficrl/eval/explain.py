"""Per-decision explanation log (ROADMAP 6.4; plan phase6-ugmarl D7).

:class:`DecisionRecorder` turns ``agent.last_explanations()`` into one row per **ready**
intersection per decision: σ, τ, whether the gate fired, the learned and the Max-Pressure
choice, the phase taken, the five attention weights (self, E, N, W, S), the gap between the
best and second-best mean Q, and the observed vehicle count on the incoming lanes.
``scripts/evaluate.py --log-decisions`` writes ``decisions_s<seed>.csv``; the Phase 8
figures (attention heatmaps, gate-firing analysis) read these files. Attention weights and σ
describe what the model computed — associations, not causes.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SLOTS = ("self", "E", "N", "W", "S")
COLUMNS = (
    "t",
    "iid",
    "sigma",
    "tau",
    "fallback",
    "greedy_phase",
    "mp_phase",
    "phase_taken",
    *(f"attn_{s}" for s in SLOTS),
    "q_gap",
    "in_veh_observed",
)


class DecisionRecorder:
    def __init__(self, env: Any) -> None:
        inter = env.features.roadnet.intersections
        self._in_lanes = {iid: inter[iid].in_lanes for iid in env.intersection_ids}
        self.rows: list[dict[str, Any]] = []

    def record(
        self,
        t: int,
        info: Mapping[str, Any],
        actions: Mapping[str, int],
        expl: Mapping[str, Any],
    ) -> None:
        nan = float("nan")
        counts = info.get("lane_count", {})
        att = expl.get("attention") or {}
        for iid in info["ready"]:
            q = expl.get("q_mean", {}).get(iid)
            if q is not None and len(q) > 1:
                top = np.sort(np.asarray(q, dtype=float))[-2:]
                gap = float(top[1] - top[0])
            else:
                gap = nan
            a = att.get(iid)
            row = {
                "t": int(t),
                "iid": iid,
                "sigma": expl.get("sigma", {}).get(iid, nan),
                "tau": nan if expl.get("tau") is None else float(expl["tau"]),
                "fallback": bool(expl.get("fallback", {}).get(iid, False)),
                "greedy_phase": expl.get("greedy_phase", {}).get(iid),
                "mp_phase": expl.get("mp_phase", {}).get(iid),
                "phase_taken": int(actions[iid]),
                **{f"attn_{s}": (nan if a is None else float(a[k])) for k, s in enumerate(SLOTS)},
                "q_gap": gap,
                "in_veh_observed": float(sum(counts.get(ln, 0) for ln in self._in_lanes[iid])),
            }
            self.rows.append(row)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=list(COLUMNS))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame().to_csv(path, index=False)
        return path
