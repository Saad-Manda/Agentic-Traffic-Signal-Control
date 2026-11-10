"""Robustness hooks (ROADMAP 4.3; definitions frozen in docs/evaluation_protocol.md §6).

* **Sensor noise** — Gaussian with standard deviation σ added to every capacity-scaled lane
  reading x̃ = x / C_l, clipped to [0, 1], reported back in vehicles (x̃′ · C_l).
* **Detector dropout** — each lane detector independently reports 0 (count and waiting) with
  probability p at each reading.
* **Demand surge** — each vehicle of the flow duplicated with probability ``surge − 1``
  (same route, start time + 1 s); the derived flow file lives in the run's work directory.

Perturbations change what **every** controller observes (Max-Pressure reads the same
detectors); the reward and the metrics always use the true simulator state.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import numpy as np


class SensorModel:
    def __init__(
        self,
        capacity: Mapping[str, int],
        noise_std: float = 0.0,
        dropout_p: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")
        if not 0 <= dropout_p <= 1:
            raise ValueError(f"dropout_p must be in [0, 1], got {dropout_p}")
        if (noise_std or dropout_p) and rng is None:
            raise ValueError("a seeded rng is required when noise or dropout is on")
        self.capacity = dict(capacity)
        self.noise_std = float(noise_std)
        self.dropout_p = float(dropout_p)
        self.rng = rng

    @property
    def active(self) -> bool:
        return self.noise_std > 0 or self.dropout_p > 0

    def observe(
        self, lane_count: Mapping[str, float], lane_waiting: Mapping[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Observed ``(lane_count, lane_waiting)``; identity when the model is inactive."""
        if not self.active:
            return dict(lane_count), dict(lane_waiting)
        lanes = list(lane_count)
        cap = np.array([self.capacity[ln] for ln in lanes], dtype=float)
        count = np.array([lane_count[ln] for ln in lanes], dtype=float)
        wait = np.array([lane_waiting.get(ln, 0) for ln in lanes], dtype=float)
        if self.noise_std > 0:
            count = self._noisy(count, cap)
            wait = self._noisy(wait, cap)
        if self.dropout_p > 0:
            dead = self.rng.random(len(lanes)) < self.dropout_p
            count[dead] = 0.0
            wait[dead] = 0.0
        return dict(zip(lanes, count.tolist(), strict=True)), dict(
            zip(lanes, wait.tolist(), strict=True)
        )

    def _noisy(self, x: np.ndarray, cap: np.ndarray) -> np.ndarray:
        scaled = np.clip(x / cap + self.rng.normal(0.0, self.noise_std, x.shape), 0.0, 1.0)
        return scaled * cap


def surge_flow(
    roadnet_path: Path, flow_path: Path, out_dir: Path, surge: float, rng: np.random.Generator
) -> tuple[Path, Path, int]:
    """Write a surged copy of ``flow_path`` (and the roadnet) into ``out_dir``.

    Returns ``(roadnet copy, surged flow, vehicles added)``. ``surge`` is the expected demand
    multiplier, in [1, 2].
    """
    if not 1.0 <= surge <= 2.0:
        raise ValueError(f"surge must be in [1, 2], got {surge}")
    rate = surge - 1.0
    flows = json.loads(Path(flow_path).read_text())
    out: list[dict] = []
    added = 0
    for v in flows:
        out.append(v)
        if rng.random() < rate:
            dup = json.loads(json.dumps(v))
            dup["startTime"] = v["startTime"] + 1
            dup["endTime"] = v["endTime"] + 1
            out.append(dup)
            added += 1
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roadnet_copy = out_dir / Path(roadnet_path).name
    shutil.copyfile(roadnet_path, roadnet_copy)
    surged = out_dir / f"surge{surge:g}_{Path(flow_path).name}"
    surged.write_text(json.dumps(out))
    return roadnet_copy, surged, added
