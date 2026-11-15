"""Multi-intersection CityFlow environment."""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trafficrl.data.roadnet import YELLOW_PHASE, Roadnet, load_roadnet
from trafficrl.data.splits import SplitError, flow_files
from trafficrl.envs.engine import make_engine
from trafficrl.envs.features import FeatureBuilder, FeatureSpec
from trafficrl.envs.metrics import EpisodeTracker
from trafficrl.envs.perturb import SensorModel, surge_flow
from trafficrl.utils.seed import make_rng, seed_cityflow

FINAL_SPLITS = ("test", "heldout")
PERTURB_SEED_OFFSET = 10_000


@dataclass(frozen=True)
class Transition:
    """One completed decision interval of one intersection."""

    iid: str
    obs: dict[str, np.ndarray]
    action: int
    reward: float
    next_obs: dict[str, np.ndarray]
    done: bool
    duration_s: int


@dataclass
class _Clock:
    green: int | None = None
    left: int = 0
    yellow: bool = False
    green_elapsed: int = 0


class TrafficEnv:
    """Use :func:`make_env` to build one from a config (applies the split guard)."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        seed: int,
        roadnet: Roadnet,
        engine_factory: Callable[[], Any],
        *,
        track: str = "full",
        data_info: Mapping[str, Any] | None = None,
        workdir: tempfile.TemporaryDirectory | None = None,
    ) -> None:
        sim = cfg["sim"]
        self.seed = seed
        self.roadnet = roadnet
        self.features = FeatureBuilder(roadnet, FeatureSpec.from_config(cfg))
        self.delta_t = int(sim["decision_interval"])
        self.yellow_s = int(sim["yellow_seconds"])
        self.episode_s = int(sim["episode_seconds"])
        if sim.get("reward", "pressure") != "pressure":
            raise ValueError(f"only reward 'pressure' is implemented, got {sim['reward']!r}")
        if self.delta_t <= 0 or self.yellow_s < 0 or self.episode_s <= 0:
            raise ValueError("decision_interval, yellow_seconds, episode_seconds must be positive")
        rob = cfg.get("robustness", {}) or {}
        self.sensor = SensorModel(
            self.features.capacity,
            noise_std=float(rob.get("noise_std", 0.0)),
            dropout_p=float(rob.get("dropout_p", 0.0)),
            rng=make_rng(seed + PERTURB_SEED_OFFSET),
        )
        self.track = track
        self.data_info = dict(data_info or {})
        self._workdir = workdir
        self._engine_factory = engine_factory
        self.engine = None

        self.intersection_ids: list[str] = list(roadnet.intersections)
        self.phase_ids: tuple[int, ...] = self.features.phase_ids
        self.obs_dim: int = self.features.dim
        self._in_lanes = [ln for i in roadnet.intersections.values() for ln in i.in_lanes]
        self._episodes = 0

    def neighbors(self, iid: str) -> list[str]:
        """Signalised neighbours of ``iid`` in direction order E, N, W, S (boundary dropped)."""
        return [n for n in self.roadnet.intersections[iid].neighbors if n is not None]

    def neighbor_index(self) -> tuple[np.ndarray, np.ndarray]:
        """``(index, mask)`` of each intersection's neighbours, order E, N, W, S."""
        pos = {iid: k for k, iid in enumerate(self.intersection_ids)}
        n = len(self.intersection_ids)
        index = np.zeros((n, 4), dtype=np.int64)
        mask = np.zeros((n, 4), dtype=bool)
        for k, iid in enumerate(self.intersection_ids):
            for d, nb in enumerate(self.roadnet.intersections[iid].neighbors):
                if nb is not None:
                    index[k, d], mask[k, d] = pos[nb], True
        return index, mask

    def reset(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self.engine is None:
            self.engine = self._engine_factory()
        else:
            self.engine.reset()
        seed_cityflow(self.engine, self.seed)
        self._episodes += 1
        self.t = 0
        self.clock = {iid: _Clock() for iid in self.intersection_ids}
        self.tracker = EpisodeTracker(self._in_lanes, self.track)
        self._pending: dict[str, tuple[dict[str, np.ndarray], int, int]] = {}
        self.done = False
        ready = list(self.intersection_ids)
        obs, info = self._observe(ready, [])
        return obs, info

    def step(
        self, actions: Mapping[str, int]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict[str, Any]]:
        if self.engine is None or self.done:
            raise RuntimeError("call reset() before step() (episode finished or not started)")
        for iid in self._ready:
            if iid not in actions:
                raise KeyError(f"no action for ready intersection {iid}")
            self._apply(iid, int(actions[iid]))
            self._pending[iid] = (self._last_obs, int(actions[iid]), self.t)

        ready: list[str] = []
        while not ready and self.t < self.episode_s:
            self.engine.next_step()
            self.t += 1
            if self.tracker.full:
                self.tracker.on_second(self.engine.get_vehicle_speed())
            for iid, c in self.clock.items():
                c.left -= 1
                if not c.yellow:
                    c.green_elapsed += 1
                if c.left > 0:
                    continue
                if c.yellow:
                    c.yellow, c.left, c.green_elapsed = False, self.delta_t, 0
                    self.engine.set_tl_phase(iid, c.green)
                else:
                    ready.append(iid)
        self.done = self.t >= self.episode_s
        closing = list(self.intersection_ids) if self.done else ready
        obs, info = self._observe([] if self.done else ready, closing)
        rewards = {tr.iid: tr.reward for tr in info["transitions"]}
        return obs, rewards, self.done, info

    def _apply(self, iid: str, phase: int) -> None:
        if phase not in self.phase_ids:
            raise ValueError(f"{iid}: action {phase} is not a green phase {self.phase_ids}")
        c = self.clock[iid]
        if c.green is None or phase == c.green or self.yellow_s == 0:
            if phase != c.green:
                c.green_elapsed = 0
                if c.green is not None:
                    self.tracker.on_switch()
            c.green, c.left, c.yellow = phase, self.delta_t, False
            self.engine.set_tl_phase(iid, phase)
        else:
            c.green, c.left, c.yellow, c.green_elapsed = phase, self.yellow_s, True, 0
            self.engine.set_tl_phase(iid, YELLOW_PHASE)
            self.tracker.on_switch()

    def _observe(
        self, ready: list[str], closing: list[str]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        true_count = self.engine.get_lane_vehicle_count()
        true_wait = self.engine.get_lane_waiting_vehicle_count()
        self.tracker.on_step(true_wait)
        self.true_count, self.true_wait = true_count, true_wait
        count, wait = self.sensor.observe(true_count, true_wait)
        phases = {
            iid: c.green if c.green is not None else self.phase_ids[0]
            for iid, c in self.clock.items()
        }
        obs = self.features.observations(count, wait, phases)

        transitions = []
        for iid in closing:
            if iid not in self._pending:
                continue
            prev_obs, action, t0 = self._pending.pop(iid)
            transitions.append(
                Transition(
                    iid=iid,
                    obs=prev_obs,
                    action=action,
                    reward=-abs(self.features.intersection_pressure(iid, true_count)),
                    next_obs=obs,
                    done=self.done,
                    duration_s=self.t - t0,
                )
            )
        self._ready = ready
        self._last_obs = obs
        info = {
            "t": self.t,
            "ready": list(ready),
            "phase": phases,
            "in_yellow": {iid: c.yellow for iid, c in self.clock.items()},
            "green_elapsed": {iid: c.green_elapsed for iid, c in self.clock.items()},
            "lane_count": count,
            "transitions": transitions,
        }
        return obs, info

    def metrics(self) -> dict[str, float | None]:
        """Episode metrics; call after the episode is done."""
        return self.tracker.result(
            self.engine.get_average_travel_time(), self.engine.get_vehicle_count()
        )


def make_env(
    cfg: Mapping[str, Any],
    seed: int,
    *,
    allow_final: bool = False,
    track: str = "full",
) -> TrafficEnv:
    """Build the env for ``cfg['data']``.

    ``test`` / ``heldout`` splits need ``allow_final=True``.
    """
    data = cfg["data"]
    dataset, split = data["dataset"], data["split"]
    if split in FINAL_SPLITS and not allow_final:
        raise SplitError(
            f"{dataset}/{split} is a final-evaluation split; it is opened only by "
            "scripts/evaluate.py --final"
        )
    roadnet_path, flows = flow_files(dataset, split)
    index = int(data.get("flow_index", 0))
    if not 0 <= index < len(flows):
        raise SplitError(f"{dataset}/{split} has {len(flows)} flow file(s), index {index}")
    flow_path = flows[index]

    workdir = tempfile.TemporaryDirectory(prefix="trafficrl_env_")
    wd = Path(workdir.name)
    info: dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "flow_file": flow_path.name,
    }
    surge = float((cfg.get("robustness", {}) or {}).get("surge", 1.0))
    if surge != 1.0:
        roadnet_path, flow_path, added = surge_flow(
            roadnet_path, flow_path, wd / "surge", surge, make_rng(seed + PERTURB_SEED_OFFSET + 1)
        )
        info.update(
            surge_vehicles_added=added,
            surge_flow_sha256=hashlib.sha256(flow_path.read_bytes()).hexdigest(),
        )
    threads = int(cfg["sim"].get("threads", 1))

    def factory():
        return make_engine(roadnet_path, flow_path, seed, threads, wd)

    return TrafficEnv(
        cfg,
        seed,
        load_roadnet(roadnet_path),
        factory,
        track=track,
        data_info=info,
        workdir=workdir,
    )
