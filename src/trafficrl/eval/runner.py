"""Shared evaluation loop for every agent (ROADMAP 4.6; protocol §5–§7).

:func:`run_episode` plays one episode and adds the agent-side metrics (decision latency,
fallback rate) to the env metrics. :func:`evaluate` runs one agent on one split for several
seeds and writes, under ``<results_dir>/<run_id>/``:

* ``metrics.csv``  — one row per seed;
* ``summary.csv``  — per metric: mean, std, 95 % CI, n, deterministic flag;
* ``config.yaml``, ``env.json`` (software + hardware), ``data.json`` (flow file, surge hash);

and appends one row per seed to ``docs/experiment_log.csv``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trafficrl.agents import make_agent
from trafficrl.data.splits import REPO_ROOT
from trafficrl.envs.traffic_env import FINAL_SPLITS, make_env
from trafficrl.utils.config import save_config
from trafficrl.utils.runlog import EXPERIMENT_LOG, append_experiment_row, save_env_info
from trafficrl.utils.seed import seed_everything
from trafficrl.utils.stats import summarise

PRIMARY_METRIC = "avg_travel_time_s"


def run_episode(env, agent, *, explore: bool = False) -> dict[str, Any]:
    obs, info = env.reset()
    latency_ms: list[float] = []
    decisions = fallbacks = 0
    done = False
    while not done:
        t0 = time.perf_counter()
        actions = agent.act(obs, info, explore=explore)
        latency_ms.append((time.perf_counter() - t0) * 1e3)
        flags = agent.last_explanations().get("fallback", {})
        decisions += len(info["ready"])
        fallbacks += sum(bool(flags.get(iid, False)) for iid in info["ready"])
        obs, _, done, info = env.step(actions)
        agent.observe(info["transitions"])
    m = env.metrics()
    m.update(
        decisions=decisions,
        fallback_rate=fallbacks / decisions if decisions else 0.0,
        decision_latency_ms=float(np.mean(latency_ms)),
        decision_latency_ms_p95=float(np.percentile(latency_ms, 95)),
    )
    return m


def resolve_checkpoint(checkpoint: str | Path | None, seed: int) -> Path | None:
    """``{seed}`` in the path is replaced by the evaluation seed, so each seed loads the model
    trained with that seed (protocol §6: a seed fixes training **and** evaluation)."""
    return None if checkpoint is None else Path(str(checkpoint).replace("{seed}", str(seed)))


def condition_label(cfg: Mapping[str, Any]) -> str:
    rob = cfg.get("robustness", {}) or {}
    parts = []
    if rob.get("noise_std", 0):
        parts.append(f"noise{rob['noise_std']:g}")
    if rob.get("dropout_p", 0):
        parts.append(f"drop{rob['dropout_p']:g}")
    if rob.get("surge", 1.0) != 1.0:
        parts.append(f"surge{rob['surge']:g}")
    return "-".join(parts) or "clean"


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in metrics.columns:
        if col == "seed":
            continue
        vals = pd.to_numeric(metrics[col], errors="coerce").dropna()
        if len(vals):
            rows.append({"metric": col, **summarise(vals.tolist())})
    return pd.DataFrame(rows)


def evaluate(
    cfg: Mapping[str, Any],
    agent_name: str,
    seeds: Sequence[int],
    *,
    final: bool = False,
    checkpoint: str | Path | None = None,
    track: str = "full",
    note: str = "",
    log_path: Path = EXPERIMENT_LOG,
) -> tuple[Path, pd.DataFrame]:
    split = cfg["data"]["split"]
    if split in FINAL_SPLITS and not final:
        raise PermissionError(
            f"split {split!r} is final-evaluation only: pass final=True (--final) in Phase 7"
        )
    condition = condition_label(cfg)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{agent_name}_{cfg['data']['dataset']}-{split}_{condition}_{stamp}"
    out = REPO_ROOT / cfg["output"]["results_dir"] / run_id
    out.mkdir(parents=True, exist_ok=True)
    save_config({**cfg, "eval": {**cfg.get("eval", {}), "seeds": list(seeds)}}, out / "config.yaml")
    save_env_info(out / "env.json")

    sim = cfg["sim"]
    hyper = {
        "decision_interval_s": sim["decision_interval"],
        "yellow_s": sim["yellow_seconds"],
        "episode_s": sim["episode_seconds"],
        "robustness": dict(cfg.get("robustness", {}) or {}),
        "agent": dict((cfg.get("agents", {}) or {}).get(agent_name, {}) or {}),
        "track": track,
    }
    rows, data_info = [], {}
    for seed in seeds:
        run_cfg = {**cfg, "seed": int(seed)}
        seed_everything(int(seed), cfg.get("deterministic", True))
        t0 = time.perf_counter()
        env = make_env(run_cfg, int(seed), allow_final=final, track=track)
        agent = make_agent(agent_name, env, run_cfg)
        ckpt = resolve_checkpoint(checkpoint, int(seed))
        if ckpt is not None:
            agent.load(ckpt)
        m = run_episode(env, agent)
        runtime = time.perf_counter() - t0
        data_info[int(seed)] = env.data_info
        rows.append({"seed": int(seed), **m, "runtime_s": round(runtime, 3)})
        notes = "; ".join(x for x in (condition, f"final={final}" if final else "", note) if x)
        append_experiment_row(
            run_id=f"{run_id}-s{seed}",
            split=split,
            seed=int(seed),
            model=agent_name,
            hyperparameters={
                **hyper,
                "flow_file": env.data_info["flow_file"],
                **({"checkpoint": str(ckpt)} if ckpt is not None else {}),
            },
            metric_name=PRIMARY_METRIC,
            metric_value=m[PRIMARY_METRIC],
            validation_metric=m[PRIMARY_METRIC] if split == "val" else None,
            runtime_s=runtime,
            notes=notes,
            path=log_path,
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    summary_table(metrics).to_csv(out / "summary.csv", index=False)
    (out / "data.json").write_text(json.dumps(data_info, indent=2) + "\n")
    return out, metrics
