"""Training loop for learned agents (ROADMAP 5.4; plan phase5-rl-baselines D7).

:func:`train` trains one agent on the **train** flow and selects its checkpoint on the
**validation** flow (greedy episode every ``train.val_every`` episodes). Output under
``<output.results_dir>/<run_name>/``:

* ``train_log.csv`` — per training episode: train travel time, mean raw reward, ε, loss, …
* ``val_log.csv``   — per validation episode: every metric of :func:`run_episode`
* ``checkpoints/best.pt`` (lowest val average travel time) and ``checkpoints/last.pt``
* ``tb/`` (TensorBoard), ``config.yaml`` (resolved), ``env.json``, ``summary.json``

and one row in ``docs/experiment_log.csv``. Test / held-out flows are never opened:
:func:`make_env` is called without ``allow_final`` and the split names are fixed here.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from trafficrl.agents import LEARNED_AGENTS, make_agent
from trafficrl.agents.core import LinearSchedule, n_params
from trafficrl.data.splits import REPO_ROOT
from trafficrl.envs.traffic_env import make_env
from trafficrl.eval.runner import PRIMARY_METRIC, run_episode
from trafficrl.utils.config import save_config
from trafficrl.utils.runlog import EXPERIMENT_LOG, append_experiment_row, save_env_info
from trafficrl.utils.seed import seed_everything

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"


class TrainSplitError(ValueError):
    """Training was asked to run on a split other than ``train``."""


def train_episode(env, agent) -> dict[str, float]:
    """One exploring episode; the agent stores experience and learns as it goes."""
    obs, info = env.reset()
    agent.training = True
    rewards: list[float] = []
    stats: list[dict[str, float]] = []
    done = False
    while not done:
        actions = agent.act(obs, info, explore=True)
        obs, _, done, info = env.step(actions)
        agent.observe(info["transitions"])
        rewards.extend(tr.reward for tr in info["transitions"])
        s = agent.update()
        if s:
            stats.append(s)
    agent.training = False
    m = env.metrics()
    out = {
        "train_avg_travel_time_s": m[PRIMARY_METRIC],
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "transitions": len(rewards),
    }
    for key in ("loss", "q_mean", "td_abs", "grad_norm"):
        out[key] = float(np.mean([s[key] for s in stats])) if stats else float("nan")
    return out


def _val_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    v = copy.deepcopy(dict(cfg))
    v["data"] = {**v["data"], "split": VAL_SPLIT}
    return v


def train(
    cfg: Mapping[str, Any],
    *,
    run_name: str | None = None,
    log_path: Path = EXPERIMENT_LOG,
    note: str = "",
    verbose: bool = True,
) -> Path:
    tcfg = dict(cfg["train"])
    agent_name = tcfg["agent"]
    if agent_name not in LEARNED_AGENTS:
        raise ValueError(f"train.agent must be one of {LEARNED_AGENTS}, got {agent_name!r}")
    if cfg["data"]["split"] != TRAIN_SPLIT:
        raise TrainSplitError(
            f"training uses the {TRAIN_SPLIT!r} split only (got {cfg['data']['split']!r}); "
            "checkpoints are selected on 'val', test / held-out are never used here"
        )
    seed = int(cfg["seed"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = run_name or f"{agent_name}_s{seed}_{stamp}"
    out = REPO_ROOT / cfg["output"]["results_dir"] / run_name
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    save_env_info(out / "env.json")

    torch.set_num_threads(int(tcfg.get("torch_threads", 1)))
    seed_everything(seed, cfg.get("deterministic", True))
    device = torch.device(tcfg.get("device", "cpu"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_env = make_env(cfg, seed, track="light")
    val_env = make_env(_val_cfg(cfg), seed, track="light")
    agent = make_agent(agent_name, train_env, cfg)

    episodes, val_every = int(tcfg["episodes"]), int(tcfg["val_every"])
    eps = LinearSchedule(
        tcfg["eps_start"], tcfg["eps_end"], float(tcfg["eps_decay_frac"]) * episodes
    )
    from torch.utils.tensorboard import SummaryWriter  # lazy: slow import

    writer = SummaryWriter(str(out / "tb"))
    train_rows, val_rows = [], []
    best = (float("inf"), 0)
    t_start = time.perf_counter()
    for ep in range(1, episodes + 1):
        agent.epsilon = eps.value(ep - 1)
        t0 = time.perf_counter()
        row = {"episode": ep, "epsilon": agent.epsilon, **train_episode(train_env, agent)}
        row.update(grad_steps=agent.grad_steps, wall_s=time.perf_counter() - t0)
        train_rows.append(row)
        for k, v in row.items():
            if k != "episode":
                writer.add_scalar(f"train/{k}", v, ep)
        msg = f"ep {ep:3d} eps {agent.epsilon:.2f} train ATT {row['train_avg_travel_time_s']:.1f}"
        if ep % val_every == 0 or ep == episodes:
            m = run_episode(val_env, agent)
            val_rows.append({"episode": ep, **m})
            att = m[PRIMARY_METRIC]
            writer.add_scalar("val/avg_travel_time_s", att, ep)
            if att < best[0]:
                best = (att, ep)
                agent.save(out / "checkpoints" / "best.pt")
            msg += f" | val ATT {att:.1f} (best {best[0]:.1f} @ {best[1]})"
        if verbose:
            print(msg + f" [{row['wall_s']:.1f}s]", flush=True)
    agent.save(out / "checkpoints" / "last.pt")
    writer.close()
    runtime = time.perf_counter() - t_start

    pd.DataFrame(train_rows).to_csv(out / "train_log.csv", index=False)
    pd.DataFrame(val_rows).to_csv(out / "val_log.csv", index=False)
    summary = {
        "run_name": run_name,
        "agent": agent_name,
        "seed": seed,
        "episodes": episodes,
        "best_val_avg_travel_time_s": best[0],
        "best_episode": best[1],
        "last_val_avg_travel_time_s": val_rows[-1][PRIMARY_METRIC],
        "training_time_s": round(runtime, 2),
        "params": n_params(agent.net),
        "grad_steps": agent.grad_steps,
        "device": str(device),
        "peak_gpu_mem_mb": (
            round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
            if device.type == "cuda"
            else None
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    hyper = {k: v for k, v in tcfg.items() if k != "agent"}
    hyper.update(
        agent_params=dict((cfg.get("agents", {}) or {}).get(agent_name, {}) or {}),
        flow_file=train_env.data_info["flow_file"],
        val_flow_file=val_env.data_info["flow_file"],
    )
    append_experiment_row(
        run_id=run_name,
        split=TRAIN_SPLIT,
        seed=seed,
        model=agent_name,
        hyperparameters=hyper,
        metric_name=f"{PRIMARY_METRIC} (val, best checkpoint)",
        metric_value=best[0],
        validation_metric=best[0],
        runtime_s=runtime,
        notes="; ".join(x for x in ("dev-WSL", f"best episode {best[1]}/{episodes}", note) if x),
        path=log_path,
    )
    return out
