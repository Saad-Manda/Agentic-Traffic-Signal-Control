"""Evaluation loop shared by every agent."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trafficrl.agents import make_agent, model_label
from trafficrl.agents.ugmarl.gate import gate_file_for, load_gate
from trafficrl.data.splits import REPO_ROOT
from trafficrl.envs.traffic_env import FINAL_SPLITS, make_env
from trafficrl.utils.config import save_config
from trafficrl.utils.runlog import EXPERIMENT_LOG, append_experiment_row, save_env_info
from trafficrl.utils.seed import seed_everything
from trafficrl.utils.stats import summarise

PRIMARY_METRIC = "avg_travel_time_s"


def run_episode(
    env, agent, *, explore: bool = False, recorder=None, step_recorder=None
) -> dict[str, Any]:
    """Play one episode; ``recorder`` (a :class:`DecisionRecorder`) logs every decision,
    ``step_recorder`` (a :class:`StepRecorder`) the true network state at every env step."""
    obs, info = env.reset()
    if step_recorder is not None:
        step_recorder.record(info)
    latency_ms: list[float] = []
    decisions = fallbacks = 0
    done = False
    while not done:
        t0 = time.perf_counter()
        actions = agent.act(obs, info, explore=explore)
        latency_ms.append((time.perf_counter() - t0) * 1e3)
        expl = agent.last_explanations()
        if recorder is not None:
            recorder.record(info["t"], info, actions, expl)
        flags = expl.get("fallback", {})
        decisions += len(info["ready"])
        fallbacks += sum(bool(flags.get(iid, False)) for iid in info["ready"])
        obs, _, done, info = env.step(actions)
        if step_recorder is not None:
            step_recorder.record(info)
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
    """Replace ``{seed}`` in the checkpoint path with the evaluation seed."""
    return None if checkpoint is None else Path(str(checkpoint).replace("{seed}", str(seed)))


class GateError(ValueError):
    """The gate is on but no threshold τ could be found."""


def resolve_gate(
    cfg: Mapping[str, Any], checkpoint: Path | None, seed: int
) -> tuple[float | None, str]:
    """``(τ, source)`` for UG-MARL in ``cfg`` and one seed; ``(None, "off")`` without a gate."""
    p = dict((cfg.get("agents", {}) or {}).get("ugmarl", {}) or {})
    if not (p.get("use_gate", True) and p.get("use_ensemble", True) and int(p.get("heads", 5)) > 1):
        return None, "off"
    gate = dict(p.get("gate") or {})
    if gate.get("tau") is not None:
        return float(gate["tau"]), "config"
    spec = gate.get("file") or "auto"
    if spec == "auto":
        if checkpoint is None:
            raise GateError("gate on with gate.file=auto but no --checkpoint to find gate.json")
        f = gate_file_for(checkpoint)
    else:
        f = Path(str(spec).replace("{seed}", str(seed)))
    if not f.exists():
        raise GateError(
            f"gate on but {f} does not exist: run scripts/calibrate_gate.py, set "
            "agents.ugmarl.gate.tau, or evaluate −gate with agents.ugmarl.use_gate=false"
        )
    return float(load_gate(f)["tau"]), str(f)


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
    log_decisions: bool = False,
    log_trace: bool = False,
) -> tuple[Path, pd.DataFrame]:
    split = cfg["data"]["split"]
    if split in FINAL_SPLITS and not final:
        raise PermissionError(
            f"split {split!r} is final-evaluation only: pass final=True (--final)"
        )
    if log_trace and track != "full":
        raise ValueError("log_trace needs track='full' (per-vehicle log)")
    condition = condition_label(cfg)
    label = model_label(agent_name, cfg)
    if agent_name == "ugmarl":
        for seed in seeds:
            resolve_gate(cfg, resolve_checkpoint(checkpoint, int(seed)), int(seed))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{label}_{cfg['data']['dataset']}-{split}_{condition}_{stamp}"
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
        gate_info: dict[str, Any] = {}
        if agent_name == "ugmarl":
            agent.tau, source = resolve_gate(run_cfg, ckpt, int(seed))
            gate_info = {"tau": agent.tau, "tau_source": source}
        recorder = None
        if log_decisions:
            from trafficrl.eval.explain import DecisionRecorder

            recorder = DecisionRecorder(env)
        stepper = None
        if log_trace:
            from trafficrl.eval.explain import StepRecorder

            stepper = StepRecorder(env)
        m = run_episode(env, agent, recorder=recorder, step_recorder=stepper)
        if recorder is not None:
            recorder.save(out / f"decisions_s{seed}.csv")
        if stepper is not None:
            stepper.save(out / f"trace_s{seed}.csv")
            env.tracker.vehicle_frame().to_csv(out / f"vehicles_s{seed}.csv", index=False)
        runtime = time.perf_counter() - t0
        data_info[int(seed)] = env.data_info
        rows.append({"seed": int(seed), **m, **gate_info, "runtime_s": round(runtime, 3)})
        notes = "; ".join(x for x in (condition, f"final={final}" if final else "", note) if x)
        append_experiment_row(
            run_id=f"{run_id}-s{seed}",
            split=split,
            seed=int(seed),
            model=label,
            hyperparameters={
                **hyper,
                **gate_info,
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
