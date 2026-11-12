"""Decision recorder and gate resolution in evaluation (plan phase6-ugmarl D5, D7)."""

import csv
import json

import numpy as np
import pytest
import torch

from tests.envs.test_traffic_env import toy_env
from trafficrl.agents import UGMARLAgent, make_agent
from trafficrl.agents.ugmarl.gate import save_gate
from trafficrl.data.splits import REPO_ROOT
from trafficrl.eval import runner
from trafficrl.eval.explain import COLUMNS, DecisionRecorder, StepRecorder
from trafficrl.eval.runner import GateError, evaluate, resolve_gate, run_episode
from trafficrl.utils.config import load_config

SMALL = {"heads": 3, "attn_dim": 8, "head_hidden": 16}
EVAL_CFG = REPO_ROOT / "configs" / "eval_ugmarl.yaml"


def busy_toy(episode=120):
    env, engine = toy_env(episode=episode)
    rng = np.random.default_rng(0)
    engine.counts = {ln: int(rng.integers(0, 10)) for ln in engine.lanes}
    return env


def test_one_row_per_ready_decision_and_fallback_rate_matches():
    env = busy_toy()
    torch.manual_seed(0)
    agent = UGMARLAgent(env, SMALL)
    obs, info = env.reset()
    agent.act(obs, info)
    agent.tau = float(np.median([agent.last_explanations()["sigma"]["C"]]))  # some firing
    rec = DecisionRecorder(env)
    m = run_episode(env, agent, recorder=rec)
    df = rec.frame()
    assert list(df.columns) == list(COLUMNS) and len(df) == m["decisions"] == 8
    assert df["fallback"].mean() == pytest.approx(m["fallback_rate"])
    att = df[[c for c in COLUMNS if c.startswith("attn_")]].to_numpy()
    assert np.allclose(att.sum(axis=1), 1.0) and (att[:, 1:] == 0).all()  # toy: no neighbours
    assert (df["phase_taken"] == np.where(df["fallback"], df["mp_phase"], df["greedy_phase"])).all()
    assert (df["q_gap"] >= 0).all() and (df["in_veh_observed"] > 0).all()


def test_recorder_works_for_agents_without_explanations():
    env = busy_toy(60)
    rec = DecisionRecorder(env)
    run_episode(env, make_agent("max_pressure", env, {}), recorder=rec)
    df = rec.frame()
    assert len(df) == 4 and df["sigma"].isna().all() and not df["fallback"].any()


def ucfg(*extra):
    return load_config(EVAL_CFG, list(extra))


def test_resolve_gate_sources(tmp_path):
    ckpt = tmp_path / "ugmarl_final_s3" / "checkpoints" / "best.pt"
    assert resolve_gate(ucfg("agents.ugmarl.use_gate=false"), ckpt, 3) == (None, "off")
    assert resolve_gate(ucfg("agents.ugmarl.use_ensemble=false"), ckpt, 3) == (None, "off")
    assert resolve_gate(ucfg("agents.ugmarl.gate.tau=0.5"), ckpt, 3) == (0.5, "config")
    with pytest.raises(GateError):
        resolve_gate(ucfg(), ckpt, 3)  # no gate.json yet -> refused, never silently ungated
    with pytest.raises(GateError):
        resolve_gate(ucfg(), None, 3)
    save_gate(tmp_path / "ugmarl_final_s3" / "gate.json", {"p": 90, "tau": 0.25})
    assert resolve_gate(ucfg(), ckpt, 3)[0] == 0.25
    save_gate(tmp_path / "g_s3.json", {"tau": 0.75})
    cfg = ucfg(f"agents.ugmarl.gate.file={tmp_path}/g_s{{seed}}.json")
    assert resolve_gate(cfg, None, 3)[0] == 0.75


@pytest.fixture
def toy_make_env(monkeypatch):
    def fake(cfg, seed, *, allow_final=False, track="full"):
        env = busy_toy(60)
        env.data_info = {"flow_file": "val.json"}
        return env

    monkeypatch.setattr(runner, "make_env", fake)


def test_evaluate_resolves_tau_per_seed_and_logs_decisions(tmp_path, toy_make_env):
    env = busy_toy(60)
    for s, tau in ((0, -1.0), (1, 1e9)):  # seed 0 always falls back, seed 1 never
        torch.manual_seed(s)
        a = UGMARLAgent(env, SMALL)
        a.save(tmp_path / f"m_s{s}" / "checkpoints" / "best.pt")
        save_gate(tmp_path / f"m_s{s}" / "gate.json", {"p": 90, "tau": tau})
    cfg = ucfg(
        *(f"agents.ugmarl.{k}={v}" for k, v in SMALL.items()), f"output.results_dir={tmp_path}"
    )
    log = tmp_path / "log.csv"
    out, metrics = evaluate(
        cfg, "ugmarl", [0, 1], checkpoint=str(tmp_path / "m_s{seed}/checkpoints/best.pt"),
        log_path=log, log_decisions=True,
    )  # fmt: skip
    assert metrics["fallback_rate"].tolist() == [1.0, 0.0]
    assert metrics["tau"].tolist() == [-1.0, 1e9]
    for s in (0, 1):
        assert (out / f"decisions_s{s}.csv").exists()
    rows = list(csv.DictReader(log.open()))
    assert [r["model"] for r in rows] == ["ugmarl", "ugmarl"]
    assert json.loads(rows[0]["hyperparameters"])["tau"] == -1.0
    assert out.name.startswith("ugmarl_jinan-val_clean_")


def test_evaluate_refuses_gate_without_tau_before_writing(tmp_path, toy_make_env):
    cfg = ucfg(f"output.results_dir={tmp_path}")
    with pytest.raises(GateError):
        evaluate(cfg, "ugmarl", [0], checkpoint=str(tmp_path / "x/checkpoints/best.pt"),
                 log_path=tmp_path / "log.csv")  # fmt: skip
    assert not (tmp_path / "log.csv").exists() and not any(tmp_path.iterdir())


def test_nogate_variant_name_in_run_and_log(tmp_path, toy_make_env):
    env = busy_toy(60)
    UGMARLAgent(env, SMALL).save(tmp_path / "m" / "checkpoints" / "best.pt")
    cfg = ucfg(
        *(f"agents.ugmarl.{k}={v}" for k, v in SMALL.items()),
        "agents.ugmarl.use_gate=false",
        f"output.results_dir={tmp_path}",
    )
    out, metrics = evaluate(cfg, "ugmarl", [0], checkpoint=str(tmp_path / "m/checkpoints/best.pt"),
                            log_path=tmp_path / "log.csv")  # fmt: skip
    assert out.name.startswith("ugmarl-nogate_") and metrics["fallback_rate"].tolist() == [0.0]
    assert list(csv.DictReader((tmp_path / "log.csv").open()))[0]["model"] == "ugmarl-nogate"


def test_step_recorder_logs_true_state_and_leaves_metrics_unchanged(tmp_path):
    def play(with_recorder):
        env = busy_toy(episode=60)
        rec = StepRecorder(env) if with_recorder else None
        m = run_episode(env, make_agent("max_pressure", env, {}), step_recorder=rec)
        m = {k: v for k, v in m.items() if not k.startswith("decision_latency")}
        return env, rec, m

    env, rec, m = play(True)
    assert play(False)[2] == m  # read-only: same metrics with and without the recorder
    f = rec.frame()
    assert list(f.columns[:5]) == ["t", "iid", "phase", "in_yellow", "ready"]
    assert sum(c.startswith("q_") for c in f.columns) == 12
    assert f["t"].iloc[0] == 0 and f["t"].iloc[-1] == 60 and set(f["iid"]) == {"C"}
    assert bool(f["ready"].iloc[0]) and not bool(f["ready"].iloc[-1])  # nothing decides at the end
    lanes = env.features.roadnet.intersections["C"].in_lanes
    assert f.loc[0, [f"n_{k}" for k in range(12)]].tolist() == [env.true_count[ln] for ln in lanes]
    out = rec.save(tmp_path / "trace.csv")
    assert len(list(csv.DictReader(out.open()))) == len(f)
