"""Training loop (ROADMAP 5.4): split guards, outputs, no val experience, determinism."""

import csv
import json
import subprocess
import sys

import pytest
import torch

from tests.envs.test_traffic_env import toy_env
from trafficrl.agents import make_agent
from trafficrl.data.splits import REPO_ROOT
from trafficrl.eval.runner import run_episode
from trafficrl.train import loop
from trafficrl.train.loop import TrainSplitError, train, train_episode
from trafficrl.utils.config import load_config

CFG = REPO_ROOT / "configs" / "dqn.yaml"
FAST = ["train.episodes=3", "train.val_every=2", "train.learning_starts=0", "train.batch_size=2"]


@pytest.fixture
def fake_envs(monkeypatch):
    """Replace make_env with toy envs and record every split / allow_final request."""
    calls = []

    def fake_make_env(cfg, seed, *, allow_final=False, track="full"):
        calls.append((cfg["data"]["split"], allow_final, track))
        env, engine = toy_env(episode=60)
        engine.speeds = {"v": 0.0}
        env.data_info = {"flow_file": f"{cfg['data']['split']}.json"}
        return env

    monkeypatch.setattr(loop, "make_env", fake_make_env)
    return calls


def cfg_for(tmp_path, *extra):
    return load_config(CFG, [*FAST, f"output.results_dir={tmp_path}", *extra])


@pytest.mark.parametrize("split", ["val", "test", "heldout"])
def test_refuses_any_split_but_train(split, tmp_path, fake_envs):
    with pytest.raises(TrainSplitError):
        train(cfg_for(tmp_path, f"data.split={split}"), log_path=tmp_path / "log.csv")
    assert fake_envs == []


def test_script_exit_code_2_on_test_split():
    r = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "--config",
            str(CFG),
            "data.split=test",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert r.returncode == 2 and "train" in r.stderr


def test_train_writes_outputs_and_one_log_row(tmp_path, fake_envs):
    log = tmp_path / "log.csv"
    out = train(cfg_for(tmp_path), run_name="r", log_path=log, verbose=False)
    assert sorted(fake_envs) == [("train", False, "light"), ("val", False, "light")]
    for name in (
        "config.yaml",
        "env.json",
        "summary.json",
        "train_log.csv",
        "val_log.csv",
        "checkpoints/best.pt",
        "checkpoints/last.pt",
    ):
        assert (out / name).exists(), name
    s = json.loads((out / "summary.json").read_text())
    assert s["episodes"] == 3 and s["best_episode"] in (2, 3) and s["params"] == 23_304
    val_eps = [int(r["episode"]) for r in csv.DictReader((out / "val_log.csv").open())]
    assert val_eps == [2, 3]
    rows = list(csv.DictReader(log.open()))
    assert len(rows) == 1 and rows[0]["split"] == "train" and rows[0]["model"] == "dqn"
    assert float(rows[0]["validation_metric"]) == s["best_val_avg_travel_time_s"]


def test_validation_episodes_never_enter_the_buffer():
    env, engine = toy_env(episode=60)
    cfg = load_config(CFG, ["train.learning_starts=0"])
    agent = make_agent("dqn", env, cfg)
    train_episode(env, agent)
    n = agent.buffer.total
    assert n > 0 and not agent.training
    run_episode(env, agent)
    assert agent.buffer.total == n


def test_same_seed_same_weights(tmp_path, fake_envs):
    def weights(name, seed):
        out = train(
            cfg_for(tmp_path, f"seed={seed}"),
            run_name=name,
            log_path=tmp_path / "log.csv",
            verbose=False,
        )
        ckpt = torch.load(out / "checkpoints" / "last.pt", weights_only=False)
        return torch.cat([v.flatten() for v in ckpt["net"].values()])

    assert torch.equal(weights("a", 0), weights("b", 0))
    assert not torch.equal(weights("a", 0), weights("c", 1))
