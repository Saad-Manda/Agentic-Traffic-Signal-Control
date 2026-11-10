import pytest

from tests.envs.test_traffic_env import toy_env
from trafficrl.agents import AGENTS, Agent, make_agent
from trafficrl.data.splits import REPO_ROOT
from trafficrl.utils.config import load_config


@pytest.mark.parametrize("name", AGENTS)
def test_every_agent_implements_the_interface(name, tmp_path):
    env, _ = toy_env()
    agent = make_agent(name, env, load_config(REPO_ROOT / "configs" / "base.yaml"))
    assert isinstance(agent, Agent) and agent.name == name
    obs, info = env.reset()
    done = False
    while not done:
        actions = agent.act(obs, info)
        assert set(actions) == set(env.intersection_ids)
        assert all(a in env.phase_ids for a in actions.values())
        obs, _, done, info = env.step(actions)
        agent.observe(info["transitions"])
    assert agent.update() == {} and agent.last_explanations() == {}
    agent.save(tmp_path / "x")
    agent.load(tmp_path / "x")


def test_unknown_agent():
    env, _ = toy_env()
    with pytest.raises(ValueError, match="unknown agent"):
        make_agent("dqn", env, {})
