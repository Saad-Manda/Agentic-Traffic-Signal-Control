# 015 — Phase 4: environment, metrics, robustness hooks, rule baselines, evaluation
**Date:** 2026-11-10   **Tag:** code, experiment, decision   **ROADMAP:** 4.1–4.6   **Status:** in-progress

## What
Executing `.claude/plans/phase4-env-metrics-rule-baselines.md`: the shared CityFlow
environment wrapper, metrics, robustness hooks, Fixed-Time / Max-Pressure agents and
`scripts/evaluate.py`, then Fixed-Time / Max-Pressure numbers on the train (sanity) and
validation flows.

## Why
ROADMAP Phase 4 (R4, G7, G9): every later agent (DQN, IDQN, UG-MARL, ablations) needs one
environment and one evaluation loop so that results are comparable.

## How
In progress (plan steps 1–9).

## Result
In progress.

## Files changed
| File | Action | What changed | Why |
|------|--------|--------------|-----|
| `src/trafficrl/envs/engine.py` | created | `make_engine` moved here verbatim from `data/trace.py` | the env must not import from an EDA module |
| `src/trafficrl/data/trace.py` | modified | `make_engine` removed and re-imported from `envs.engine`; unused `json` / `seed_cityflow` imports dropped | keeps `collect_traces.py` and its tests working unchanged |
| `src/trafficrl/agents/base.py` | created | `Agent` protocol (`act(obs, info, explore)`, `observe`, `update`, `save`, `load`, `last_explanations`) and `RuleAgent` no-op base | shared interface (plan D5) |
| `src/trafficrl/envs/traffic_env.py` | created | `TrafficEnv` (per-intersection clocks, yellow on top of Δt, `neighbors`, `neighbor_index`, `reset`/`step`/`metrics`), `Transition` dataclass, `make_env` (split guard: test/heldout need `allow_final=True`; surge flow in a temp workdir) | ROADMAP 4.1, plan D2 |
| `src/trafficrl/envs/metrics.py` | created | `EpisodeTracker` (`on_second`, `on_step`, `on_switch`, `result`), `track` levels full / light | ROADMAP 4.2, protocol §5 definitions |
| `src/trafficrl/envs/perturb.py` | created | `SensorModel` (noise on capacity-scaled readings, per-lane dropout), `surge_flow` | ROADMAP 4.3, protocol §6 definitions |
| `src/trafficrl/utils/stats.py` | created | `summarise` (mean, std, 95 % CI, deterministic flag), `t975` with a Student-t table | mean ± std and 95 % CI over seeds without a scipy dependency |
| `tests/envs/fake_engine.py` | created | `FakeEngine` stand-in for `cityflow.Engine` (records phase calls with time) | fast timing tests without the simulator |
| `tests/envs/test_traffic_env.py` | created | timing (keep / change / yellow / episode end), pending-green one-hot, reward from true counts, split guard, engine reuse; slow Jinan tests: topology vs `neighbor_index`, 5 s grid, determinism, reset = fresh engine, surge on val | verifies plan D2 |
| `tests/envs/test_metrics.py`, `tests/envs/test_perturb.py`, `tests/utils/test_stats.py` | created | hand-fed tracker values; noise bounds / seed / σ, dropout rate, surge duplicates; t-values, known CI, deterministic sample | verifies D3, D4 |
| `worklog/015-…`, `worklog/README.md` | created / modified | this entry and its index row | work-log rule |

## Commits
Not committed yet.

## Next
Plan steps 2–9.
