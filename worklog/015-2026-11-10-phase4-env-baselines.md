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
| `src/trafficrl/agents/baselines/fixed_time.py` | created | `FixedTimeAgent`: phases 1 → 8, keep while `green_elapsed` < `green_seconds` (30 s), rejects a green not a multiple of Δt | ROADMAP 4.4, protocol §4 |
| `src/trafficrl/agents/baselines/max_pressure.py` | created | `MaxPressureAgent.choose` / `act` from observed `info["lane_count"]`, ties → lowest id | ROADMAP 4.5; reused as the UG-MARL fallback |
| `src/trafficrl/agents/__init__.py` | modified | `AGENTS`, `make_agent(name, env, cfg)` registry reading `cfg["agents"][name]` | one construction path for `evaluate.py` |
| `tests/agents/__init__.py`, `tests/agents/test_rule_agents.py`, `tests/agents/test_interface.py` | created | Max-Pressure choice and ties; Fixed-Time timing in the env (decisions at 0, 15, 30, 50, 65 s; order 1 … 8, 1); every registered agent conforms to the interface | testing.md: interface conformance, Max-Pressure |
| `src/trafficrl/eval/__init__.py`, `src/trafficrl/eval/runner.py` | created | `run_episode` (env metrics + decision latency mean / p95 + fallback rate), `condition_label`, `summary_table`, `evaluate` (per seed: seeded env + agent, outputs, one experiment-log row; refuses test / heldout without `final=True`) | ROADMAP 4.6, plan D6 |
| `scripts/evaluate.py` | created | CLI: `--agent --split --dataset --seeds --checkpoint --track --final --note` + overrides; exits 2 on test / heldout without `--final`; prints mean ± std / CI | ROADMAP 4.6 |
| `configs/base.yaml` | modified | new blocks `robustness` (noise_std, dropout_p, surge), `agents` (fixed_time.green_seconds = 30, max_pressure), `eval` (seeds 0–4, track full); frozen keys untouched | parameters in config, not code |
| `configs/eval_baselines.yaml` | created | Jinan val, seeds 0–4, results to `results/eval` | Phase 4 baseline runs |
| `tests/eval/__init__.py`, `tests/eval/test_evaluate.py` | created | episode metrics, condition label, deterministic summary, CLI + API refuse test / heldout, slow end-to-end outputs + log rows (temp log) | verifies D6 and the leakage guard |
| `worklog/015-…`, `worklog/README.md` | created / modified | this entry and its index row | work-log rule |

## Commits
Not committed yet.

## Next
Plan steps 2–9.
