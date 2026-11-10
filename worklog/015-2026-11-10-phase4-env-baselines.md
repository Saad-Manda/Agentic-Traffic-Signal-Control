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
| `worklog/015-…`, `worklog/README.md` | created / modified | this entry and its index row | work-log rule |

## Commits
Not committed yet.

## Next
Plan steps 2–9.
