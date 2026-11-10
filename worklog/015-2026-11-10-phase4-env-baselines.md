# 015 — Phase 4: environment, metrics, robustness hooks, rule baselines, evaluation
**Date:** 2026-11-10   **Tag:** code, experiment, decision   **ROADMAP:** 4.1–4.6   **Status:** done

## What
Executing `.claude/plans/phase4-env-metrics-rule-baselines.md`: the shared CityFlow
environment wrapper, metrics, robustness hooks, Fixed-Time / Max-Pressure agents and
`scripts/evaluate.py`, then Fixed-Time / Max-Pressure numbers on the train (sanity) and
validation flows.

## Why
ROADMAP Phase 4 (R4, G7, G9): every later agent (DQN, IDQN, UG-MARL, ablations) needs one
environment and one evaluation loop so that results are comparable.

| Decision | Chosen | Rejected, and why |
|---|---|---|
| Env timing | per-intersection clocks (user, 10 Nov) | synchronous step with yellow outside Δt (non-changers get 20 s green, Fixed-Time not 30 s, EDA numbers not reproducible); yellow inside Δt (contradicts protocol §2) |
| Splits in Phase 4 | train (sanity) + val (user) | test now (breaks "test once, all agents together") |
| `act(obs, info, explore)` | `info` carries observed raw counts, phase, green elapsed | Max-Pressure from the capacity-normalised pressure features (protocol defines it on raw counts); agents reading the engine (architecture rule) |
| Noise / dropout on lane readings | one `SensorModel` feeds features and Max-Pressure | noise on the 44-vector (Max-Pressure would see clean data, protocol §6 says every controller sees it) |
| Test guard | `make_env(allow_final=True)` code argument | config key (reachable by a CLI override) |
| CI | hand-written Student-t table | scipy dependency for one function |

## How
Plan steps 1–9, one commit per step group. `TrafficEnv` advances the engine 1 s at a time;
each intersection has a clock (green / pending green, seconds left, yellow flag, green
elapsed). Fast tests drive the env with `tests/envs/fake_engine.FakeEngine` on the hand-built
toy roadnet; slow tests use the Jinan train / val flows with 300 s episodes.

Runs (all `configs/eval_baselines.yaml`, 3600 s episodes, Δt 15 s, yellow 5 s, `track=full`):
```bash
python scripts/evaluate.py --agent max_pressure --split train --seeds 0 --note "env sanity check vs EDA trace"
python scripts/evaluate.py --agent fixed_time   --split train --seeds 0 --note "env sanity check vs EDA trace"
python scripts/evaluate.py --agent max_pressure --split train --seeds 0 --note "hook smoke test; not reported" robustness.noise_std=0.1   # also dropout_p=0.1, surge=1.5
python scripts/evaluate.py --agent fixed_time   --split val --note "Phase 4 validation baseline"   # seeds 0-4
python scripts/evaluate.py --agent max_pressure --split val --note "Phase 4 validation baseline"   # seeds 0-4
```

## Result
- `pytest`: 128 passed (incl. slow); `ruff check src tests scripts` clean.
- Tracking cost, one Jinan train episode: `light` 1.9 s, `full` 4.2 s (same travel time).
- **Validation (Jinan `real_2000`, seeds 0–4; identical for every seed → deterministic):**

| Agent | Avg travel time | Avg waiting | Avg queue (veh/lane) | Throughput | Max per-vehicle wait | Latency |
|---|---|---|---|---|---|---|
| Fixed-Time | 434.84 s | 187.78 s | 1.59 | 3991 | 1124 s | < 0.01 ms |
| Max-Pressure | **326.11 s** | 83.42 s | 0.72 | 4106 | 1334 s | 0.13 ms |

  Max-Pressure is 25 % faster on average but has the *higher* worst-case wait (1334 vs
  1124 s): it can starve low-pressure approaches. Relevant to H2 / fairness (ROADMAP 9.8).
- **Train sanity check:** Max-Pressure 349.67 s, Fixed-Time 516.79 s vs EDA traces 344.13 /
  515.21 s (+1.6 % / +0.3 %, inside the ±3 % plan tolerance); throughput 5602 / 4971 vs EDA
  completions 5605 / 4969. Cause found: `collect_trace` gives the **first** green 14 s (its
  counter is decremented before the first engine step). Copying that into the env (scratch
  monkeypatch, not committed) reproduces Max-Pressure's **344.1295 s exactly**, so the env and
  the trace are otherwise identical. The env keeps the correct 15 s; EDA results stand as
  descriptive data.
- **Hook smoke test** (train, seed 0, Max-Pressure; not a result): noise 0.1 → 395.56 s,
  dropout 0.1 → 339.87 s (*better* than clean 349.67 s), surge 1.5 → 512.49 s with 3182
  added vehicles (50.5 %), 9375 entered. The dropout effect is one seed on train; watch it in
  7.3 (degradation of Max-Pressure need not be monotonic).
- 15 rows appended to `docs/experiment_log.csv` (2 sanity, 3 smoke, 10 validation). No test
  or held-out flow was opened.
- Not done: optional per-step trace for error analysis (Phase 8 will add it).

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
| `docs/experiment_log.csv` | modified | 15 rows: `max_pressure` / `fixed_time` train sanity (seed 0), 3 hook smoke runs, 10 validation rows (seeds 0–4) | guidelines: every run logged |
| `.claude/plans/phase4-env-metrics-rule-baselines.md` | created / modified | plan; status executed, steps ticked, "Deviations during execution" section | plan record |
| `ROADMAP.md` | modified | 4.1–4.6 ticked with validation numbers; final-gate "Simple baseline established" ticked; 2 Decision Log rows (env timing; `act` signature, perturbation semantics, test guard) | progress + decisions |
| `docs/evaluation_protocol.md` | modified | §9 row: queue sampled at every env step (clarification, no value changed) | asynchronous decisions make "decision step" ambiguous |
| `docs/how_it_works.md` | modified | §4.1 paragraph "Each intersection keeps its own clock" | keep the explainer in sync with the env |
| `.claude/reference/api.md` | modified | Environment, Agent, Evaluation sections rewritten as implemented (`make_env`, `TrafficEnv`, `Transition`, metrics / perturb / engine, `act(obs, info, explore)`, registry, runner, stats); `evaluate.py` script line | reference must match the code |
| `.claude/reference/architecture.md` | modified | simulator layer: per-intersection timing, perturbation semantics, test guard, extra engine calls | architecture record |
| `.claude/reference/testing.md` | modified | env, metrics / perturb / stats, rule-agent and evaluation test bullets | test inventory |
| `CLAUDE.md` | modified | Phase 4 `evaluate.py` command; planned evaluate line takes `--agent`; structure lists `envs/*` and `eval/`; env-timing note in Architecture | project instructions current |
| `README.md` | modified | new §7 "Environment and rule baselines" (commands, outputs, expected numbers, EDA off-by-one note); Tests → §8; layout lines | reproducibility (G12, ROADMAP 9.4) |
| `worklog/015-…`, `worklog/README.md` | created / modified | this entry and its index row | work-log rule |

## Commits
- `4cbb5c7` refactor(env): move engine factory to envs, add shared agent interface
- `1147d0d` feat(env): add CityFlow env wrapper, metrics and robustness hooks
- `6459a2e` feat(baselines): add Fixed-Time and Max-Pressure agents
- `d7b1459` feat(eval): add evaluation harness and scripts/evaluate.py
- `9650f51` exp(baselines): Fixed-Time and Max-Pressure on Jinan train and val
- `3e6a50b` docs: record Phase 4 env, interfaces and evaluation

## Next
- Phase 5: DQN core, shared-parameter DQN and IDQN on top of `Transition` (train with
  `track="light"`), `scripts/train.py` with validation checkpoint selection.
- Phase 8: add the per-step trace to `evaluate.py` for error analysis.
- Native Ubuntu (0.1b): re-run the validation baselines; numbers must match exactly.
