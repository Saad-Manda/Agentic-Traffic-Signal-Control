# Testing

Runner: `pytest` (tests in `tests/`, mirroring `src/trafficrl/`). Lint: `ruff check src tests`.

## Run subsets

```bash
pytest                              # all
pytest tests/agents                 # agent tests only
pytest -k gate                      # by keyword
pytest -m "not slow"                # skip simulator-heavy tests (mark them @pytest.mark.slow)
```

## What to test

- **Interface conformance**: every agent implements `act/observe/update/save/load`
  (`reference/api.md`) and returns a valid phase per intersection.
- **Max-Pressure**: picks the max-pressure phase on hand-built states.
- **UG-MARL gate**: σ ≤ τ → RL action; σ > τ → Max-Pressure action; ensemble σ computed correctly.
- **Attention**: weights sum to 1 over N(i) ∪ {i}; shapes correct.
- **Metrics**: travel time / wait / throughput computed correctly on a toy episode.
- **Seeding**: same config + seed → identical short rollout.
- **No data leakage**: train/val/test split files have no overlapping vehicles; normalization
  stats computed only from train; evaluation code refuses to load test data in tuning mode.
- **Roadnet** (`tests/data/test_roadnet.py`): canonical lane order, movements, phases 1–8,
  neighbours, capacities on a hand-built roadnet (`tests/data/toy.py`) and on both real files.
- **Audit** (`tests/data/test_audit.py`): one synthetic defect per check is counted; a clean
  flow gives zero.
- **Features** (`tests/envs/test_features.py`): exact 44-vector on hand-built counts, bounds,
  clipping, group selection, pressure definitions, builder works with no flow file present.
- **EDA leakage guards** (`tests/data/test_trace.py`): traces refuse every dataset / split but
  Jinan train; the audit plots train only; the EDA notebook names no other split or file.
- **Run log** (`tests/utils/test_runlog.py`): header once, rows appended, `env_info` keys.
- **Env** (`tests/envs/test_traffic_env.py`, fake engine in `tests/envs/fake_engine.py`):
  keep → next decision after 15 s, change → yellow 5 s + green, 5 s grid, episode end closes
  transitions, pending-green one-hot, reward from true counts under noise, split guard; slow
  Jinan tests: `neighbor_index` vs `neighbors`, determinism, reset = fresh engine, surge on val.
- **Metrics / perturb / stats** (`tests/envs/test_metrics.py`, `test_perturb.py`,
  `tests/utils/test_stats.py`): hand-fed tracker values; noise bounds, σ, seeding; dropout
  rate; surge duplicates; t-values, known CI, deterministic flag.
- **Rule agents** (`tests/agents/test_rule_agents.py`, `test_interface.py`): Max-Pressure
  choice + ties; Fixed-Time decision times (0, 15, 30, 50, 65 s) and order; every registered
  agent conforms to the interface.
- **Evaluation** (`tests/eval/test_evaluate.py`): CLI and API refuse test / heldout without
  `--final`; slow end-to-end run writes metrics / summary / log rows (temp log).
- **Smoke (slow)**: 300-step CityFlow trace is identical for the same seed; plus a few training steps on a tiny network run end-to-end without error.

Experiment evaluation (not unit tests): results across ≥ 3 seeds, reported as mean ± std **and
95% CI**, saved to `results/<experiment>/metrics.csv`; every run also appends a row to
`docs/experiment_log.csv` and saves software versions + hardware.

## Reproducibility check (guidelines G12, ROADMAP 9.7)

Before submission: clone into a fresh directory, follow only the README, regenerate the main
comparison table (7.1) and one figure. Pass = results within seed variance of the reported
numbers, with no hidden manual steps. Log the outcome in `worklog/`.
- **Q-learning core** (`tests/agents/test_core.py`): replay wrap-around, per-group sampling,
  seeded batches; `StackedMLPQNet` = loop of MLPs; hand-computed Double-DQN SMDP target
  (15 s vs 20 s), bootstrap at the episode cut; observe only in training mode; update
  schedule; target sync; ε exploration covers all phases, greedy is deterministic.
- **DQN / IDQN** (`tests/agents/test_dqn_idqn.py`): param counts, shared vs per-intersection
  groups, IDQN gradient isolation between intersections, save / load round trip and wrong-agent
  refusal, same seed → same weights, `make_agent` reads the `train:` block.
- **Training** (`tests/train/test_loop.py`, fake engine via monkeypatched `make_env`): refuses
  val / test / held-out as training split (script exit 2), opens only train + val without
  `allow_final`, writes every output + one log row, val episodes never enter the buffer, same
  seed → identical checkpoint. **Tuning** (`tests/train/test_tune.py`): search file complete,
  sampler deterministic / distinct / in-space, selection rule. **Report helpers**
  (`tests/eval/test_report.py`).
