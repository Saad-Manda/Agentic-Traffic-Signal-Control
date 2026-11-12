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
- **Metrics**: travel time / wait / throughput computed correctly on a toy episode.
- **Seeding**: same config + seed → identical short rollout.
- **No data leakage**: train/val/test split files have no overlapping vehicles; nothing is
  fitted (capacity scaling); evaluation / calibration code refuses test data outside `--final`.
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
- **Core, Phase 6 additions** (`tests/agents/test_core.py`): replay extras round trip /
  wrap-around / column check; extras do not change the sampling stream; masked ensemble loss
  by hand (incl. an empty head) and = Phase 5 loss for K = 1; K = 1 learner stores no mask
  and its loss equals `smooth_l1` on the same batch (DQN / IDQN regression); Bernoulli(0.8)
  masks; seeded mask stream; shared checkpoints load on other networks, independent do not.
- **UG-MARL network** (`tests/agents/test_ugmarl_network.py`): encoder slots E, N, W, S and
  boundary zeros; `encode_transition` = row of `encode_all`; α sums to 1 and is 0 on absent
  slots; a neighbour's state changes Q only with attention; stacked heads = loop of heads;
  a head's loss reaches only that head (+ encoder); parameter-count formula.
- **UG-MARL agent / gate** (`tests/agents/test_ugmarl_agent.py`, `test_gate.py`): variant
  names; flags build the right network; τ = None ungated; −ensemble σ = 0, never fires;
  **τ = −1 reproduces Max-Pressure's actions over an episode**, τ = ∞ = gate off; strict σ > τ;
  exploration never gated; architecture-checked checkpoints (`use_gate` may differ); shared
  agent loads on a network of another size; determinism; σ by hand; percentiles; smaller-p
  tie rule; `gate.json` round trip.
- **Explanations / gate in evaluation** (`tests/eval/test_explain.py`): one recorder row per
  ready decision, fallback share = `fallback_rate`, phase taken = fallback ? MP : greedy;
  τ sources (config, `gate.json`, `{seed}` file, off) and refusal without τ before any output;
  per-seed τ in `evaluate`; `-nogate` run / log names. **Tuning**: `tie_break`, full grid.
  **Calibration script** (`tests/scripts/test_calibrate_gate.py`): refuses train / test /
  held-out, variants without a gate, missing checkpoints.
- **Real-simulator checks (not unit tests, worklog/017):** DQN training bit-identical before /
  after the core change; τ = 0 on Jinan val = Max-Pressure's 326.11 s exactly.
