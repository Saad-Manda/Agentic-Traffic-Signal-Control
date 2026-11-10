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
- **Smoke (slow)**: 300-step CityFlow trace is identical for the same seed; plus a few training steps on a tiny network run end-to-end without error.

Experiment evaluation (not unit tests): results across ≥ 3 seeds, reported as mean ± std **and
95% CI**, saved to `results/<experiment>/metrics.csv`; every run also appends a row to
`docs/experiment_log.csv` and saves software versions + hardware.

## Reproducibility check (guidelines G12, ROADMAP 9.7)

Before submission: clone into a fresh directory, follow only the README, regenerate the main
comparison table (7.1) and one figure. Pass = results within seed variance of the reported
numbers, with no hidden manual steps. Log the outcome in `worklog/`.
