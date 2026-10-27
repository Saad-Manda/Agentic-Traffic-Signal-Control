# Testing

Runner: `pytest` (tests in `tests/`, mirroring `src/trafficrl/`). Lint: `ruff check src tests`.

## Run subsets

```powershell
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
- **Smoke (slow)**: a few training steps on a tiny network run end-to-end without error.

Experiment evaluation (not unit tests): results across ≥ 3 seeds, mean ± std, saved to
`results/<experiment>/metrics.csv`.
