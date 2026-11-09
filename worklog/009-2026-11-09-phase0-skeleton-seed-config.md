# 009 — Phase 0: repo skeleton, pinned environment, seeding + config utilities
**Date:** 2026-11-09   **Tag:** setup   **ROADMAP:** 0.2, 0.3   **Status:** done

## What
Created the `src/trafficrl` package skeleton, `pyproject.toml` (editable install + ruff/pytest
config) and an exact-pinned `requirements.txt`. Installed the ML stack into `.venv` (WSL).
Added the seeding utility, the YAML config loader, `configs/base.yaml` and their tests.

## Why
ROADMAP 0.2–0.3: every later phase imports from `src/trafficrl`; pins keep WSL (dev) and native
Ubuntu (final numbers) identical (G12 reproducibility).

Decisions (asked 9 Nov):
- **pyproject + `pip install -e .`** over requirements-only: notebooks/scripts `import
  trafficrl` with no `sys.path` hacks; one place for ruff/pytest config.
- **Exact `==` pins** (full `pip freeze`, incl. transitive) over `>=` bounds: the two machines
  must not drift.
- **Strict GPU determinism** (`torch.use_deterministic_algorithms`, cuDNN deterministic,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`) over seed-only: small MLPs lose ~nothing and identical
  re-runs make debugging and the fresh-clone check (9.7) meaningful.
- Config loader kept minimal (plain dicts + `base:` inheritance + dotted overrides) instead of
  Hydra/OmegaConf: no extra dependency, enough for one-YAML-per-run.
- 0.1b on native Ubuntu stays open — must be run by the user after booting native Ubuntu.

## How
- `pip install torch numpy pandas matplotlib seaborn pyyaml pytest jupyterlab tensorboard`
  → torch **2.15.0+cu132** (CUDA 13.2 bundled), `torch.cuda.is_available() == True` on the
  RTX 4050 through the WSL driver (610.62). numpy 2.5.4, pandas 3.1.1.
- `pip freeze --exclude-editable`, minus CityFlow (built from source) → `requirements.txt`
  (138 pins + header comment).
- Skeleton: `src/trafficrl/{envs,agents/baselines,agents/ugmarl,utils}`, `configs/`,
  `tests/`, `notebooks/`; README install steps + layout updated; `*.egg-info/` ignored.

- `utils/seed.py`: `seed_everything(seed, deterministic=True)`, `seed_cityflow(engine, seed)`,
  `make_rng(seed)`; validates seed ∈ [0, 2³²).
- `utils/config.py`: `load_config(path, overrides)` with `base:` chains (cycle detection),
  `"a.b=val"` overrides parsed as YAML, required integer `seed`; `save_config` for the resolved
  config.
- `configs/base.yaml`: seed, deterministic, sim (1 thread, 3600 s, Δt = 10 s, 5 s yellow —
  marked PROVISIONAL until the 3.5 freeze), data split placeholder.

## Result
`pip install -e .` works; **29 tests pass** (`pytest`, 4 s), `ruff check` + `ruff format
--check` clean. The CUDA determinism test runs on the RTX 4050.

Dead end worth knowing: the first override test failed because PyYAML (YAML 1.1) loads `5e-4`
— no decimal point — as the **string** `"5e-4"`. A config line `lr: 1e-3` would have silently
become text. Fixed with a SafeLoader subclass adding a float resolver for scientific
notation; regression test `test_scientific_notation_is_float`.

## Files / commits
`pyproject.toml`, `requirements.txt`, `src/trafficrl/**/__init__.py`, `notebooks/.gitkeep`,
`README.md`, `.gitignore`, `ROADMAP.md` · commit 028f932 (0.2)
`src/trafficrl/utils/{seed,config}.py`, `configs/base.yaml`, `tests/utils/`,
`.claude/reference/api.md`, `CLAUDE.md` · commit 31a9384 (0.3)

## Next
- User: run 0.1b on native Ubuntu (README steps 1–6 + `scripts/bench_cityflow.py`), record
  numbers in worklog/007; that closes Phase 0.
- Phase 1 dataset selection + split plan (due 10 Nov).
