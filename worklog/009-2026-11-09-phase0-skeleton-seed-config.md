# 009 — Phase 0: repo skeleton, pinned environment, seeding + config utilities
**Date:** 2026-11-09   **Tag:** setup   **ROADMAP:** 0.2, 0.3   **Status:** in-progress

## What
Created the `src/trafficrl` package skeleton, `pyproject.toml` (editable install + ruff/pytest
config) and an exact-pinned `requirements.txt`. Installed the ML stack into `.venv` (WSL).

## Why
ROADMAP 0.2: every later phase imports from `src/trafficrl`; pins keep WSL (dev) and native
Ubuntu (final numbers) identical (G12 reproducibility).

Decisions (asked 9 Nov):
- **pyproject + `pip install -e .`** over requirements-only: notebooks/scripts `import
  trafficrl` with no `sys.path` hacks; one place for ruff/pytest config.
- **Exact `==` pins** (full `pip freeze`, incl. transitive) over `>=` bounds: the two machines
  must not drift.
- 0.1b on native Ubuntu stays open — must be run by the user after booting native Ubuntu.

## How
- `pip install torch numpy pandas matplotlib seaborn pyyaml pytest jupyterlab tensorboard`
  → torch **2.15.0+cu132** (CUDA 13.2 bundled), `torch.cuda.is_available() == True` on the
  RTX 4050 through the WSL driver (610.62). numpy 2.5.4, pandas 3.1.1.
- `pip freeze --exclude-editable`, minus CityFlow (built from source) → `requirements.txt`
  (138 pins + header comment).
- Skeleton: `src/trafficrl/{envs,agents/baselines,agents/ugmarl,utils}`, `configs/`,
  `tests/`, `notebooks/`; README install steps + layout updated; `*.egg-info/` ignored.

## Result
`pip install -e .` works; `ruff check src tests` clean.

## Files / commits
`pyproject.toml`, `requirements.txt`, `src/trafficrl/**/__init__.py`, `notebooks/.gitkeep`,
`README.md`, `.gitignore`, `ROADMAP.md`

## Next
0.3 seeding + config loader.
