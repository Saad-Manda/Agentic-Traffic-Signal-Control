# 029 — Repo cleanup, step 3: keep only the data-split tests (2026-11-15)

## What
Removed every test except `tests/data/test_splits.py`: 35 test files → 1, 275 tests → 10.

## Why
User decision: the user does not normally write test files, so a 275-test suite does not
match the rest of the submission. The split / leakage test stays because "no data leakage"
is the claim most worth defending in the viva, and it runs on its own in under a second.
The gate behaviour (τ = −1 → Max-Pressure, τ = ∞ → ungated) can still be shown with
`evaluate.py` with and without `agents.ugmarl.use_gate=false`. Everything removed is on
branch `archive/full-dev`.

## How
- First pass (agreed split): removed `tests/scripts/`, `tests/test_notebooks.py`,
  `tests/eval/` (all 9 files), `tests/data/test_{audit,trace,roadnet}.py`,
  `tests/train/test_tune.py` and `tests/utils/test_{config,runlog,stats}.py` (120 tests left).
- Second pass: removed `tests/{agents,envs,train,utils}/` and `tests/data/toy.py`.
- `test_splits.py` imports only `trafficrl.data.splits`, so it needs no helper.

## Result
`pytest`: 10 passed (the 2 tests that need `data/raw/` skip on a fresh clone).
`ruff check src tests scripts`: all checks passed.

## Files changed
| File | Action | What changed | Why |
|---|---|---|---|
| `tests/agents/*` (7), `tests/envs/*` (4 + `fake_engine.py`), `tests/eval/*` (9), `tests/scripts/*` (4), `tests/train/*` (2), `tests/utils/*` (4), `tests/data/test_{audit,trace,roadnet}.py`, `tests/data/toy.py`, `tests/test_notebooks.py` | deleted | whole files (+ package `__init__.py`) | user decision: keep only the split tests |
| `pyproject.toml` | modified | removed the `slow` pytest marker | no test uses it now |
| `README.md` | modified | "Tests and lint" and layout lines describe the split checks | match the remaining test |
| `notebooks/01_eda_train.ipynb` | modified | first cell: dropped the sentence about `tests/data/test_trace.py` | test removed |
| `CLAUDE.md` | modified | Testing section | match the remaining test |
| `worklog/029-…`, `worklog/README.md` | created / modified | this entry and its index row | mandatory work log |

## Commits
See `git log` (committed with this entry).

## Next
Remove `plot_training.py` (pending user answer); clean up the code (ROADMAP / plan IDs in
docstrings); remove `PRD.md`, `ROADMAP.md`, `worklog/`, `.claude/`, `CLAUDE.md` last.
