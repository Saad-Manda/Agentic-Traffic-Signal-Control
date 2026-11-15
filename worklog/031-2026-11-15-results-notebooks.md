# 031 — Repo cleanup, step 5: result and analysis scripts become notebooks (2026-11-15)

## What
The five scripts that read results or run the validation-only checks are now notebooks:
- `scripts/{plot_training,val_table,final_tables}.py`, plus the table helpers of
  `eval/report.py` → new **`notebooks/02_results.ipynb`**.
- `scripts/{sigma_check,attention_faithfulness}.py`, `eval/faithfulness.py` and the
  faithfulness section of the analysis notebook → new **`notebooks/04_gate_attention.ipynb`**.
- `02_analysis.ipynb` was renamed **`03_error_analysis.ipynb`**.

`scripts/` went from 12 files to 7 (train, tune, evaluate, calibrate_gate, robustness,
lock_models, tau_sweep). `scripts/` + `src/trafficrl/eval/` are now 1,830 lines.

## Why
User decision (layout agreed in chat): a student keeps result tables, plots and small
analysis studies in notebooks and keeps only the long-running pipeline as scripts.
`eval/hypotheses.py` stays a module because three notebooks use it, and it holds the
verdict rules fixed before the test runs.

## How
- Two scratchpad builders pull the function source straight from the old files (`ast`
  segments) and replace `args.*` with notebook constants.
  - `write()` now saves the CSV and shows the readable table inline, instead of writing a
    `.md` file.
  - ROADMAP / worklog / protocol references were dropped from the table titles.
- `03_error_analysis`:
  - `study_runs` / `load_study` were inlined next to `Study`.
  - `finish()` saves PNG + PDF only, with no `.md` caption.
  - The unused `per_seed` helper and the `INTERPRETATION` constant were removed.
  - The faithfulness cells (heading, helper, display, reading) moved to `04`.
- `04` re-simulates the 10 validation episodes (no cache). It asserts the validation split,
  and that each faithfulness episode reproduces the `val_phase6.csv` travel time.
- `eval/report.py` keeps `fmt`, `condition_table`, `condition_name` and `finished_run`
  (used by `robustness.py`, `tau_sweep.py` and the notebooks): 151 → 61 lines.
- Dev-only outputs:
  - `docs/results/val_phase5.csv` deleted. It was built from WSL Phase 5 eval runs that
    are not on this machine, so it could not be rebuilt even by the old script, and
    nothing uses it.
  - `learning_curves_phase6.png` became `learning_curves.png`, replacing the WSL dev
    figure.

## Result
- **Baseline first:** the old scripts reproduced every committed file except
  `val_phase5.csv` and the dev `learning_curves.png` (the reason is above).
- **`02_results`:**
  - `val_phase6.csv`, all `test_*.csv`, `heldout_generalisation{,_all}.csv` are
    byte-identical.
  - The new `learning_curves.png` is byte-identical to the old `learning_curves_phase6.png`.
  - Verdicts: H1 supported, H2 supported, H3 not supported, smallest worst-case regret
    UG-MARL.
- **`04_gate_attention`:** `sigma_validity.csv`, `sigma_validity_td.csv` and
  `attention_faithfulness.csv` are byte-identical, as are `sigma_validity.png` and
  `attention_faithfulness.png`. The PDF differed only in `CreationDate`, so it was restored.
- **`03_error_analysis`:** re-executed; CSVs and PNGs are unchanged and the PDFs were
  restored.
- `ruff` (src / tests / scripts, and notebooks `--select F`) passes. `pytest`: 10 passed.

## Files changed
| File | Action | What changed | Why |
|---|---|---|---|
| `scripts/{plot_training,val_table,final_tables}.py` | deleted | whole files | now `02_results.ipynb` |
| `scripts/{sigma_check,attention_faithfulness}.py`, `src/trafficrl/eval/faithfulness.py` | deleted | whole files | now `04_gate_attention.ipynb` |
| `notebooks/02_results.ipynb` | created | learning curves, validation table, final tables 7.1–7.6 + verdicts | layout |
| `notebooks/04_gate_attention.ipynb` | created | σ under degradation, σ vs TD, attention faithfulness + figure + reading | layout |
| `notebooks/02_analysis.ipynb` → `03_error_analysis.ipynb` | renamed + modified | title; `load_study` inlined; `finish()` without `.md`; faithfulness section removed; heading §9 → §8; intro; unused `Callable` import | layout |
| `src/trafficrl/eval/report.py` | modified | removed 6 reader functions and unused imports; one-line docstring | moved to notebooks |
| `src/trafficrl/eval/hypotheses.py` | modified | docstring points to `02_results` | script removed |
| `scripts/calibrate_gate.py`, `scripts/robustness.py` | modified | docstring / message point to the notebooks | scripts removed |
| `configs/analysis.yaml`, `configs/final_test.yaml` | modified | comments point to the notebooks | scripts removed |
| `docs/results/val_phase5.csv` | deleted | WSL dev table | cannot be rebuilt; unused |
| `docs/figures/training/learning_curves.png` / `learning_curves_phase6.png` | modified / deleted | native figure under the plain name | drop the phase name |
| `README.md` | modified | steps 7–10 commands, repository layout | match the notebooks |
| `CLAUDE.md` | modified | commands, layout, conventions | match the notebooks |
| `worklog/031-…`, `worklog/README.md` | created / modified | this entry and its index row | mandatory work log |

## Commits
`3d2672f` refactor(notebooks): turn result and analysis scripts into notebooks

## Next
Code cleanup pass over the remaining scripts / src (ROADMAP / plan / worklog references in
docstrings and comments); remove the Claude / process files last.
