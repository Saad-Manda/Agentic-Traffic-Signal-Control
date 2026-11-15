# 030 — Repo cleanup, step 4: move the notebook-only modules into the notebooks (2026-11-15)

## What
Moved the code that only the two notebooks used into the notebooks themselves and deleted
the five modules: `data/{audit,eda,trace}.py` (613 lines) → `01_eda_train.ipynb`, and
`eval/{analysis,figures}.py` (1,729 lines) → `02_analysis.ipynb`.

## Why
User decision: a student project keeps its EDA and analysis logic in the notebook, and the
old "notebooks hold no logic" rule is what created these modules. About 40 % of `src/` was
there only to feed a notebook.

## How
- **01_eda_train:** a "Helper functions" cell defines the 10 EDA helpers it uses
  (`train_flow` with the train-only guard, `arrival_counts`, `movement_demand`,
  `feature_arrays`, …). A new "Simulator rollouts" cell runs the 12 rule-controller
  rollouts in memory (Fixed-Time / Max-Pressure / random × seeds 0–2, plus a Max-Pressure
  Δt sweep). Before, the notebook read traces from `results/eda/`, which was no longer
  on disk and had no script left to make it. Rows are recorded every **10 s**, the
  `decision_interval` in `base.yaml` when the EDA first ran (8766017); with the current
  15 s, three tables differed in the second decimal.
- **02_analysis:** a scratchpad inliner placed each `analysis` / `figures` definition (81)
  in a code cell right before its first use, together with its dependencies, and stripped
  the `A.` / `F.` prefixes. It checked that no notebook variable shadows an inlined name.
  The setup lines were split from the imports cell, and the imports were merged into one
  cell. `trafficrl.eval.{faithfulness,hypotheses,report}` stay as modules because
  scripts use them.
- Notebooks normalised with `nbformat` (cell ids).

## Result
- Both notebooks execute top to bottom (`01` ≈ 1 min, `02` ≈ 1 min).
- `01`: all 10 compared code-cell outputs identical to the saved ones; all 8 EDA PNGs
  byte-identical.
- `02`: all 23 matched cells give identical text / table output; all result PNGs
  byte-identical. PDFs differ only in `CreationDate`, so the originals were restored.
- `ruff check src tests scripts` and `ruff check notebooks --select F`: pass. `pytest`: 10 passed.

## Files changed
| File | Action | What changed | Why |
|---|---|---|---|
| `src/trafficrl/data/{audit,eda,trace}.py` | deleted | whole modules | moved into notebook 01 (only `arrival_counts`, the EDA aggregations, `train_flow`, the chooser and the rollout loop were used) |
| `src/trafficrl/eval/{analysis,figures}.py` | deleted | whole modules | moved into notebook 02 |
| `notebooks/01_eda_train.ipynb` | modified | new imports cell, helper cell, setup cell, rollout cell (+2 markdown); `eda.` prefixes dropped; intro "Inputs" text; re-executed | self-contained EDA |
| `notebooks/02_analysis.ipynb` | modified | 24 helper cells inserted before first use; one imports cell; setup split out; intro bullet about `src/` functions removed; re-executed | self-contained analysis |
| `README.md` | modified | §10 note: helpers are defined in the notebook | module names gone |
| `CLAUDE.md` | modified | conventions (notebooks define their own helpers), structure lines for `notebooks/`, `data/`, `eval/` | match the code |
| `worklog/030-…`, `worklog/README.md` | created / modified | this entry and its index row | mandatory work log |

## Commits
Not committed yet.

## Next
User to decide which other scripts / modules move into notebooks (see the list given in chat).
