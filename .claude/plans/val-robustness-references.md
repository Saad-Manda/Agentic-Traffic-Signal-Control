# Plan — 6.8 Validation robustness references (ROADMAP 6.8)

Status: **executed 12 Nov 2026** — outcome mixed: (a) holds against DQN / IDQN, tie with
Max-Pressure under noise (worklog/019). Tick steps `[x]` during execution and note
deviations under each step.

## Goal

Phase 6 measured the gate under degraded detectors only against the same model without the
gate (−gate: 599 / 909 s at noise 0.1 / 0.2; gated: 376 / 423 s). The baselines were never run
under those conditions, so "UG-MARL is more robust than existing methods" is not yet shown.
This plan runs **every baseline under the same validation perturbations** and puts them in one
table next to UG-MARL and −gate, **before** the test flows are opened in Phase 7.

| Output | Serves |
|---|---|
| `scripts/robustness.py` — model × condition evaluation + table, driven by a spec file | F6, F8, R5, G9, G11 |
| `docs/results/val_robustness.{csv,md}` — 6 models × 5 conditions on Jinan val | F8, R6, G11 |
| The same script is reused for ROADMAP 7.3 (test, `--final`, + surge) with a second spec | F8, G10 |

Out of scope: any retraining, any change to checkpoints / K / d / p\* / `gate.json`, the
evaluation protocol, test or held-out flows, demand surge (untuned test condition, 7.3),
the −attention / −ensemble ablations (7.2 / 7.3).

## Decisions

| # | Topic | Decision | Reason |
|---|---|---|---|
| V1 | Models | Fixed-Time, Max-Pressure, DQN, IDQN, UG-MARL, UG-MARL −gate | All four baselines plus the two rows the claim is about. |
| V2 | Conditions | clean, noise 0.1, noise 0.2, dropout 10 %, dropout 20 % | Exactly the conditions of `configs/search/ugmarl_gate.yaml`; nothing new is introduced on validation. |
| V3 | Seeds | 0–4; learned seed s = checkpoint trained with seed s (`best.pt`) | Same as `docs/results/val_phase6.md`. Rule-based agents are also run on 5 seeds: their decisions are deterministic, but noise / dropout draws depend on the seed, so std may be > 0 under perturbation. |
| V4 | UG-MARL rows | **Re-run** UG-MARL and −gate through the new script (τ from each `gate.json`, p = 80) instead of copying the calibration table | One code path for all rows; the re-run must reproduce the calibration numbers (determinism check, Verification step 4); the script is then ready for 7.3 unchanged. Cost: 50 extra episodes. |
| V5 | Clean column | Re-run clean for every model too | Must reproduce `val_phase6.md` (second determinism check); keeps the table self-contained. |
| V6 | No selection | Nothing is chosen from this table | ROADMAP 6.8 rules; readings (a) / (b) were written in the ROADMAP before the runs. |

## Design

**Spec file** `configs/robustness_val.yaml` (written and committed before the first run):

```yaml
study: val_robustness
dataset: jinan
split: val
seeds: [0, 1, 2, 3, 4]
tag: final                       # training runs <model>_<tag>_s<seed>
models:                          # row order of the table
  fixed_time:    {config: configs/eval_baselines.yaml}
  max_pressure:  {config: configs/eval_baselines.yaml}
  dqn:           {config: configs/eval_rl_baselines.yaml}
  idqn:          {config: configs/eval_rl_baselines.yaml}
  ugmarl:        {config: configs/eval_ugmarl.yaml}
  ugmarl-nogate: {config: configs/eval_ugmarl.yaml, agent: ugmarl, train_model: ugmarl,
                  overrides: [agents.ugmarl.use_gate=false]}
conditions:                      # same as configs/search/ugmarl_gate.yaml
  - {}
  - {robustness.noise_std: 0.1}
  - {robustness.noise_std: 0.2}
  - {robustness.dropout_p: 0.1}
  - {robustness.dropout_p: 0.2}
```

**`scripts/robustness.py`**

```
python scripts/robustness.py --spec configs/robustness_val.yaml [--workers 5] \
    [--out docs/results/val_robustness] [--table-only] [--final]
```

1. Read the spec. `split` in `FINAL_SPLITS` without `--final` → exit 2 (same rule as
   `evaluate.py`; `--final` is passed through). Missing checkpoint or `gate.json` → exit 2
   before anything runs.
2. One job per (model, condition): `scripts/evaluate.py --config <config> --agent <agent>
   --split <split> --seeds … [--checkpoint results/train/<train_model>_<tag>_s{seed}/checkpoints/best.pt]
   --note "<study>: <model>, <condition>" <model overrides> <condition overrides>
   output.results_dir=results/robustness/<study>/eval/<model>__<condition>`.
   Jobs run with the existing `trafficrl.train.tune.run_parallel` (logs in
   `results/robustness/<study>/logs/`). `evaluate.py` writes the experiment-log rows and
   `env.json` as usual — no new evaluation code.
3. A (model, condition) directory that already holds a finished run is skipped (resume after
   a crash); `--table-only` runs no job.
4. Table: `trafficrl.eval.report.condition_table(runs)` reads each run's `metrics.csv` and
   returns one row per (model, condition): travel time mean / std / 95 % CI (Student t,
   `utils.stats.summarise`), max per-vehicle wait (mean over seeds), fallback rate, n seeds.
   The script writes `<out>.csv` (long format) and `<out>.md`: intro paragraph (flows, seeds,
   what is locked, "validation, optimistic"), a travel-time table (model × condition), a
   max-wait table (model × condition), and the degradation "noise 0.2 − clean" per model.

No change to agents, env, runner or the agent interface. No new maths.

## Justification

- **Why on validation, before Phase 7:** test flows are run once. If the baselines turn out
  robust on their own, validation is the only place where that can still inform a decision.
  Alternative "wait for 7.3" rejected for that reason (user decision, 12 Nov).
- **Why a script, not 20 manual `evaluate.py` calls:** every table must come from a command;
  7.3 needs the same thing on test, so the code is written once.
- **Why not extend `val_table.py`:** it is one row per model on the clean flow with training
  columns; a model × condition layout is a different table. **Why not extend
  `calibrate_gate.py`:** it selects p and writes `gate.json`; a script that must not select
  anything should not share that code path.
- **Why re-run UG-MARL instead of reusing calibration evaluations (V4):** avoids a table
  stitched from two pipelines, and gives a reproduction check for free.

## Files

| File | Action | Purpose |
|---|---|---|
| `configs/robustness_val.yaml` | create | Spec: models, seeds, conditions (committed before the runs) |
| `scripts/robustness.py` | create | Job builder, parallel run, guards, table writer |
| `src/trafficrl/eval/report.py` | modify | `condition_table()` — pure reader of finished runs |
| `tests/scripts/test_robustness.py` | create | Job construction, guards |
| `tests/eval/test_report.py` | modify | `condition_table` on fake `metrics.csv` files |
| `docs/results/val_robustness.{csv,md}` | create (generated) | The table |
| `docs/experiment_log.csv` | modify (by `evaluate.py`) | 150 rows, all `val` |
| `ROADMAP.md` | modify | 6.8 command wording → `scripts/robustness.py`; tick; Decision Log row with the outcome |
| `README.md`, `CLAUDE.md` | modify | Command added to the script list |
| `worklog/019-2026-11-12-val-robustness-references.md`, `worklog/README.md` | create / modify | Work log |

## Config

New file `configs/robustness_val.yaml` only (keys above). No new key in `base.yaml`; the
conditions use the existing `robustness.noise_std` / `robustness.dropout_p`. Seeds 0–4 are
passed to `evaluate.py`, which seeds Python, NumPy, PyTorch, CityFlow and the perturbation RNG.

## Data-leakage check

- Reads the **validation** flow only (`real_2000`). The script refuses test / held-out without
  `--final`, and this plan never passes `--final`.
- Nothing is fitted or selected: checkpoints, K, d, p\* and τ were fixed in Phase 6; the table
  is read against readings (a) / (b) written in ROADMAP 6.8 beforehand.
- Validation was already used for selection, so these numbers are labelled optimistic; the
  reported robustness result remains 7.3 on test.

## Tests

- `test_robustness.py`: argv per model (rule agents have no `--checkpoint`; −gate carries
  `use_gate=false` and the full model's checkpoints; condition overrides present; separate
  results dir per job); test split without `--final` → exit 2; missing checkpoint → exit 2;
  finished job is skipped.
- `test_report.py`: `condition_table` returns the right mean / std / CI / n for two fake runs
  and marks a zero-variance row as deterministic.

## Steps and verification

1. [x] `configs/robustness_val.yaml`; commit (spec before runs).
2. [x] `condition_table` + test; `scripts/robustness.py` + tests.
   `pytest tests/scripts/test_robustness.py tests/eval/test_report.py` and `ruff check src tests scripts` pass.
3. [x] Smoke: a throw-away spec (Max-Pressure + DQN, seed 0, clean + noise 0.1) with a
   scratch output dir → 4 runs, table written; rows labelled smoke in the log note.
   *Deviation:* the first smoke used a results dir outside the repo; the 4 episodes ran and
   were logged, then `evaluate.py` failed printing a repo-relative path. Re-run with the
   default `results/robustness/` → 8 smoke rows in the log instead of 4.
4. [x] Full run: `python scripts/robustness.py --spec configs/robustness_val.yaml --workers 5`
   → 30 jobs, 150 episodes. Checks:
   - clean column equals `docs/results/val_phase6.md` (Fixed-Time 434.8, Max-Pressure 326.1,
     DQN 299.8 ± 1.0, IDQN 312.7 ± 2.5, UG-MARL 302.3 ± 1.1, −gate 301.8 ± 0.8);
   - UG-MARL / −gate perturbed columns equal `docs/results/gate_calibration.md`
     (p = 80 and ungated rows);
   - 150 new `val` rows in `docs/experiment_log.csv`, none `test` / `heldout`.
   A mismatch is investigated and logged before the table is used.
   *Result:* all three checks pass (158 rows with the smoke runs).
5. [x] Read the table against ROADMAP 6.8 (a) / (b); report to the user. Outcome (b) → stop,
   the next step is the user's decision.
6. [x] ROADMAP (tick 6.8, Decision Log), README / CLAUDE.md commands, worklog/019, commits
   per `.claude/commands/commit.md`.

## Experiment log

150 rows appended by `evaluate.py` (6 models × 5 conditions × 5 seeds), split `val`,
note `val_robustness: <model>, <condition>`; plus ≤ 4 smoke rows. No tuning, no search space
needed (nothing is selected).

## Risks / open questions

- **Decision latency** is distorted by parallel workers → not reported in this table.
- **Max wait is noisy** over 5 seeds (std ≈ 120–150 s on clean val); the reading rests on
  travel time, max wait is shown as supporting evidence.
- If the clean or UG-MARL columns do not reproduce earlier tables, the cause (nondeterminism,
  config drift) must be found first; that would also matter for Phase 7.
- Outcome (b) is possible: Max-Pressure reads the same noisy detectors, but a greedy
  pressure rule may degrade less than a learned Q-function; the gated model at noise 0.2
  falls back 96 % of the time, so it can at best match Max-Pressure there.

## Worklog

Execution ends with `worklog/019-2026-11-12-val-robustness-references.md` (What / Why / How /
Result / Files changed per file / Commits / Next) and its row in `worklog/README.md`.
