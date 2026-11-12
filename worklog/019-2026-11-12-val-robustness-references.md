# 019 — Validation robustness references (ROADMAP 6.8)
**Date:** 2026-11-12   **Tag:** decision, code, experiment   **ROADMAP:** 6.8   **Status:** done

## What
After Phase 6 (UG-MARL 302.3 s vs DQN 299.8 s on clean validation) the user asked what to do.
Decision: no retuning to beat DQN; first measure what was missing — **every baseline under the
detector perturbations used for the gate calibration**, on validation, before the test flows
are opened. Added ROADMAP task 6.8, a plan, a spec file, `scripts/robustness.py` and the table
`docs/results/val_robustness.md` (6 models × 5 conditions × 5 seeds, 150 episodes).

## Why
The gate's robustness was shown only against the same model without the gate. "Better than
existing methods" needs the baselines under the same conditions. 7.3 does this on test, but
test is run once; validation is the only place where a bad answer can still inform a decision
(user decision, 12 Nov: keep 6.8).

## How
- ROADMAP 6.8 and a Decision Log row written first, with the readings fixed before the runs:
  (a) DQN / IDQN degrade like −gate and Max-Pressure under noise is not better than gated
  UG-MARL → go to Phase 7; (b) otherwise → finding, user decides.
- Plan `.claude/plans/val-robustness-references.md`; spec `configs/robustness_val.yaml`
  committed before any run (`39fca33`).
- `scripts/robustness.py`: one `scripts/evaluate.py` process per (model, condition) over all
  seeds, run with `trafficrl.train.tune.run_parallel`; refuses test / held-out without
  `--final`; checks checkpoints and `gate.json` first; skips finished jobs; builds the table
  with the new `trafficrl.eval.report.condition_table`. No change to agents, env or runner.
- Smoke (Max-Pressure + DQN, seed 0, clean + noise 0.1). The first attempt wrote results to
  the session scratchpad: the episodes ran and were logged, then `evaluate.py` failed on
  printing a repo-relative path. Re-run with the default `results/robustness/` worked
  (8 smoke rows in the log, note `smoke_val_robustness: …`).
- Full run: `python scripts/robustness.py --spec configs/robustness_val.yaml --workers 5`,
  30 jobs, 3 min 37 s, native Ubuntu, CPU.

## Result
Checks: the clean column equals `docs/results/val_phase6.md` and the UG-MARL / −gate rows
equal `docs/results/gate_calibration.md` (p = 80, ungated) to the printed digit — the pipeline
is deterministic. 158 new log rows, all `val`.

Average travel time (s), mean ± std over seeds 0–4:

| Model | clean | noise 0.1 | noise 0.2 | drop 10 % | drop 20 % |
|---|---|---|---|---|---|
| Fixed-Time | 434.8 | 434.8 | 434.8 | 434.8 | 434.8 |
| Max-Pressure | 326.1 | 371.7 ± 4.6 | 414.9 ± 6.8 | 322.8 ± 1.3 | 321.7 ± 1.1 |
| DQN | 299.8 ± 1.0 | 553.3 ± 32.2 | 854.9 ± 35.7 | 304.8 ± 1.8 | 309.3 ± 1.9 |
| IDQN | 312.7 ± 2.5 | 528.5 ± 65.7 | 766.8 ± 53.5 | 319.1 ± 2.0 | 326.6 ± 1.9 |
| UG-MARL | 302.3 ± 1.1 | 376.5 ± 4.2 | 423.4 ± 7.5 | 306.0 ± 1.5 | 310.6 ± 2.4 |
| −gate | 301.8 ± 0.8 | 599.2 ± 54.9 | 908.8 ± 48.3 | 306.1 ± 1.9 | 310.3 ± 2.9 |

Max wait at noise 0.2: DQN 3407 s, IDQN 2875 s, −gate 3307 s, Max-Pressure 1277 s, UG-MARL
1238 s. Fallback rate of UG-MARL: 21 % clean, 81 % / 96 % under noise, 23 % / 25 % under dropout.

Reading against the pre-written rules — **mixed**:
- **(a) holds against the learned baselines.** DQN and IDQN lose 250–550 s under noise, like
  −gate; UG-MARL loses 74 / 121 s.
- **Tie with Max-Pressure under noise (the (b) element).** Max-Pressure is 5–9 s lower than
  UG-MARL at noise 0.1 / 0.2 with overlapping 95 % CIs. This is the ceiling of the design: at
  81–96 % fallback the gated model *is* mostly Max-Pressure.
- Under dropout UG-MARL stays with DQN and 12–17 s ahead of Max-Pressure.
- No baseline is good everywhere: DQN is best on clean / dropout and worst under noise;
  Max-Pressure is best under noise and 24 s behind on clean. UG-MARL is within 2.5 s of the
  best learned model on clean / dropout and within 9 s of Max-Pressure under noise. Mean over
  the five conditions: UG-MARL 344 s, Max-Pressure 351 s, Fixed-Time 435 s, DQN 464 s — but
  p\* was selected on this score, so it is optimistic for UG-MARL.
- Fixed-Time does not read detectors, so it is unchanged; at noise 0.2 it is only 12 s behind
  UG-MARL.
- Associations on one validation flow; the reported result is 7.3 on test.

Nothing was selected or retuned.

## Files changed
| File | Action | What changed | Why |
|---|---|---|---|
| `ROADMAP.md` | modified | New task 6.8 (rules, command, readings (a) / (b), result, ticked); two Decision Log rows (12 Nov: decision; 12 Nov: result) | User asked for the roadmap entry before any run; outcome recorded |
| `.claude/plans/val-robustness-references.md` | created | Plan: decisions V1–V6, design, leakage check, steps (ticked, deviation noted) | User asked for a plan before implementation |
| `configs/robustness_val.yaml` | created | Spec: 6 models (configs, −gate override), seeds 0–4, 5 conditions | Committed before the runs |
| `scripts/robustness.py` | created | `build_jobs`, `missing_inputs`, `finished_run`, `write_md`, `main` (`--spec/--workers/--out/--results-dir/--train-dir/--table-only/--final`) | Model × condition table from one command; reusable for 7.3 |
| `src/trafficrl/eval/report.py` | modified | `condition_table()`; `Mapping` import | Pure reader behind the table |
| `tests/scripts/test_robustness.py` | created | Job argv per model, missing checkpoint / `gate.json`, final splits refused, finished-job lookup, `--table-only` | Guards and job construction |
| `tests/eval/test_report.py` | modified | `test_condition_table`; import list | New helper |
| `docs/results/val_robustness.{csv,md}` | created (generated) | Travel time, max wait, fallback rate and increase over clean, model × condition | The 6.8 table |
| `docs/experiment_log.csv` | modified (by `evaluate.py`) | +158 rows, all `val`: 150 `val_robustness: …`, 8 `smoke_val_robustness: …` | Every run logged |
| `README.md`, `CLAUDE.md` | modified | `scripts/robustness.py` command; script list in the structure block | Commands documented |
| `worklog/019-2026-11-12-val-robustness-references.md`, `worklog/README.md` | created / modified | This entry; index row | Work log rule |

Not tracked (gitignored): `results/robustness/{val_robustness,smoke_val_robustness}/`.

## Commits
`39fca33` plan + ROADMAP 6.8 + spec · code and results commits: `git log --oneline -3 -- scripts/robustness.py docs/results/val_robustness.md`.

## Next
- **User decision:** Phase 7 as locked (claim: "tracks the better of DQN and Max-Pressure in
  every condition"; not "beats Max-Pressure under noise"), or one time-boxed, pre-registered
  improvement attempt first.
- 7.3: second spec (`split: test`, seeds 0–2, + surge, + ablations) for `scripts/robustness.py --final`.
- Possible small fix: `scripts/evaluate.py` fails after the run when `output.results_dir` is
  outside the repo (path printing only).
