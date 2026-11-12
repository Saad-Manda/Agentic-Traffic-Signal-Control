# 020 — Phase 7: final experiments on the test flow and the held-out city
**Date:** 2026-11-12   **Tag:** decision, code, experiment   **ROADMAP:** 7.1–7.6   **Status:** done

## What
Planned and ran Phase 7: every model, locked as selected on validation, was evaluated once on
the Jinan test flow (8 models × 6 conditions × 5 seeds = 240 episodes) and zero-shot on
Hangzhou (7 models × 3 flows × 5 seeds = 105 episodes). Six result tables and the hypothesis
verdicts were produced by code written before the test flow was opened.

## Why
ROADMAP 7.1–7.6 (R5, R6, G10, G11). Plan: `.claude/plans/phase7-final-experiments.md`.

Decisions (ROADMAP Decision Log, 12 Nov):
- **Models as locked; no improvement attempt first** (user decision). The tie with
  Max-Pressure under noise (6.8) is the ceiling of a gate that falls back to Max-Pressure.
  Rejected: "run the test, then improve if UG-MARL loses" — a change chosen because of a test
  result uses the test flow for selection. Rejected: changing the architecture before the
  test — a 2.5 s gap to DQN on validation is too small to turn into a clear win, and it would
  need retraining, re-tuning and re-calibration 5 days before submission. A detector-smoothing
  extension is pre-specified in the plan but not started.
- **Seeds 0–4 for 7.3 / 7.4** (protocol said 0–2; user decision). Narrower CIs; the clean
  column of 7.3 is the same runs as 7.1 / 7.2, so each test episode runs once.
- **H2 read in three levels** (user decision): (i) max wait clean, (ii) max wait at noise 0.2,
  (iii) travel-time increase clean → noise 0.2. PRD §1c did not say where max wait is compared.
- **One pass, everything logged, `--workers 1`, lock manifest** (plan P5–P8). Rejected: a
  later deterministic re-run for the Phase 8 logs (a second pass over the test flow); a
  separate latency benchmark; `run_ablation.py` (ablations are rows of the same grid).

## How
1. Decisions and readings written to ROADMAP, protocol §6 / §9 and PRD §1c; committed.
2. Code: `--log-trace` (per-step and per-vehicle logs, wait p95 / p99), lock manifest,
   runner extensions (named conditions, logging flags, manifest check, unfinished-run guard),
   `trafficrl.eval.hypotheses`, `scripts/final_tables.py`. 240 tests pass, ruff clean.
3. Hangzhou code path tested without Hangzhou flows: Hangzhou roadnet + synthetic flow of
   two-road routes, weights from a Jinan-shaped agent (`tests/eval/test_zero_shot.py`).
4. Smoke on **validation**: the test spec with `split: val` (study `smoke_final`, 240
   episodes, `--workers 5`). All 30 cells shared with `val_robustness.csv` and the 8 clean
   rows of `val_phase6.csv` reproduce with difference 0.
5. `python scripts/lock_models.py --write` → `models/manifest.json` (25 runs); committed;
   tree clean at `94c1bbb`.
6. `python scripts/robustness.py --spec configs/final_test.yaml --final --workers 1 --results-dir results/final`
7. `python scripts/robustness.py --spec configs/final_heldout.yaml --final --workers 1 --results-dir results/final`
8. `python scripts/final_tables.py --study all`

Settings: native Ubuntu, CPU, one process, 3600 s episodes, Δt = 15 s, seeds 0–4 (seed s =
checkpoint trained with seed s), τ from each model's Jinan-validation `gate.json` (p = 80).

## Result
No job failed and nothing was re-run. 240 `test` and 105 `heldout` rows in the experiment
log, all at commit `94c1bbb`. Fixed-Time and Max-Pressure are identical across seeds on the
clean flow.

Average travel time (s), mean ± std over 5 seeds:

| Model | Val | Test clean | Test noise 0.1 | Test noise 0.2 | Test drop 20 % | Test surge ×1.5 | Hangzhou (3 flows) |
|---|---|---|---|---|---|---|---|
| Fixed-Time | 434.8 | 462.2 | 462.2 | 462.2 | 462.2 | 661.4 | 667.5 |
| Max-Pressure | 326.1 | 320.8 | 366.9 | 413.7 | 317.2 | 435.2 | 431.8 |
| DQN | 299.8 | 292.2 ± 0.6 | 569.8 | 828.2 | 304.8 | 373.0 | 410.4 ± 7.1 |
| IDQN | 312.7 | 307.9 ± 1.4 | 543.3 | 758.9 | 322.3 | 455.3 | not transferable |
| UG-MARL | 302.3 | 295.4 ± 1.1 | 370.4 | 420.5 | 305.1 | 375.0 | 410.0 ± 2.9 |
| −gate | 301.8 | 295.0 ± 1.0 | 597.7 | 870.2 | 305.4 | 376.4 | 420.1 ± 14.6 |
| −attention | 301.9 | 294.2 ± 1.1 | 369.3 | 419.3 | 304.4 | 376.3 | 403.8 ± 1.2 |
| −ensemble | 303.7 | 297.3 ± 1.5 | 611.9 | 864.9 | 311.6 | 395.1 | 423.7 ± 14.9 |

Verdicts (rules fixed before the runs):
- **H1 supported.** UG-MARL's CI [294.1, 296.8] is below Max-Pressure (320.8) and IDQN's CI
  [306.1, 309.7].
- **H2 supported by the rule, with a caveat.** (ii) max wait at noise 0.2: 1493 vs 3306 s
  (paired −1813 s, CI [−2306, −1320]); (iii) travel-time increase: +125 vs +575 s (paired
  −450 s, CI [−512, −388]). (i) clean max wait 896 vs 910 s counts as "lower" under the rule,
  but the paired difference is −15 ± 296 s — no real difference on clean data. On validation
  the same comparison went the other way (761 vs 691 s).
- **H3 not supported.** −attention 294.2 ± 1.1 s vs full 295.4 ± 1.1 s; −attention also has
  61 % fewer parameters and is ahead on Hangzhou (403.8 vs 410.0 s). Negative result.
- **Minimum outcome met** in all six conditions (UG-MARL max wait ≤ Max-Pressure's; clean
  896 vs 1219 s).
- **UG-MARL is worse than DQN on clean test data**: +3.2 s, CIs separated (validation: +2.5 s,
  CIs touching).
- **Worst-case regret over the six conditions:** UG-MARL 6.8 s, Max-Pressure 62.2 s,
  Fixed-Time 288.4 s, IDQN 345.2 s, DQN 414.6 s. The claim written in advance holds: smallest
  worst-case regret, and no win over Max-Pressure under noise (+3.5 / +6.8 s, CIs overlap).

Other findings:
- Ensemble without gate does not help: −ensemble and −gate both collapse under noise. The
  robustness comes from the gate, which needs the ensemble's σ.
- Surge ×1.5 (never tuned on): UG-MARL 375 s ≈ DQN 373 s, Max-Pressure 435 s. The gate fires
  on 55 % of decisions and max wait is 2162 s (−gate 1430 s, DQN 1161 s, Max-Pressure 3260 s),
  so under surge the fallback raises the worst wait towards Max-Pressure's.
- Hangzhou: the learned shared policies transfer (410 s vs Max-Pressure 432 s, Fixed-Time
  668 s). The gate fires on 34 % of decisions and narrows the seed spread (± 2.9 vs ± 14.6 s
  without the gate). UG-MARL's max wait (2262 s) is below Max-Pressure's (2734 s) but above
  DQN's (1660 s).
- Gate statistics (test, clean): fallback 20.6 %; the learned phase differs from the
  Max-Pressure phase in 59 % of decisions; the gate changes the action in 12.4 %.
- Attention (clean): mean weight 0.82 on the own intersection, 0.05–0.08 per neighbour.
- Efficiency: 0.81 ms per joint decision for UG-MARL (DQN 0.22, Max-Pressure 0.14), 157,720
  parameters, 18 min training per run on CPU; GPU not used.

Limits of these numbers: one test flow and one held-out city; validation → test scores drop
by 5–8 s for every learned model and Max-Pressure, so the flows differ in difficulty; max
wait is a single-vehicle statistic with std 85–170 s over seeds. The smoke run showed the
surge condition on validation before the test run; nothing was selected from it.

## Files changed
| File | Action | What changed | Why |
|------|--------|--------------|-----|
| `.claude/plans/phase7-final-experiments.md` | created | Plan: decisions P1–P11, readings, design, steps (ticked, results and deviations noted) | Plan before execution |
| `ROADMAP.md` | modified | Phase 7: set-up block (studies, commands, readings), result table, 7.1–7.6 ticked; final gate: 3 lines ticked; 5 Decision Log rows (path, seeds, readings, execution design, result) | Record decisions before the runs and results after |
| `docs/evaluation_protocol.md` | modified | §6 seeds 0–4 for 7.3 / 7.4; §9 two rows (seed change; reading of §8, regret, wait percentiles) | Protocol changes need a dated row |
| `PRD.md` | modified | §1c: seeds used; H2 three-level reading | Success criterion made unambiguous before the runs |
| `configs/final_test.yaml`, `configs/final_heldout.yaml` | created | Specs: models, seeds, conditions / flows, logging flags | Committed before the runs |
| `src/trafficrl/envs/metrics.py` | modified | `EpisodeTracker`: `seconds`, `enter_t`, `leave_t`; `vehicle_frame()`; `wait_p95_s`, `wait_p99_s` in `result()` | Per-vehicle log and wait tail |
| `src/trafficrl/envs/traffic_env.py` | modified | `_observe` keeps `true_count` / `true_wait` as attributes | Read by `StepRecorder` |
| `src/trafficrl/eval/explain.py` | modified | `StepRecorder` (per step × intersection: phase, yellow, ready, true queue and count per lane) | Phase 8 error analysis without a second test run |
| `src/trafficrl/eval/runner.py` | modified | `run_episode(step_recorder=)`; `evaluate(log_trace=)` writes `trace_s*.csv`, `vehicles_s*.csv` | Same |
| `scripts/evaluate.py` | modified | `--log-trace` | Same |
| `src/trafficrl/eval/lock.py`, `scripts/lock_models.py` | created | `build_manifest`, `verify_manifest`, `uncovered`; CLI `--write` / `--verify` | Proof that models did not change after the test flow was opened |
| `models/manifest.json` | created (generated) | 25 runs: SHA-256 of `best.pt`, `config.yaml`, `gate.json`; τ, p, best validation score | Lock |
| `scripts/robustness.py` | modified | Named conditions, `log_decisions` / `log_trace` / `table` spec keys, `training_runs`, `unfinished_runs`, `--manifest`, `--rerun-failed`, manifest check under `--final`; helpers moved to `report.py` | One runner for 7.1–7.4 with final-run guards |
| `src/trafficrl/eval/report.py` | modified | `condition_name`, `finished_run`, `study_runs`, `load_study` | Shared by runner and table script |
| `src/trafficrl/eval/hypotheses.py` | created | `ci_below`, `compare`, `paired`, `h1`, `h2`, `h3`, `minimum_outcome`, `regret_table`, `regret_summary` | Verdict rules as code, written before the runs |
| `scripts/final_tables.py` | created | Six tables + verdict printout (pure reader) | 7.1–7.6 outputs |
| `src/trafficrl/utils/runlog.py` | modified | `git_commit`: experiment log excluded from the dirty check | Rows after the first of a study were marked `-dirty` |
| `tests/scripts/test_robustness.py`, `tests/envs/test_metrics.py`, `tests/eval/test_explain.py` | modified | Named conditions / flags / manifest / unfinished-job guard; vehicle frame and percentiles; `StepRecorder` | New behaviour |
| `tests/eval/test_lock.py`, `tests/eval/test_hypotheses.py`, `tests/eval/test_zero_shot.py`, `tests/scripts/test_final_tables.py` | created | Manifest; every verdict level; Hangzhou roadnet with synthetic flow; tables from fake studies | New code |
| `docs/results/test_{main,ablation,robustness,efficiency,safety_explain}.{csv,md}`, `docs/results/heldout_generalisation.{csv,md}` | created (generated) | Final tables | Results |
| `docs/experiment_log.csv` | modified (by `evaluate.py`) | +585 rows: 240 `val` (`smoke_final: …`), 240 `test` (`final_test: …`), 105 `heldout` (`final_heldout: …`) | Every run logged |
| `README.md` | modified | New §10 final evaluation (commands, logs, expected output); tests section renumbered to 11 | Commands for every table |
| `CLAUDE.md` | modified | Phase 7 commands (replacing the planned `run_ablation.py`), structure block (eval modules, scripts, `models/`), notes (Phase 7 done, seeds) | Keep project instructions current |
| `.claude/reference/api.md` | modified | `--log-trace`, `robustness.py` spec keys and guards, `final_tables.py`, `hypotheses`; `run_ablation.py` TODO removed | Interface reference |
| `worklog/020-2026-11-12-phase7-final-experiments.md`, `worklog/README.md` | created / modified | This entry; index row | Work log rule |

Not tracked (gitignored): `results/final/{smoke_final,final_test,final_heldout}/` — keep; Phase 8
reads the per-episode logs.

## Commits
`753527e` decisions and readings · `bf9dbdc` logging · `291a75f` lock manifest and guards ·
`2ed075c` verdict rules and tables · `ab6c302` zero-shot test · `92d080f` specs ·
`6336d7b` dirty-check fix · `94c1bbb` manifest + smoke (the commit the final runs ran on) ·
`cfafe81` final results · closing docs commit: `git log --oneline -1 -- worklog/020-2026-11-12-phase7-final-experiments.md`.

## Next
- Phase 8: figures and written interpretation from `results/final/` (8.1–8.4); the τ
  sensitivity sweep of 8.4 is on validation only.
- Report wording to keep: H2's clean part is "no difference"; UG-MARL does not beat DQN on
  clean data or Max-Pressure under noise; attention gives no gain (H3 negative).
- Optional detector-smoothing extension: only with a separate go-ahead and plan.
