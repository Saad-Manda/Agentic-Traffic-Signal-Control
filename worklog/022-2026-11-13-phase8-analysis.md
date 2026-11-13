# 022 — Phase 8: analysis of the final results (figures, error analysis, τ sweep, attention check)

**Date:** 2026-11-13   **Tag:** decision, code, experiment, report   **ROADMAP:** 8.1–8.4   **Status:** done

## What
Planned and executed Phase 8: a reader analysis of the Phase 7 results in
`notebooks/02_analysis.ipynb` (18 figures, error-analysis tables, a short reading under each
figure), two validation-only studies (τ sensitivity sweep, attention faithfulness check) and
the written interpretation `docs/analysis.md`. No test or held-out episode was run.

## Why
ROADMAP 8.1–8.4 / guidelines G11: results must be compared in charts, failures located and
interpreted, sensitivity to seed / τ / shift reported, and attention treated as an association
that is checked. Plan: `.claude/plans/phase8-analysis.md`. Decisions taken with the user
(13 Nov, Decision Log):

- **Both baselines in both conditions** for the error analysis (UG-MARL vs DQN and vs
  Max-Pressure, clean and noise 0.2; −gate as the paired row). Rejected: one "best" baseline
  (it changes with the condition).
- **τ sweep by new validation runs at p = 50, 60, 70.** Rejected: reusing 80–99 only (the
  chosen value would sit at the edge of the sweep); sweeping on test (selection on test).
- **Read-only route join** from the test / Hangzhou flow files. Rejected: saved logs only.
- **Analysis in a notebook** (the first plan used scripts): notebook = front end, all logic in
  tested functions under `src/trafficrl/eval/`; the rule "notebooks only for EDA" relaxed.
  Rejected: logic in notebook cells (untestable, fails the fresh-clone test).
- Plan decisions: no closed-loop perturbation for the attention check (offline slot
  replacement instead; knock-out as second variant); cases picked by rule; readings for the
  two validation studies written before the runs; nothing changed because of a finding.

## How
- `src/trafficrl/eval/analysis.py` — pure functions on `trace_s*`, `vehicles_s*`,
  `decisions_s*` of `results/final/`; `Study` / `Context` for the notebook.
- `src/trafficrl/eval/figures.py` — one function per figure (PNG + PDF + caption `.md`).
- `configs/analysis.yaml` — reader settings (models, conditions, window 600 s, case window 300 s).
- **τ sweep:** `configs/search/tau_sensitivity.yaml` committed first (`57dd6ea`), then
  `python scripts/tau_sweep.py --search configs/search/tau_sensitivity.yaml --workers 5`:
  Jinan validation (`real_2000`), seeds 0–4, 3600 s, p ∈ {50, 60, 70} × {clean, noise 0.1,
  noise 0.2, dropout 10 %, dropout 20 %} = 75 episodes; τ from the calibration's σ sample
  (`results/tune/ugmarl_gate_ugmarl_final/sigma_val.csv`; p = 80 checked against every
  `gate.json` first). `python scripts/lock_models.py --verify` afterwards.
- **Faithfulness:** `python scripts/attention_faithfulness.py --tag final`: one clean
  validation episode per seed-model (gate on), encoder inputs of all ≈ 2180 decisions
  recorded; offline, one slot replaced by the same slot of another decision of the same
  intersection (8 draws) or masked; outcomes |Δμ|, action change, Δσ.
- **Notebook:** `jupyter nbconvert --to notebook --execute --inplace notebooks/02_analysis.ipynb`
  (≈ 2 min, native Ubuntu).
- Checks: `pytest` 276 passed, 1 skipped; `ruff check src tests scripts` clean.

## Result
**Reproduction check — and a definition learned.** The first check of per-vehicle travel time
against `metrics.csv` failed for congested episodes (up to 130 s off). Cause: the per-vehicle
log only sees vehicles inside the network; the simulator's average counts every scheduled
vehicle from its scheduled start, including vehicles blocked outside or never entering.
Fixed by computing travel time from the flow file's start times (`analysis.trips`): all 200
non-surge test episodes and 105 held-out episodes then reproduce (largest difference
0.00005 s). Consequence: **the waiting-time metrics (incl. max wait) count waiting inside the
network only.**

**Findings** (details and numbers: `docs/analysis.md`):
1. Deficit to DQN on clean data (3.2 s): more waiting at all 12 intersections (+0 to +12 %,
   clear at 4), +2.5 to +4.5 s in five of six windows, growing with route length (+5.6 s
   through 4+ intersections), carried by turning trips. Not caused by the gate (gate on vs
   off level); all four UG-MARL variants are behind DQN → network family / inherited
   hyperparameters, not isolated further.
2. Against Max-Pressure, clean: 17–40 % less waiting at every intersection.
3. Noise 0.2: ungated learned policies leave 584–795 of 5494 vehicles outside the network
   (blocks of 39–48 min); UG-MARL and Max-Pressure none. UG-MARL level with Max-Pressure at
   11 of 12 intersections; 9–16 s slower on long trips.
4. Gate under the demand surge: fallback 38 → 68 % over the hour, max wait 2162 s vs 1430 s
   without the gate (DQN 1161 s, Max-Pressure 3260 s). Same pattern on Hangzhou `real_5734`
   (max wait 3167 s ≈ Max-Pressure 3182 s).
5. After a gate override on clean data the queue grows by 0.43 vehicles in the next minute vs
   −0.20 without (difference +0.62, CI [0.47, 0.77]); sign reverses under noise. Association.
6. **τ sweep:** score 343.23 / 343.84 / 342.27 s for p = 50 / 60 / 70 vs 343.73 s at
   p\* = 80 (Max-Pressure 351.44 s, ungated 485.24 s). Pre-fixed reading: insensitive within
   5 s; lower end of the grid was binding (p = 70, −1.46 s). p\* stays 80.
7. **Attention faithfulness:** replacement — consistent with the weights: |Δμ| 0.0061
   (highest-α neighbour) vs 0.0022 (lowest-α), paired CI [0.0011, 0.0066]; own observation
   0.0415 (action changes in 80 % of decisions vs 10 % / 4 %). Knock-out — not consistent
   (CI [−0.032, 0.068]).
8. Flip-flops: UG-MARL 21.5 % of decisions, DQN 20.2 %, Max-Pressure 20.3 %, −gate 23.7 %.
   Seed range of clean travel time 2.8 s (DQN 1.6 s); of max wait 421 s.
9. Cases (rule, seed 0): win `intersection_1_2` min 40–45, loss `intersection_1_1` min 10–15,
   gate `intersection_1_3` min 0–5.

**Failures / blemishes.**
- Experiment-log commit column of the 75 sweep rows: 10 × `57dd6ea`, 19 × `57dd6ea-dirty`,
  46 × `4ab9ac9`. New, unrelated files (faithfulness module) were written and committed
  while the sweep ran; the code the sweep used did not change. Lesson: do not touch the tree
  during a logged run.
- The τ table was first written with the ungated row in the middle; `--tables-only` was added
  to rebuild it from the finished runs (values identical).
- The surge flow is generated at run time, so it has no route join; its breakdown uses time
  inside the network.
- No on-disk cache was built (not needed).

## Files changed
| File | Action | What changed | Why |
|------|--------|--------------|-----|
| `.claude/plans/phase8-analysis.md` | created | Plan: goal, decisions A1–A12, readings fixed in advance, design, files, steps; later ticked with results and deviations | Plan before non-trivial work; record of what was decided before the runs |
| `ROADMAP.md` | modified | Decision Log: Phase 8 set-up, readings, result row; 8.1–8.4 ticked with pointers; final-gate line "Results and failures interpreted" ticked | Decisions and progress record |
| `src/trafficrl/eval/analysis.py` | created | Log readers; `vehicle_times`, `trips` (scheduled-start travel time, route join + checks), `window_table`, `intersection_table`, `network_queue`, `approach_table`, `unit_table`, `flip_flops`, `route_groups`, `route_table`, `entry_summary`, `seed_stats`, `seed_difference`, `after_decision_queue`, `gate_effect`, `median_seed`, `pick_cases`, `case_series`; `Study`, `breakdowns`, `comparison`, `stability_table`, `seed_spread`; notebook helpers `Context`, `reproduction_check`, `comparisons`, `comparison_overview`, `entry_table`, `gate_effect_table`, `fallback_by_window`, `inside_windows`, `worst_vehicle`, `approach_overview`, `heldout_overview`, `model_order`, `select_cases`, `case_inputs`, `write_error_report`, `show` | All 8.4 logic in one tested module so the notebook holds none |
| `src/trafficrl/eval/figures.py` | created | `fig_main_comparison`, `fig_ablation`, `fig_wait_tail`, `fig_generalisation`, `fig_seed_spread`, `fig_robustness`, `fig_regret`, `fig_gate_firing`, `fig_attention_map`, `fig_error_intersections`, `fig_error_windows`, `fig_error_routes`, `fig_case`, `fig_tau_sensitivity`, `fig_faithfulness`; `finish` (PNG + PDF + caption), `grid_panel` | 8.1 / 8.2 / 8.4 figures as functions |
| `src/trafficrl/eval/faithfulness.py` | created | `outputs`, `donors`, `replace_slot`, `knock_out`, `effect`, `slot_effects`, `rank_slots`, `seed_summary`, `verdict` | Attention faithfulness check (gap analysis A8) |
| `configs/analysis.yaml` | created | Studies, agents, ablations, baselines, conditions, window sizes, output paths | Reader settings out of the notebook |
| `configs/search/tau_sensitivity.yaml` | created | Percentiles 50 / 60 / 70, seeds, five validation conditions, σ source, reading thresholds | Search space written before the runs (G8) |
| `scripts/tau_sweep.py` | created | Spec checks (val only, no clash with the calibration grid), τ from the calibration σ sample with a p = 80 check, jobs through `evaluate.py`, table + reading by code, `--tables-only` | τ sensitivity (8.4) without any path that writes `gate.json` |
| `scripts/attention_faithfulness.py` | created | Records encoder inputs of one validation episode per seed, checks the Phase 6 travel time, runs the perturbations, writes table + log rows | Faithfulness check (8.2) |
| `notebooks/02_analysis.ipynb` | created | 63 cells: set-up + reproduction check, recap, 8.1 / 8.2 figures, error analysis, cases, stability, τ and shift, faithfulness, report tables, summary; executed, outputs kept | Analysis front end the user asked for |
| `tests/eval/test_analysis.py` | created | 14 tests on hand-made logs (trips, join refusals, windows, queues, flip-flops, approaches, differences, gate effect, case rule) | Verify the breakdown logic |
| `tests/eval/test_figures.py` | created | Fake study; every figure function writes its three files; notebook helpers | Figures and helpers run without `results/final` |
| `tests/eval/test_faithfulness.py` | created | Self-replacement = 0, knock-out α = 0, donors, ranking among present neighbours, verdict rule | Verify the check |
| `tests/scripts/test_tau_sweep.py` | created | Spec valid, other splits refused, grid clash refused, thresholds, job arguments, reading rule | Guards and the pre-fixed reading |
| `tests/test_notebooks.py` | created | No `def` / `class` / `lambda`, no simulation or script imports, a reading after every figure | Keep the notebook a front end |
| `docs/analysis.md` | created | Planned vs implemented vs evidence; where UG-MARL wins / loses and why; σ and τ; attention; stability; 10 limitations; consequences for the report | ROADMAP 8.3 |
| `docs/results/tau_sensitivity.{csv,md}` | created (generated) | Score, per-condition travel time, fallback / max wait, new thresholds, reading | 8.4 result |
| `docs/results/attention_faithfulness.{csv,md}` | created (generated) | Per-seed outcomes for hi / lo / self, both perturbations, readings | 8.2 result |
| `docs/results/error_analysis.md`, `error_analysis_*.csv` (13) | created (generated) | Entry blocking, intersections, windows, route length / turns, gate effect, surge windows, approaches, stability, seeds, held-out, cases | 8.4 tables for the report |
| `docs/figures/results/*.{png,pdf,md}` (18 figures) | created (generated) | The figures listed above, `error_intersections_heldout`, three cases | 8.1 / 8.2 / 8.4 figures for report and poster |
| `docs/experiment_log.csv` | modified (by the runs) | 75 `tau_sensitivity` rows + 5 `attention_faithfulness` rows, split `val` | Every run logged |
| `docs/literature/06-gap-analysis.md` | modified | A8 → done with result; A6 mentions the extended sweep; UG-MARL row wording on the faithfulness check | Gap table in line with what was done |
| `README.md` | modified | New §11 (Phase 8 commands, guards, expected outputs), tests → §12, layout lines for notebooks and docs | Reproduction commands (9.4) |
| `CLAUDE.md` | modified | Phase 8 commands; structure (notebooks, eval modules, scripts, docs); notebook convention reworded; "Phase 8 is done" note with the findings to keep straight | Project instructions in line with the repo |
| `worklog/022-2026-11-13-phase8-analysis.md`, `worklog/README.md` | created / modified | This entry; index row | Work log rule |

## Commits
- `ecb72d0` docs(plan): Phase 8 analysis plan, decisions and readings
- `c7abd71` feat(eval): error-analysis functions on the saved Phase 7 logs
- `2251916` feat(eval): Phase 8 figure functions and study-level breakdowns
- `57dd6ea` exp(ugmarl): spec and runner for the validation tau sensitivity sweep
- `4ab9ac9` feat(eval): attention faithfulness check (offline slot perturbation)
- `ee50373` exp(ugmarl): tau sensitivity on validation, p = 50, 60, 70
- `0c24d65` exp(ugmarl): attention faithfulness check on validation
- `2c51f34` feat(eval): Phase 8 analysis notebook, figures and error-analysis tables
- closing docs commit: interpretation, ROADMAP, README, CLAUDE.md, gap table, this entry
  (hash: see `git log`, the commit after `2c51f34`)

## Next
- Phase 9: report Ch 4 from `docs/analysis.md` and `docs/figures/results/`; state the
  inside-the-network definition wherever max wait is quoted.
- 2.6 final gap-table check against the PDFs (by 15 Nov).
- Optional and not started: detector smoothing extension; a fallback with a wait bound is
  future work only.
