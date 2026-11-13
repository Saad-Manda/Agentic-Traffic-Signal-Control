# Plan — Phase 8 analysis (ROADMAP 8.1–8.4)

Status: **planned 13 Nov 2026, not started.** Tick steps `[x]` during execution and note
deviations under each step.

## Goal

Turn the Phase 7 results into the figures, tables and written interpretation that the report
(Ch 4), the poster and the viva need — **without running the test or held-out flows again**.

| Output | ROADMAP | Serves |
|---|---|---|
| Comparison figures: main bar chart, ablation, generalisation, wait tail | 8.1 | F6, R6, G9 |
| Robustness degradation plot, regret figure, gate figures, attention heatmap | 8.2 | F8, F10, R6, G11 |
| Attention faithfulness check (gap analysis A8), validation only | 8.2 | F10, G11 |
| Error analysis: per intersection, per time window, per route group; representative cases; unstable decisions; sensitivity to seed, τ and city | 8.4 | G11, R6 |
| Written interpretation `docs/analysis.md` (why UG-MARL wins / loses, limitations) | 8.3 | R6, G11 |

Out of scope: any training, retuning, re-calibration or `--final` run; SHAP (optional, not
done — Decision Log 12 Nov); the detector-smoothing extension; report / poster text (Phase 9);
the demo (9.3).

## What already exists (inputs)

- `docs/results/test_*.csv`, `heldout_generalisation.csv`, `val_phase6.csv`,
  `val_robustness.csv`, `gate_calibration.csv` — seed-level statistics.
- `results/final/final_test/eval/<model>__<condition>/<run>/` (48 jobs) and
  `results/final/final_heldout/eval/<model>__<flow>/<run>/` (21 jobs), per seed:
  - `metrics.csv` — one row per seed;
  - `trace_s<seed>.csv` — one row per env step × intersection: `t, iid, phase, in_yellow,
    ready, q_0…q_11` (true waiting vehicles per incoming lane), `n_0…n_11` (true counts);
  - `vehicles_s<seed>.csv` — `vehicle, enter_t, leave_t, wait_s`;
  - `decisions_s<seed>.csv` (UG-MARL, −gate, −attention) — `t, iid, sigma, tau, fallback,
    greedy_phase, mp_phase, phase_taken, attn_self/E/N/W/S, q_gap, in_veh_observed`.
- `results/tune/ugmarl_gate_ugmarl_final/sigma_val.csv` — σ of every decision of the clean,
  ungated validation episode of each seed-model (the sample the calibration used).

Limit of these logs, stated in the analysis: **travel time is per vehicle, not per
intersection.** Per-intersection results are therefore queue and waiting-vehicle-seconds from
the traces; travel time is broken down by entry-time window and by route group.

## Decisions

Decided with the user on 13 Nov unless marked *(plan)*.

| # | Topic | Decision | Reason |
|---|---|---|---|
| A1 | "Best baseline" in 8.4 | UG-MARL is compared with **both DQN and Max-Pressure, in both conditions** (clean and noise 0.2) — four comparisons. −gate is added as a paired row (same checkpoints). | The best baseline changes with the condition (DQN clean, Max-Pressure under noise). Showing all four also shows where each baseline loses. |
| A2 | τ sensitivity | **New validation runs at p = 50, 60, 70** (5 seeds × 5 conditions = 75 episodes), joined with the existing p = 80 / 90 / 95 / 99, ungated and Max-Pressure (τ = 0) rows. Evaluation only; `gate.json`, p\* = 80 and `models/manifest.json` are not touched. | The chosen value is at the lower edge of the calibration grid; a sweep "around" it needs points below. |
| A3 | Route groups | **Allowed, read-only:** the Jinan test and Hangzhou flow files are read (through `trafficrl.data.splits.flow_files`) only to join each vehicle's route onto the saved per-vehicle logs. No simulation. Dated Decision Log row. | "Error by subgroup" (G11). The final evaluation is finished and nothing is selected from the result. |
| A4 | Scope | **Full 8.1–8.4 with a cut order** (below). | User decision. |
| A5 | No new test episodes *(plan)* | Everything on the test / held-out flows is read from `results/final/`. New simulation runs are on the **validation flow only** (A2, A6). | Protocol: test flows run once. |
| A6 | Faithfulness check *(plan)* | Offline counterfactual on validation: record the encoder inputs of one clean validation episode per seed-model, then re-run the network with one neighbour's observation replaced (design below). No closed-loop re-simulation. | Isolates the network's response to one input; a closed-loop run would mix it with different traffic. Cheap (forward passes only). |
| A7 | Case selection by rule *(plan)* | The three representative cases are picked by a rule fixed here (below), not by eye. | Avoids choosing favourable examples (guidelines mistake 5). |
| A8 | Readings written first *(plan)* | The readings for the τ sweep and the faithfulness check are fixed in this plan, before those runs. | Same practice as 6.8 and Phase 7. |
| A9 | Nothing is changed because of Phase 8 *(plan)* | If the analysis shows something that could be improved (e.g. a better p), it is reported as a finding and future work. The locked model, the test tables and the verdicts stay. | A change chosen after the test result uses the test flow for selection (Decision Log 12 Nov). |
| A10 | Wording *(plan)* | Attention weights, σ and gate statistics are described as associations. Differences are called clear only when the 95 % CIs are separated (or the paired CI excludes 0 for gate vs −gate); otherwise "level". | ROADMAP 8.3, G11. |
| A12 | Notebook | **The reader analysis (8.1, 8.2 figures, 8.4 error analysis) lives in `notebooks/02_analysis.ipynb`**, executed top to bottom by one `nbconvert` command. The notebook holds no logic: every computation and every figure is a function in `src/trafficrl/eval/` (tested); the notebook calls it, shows the result and carries the interpretation under each figure. The two steps that launch validation runs (τ sweep, faithfulness) stay scripts; the notebook reads their outputs. The rule "notebooks only for EDA" is relaxed with a dated Decision Log row. | User decision (13 Nov): figures next to their interpretation, easier to explore and to show in the viva. Logic stays in `src/` so it is tested and the README command still regenerates everything. |
| A11 | Figure format *(plan)* | PNG (160 dpi) + PDF per figure in `docs/figures/results/`, each with a `.md` file: caption, values, short interpretation. Style and colours from `trafficrl.utils.plotting`. | Existing convention (`docs/figures/ugmarl/`); PDF for LaTeX. |

## Readings fixed before the new validation runs

**τ sweep (A2).** Score = mean travel time over seeds × the five validation conditions (the
calibration score). 
- *Insensitive:* every p in 50–90 scores within 5 s of p\* = 80 → "the result does not depend
  on the exact threshold".
- *Lower p better:* some p < 80 beats p\* = 80 by more than the 1 s tie tolerance → reported
  as "the calibration grid's lower end was binding; a lower threshold would have been chosen
  on validation". p\* stays 80 and the test flow is not re-run (A9).
- *Lower p worse:* reported as the clean-data cost of falling back more often.
- The clean column and the noise columns are also shown separately (the trade-off: more
  fallback costs on clean data and helps under noise, with Max-Pressure as the τ = 0 end).

**Attention faithfulness (A6).** Per seed, mean |Δμ| when the highest-α neighbour is replaced
vs when the lowest-α neighbour is replaced.
- *Consistent with the weights:* paired difference (high − low) over the 5 seeds > 0 with a
  95 % CI that excludes 0.
- *Not consistent:* otherwise → "the attention weights do not rank the neighbours' influence
  on Q"; attention heatmaps are then shown with that caveat.
- Reported next to it: the same quantity for the intersection's **own** observation (upper
  reference) and the share of decisions whose greedy action changes. Expected from H3 and the
  mean self-weight of 0.82: neighbour effects are small in absolute terms. Either result is
  reported; it is an association check, not a causal claim.

## Design

### 8.1 / 8.2 — figures (`eval/figures.py`, shown in the notebook, pure reader)

One function per figure in `src/trafficrl/eval/figures.py` (`fig_main_comparison(study, out)`, …): it
returns the matplotlib figure and writes PNG + PDF + `.md`. Notebook §2–§3 call them in
order. Seed-level values come from
`report.load_study` on `results/final`; nothing is recomputed that `final_tables.py` already
defines (regret via `eval.hypotheses`).

| ID | File | Content |
|---|---|---|
| R1 | `main_comparison` | Clean test, average travel time, 5 agents: bar = mean, error bar = 95 % CI, dots = seeds; Max-Pressure and Fixed-Time marked deterministic. Second panel: max per-vehicle wait. |
| R2 | `robustness_degradation` | Travel time vs noise (0, 0.1, 0.2) and vs dropout (0, 10, 20 %): one line per model with CI band, UG-MARL solid and −gate dashed; surge ×1.5 as a third panel (dots). |
| R3 | `regret` | Regret per model × condition (heat grid with numbers) + largest and mean regret per model. |
| R4 | `ablation` | Full / −gate / −attention / −ensemble: clean and noise 0.2 travel time and max wait (dot plot with CI). The table itself is `test_ablation.md`. |
| R5 | `wait_tail` | Empirical distribution of per-vehicle wait (share of vehicles waiting longer than x, log y), clean and noise 0.2, 5 agents + −gate; seeds pooled, p95 / p99 / max marked. |
| R6 | `generalisation` | Hangzhou, three flows: travel time per model with CI; fallback rate of UG-MARL per flow as text. |
| R7 | `gate_firing` | (a) σ / τ distribution per condition; (b) fallback rate per 5-minute window per condition; (c) fallback rate per intersection on the 3 × 4 grid, clean and noise 0.2. |
| R8 | `attention_map` | Mean attention weight per intersection and slot, drawn on the 3 × 4 network layout (clean test); −attention has none. |
| R9 | `seed_spread` | Per-seed clean travel time and max wait for the learned models (strip plot) — the seed-sensitivity figure of 8.4. |

Efficiency (7.5) stays a table. The Phase 9 report picks from R1–R9 within its page budget.

### 8.2 — attention faithfulness (`scripts/attention_faithfulness.py`, `eval/faithfulness.py`)

1. For each seed s: one clean validation episode with UG-MARL s as deployed (gate on, τ from
   `gate.json`); at every decision store the encoder input of each ready intersection
   (`NeighborhoodEncoder.encode_all` output: 5 slots × (44 values + present flag)), α, μ, σ.
   The episode must reproduce the validation travel time in `val_phase6.csv` (check).
2. For every stored decision with ≥ 2 present neighbours: slot *hi* = largest α among present
   neighbours, *lo* = smallest, plus *self* as reference.
3. **Replacement perturbation:** the 44 values of the chosen slot are replaced by the same
   slot's values from another decision of the same intersection in the same episode, drawn at
   random (R = 8 draws, RNG seeded with s); the full network is run again, so α is recomputed.
   **Knock-out perturbation** (second variant): the slot's present flag is set to 0.
4. Outcomes per decision: |Δμ(a\*)| (a\* = original greedy action), max over actions of |Δμ|,
   greedy action changed (0 / 1), Δσ.
5. Statistics: per-seed means for hi / lo / self; paired difference hi − lo over seeds with
   95 % CI (`hypotheses.paired`); Spearman ρ between α and |Δμ(a\*)| over all neighbour slots,
   per seed. Output `docs/results/attention_faithfulness.{csv,md}` + figure
   `docs/figures/results/attention_faithfulness`.

No learned or selected quantity; validation flow only.

### 8.4 — error analysis (`eval/analysis.py`, shown in the notebook, pure reader)

Pure functions in `eval/analysis.py` (frames in, frames out), called from notebook §4–§7;
`analysis.write_error_report(...)` writes `docs/results/error_analysis.{md,csv}` from the same
frames:

- `vehicle_times(vehicles, horizon=3600)` — travel time per vehicle (`leave_t − enter_t`;
  vehicles still inside counted to the horizon). **Check:** its mean reproduces
  `avg_travel_time_s` in `metrics.csv` for every model; if the simulator's definition differs,
  the function is aligned with it and the definition is stated.
- `window_table(vehicles, width=600)` — travel time, wait, vehicles entered and completed per
  entry-time window.
- `intersection_table(trace)` — per intersection: mean waiting vehicles, waiting-vehicle-seconds
  (Σ over steps of Σ q_k × step length), phase switches, flip-flops. **Check:** the network
  mean reproduces `avg_queue_veh`.
- `approach_table(trace)` — the same per approach (E, N, W, S; three lanes each) — used for
  the fairness paragraph of 9.8 (is a minor approach left waiting?).
- `flip_flops(trace)` — a flip-flop = phase A → B → A where B was held for exactly one
  decision interval; reported as share of decisions, per model and condition.
- `route_table(vehicles, flow)` — joins route length (number of intersections passed: 1, 2,
  3, 4+) and entry side (N / E / S / W boundary) from the flow file. Vehicle `flow_<i>_<j>` =
  j-th vehicle of flow entry i; **check:** every joined `enter_t` is consistent with that
  entry's start time, else the join is dropped and reported as not possible.
- `pick_cases(...)` — rule A7.

Tables (`docs/results/error_analysis.md` + CSVs), all with mean ± std [95 % CI] over seeds:

1. **Four comparisons (A1):** UG-MARL vs DQN and vs Max-Pressure, clean and noise 0.2 — per
   intersection (difference in waiting-vehicle-seconds), per 10-minute window (difference in
   travel time), per route group. −gate as a paired row with paired CIs.
2. **Where the 3.2 s to DQN goes:** the same breakdown restricted to clean, plus the
   ablations' clean numbers (−gate 295.0, −attention 294.2, −ensemble 297.3 s) to separate
   "cost of the gate" from "cost of the architecture / training".
3. **Gate on clean data:** decisions where the gate fired and changed the action — queue in
   the following 60 s at that intersection vs comparable decisions where it did not fire
   (same intersection, same 10-minute window). Association only.
4. **Unstable decisions:** flip-flop share and phase switches per model and condition;
   per-seed table of travel time and max wait (min, max, range over seeds).
5. **Surge:** why UG-MARL's max wait is 2162 s (DQN 1161 s) with 55 % fallback — per-window
   fallback rate and wait tail under surge.
6. **Shift:** Hangzhou per flow and per intersection — fallback rate, σ / τ, travel time vs
   DQN and Max-Pressure; whether the model order matches Jinan test.

Figures: `error_intersections` (3 × 4 grids of differences, four comparisons),
`error_windows` (difference per window with CI), `error_routes`, and the cases below.

**Representative cases (A7).** Unit = (intersection, 5-minute window) on the clean test flow.
Seed = the UG-MARL seed whose clean travel time is the median of the five; DQN and
Max-Pressure of the same seed number.
- *Win:* unit with the most negative waiting-vehicle-seconds difference UG-MARL − DQN.
- *Loss:* unit with the most positive difference.
- *Gate fired:* unit with the most decisions where the gate fired and changed the action.
Each case = one figure `case_<win|loss|gate>`: queue per approach over time for UG-MARL and
the baseline, phases taken, σ(s) against τ, gate firings marked; caption states the rule.

### 8.4 — τ sensitivity (`scripts/tau_sweep.py`, `configs/search/tau_sensitivity.yaml`)

- Spec (written and committed before the runs): `percentiles: [50, 60, 70]`, the five
  validation conditions of `ugmarl_gate.yaml`, seeds 0–4, model `ugmarl_final`.
- τ_{p,s} = p-th percentile of `sigma_val.csv` for seed s (no new σ pass). **Check before the
  runs:** p = 80 computed this way equals τ in each `gate.json`.
- Threshold files go to `results/analysis/tau_sensitivity/taus/p<p>_s{seed}.json`; each job is
  `scripts/evaluate.py --split val` with `agents.ugmarl.gate.file=<that file>` (the mechanism
  the calibration used). The script has no code path that writes `gate.json`; it refuses any
  split other than `val`.
- Output: `docs/results/tau_sensitivity.{csv,md}` (new rows + the existing calibration rows +
  Max-Pressure from `val_robustness.csv`, source column) and figure `tau_sensitivity`
  (travel time and fallback rate vs p, clean / noise / dropout panels, p\* marked).
- After the runs: `python scripts/lock_models.py --verify` must pass.

### The notebook (`notebooks/02_analysis.ipynb`)

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/02_analysis.ipynb`
(committed with outputs, like `01_eda_train.ipynb`). Reads `configs/analysis.yaml`. Sections:

0. Set-up: paths, models, conditions; **reproduction checks** (per-vehicle travel time and
   trace queue reproduce `metrics.csv`; cells assert, so a failed check stops the run).
1. Results recap: the Phase 7 tables loaded from `docs/results/` (no recomputation).
2. Comparison figures R1, R4, R5, R6 (8.1).
3. Robustness and explainability figures R2, R3, R7, R8 (8.2).
4. Error analysis: four comparisons per intersection, window and route group (8.4).
5. Representative cases (rule A7 printed with the chosen units).
6. Unstable decisions and seed spread (R9).
7. Sensitivity: τ sweep and distribution shift (reads `tau_sensitivity.csv`, held-out logs).
8. Attention faithfulness (reads `attention_faithfulness.csv`).
9. Summary of findings → pointer to `docs/analysis.md`.

Rules: code cells contain calls and display only (no function definitions, no numbers typed
by hand); a markdown cell with a short interpretation under every figure; §7 and §8 show a
clear "run the script first" message if their CSV is missing. The notebook never simulates
and never imports `make_env`.

### 8.3 — written interpretation (`docs/analysis.md`)

Written last, from the tables above; every number carries its source file. Sections:
1. What the evidence shows, per hypothesis (H1, H2 with its clean-flow caveat, H3 negative).
2. Why UG-MARL is 3.2 s behind DQN on clean data.
3. Why it ties Max-Pressure under noise (the fallback's ceiling) and what the gate buys
   (worst-case regret 6.8 s).
4. Surge and Hangzhou: where the gate helps and where it costs (max wait).
5. Attention: H3, the weights, the faithfulness result.
6. Unstable decisions and seed spread.
7. Limitations and what remains uncertain (σ over 3 heads, σ vs TD error ρ ≈ 0.03, validation
   reuse, one city family, simulator fidelity, 5 seeds).
Kept separate: planned / implemented / shown by evidence / uncertain (ROADMAP 9.1 rule).

## Justification

- **Pure readers for everything on the test flow:** the Phase 7 pass logged per-step,
  per-vehicle and per-decision data for this purpose (P6). Rejected: deterministic re-runs
  with extra logging — bit-identical, but a second pass over the test flow.
- **Both baselines in both conditions (A1)** rather than one "best" baseline: the user's
  choice; it avoids picking the comparison that flatters UG-MARL.
- **τ sweep by new validation runs (A2)** rather than reusing 80–99 only: the selected value
  would sit at the edge. Rejected: sweeping on the test flow (selection on test).
- **Replacement from the same slot and intersection (A6)** rather than zeroing or Gaussian
  noise: zeros and noise are inputs the network never saw, so a large response could be an
  out-of-distribution effect. Knock-out is kept as a second variant because it is the common
  form of the check. Rejected: closed-loop perturbation (not attributable to one decision);
  SHAP (optional, out of time).
- **Cases by rule (A7)** rather than hand-picked.
- **Travel time by window and route, queue by intersection:** the per-vehicle log has no
  per-intersection delay; adding it would need a new test pass. Stated as a limitation.
- **Notebook as the front end, logic in `src/` (A12):** the user wants the analysis in a
  notebook. Rejected: logic written in notebook cells (not testable, depends on cell order,
  fails the fresh-clone test 9.7); scripts only (the first version of this plan — no place
  where figure and interpretation sit together); runs launched from the notebook (parallel
  jobs, experiment-log rows and split guards already live in `scripts/`).

## Files

| File | Action | Purpose |
|---|---|---|
| `src/trafficrl/eval/analysis.py` | create | Breakdown functions (windows, intersections, approaches, routes, flip-flops, case rule) |
| `src/trafficrl/eval/faithfulness.py` | create | Slot replacement / knock-out and outcome measures |
| `src/trafficrl/eval/figures.py` | create | One function per figure (R1–R9, error figures, cases, τ, faithfulness) |
| `notebooks/02_analysis.ipynb` | create | Phase 8 reader analysis: calls, figures, interpretation (A12) |
| `scripts/attention_faithfulness.py` | create | Validation episode + counterfactual + table / figure |
| `scripts/tau_sweep.py` | create | Validation τ sweep + table / figure |
| `configs/search/tau_sensitivity.yaml` | create | Sweep spec (before the runs) |
| `configs/analysis.yaml` | create | Study paths, models, conditions, window widths, case rule, R, figure list |
| `src/trafficrl/utils/plotting.py` | modify | Helpers shared by the new figures (grid panel, CI bar); no colour change |
| `tests/eval/test_analysis.py`, `tests/eval/test_faithfulness.py`, `tests/scripts/test_tau_sweep.py`, `tests/eval/test_figures.py`, `tests/test_notebooks.py` | create | See Tests |
| `docs/figures/results/*.{png,pdf,md}` | create (generated) | Figures with captions |
| `docs/results/error_analysis.{md,csv}`, `tau_sensitivity.{md,csv}`, `attention_faithfulness.{md,csv}` | create (generated) | Tables |
| `docs/analysis.md` | create | 8.3 written interpretation |
| `docs/literature/06-gap-analysis.md` | modify | A8 row: result of the faithfulness check |
| `docs/experiment_log.csv` | modify (by the runs) | 75 + 5 validation rows |
| `ROADMAP.md` | modify | Ticks 8.1–8.4, Decision Log rows (A1–A3, readings, outcome), final-gate line "Results and failures interpreted" |
| `README.md`, `CLAUDE.md` | modify | Phase 8 commands, new files in the structure block; convention reworded: notebooks = train-split EDA **and** the Phase 8 analysis reader, logic always in `src/` |
| `worklog/022-2026-11-13-phase8-analysis.md`, `worklog/README.md` | create / modify | Work log |

## Config

`configs/analysis.yaml` (read by the notebook; no `seed` because nothing is
simulated): `final_dir: results/final`, `studies`, `models`, `conditions`, `window_s: 600`,
`case_window_s: 300`, `out_figures: docs/figures/results`, `out_tables: docs/results`.
`configs/search/tau_sensitivity.yaml`: `percentiles`, `conditions`, `seeds: [0,1,2,3,4]`,
`sigma_source`. Faithfulness: `seeds`, `draws: 8`; the replacement RNG is seeded with the model
seed. Validation runs are seeded by `evaluate.py` as before.

## Data-leakage check

| Step | Reads | Simulates |
|---|---|---|
| Figures, error analysis | saved logs of test / held-out runs; test and Hangzhou flow files for the route join only (A3) | nothing |
| τ sweep | validation flow | validation only |
| Faithfulness | validation flow | validation only |

Nothing is fitted, tuned or selected. `gate.json`, checkpoints and `models/manifest.json` are
unchanged (`lock_models.py --verify` at the end). No `--final` command is run. A finding that
suggests a better setting is reported, not applied (A9).

## Tests

- `test_analysis.py` — hand-made frames: travel time with a vehicle still inside; window
  table sums; waiting-vehicle-seconds with uneven step lengths; flip-flop definition (A-B-A
  counted, A-B-B-A not); approach grouping; route join and its consistency check failing on a
  wrong ID; case rule returns the expected unit.
- `test_faithfulness.py` — tiny `UGMARLNet`: replacing a slot with itself gives Δ = 0;
  knock-out sets that slot's α to 0; hi / lo picked among present neighbours only;
  intersections with < 2 neighbours skipped; same seed → same draws.
- `test_tau_sweep.py` — percentiles from a σ frame; job arguments use `--split val` and a
  threshold file under `results/analysis`; a spec with another split or a path to `gate.json`
  is refused.
- `test_figures.py` — every figure function runs on a fake study directory and writes PNG +
  PDF + `.md`.
- `test_notebooks.py` — `02_analysis.ipynb` has no `def` / `class` in code cells and does not
  import `make_env` or any `scripts` module (static check of the JSON; the notebook itself is
  executed in step 4, not in pytest — it needs `results/final`).

## Steps and verification

1. [ ] **Record decisions:** ROADMAP Decision Log rows for A1–A3, A12 and the two readings; commit
   this plan.
2. [ ] **`eval/analysis.py` + tests.** `pytest tests/eval/test_analysis.py`; then on the real
   logs: `vehicle_times` reproduces `avg_travel_time_s` and `intersection_table` reproduces
   `avg_queue_veh` for all 48 test jobs (printed check, tolerance 0.01).
3. [ ] **`eval/figures.py` + tests; notebook §0–§3** (R1–R9). Check: values printed in each
   `.md` equal `docs/results/test_*.csv`; every figure has axis labels, units, caption and
   interpretation.
4. [ ] **Notebook §4–§6** (error analysis, cases, unstable decisions) and a full run:
   `jupyter nbconvert --to notebook --execute --inplace notebooks/02_analysis.ipynb`. Check:
   per-window and per-route means recombine (vehicle-weighted) to the table values of
   `test_main.md`; §7 / §8 show the "run the script first" message at this point.
5. [ ] **τ sweep:** commit `configs/search/tau_sensitivity.yaml`; p = 80 check against
   `gate.json`; `python scripts/tau_sweep.py --search configs/search/tau_sensitivity.yaml
   --workers 5` (75 validation episodes, ≈ 10 min); `python scripts/lock_models.py --verify`.
6. [ ] **Faithfulness:** `python scripts/attention_faithfulness.py --tag final` (5 validation
   episodes + forward passes). Check: episode travel times equal `val_phase6.csv`.
7. [ ] **Notebook §7–§9**, re-executed end to end with the sweep and faithfulness outputs;
   **`docs/analysis.md`** written from the outputs; read against the pre-written readings.
8. [ ] **Close:** `pytest`, `ruff check src tests scripts`; gap table A8; ROADMAP ticks and
   Decision Log result row; README / CLAUDE.md; worklog/022; commits per
   `.claude/commands/commit.md` (no Claude co-author).

Time: steps 1–4 on 13–14 Nov, steps 5–7 on 14 Nov, step 8 by 15 Nov morning; Phase 9 writing
starts in parallel from step 3.

**Cut order if late (A4), first to last:** route groups (A3) → Hangzhou per-intersection
breakdown → the "gate fired" case (the win / loss cases stay) → knock-out variant of the
faithfulness check → R6 / R9. Never cut: R1, R2, R4, the four comparisons per intersection
and window, the τ sweep, `docs/analysis.md`.

## Experiment log

- τ sweep: 75 rows, split `val`, note `tau_sensitivity: p=<p>, <condition>`.
- Faithfulness: 5 rows, split `val`, note `attention_faithfulness: seed <s>`.
- The notebook adds no rows (no run). Search space for the sweep is the spec file, committed
  before the runs; nothing is selected from it.

## Risks / open questions

- **The τ sweep may show that p < 80 scores better on validation.** That is a finding about
  the calibration grid, reported under A9; it must not lead to a new test run.
- **Faithfulness may come out "not consistent".** With a self-weight of 0.82 and H3 not
  supported this is plausible; the attention heatmap then carries that caveat and the
  explainability claim rests on the gate log and σ.
- **Per-vehicle travel time may not reproduce the simulator's average exactly** (treatment of
  vehicles still inside). Step 2 checks it first; the definition is aligned or the difference
  stated.
- **Vehicle-ID → route join** depends on CityFlow's ID scheme; if the consistency check fails
  the route breakdown is dropped (first item of the cut order anyway).
- **Small effects:** UG-MARL vs DQN is 3.2 s out of 295 s; per-intersection differences may
  all be inside seed noise. Then the answer to "where does it lose" is "spread evenly, no
  single location", and that is what is written.
- **Many comparisons:** tables are descriptive; no significance claims beyond CI separation.
- **Different seeds are different models:** case figures use one seed (the median one) and
  say so; they illustrate, the tables carry the evidence.
- **Notebook size and run time:** it reads ~530 MB of CSV logs; frames are loaded once in §0
  and cached to `results/analysis/cache/*.pkl` (gitignored; pandas pickle, no new dependency) so a re-run takes seconds.
  Outputs are committed, so figures are embedded once — acceptable (PNG, 160 dpi).
- **Time:** Phase 9 needs R1, R2, R4 and the architecture figure first — step 3 is ordered
  accordingly.

## Worklog

Execution ends with `worklog/022-2026-11-13-phase8-analysis.md` (What / Why / How / Result /
Files changed per file / Commits / Next) and its row in `worklog/README.md`.
