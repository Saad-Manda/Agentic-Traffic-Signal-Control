# 013 — Phase 3: data audit, train-split EDA, feature engineering
**Date:** 2026-11-10   **Tag:** data, code, experiment, decision   **ROADMAP:** 3.0, 3.3, 3.4   **Status:** done (3.5 protocol freeze deferred by the user)

## What
Executed Steps 1–7 of `.claude/plans/phase3-data-audit-eda-features.md`:

- **3.0 Data audit** of all six flow files and both road networks → `docs/data_audit.md`.
- **3.3 EDA** on the Jinan training flow only → `notebooks/01_eda_train.ipynb`, 8 figures in
  `docs/figures/eda/`, based on rule-controller rollouts on the train flow.
- **3.4 Feature engineering**: road-network model and a 44-value, capacity-normalised
  observation (`trafficrl.data.roadnet`, `trafficrl.envs.features`), reused by the Phase 4 env.
- Experiment log (`docs/experiment_log.csv`) and run-environment record started.

Step 8 of the plan (ROADMAP 3.5, evaluation-protocol freeze) was **not** run: the user deferred
it until after the rest of Phase 3.

## Why
ROADMAP Phase 3 (rubric R3 "dataset & preprocessing", guidelines G3, G5, G6; PRD F2). Phase 4
needs a verified data layer and a fixed observation definition, and the 3.5 freeze needs
evidence for Δt, reward and normalisation.

Decisions (user, 10 Nov, before execution) and the alternatives rejected:

| Decision | Chosen | Rejected, and why |
|---|---|---|
| Normalisation | lane-capacity scaling, nothing fitted | z-score fitted on train rollouts (depends on the behaviour policy, drifts during training, may not transfer to Hangzhou); both with a switch (one more thing to tune); none (counts up to ~100 next to 0/1 inputs) |
| Seeds for the freeze | 5 for 7.1 / 7.2, 3 for 7.3 / 7.4 | 3 everywhere (95 % CI uses t = 4.30, too wide for the non-overlap criteria); 5 everywhere (compute) |
| Δt and reward | keep Δt = 10 s and −pressure as defaults, check them in the EDA | let a sweep decide (one hour, one controller); fix without a check (decision without evidence) |
| Scope | 3.0, 3.3, 3.4 now; 3.5 later | — |

Decisions taken during execution (flagged to the user in the summary):

- The check "route ends at a boundary node" fired on 1–2 % of every file. These are trips that
  end inside the network; CityFlow simulates them normally. The check was **reclassified from
  defect to information** and execution continued instead of stopping, because the only
  handling consistent with the audit policy (never edit hash-pinned benchmark files) is to
  keep and document. Alternative rejected: excluding those trips (would change a published
  benchmark).
- Observation contents: queue + vehicle count + movement pressure + phase (44 values).
  Rejected: queue only (loses approaching vehicles); lane segments / images (larger state).
- EDA rollouts use three rule controllers (Fixed-Time, Max-Pressure, random) so that the
  feature ranges cover good and bad control. Rejected: Fixed-Time only; waiting for Phase 4.

## How
- **Roadnet model** (`data/roadnet.py`): parses intersections, movements, phases, neighbours;
  canonical lane order = incoming roads by direction of travel (E, N, W, S, from road
  geometry), then lane index (0 left, 1 through, 2 right). Capacity = ⌊lane length / 7.5 m⌋.
- **Audit** (`data/audit.py`, `scripts/audit_data.py`): 17 flow checks and 12 roadnet checks
  that return counts; information counts (trips ending inside the network, routes revisiting
  an intersection, arrival outlier bins by Tukey fences on 60 s bins, same-second arrivals on
  one entry lane, vehicle profiles). Counts for all splits; plots and demand description for
  the train split only (`PLOT_SPLITS = ("train",)`). The report is generated, not hand-written.
- **Features** (`envs/features.py`): `FeatureBuilder` reads the roadnet only. Observation =
  queue/C (12) ‖ count/C (12) ‖ movement pressure (12) ‖ one-hot green phase (8), clipped to
  [0, 1]. Also defines the raw phase pressure (Max-Pressure rule) and intersection pressure
  (reward). Config block `features:` in `configs/base.yaml`.
- **Traces** (`data/trace.py`, `scripts/collect_traces.py`, `configs/eda.yaml`): config
  `configs/eda.yaml`, split **train** (Jinan `real`), episode 3600 s, yellow 5 s, 1 simulator
  thread. Fixed-Time (30 s), Max-Pressure (Δt 10 s) and random phase (Δt 10 s), seeds 0, 1, 2
  each; Max-Pressure at Δt 5, 15, 20 s with seed 0. 12 runs, 9–15 s each. State recorded every
  10 s; every vehicle tracked (entry, exit, travel time = exit − planned start).
  `train_flow()` raises `SplitError` for any other dataset or split.
- **Notebook**: built once from a cell list with `nbformat`, then executed with
  `jupyter nbconvert --execute --inplace`. Every figure has a caption and an interpretation
  cell; the last section lists the insights.
- **Figures**: shared style in `utils/plotting.py`; the two colour sets were checked with a
  colour-blind-separation validator (all checks pass).
- A first single-seed smoke run of `collect_traces.py` wrote 3 rows to the experiment log;
  the log and its output folder were deleted before the real run, so the log holds only the
  12 real runs.

## Result
**Audit — 0 defects** in 6 flow files (31,659 records) and 2 road networks. Information:

| File (split) | Vehicles | End inside network | Revisit an intersection | Burst bins | Low bins |
|---|---|---|---|---|---|
| Jinan `real` (train) | 6295 | 77 (1.2 %) | 208 (3.3 %) | 0 | 5 |
| Jinan `real_2000` (val) | 4365 | 54 (1.2 %) | 160 (3.7 %) | 0 | 0 |
| Jinan `real_2500` (test) | 5494 | 70 (1.3 %) | 189 (3.4 %) | 0 | 0 |
| Hangzhou `real` (held-out) | 2983 | 55 (1.8 %) | 130 (4.4 %) | 5 | 0 |
| Hangzhou `real_5734` (held-out) | 6538 | 124 (1.9 %) | 277 (4.2 %) | 0 | 0 |
| Hangzhou `real_5816` (held-out) | 6984 | 124 (1.8 %) | 293 (4.2 %) | 0 | 0 |

**EDA rollouts on the train flow** (not baseline results):

| Controller | Δt (s) | Seeds | Avg travel time (s) | Mean waiting vehicles | Completed trips |
|---|---|---|---|---|---|
| Fixed-Time | 30 | 0, 1, 2 | 515.2 (identical) | 430.5 | 4969 |
| Max-Pressure | 10 | 0, 1, 2 | 374.9 (identical) | 247.4 | 5446 |
| Random phase | 10 | 0, 1, 2 | 645.8 / 656.7 / 650.2 | 591.6 / 582.7 / 579.5 | 4417 / 4463 / 4405 |
| Max-Pressure | 5 | 0 | 451.1 | 388.4 | 5063 |
| Max-Pressure | 15 | 0 | 344.1 | 190.6 | 5605 |
| Max-Pressure | 20 | 0 | 347.3 | 195.8 | 5635 |

Fixed-Time and Max-Pressure reproduce the install benchmark of worklog/007 exactly.

**Findings**
- Capacity scaling works: maximum lane occupancy 0.96 (Fixed-Time, random) and 0.57
  (Max-Pressure); no sample above 1, so clipping never acts on the train traces.
- Demand: 6,295 veh/h on average, 3,000 to 8,160 veh/h in five-minute blocks; western entries
  carry 2.6× the eastern ones; turning ratios 57 % through / 14 % left / 28 % right in every
  direction.
- Travel time is right-skewed; 13 % (Max-Pressure) to 21 % (Fixed-Time) of trips do not finish
  within the hour; queues grow all hour (no steady state).
- **Δt:** Max-Pressure is 8 % better at 15 s than at the provisional 10 s; 5 s is clearly worse.
- **Reward:** queue and |pressure| correlate at 0.65–0.82 (0.85–0.97 for network totals), but
  per intersection the correlation ranges from −0.36 to 0.96, and pressure is near zero when
  upstream and downstream lanes are both full.
- **Seeds:** the rule controllers are deterministic across seeds; only random actions vary.

**Verification:** `pytest` → 92 passed (incl. the slow simulator tests);
`ruff check src tests scripts` → clean; `python scripts/audit_data.py` → exit 0; notebook runs
top to bottom.

**Not done / limits:** Step 8 (3.5 freeze) deferred. Native-Ubuntu numbers not produced (WSL
only). The EDA covers one hour and three rule controllers; statements about what will help the
learned model are expectations, not results.

## Files changed
| File | Action | What changed | Why |
|------|--------|--------------|-----|
| `src/trafficrl/data/roadnet.py` | created | `Roadnet`, `Intersection`, `Movement` dataclasses; `parse_roadnet`, `load_roadnet`, `lane_capacity`; `YELLOW_PHASE`, `DIRECTIONS`; `RoadnetError` for intersections without 4 approaches / 12 movements | One roadnet parser for the audit, features, EDA and the Phase 4 env; canonical lane order for parameter sharing |
| `src/trafficrl/data/audit.py` | created | `audit_flow` (17 defect + 7 information counts), `audit_roadnet` (12 defect + 4 size counts), `arrival_counts`, `burst_threshold`, `low_threshold`, `defects` | ROADMAP 3.0 checks as tested, reusable functions |
| `src/trafficrl/data/trace.py` | created | `train_flow` (split guard), `make_engine`, `make_chooser` (fixed_time / max_pressure / random), `collect_trace` (lane states every 10 s, per-vehicle entry / exit, summary) | Queues and travel times exist only under a controller; needed for EDA figures 4, 6, 8–10 |
| `src/trafficrl/data/eda.py` | created | `entry_demand`, `movement_demand`, `turning_ratios`, `intersection_load`, `route_lengths`, `signalised_crossings`, `load_run`, `load_summaries`, `feature_arrays`, `network_series` | Aggregations kept in `src/` (tested / importable) so the notebook stays thin |
| `src/trafficrl/envs/features.py` | created | `FeatureSpec` (groups, `vehicle_space_m`, `clip`, `normalisation`; `from_config`), `FeatureBuilder` (`dim`, `slices`, `observation`, `observations`, `phase_pressure`, `intersection_pressure`) | ROADMAP 3.4: the observation and pressure definitions every agent and the env will share |
| `src/trafficrl/utils/runlog.py` | created | `append_experiment_row`, `env_info`, `save_env_info`, `git_commit`; log columns incl. `metric_name` / `metric_value` | Guideline G8: every run logged with versions + hardware; needed now because the EDA rollouts are runs |
| `src/trafficrl/utils/plotting.py` | created | `apply_style`, `save_figure`, colour constants, `CONTROLLER_COLOR` / `CONTROLLER_LABEL`, `BLUE_RAMP` | One figure style for audit, EDA and later results figures (not in the plan) |
| `scripts/audit_data.py` | created | Runs the audit on all splits, writes `docs/data_audit.{md,json}`, train-only figures and demand description; exit 1 on a defect | Reproducible command for the audit report |
| `scripts/collect_traces.py` | created | Expands `configs/eda.yaml` into runs, collects traces, saves outputs + resolved config + `env.json`, appends experiment-log rows | Reproducible command for the EDA rollouts |
| `scripts/bench_cityflow.py` | modified | One over-long line (the `make_engine(...)` call in `main`) wrapped; no behaviour change | The lint command now covers `scripts/`, and this pre-existing line failed it |
| `configs/base.yaml` | modified | New `features:` block (`groups`, `normalisation: lane_capacity`, `vehicle_space_m: 7.5`, `clip: true`) | Feature parameters live in config, not code |
| `configs/eda.yaml` | created | `data: jinan/train`, `eda.seeds`, `eda.controllers`, `eda.dt_sweep`, `eda.arrival_bin_seconds`, `output.results_dir` | Search values written before the first run (guidelines) |
| `notebooks/01_eda_train.ipynb` | created | 43 cells: 8 figures + 2 tables, each with caption and interpretation; insights table; executed outputs included | ROADMAP 3.3 |
| `docs/data_audit.md` | created (generated) | Scope, result, defect tables for 6 flow files and 2 roadnets, information counts with interpretation, train-split demand description (3 figures), privacy, handling | ROADMAP 3.0 deliverable |
| `docs/data_audit.json` | created (generated) | All audit counts + train-split description | Machine-readable copy; read by the notebook |
| `docs/figures/audit/jinan_train_{arrivals,route_length,entry_demand}.png` | created (generated) | 3 audit figures, train split only | Figures referenced by `docs/data_audit.md` |
| `docs/figures/eda/{02_network,03_arrival_rate,04_travel_time,06_feature_distributions,07_turning_ratios,08_load_imbalance,09_reward_check,10_decision_interval}.png` | created (generated) | 8 EDA figures | For the report and slides |
| `docs/experiment_log.csv` | created | Header + 12 rows (EDA rollouts, split = train) | Guideline G8 |
| `tests/data/toy.py` | created | Hand-built one-intersection roadnet and vehicle record | Tests must not depend on the downloaded data (not in the plan) |
| `tests/data/test_roadnet.py` | created | 7 tests: lane order, movements, phases, capacity, malformed input, both real roadnets | Verify D1 |
| `tests/data/test_audit.py` | created | 21 tests: clean flow, one case per defect, information counts, roadnet audit | Verify D2 |
| `tests/data/test_trace.py` | created | Split guard (7 cases), config check, audit plots train only, notebook scan, controller rules, 2 slow determinism tests | Leakage guards and determinism |
| `tests/envs/__init__.py`, `tests/envs/test_features.py` | created | 10 tests: exact 44-vector, bounds, clipping, groups, invalid spec, pressures, no-flow-file guard, config, real roadnets | Verify D3 |
| `tests/utils/test_runlog.py` | created | 2 tests: header once / rows appended; `env_info` keys | Verify D6 |
| `ROADMAP.md` | modified | Ticked 3.0, 3.3, 3.4 (3.4 reworded: capacity scaling, nothing fitted); Final-gate "Data audit and EDA completed" ticked; 5 Decision Log rows (audit handling, normalisation, observation, seeds, Δt / reward open point) | Progress and decisions recorded |
| `.claude/reference/architecture.md` | modified | MDP state now the concrete 44-value definition; reward and Δt with EDA evidence; Max-Pressure rule; data-flow line | Keep the reference in sync with the code |
| `.claude/reference/api.md` | modified | New sections "Data layer" and "Features"; `runlog.py` and `plotting.py` under Utilities; new scripts listed | Interfaces Phase 4 will build on |
| `.claude/reference/database.md` | modified | Rule 2 reworded (nothing fitted); rule 9 lists the code guards; audit status; layout adds `results/eda/` and `docs/figures/`; train-row use | Leakage rules match the implementation |
| `.claude/reference/testing.md` | modified | Added roadnet, audit, features, EDA-guard and run-log test descriptions | Test inventory |
| `docs/dataset_card.md` | modified | §5 now reports the full audit result; `route` row notes that 1–2 % of routes end inside the network | The card said every route runs boundary to boundary |
| `docs/how_it_works.md` | modified | Decision-loop box and a new paragraph "How the state is scaled" | CLAUDE.md asks to keep the explainer in sync with the MDP |
| `PRD.md` | modified | §9 Open Questions: Δt 10 s vs 15 s | EDA finding to settle at the freeze |
| `README.md` | modified | New §6 (audit, traces, notebook: commands, outputs, expected numbers); lint command includes `scripts`; repository layout | No hidden manual steps |
| `CLAUDE.md` | modified | Phase 3 commands; structure (`data/`, `envs/`, `utils/`, `docs/`); architecture MDP line; key files; note on open points for 3.5 | Project constitution in sync |
| `.claude/plans/phase3-data-audit-eda-features.md` | created | The plan; Steps 1–7 ticked; "Deviations during execution" section | Plan and its execution record |
| `worklog/013-2026-11-10-phase3-audit-eda-features.md`, `worklog/README.md` | created / modified | This entry; index row | Work log rule |

Not part of this work (already in the working tree before it started):
`.claude/commands/execute.md` (modified) and
`worklog/012-2026-11-10-worklog-per-file-changes.md` (untracked).

## Commits
Not committed yet.

## Next
- **User:** read `docs/data_audit.md` and `notebooks/01_eda_train.ipynb`, then decide the open
  points for 3.5: Δt (10 s / 15 s / tune Max-Pressure's interval on validation), reward
  (−pressure / −queue), and whether Fixed-Time / Max-Pressure are reported without a CI.
- Run Step 8 of the plan (protocol freeze) after sign-off, then Phase 4 (env wrapper on top of
  `FeatureBuilder`; real Fixed-Time / Max-Pressure agents must reproduce 515.2 s / 374.9 s).
- Two worklog entries carry the number 012; renumber one (confirm with the user).
- Repeat the rollouts on native Ubuntu when 0.1b is closed there.
