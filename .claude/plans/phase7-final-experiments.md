# Plan — Phase 7 final experiments (ROADMAP 7.1–7.6)

Status: **in execution since 12 Nov 2026.** Tick steps `[x]` during execution and note
deviations under each step.

## Goal

Open the Jinan **test** flow and the **Hangzhou** held-out city for the first time, once, with
every model exactly as locked on validation, and produce every number the report needs.

| Output | ROADMAP | Serves |
|---|---|---|
| `docs/results/test_main.{csv,md}` — Model \| Validation \| Test \| Training time \| Params + H1 / H3 / minimum-outcome verdicts | 7.1 | F6, R6, G10 |
| `docs/results/test_ablation.{csv,md}` — full vs −gate / −attention / −ensemble | 7.2 | F7, R5 |
| `docs/results/test_robustness.{csv,md}` — 8 models × 6 conditions, regret table, H2 verdict | 7.3 | F8, R6, G11 |
| `docs/results/heldout_generalisation.{csv,md}` — 7 models × 3 Hangzhou flows, zero-shot | 7.4 | F9, G10 |
| `docs/results/test_efficiency.{csv,md}` — latency, training time, params, memory | 7.5 | R6 |
| `docs/results/test_safety_explain.{csv,md}` — wait tail, fallback, gate and attention statistics | 7.6 | F10, G11 |
| Per-decision, per-step and per-vehicle logs of every final episode (gitignored, `results/final/`) | input of Phase 8 | 8.1–8.4 |

Out of scope: any training, retuning or re-calibration; figures and written interpretation
(Phase 8); SHAP (optional in ROADMAP, not done); the demo replay (9.3); the optional
extension in the last section.

## Decisions

Decided with the user on 12 Nov unless marked *(plan)*.

| # | Topic | Decision | Reason |
|---|---|---|---|
| P1 | Path after 6.8 | **Phase 7 runs with the models as locked.** No improvement attempt before the test flow is opened. | The tie with Max-Pressure under noise is the ceiling of the design (81–96 % of decisions are Max-Pressure there), not a tuning problem. H1–H3 do not ask for it. A round of changes now would reopen selection 5 days before submission. |
| P2 | "Test first, improve if it loses" | **Not done in that form.** Once the test result is known, any change chosen because of it uses the test flow for selection, and the improved number is no longer a test result. Validation already gave that information (6.8). A follow-up is allowed only as the pre-specified extension at the end of this plan: designed now, selected on validation only, reported as an extra labelled row; the locked model stays the headline. | Hard rule "test flows run once, after selection on validation". |
| P3 | Seeds | **0–4 for 7.1–7.4** (protocol said 0–2 for 7.3 / 7.4). Protocol §6 amended with a dated row **before** any final run. | 5 trained seeds exist, an episode takes ~6 s. 95 % CI uses t = 2.78 instead of 4.30. Same seeds as the validation table 6.8. The clean column of 7.3 is then the same runs as 7.1 / 7.2, so each test episode is run exactly once. |
| P4 | H2 reading | Three results, verdict in three levels (below). | PRD §1c does not say in which condition max wait is compared; on validation the answer differs between clean and noise. Fixed before the runs. |
| P5 | One pass *(plan)* | Two studies, each episode run once: `final_test` (Jinan test, 8 models × 6 conditions × 5 seeds = 240 episodes) and `final_heldout` (Hangzhou, 7 models × 3 flows × 5 seeds = 105 episodes). 7.1, 7.2, 7.3, 7.5 and 7.6 are all read from `final_test`. | No episode is repeated for a second table. |
| P6 | Log everything in that pass *(plan)* | Every final episode writes per-step and per-vehicle logs; UG-MARL variants also write per-decision logs. | Phase 8.4 (per-intersection / per-time-window breakdown, representative cases, gate analysis) must not need a second run on the test flow. |
| P7 | Workers *(plan)* | Final studies run with `--workers 1` on an idle machine, on native Ubuntu. | Decision latency (7.5) is then valid in the same pass. 345 episodes ≈ 1 h. |
| P8 | Lock manifest *(plan)* | `models/manifest.json` records SHA-256 of every `best.pt`, `gate.json` and training config, committed before the final runs; `--final` refuses to start if a file differs. | Proof that nothing changed after the test flow was opened. |
| P9 | Models | Test: Fixed-Time, Max-Pressure, DQN, IDQN, UG-MARL, −gate, −attention, −ensemble. Hangzhou: the same without IDQN. | IDQN has one network per Jinan intersection, so it cannot run on 16 other intersections; reported as "not transferable" (a limitation of independent learners, not a missing run). |
| P10 | Conditions | Test: clean, noise 0.1, noise 0.2, dropout 10 %, dropout 20 %, surge ×1.5 — one factor at a time. Hangzhou: clean only. | Protocol §6. No combined conditions and no perturbations on Hangzhou (not in the protocol). |
| P11 | τ on Hangzhou and under surge | τ from each model's Jinan validation `gate.json`, unchanged. | Protocol §6: zero-shot, τ from Jinan val. |

## Readings fixed before the runs

Copied into ROADMAP Phase 7 in step 1. All on the test flow, seeds 0–4, 95 % CI = Student t.
The verdicts are computed by code written and tested before the test flow is opened.

- **H1 (clean):** supported iff the upper end of UG-MARL's travel-time CI is below
  Max-Pressure's value **and** below the lower end of IDQN's CI. Otherwise not supported.
- **H3 (clean):** supported iff the upper end of UG-MARL's CI is below the lower end of
  −attention's CI. (Validation: 302.3 vs 301.9 s — expected to be a negative result.)
- **H2 (gate vs −gate, same checkpoints, so per-seed paired differences with a 95 % CI are
  also reported):**
  - (i) mean max per-vehicle wait on the clean flow lower with the gate;
  - (ii) mean max per-vehicle wait at noise 0.2 lower with the gate;
  - (iii) travel-time increase from clean to noise 0.2 smaller with the gate.
  - **Supported** = (i), (ii) and (iii). **Partly supported** = (ii) and (iii) only.
    **Not supported** = anything else.
- **Minimum acceptable outcome:** UG-MARL's mean max wait ≤ Max-Pressure's on the clean flow;
  the same comparison is listed for every condition.
- **DQN:** not part of H1. Reported as "better / level / worse" by CI overlap.
- **Robustness summary (descriptive, no threshold):** for each model and condition,
  regret = its mean travel time − the lowest mean among the five agents in that condition;
  the table reports each model's **largest** and **mean** regret over the six conditions.
  Claim written in advance from 6.8: *UG-MARL has the smallest worst-case regret; it does not
  beat Max-Pressure under noise.* If the test table says otherwise, that is what is reported.
- **Surge and Hangzhou:** no hypothesis; described as measured. For Hangzhou also: is the order
  of the models the same as on Jinan test, and how often does the gate fire in an unseen city.
- Nothing is chosen from any Phase 7 table.

## Design

### Specs (written and committed before the runs)

`configs/final_test.yaml`
```yaml
study: final_test
dataset: jinan
split: test
seeds: [0, 1, 2, 3, 4]
tag: final
log_trace: true                       # per-step + per-vehicle logs for every model
models:
  fixed_time:    {config: configs/eval_baselines.yaml}
  max_pressure:  {config: configs/eval_baselines.yaml}
  dqn:           {config: configs/eval_rl_baselines.yaml}
  idqn:          {config: configs/eval_rl_baselines.yaml}
  ugmarl:        {config: configs/eval_ugmarl.yaml, log_decisions: true}
  ugmarl-nogate: {config: configs/eval_ugmarl.yaml, agent: ugmarl, train_model: ugmarl,
                  log_decisions: true, overrides: [agents.ugmarl.use_gate=false]}
  ugmarl-noattn: {config: configs/eval_ugmarl.yaml, agent: ugmarl, log_decisions: true,
                  overrides: [agents.ugmarl.use_attention=false]}
  ugmarl-noens:  {config: configs/eval_ugmarl.yaml, agent: ugmarl,
                  overrides: [agents.ugmarl.use_ensemble=false]}
conditions:
  - {}
  - {robustness.noise_std: 0.1}
  - {robustness.noise_std: 0.2}
  - {robustness.dropout_p: 0.1}
  - {robustness.dropout_p: 0.2}
  - {robustness.surge: 1.5}
```

`configs/final_heldout.yaml` — `dataset: hangzhou`, `split: heldout`, the same models without
`idqn`, and
```yaml
conditions:
  - {name: real,      data.flow_index: 0}
  - {name: real_5734, data.flow_index: 1}
  - {name: real_5816, data.flow_index: 2}
```
The exact override spelling for −attention / −ensemble (config file vs flags) is taken from
the Phase 6 validation commands during step 3, so the model labels stay `ugmarl-noattn` /
`ugmarl-noens`.

### Code changes

1. **`scripts/robustness.py`** (runner, extended — no new evaluation logic)
   - a condition may have `name` and any config override (needed for `data.flow_index`;
     today the directory name comes from the `robustness.*` keys only, so the three Hangzhou
     flows would collide);
   - per-model `log_decisions`, spec-level `log_trace` → passed to `evaluate.py`;
   - `--results-dir` default stays; final studies are run with `--results-dir results/final`;
   - with `--final`: `models/manifest.json` must exist and verify, else exit 2;
   - with `--final`: a job directory that holds an unfinished run (no `metrics.csv`) stops the
     script with a message instead of silently re-running (a re-run needs a Decision Log row;
     `--rerun-failed` then allows it).
2. **`scripts/evaluate.py`, `trafficrl.eval.runner`** — `--log-trace` writes per seed:
   - `trace_s<seed>.csv`: one row per env step × intersection — t, iid, phase, in_yellow,
     true waiting vehicles and true vehicle count on the 12 incoming lanes (`StepRecorder` in
     `eval/explain.py`, read-only on `info` and the env's true state);
   - `vehicles_s<seed>.csv`: one row per vehicle — enter time, leave time (empty if still
     inside), total waiting time (`EpisodeTracker.vehicle_frame()`; the tracker additionally
     stores enter / leave seconds, its `result()` keys and values do not change);
   - `metrics.csv` gains `wait_p95_s`, `wait_p99_s` (from the same per-vehicle waits).
   Recorders only read; step 4 checks that results are unchanged with logging on.
3. **`scripts/lock_models.py`** — `--write` builds `models/manifest.json` (per model × seed:
   paths and SHA-256 of `best.pt`, `gate.json`, `config.yaml`; τ, p; best validation score;
   git commit); `--verify` exits 2 on any difference. `models/manifest.json` is tracked.
4. **`src/trafficrl/eval/hypotheses.py`** — pure functions on per-seed metric frames:
   `h1`, `h3`, `h2` (three sub-results, paired differences, level), `minimum_outcome`,
   `regret_table`. Uses `utils.stats.summarise`.
5. **`scripts/final_tables.py`** — pure reader of `results/final/…` and the training
   summaries; `--study final_test | final_heldout | all`. Writes the six tables listed under
   Goal. `test_main` takes its validation column from `docs/results/val_phase6.csv`.
   - 7.4 table: model × flow (mean ± std [CI]) and the mean over the three flows; fallback
     rate per flow; IDQN row "not transferable".
   - 7.5 table: decision latency mean / p95 (clean test runs, 12 intersections, CPU, one
     thread), training time (from `summary.json`; measured with 5 runs in parallel — stated),
     parameters, checkpoint size on disk, GPU memory "not used (CPU training and inference)".
   - 7.6 table, from `decisions_s*.csv` and `vehicles_s*.csv`, per condition: max / p99 / p95
     wait; fallback rate overall, per intersection and per 10-minute window; σ mean and share
     above τ; share of gate firings where the Max-Pressure phase differs from the learned
     phase; mean attention weight per slot (self, E, N, W, S). Worded as associations.

No change to agents, env dynamics, features, gate logic or any locked file.

### Hangzhou code path without Hangzhou flows

The zero-shot path (16 intersections, shared network loaded from a Jinan checkpoint) has never
been executed, and no Hangzhou flow may be simulated before the final run. Test
`tests/eval/test_zero_shot.py` therefore builds a `TrafficEnv` on the **Hangzhou roadnet with
a synthetic flow** (a few vehicles on two-road routes taken from the roadnet's road links),
runs 120 s with Max-Pressure, Fixed-Time, DQN and UG-MARL (weights saved from a Jinan-shaped
agent, loaded into the Hangzhou-shaped one) and checks 16 actions per step; IDQN must raise.
The roadnet is already read by the audit and existing tests; no file of the held-out split is
opened.

## Justification

- **One runner for all of Phase 7** instead of `run_ablation.py` + a main-comparison script:
  7.1, 7.2 and 7.3 are the same model × condition grid; the planned `run_ablation.py` is
  dropped (CLAUDE.md "planned" block updated).
- **Tables in a separate reader script:** the runner must not interpret; the reader can be
  re-run any number of times without touching the test flow.
- **Verdict code written before the runs:** the criteria cannot drift after the numbers are
  seen. Rejected: judging by hand from the tables.
- **Logging in the same pass vs. a later deterministic re-run:** a re-run is bit-identical but
  still a second pass over the test flow; logging costs a few MB.
- **Latency from the final clean runs (workers 1)** instead of a separate benchmark: one
  number, same episodes as the table. Rejected: parallel runs + separate latency script.
- **5 seeds for 7.3 / 7.4:** more evidence, decided before any final number exists; the only
  cost is a protocol amendment.
- **No combined perturbations, no perturbed Hangzhou:** not in the frozen protocol; adding
  conditions now would enlarge the result space after validation was seen.

## Files

| File | Action | Purpose |
|---|---|---|
| `ROADMAP.md` | modify | Phase 7 block: studies, commands, readings; Decision Log rows (P1–P3, P4, P5–P9); ticks |
| `docs/evaluation_protocol.md` | modify | §6 seeds 0–4 for 7.3 / 7.4; §9 change row; H2 reading and regret statistic as clarifications |
| `PRD.md` | modify | §1c: H2 three-level reading (reference to the protocol) |
| `configs/final_test.yaml`, `configs/final_heldout.yaml` | create | Specs |
| `scripts/robustness.py` | modify | Named conditions, logging flags, manifest check, unfinished-run guard |
| `scripts/evaluate.py`, `src/trafficrl/eval/runner.py` | modify | `--log-trace`, wait percentiles |
| `src/trafficrl/eval/explain.py` | modify | `StepRecorder` |
| `src/trafficrl/envs/metrics.py` | modify | enter / leave seconds, `vehicle_frame()` |
| `scripts/lock_models.py` | create | Manifest write / verify |
| `models/manifest.json` | create (generated) | Locked files and hashes |
| `src/trafficrl/eval/hypotheses.py` | create | H1–H3, minimum outcome, regret |
| `scripts/final_tables.py` | create | The six result tables |
| `tests/…` | create / modify | See Tests |
| `docs/results/test_*.{csv,md}`, `heldout_generalisation.{csv,md}` | create (generated) | Results |
| `docs/experiment_log.csv` | modify (by `evaluate.py`) | 345 final rows + smoke rows |
| `README.md`, `CLAUDE.md` | modify | Phase 7 commands; `run_ablation.py` removed from "planned" |
| `worklog/020-…-phase7-final-experiments.md`, `worklog/README.md` | create / modify | Work log |

## Config

No new key in `base.yaml`. Spec keys added: `log_trace` (study), `log_decisions` (model),
`name` (condition). Seeds 0–4 are passed to `evaluate.py`, which seeds Python, NumPy, PyTorch,
CityFlow, the perturbation and the surge RNG; seed s loads the checkpoint trained with seed s.

## Data-leakage check

- Steps 1–5 read **train / validation flows and the Hangzhou roadnet only**. Smoke runs use the
  validation flow. The test flow and the Hangzhou flows are opened in steps 6 and 7 only, by
  `robustness.py --final`, after the manifest and all code are committed.
- Nothing is fitted or selected in Phase 7: checkpoints, K, d, p\*, τ come from Phase 6
  (manifest-verified). Readings are written before the runs.
- After step 6 no file under `src/`, `configs/` or `models/` that affects the numbers is
  changed. A bug that forces a re-run is fixed, logged in the Decision Log with the reason,
  and only the affected jobs are repeated.

## Tests

- `test_robustness.py`: named condition → directory and overrides; `log_decisions` /
  `log_trace` → flags; `--final` without / with a wrong manifest → exit 2; unfinished final
  job → stops.
- `test_lock_models.py`: write then verify passes; a changed byte fails.
- `test_metrics.py`: enter / leave seconds; `result()` unchanged; percentiles.
- `test_explain.py`: `StepRecorder` rows and columns.
- `test_evaluate.py`: `--log-trace` writes both files; metrics equal with and without logging
  (fake engine).
- `test_hypotheses.py`: each verdict on hand-made numbers, including the deterministic
  baseline case, each H2 level, and regret.
- `test_zero_shot.py`: as described above.
- `test_final_tables.py`: tables from fake run directories; IDQN row on held-out.

## Steps and verification

1. [x] **Record decisions:** ROADMAP Phase 7 block (studies, readings, commands) and Decision
   Log rows; protocol §6 / §9; PRD §1c. Commit.
2. [x] **Code + tests** (changes 1–5). `pytest` and `ruff check src tests scripts` pass.
3. [x] **Specs:** `configs/final_test.yaml`, `configs/final_heldout.yaml`. Commit.
4. [x] **Smoke on validation** — a copy of the test spec with `split: val`, study
   `smoke_final`, run with `--workers 1`, logging on:
   - clean and perturbed columns equal `docs/results/val_robustness.md` and
     `val_phase6.md` for all 8 models (logging and code changes leave results unchanged);
   - `final_tables.py` builds every table from the smoke study (labelled smoke, written to the
     scratch directory, not to `docs/results/`);
   - surge ×1.5 runs to the end for every model (first time surge is used with learned agents).
   A mismatch is investigated before going on.
   *Result (12 Nov):* 48 jobs, 240 validation episodes. All 30 cells shared with
   `val_robustness.csv` and the 8 clean rows of `val_phase6.csv` reproduce with difference 0
   (travel time, std, max wait, fallback rate). Surge ×1.5 ran for every model. Smoke
   verdicts on validation: H1 supported, H3 not supported, H2 partly supported, minimum
   outcome met. *Deviations:* run with `--workers 5` (latency is not needed for the smoke);
   the surge condition was thereby seen on validation for the first time — nothing is
   selected from it. Also changed: `runlog.git_commit` ignores the experiment log itself when
   deciding `-dirty` (otherwise every row after the first of a study is marked dirty).
5. [x] **Lock:** `python scripts/lock_models.py --write`; check τ and validation scores in the
   manifest against `gate_calibration.md` / `val_phase6.md`; commit code, specs and manifest.
   Working tree clean (the experiment log must show a commit hash without `-dirty`).
6. [ ] **Final test study** (idle machine, mains power):
   `python scripts/robustness.py --spec configs/final_test.yaml --final --workers 1 --results-dir results/final`
   → 48 jobs, 240 episodes. Checks: 240 new `test` rows; rule-based agents identical across
   seeds on the clean flow; no job failed.
7. [ ] **Final held-out study:**
   `python scripts/robustness.py --spec configs/final_heldout.yaml --final --workers 1 --results-dir results/final`
   → 21 jobs, 105 episodes, 105 `heldout` rows.
8. [ ] **Tables:** `python scripts/final_tables.py --study all` → the six tables; verdicts
   printed. Read against the pre-written readings; report to the user as measured.
9. [ ] **Close:** ROADMAP ticks 7.1–7.6 and Decision Log row with the results; final-gate
   lines "Models tuned using validation only" and "Held-out evaluation completed";
   README / CLAUDE.md commands; worklog/020; commits per `.claude/commands/commit.md`
   (no Claude co-author).

Time: steps 1–5 on 13 Nov, steps 6–8 the same evening (≈ 1 h of runs), step 9 and Phase 8
from 14 Nov.

## Experiment log

345 rows from `evaluate.py`: 240 `test` (note `final_test: <model>, <condition>`, `final=True`)
and 105 `heldout` (`final_heldout: <model>, <flow>`); plus the validation smoke rows
(`smoke_final: …`). No tuning, so no search space.

## Risks / open questions

- **Expected result, from validation:** H1 likely supported (UG-MARL below IDQN and
  Max-Pressure), H3 likely not, H2 likely "partly supported", UG-MARL about level with DQN on
  clean data. The report then rests on worst-case regret and the gate's behaviour, with H3 as
  a negative result. This is accepted (ROADMAP Risks).
- **Surge ×1.5 may gridlock the network** for every controller; travel time is then dominated
  by vehicles still inside at t = 3600. Reported as measured, with throughput next to it.
- **Hangzhou is out of distribution** (other lane lengths, 16 intersections): σ may exceed τ
  almost always, making UG-MARL ≈ Max-Pressure there. That is the gate working as designed and
  is reported with the fallback rate.
- **Max wait is noisy** (std 120–150 s over 5 seeds on validation); H2 (i) may go either way.
- **Training time** was measured with 5 runs in parallel, so it is comparable between models
  only roughly; stated under the table. No retraining just for timing.
- **A crash mid-study:** finished jobs are skipped on restart; the unfinished job needs a
  Decision Log row and `--rerun-failed`.
- **Attention shows no gain on validation:** 7.6 attention statistics may be near-uniform;
  reported as such.

## Optional extension (not part of this plan — needs a separate go-ahead and plan)

Only if the report draft is on schedule by 15 Nov. Specified now so that it cannot be shaped
by the test result:

- **Idea:** detector smoothing inside UG-MARL — the policy input and the fallback's lane
  counts are a short moving average of the last m readings — aimed at the noise conditions
  where the gate hands 81–96 % of decisions to Max-Pressure on raw noisy readings.
- **Rules:** m from a grid written in `configs/` first, chosen on validation (clean + the four
  detector perturbations, as for p\*); no retraining unless the plan says so beforehand;
  Max-Pressure with the same smoothing is run as a control, so a gain from the filter alone is
  not credited to UG-MARL; one run on test.
- **Reporting:** an extra row "UG-MARL + smoothing (added after the main evaluation)". The
  locked model's rows, the hypotheses verdicts and the headline numbers do not change.

## Worklog

Execution ends with `worklog/020-2026-11-13-phase7-final-experiments.md` (What / Why / How /
Result / Files changed per file / Commits / Next) and its row in `worklog/README.md`.
