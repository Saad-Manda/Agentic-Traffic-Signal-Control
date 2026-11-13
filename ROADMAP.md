# ROADMAP — Uncertainty-Gated Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control (UG-MARL, PS16)

Solo project. Timeline: **27 Oct → 17 Nov 2026**. Reviews: **3 Nov, 10 Nov, 17 Nov**.
**Re-baselined 9 Nov** (see Decision Log) — phase dates below are the new plan:

| Dates | Work |
|---|---|
| 9–10 Nov | 0.2–0.3 skeleton, Phase 1 dataset selection + split plan, 3.0 audit, 3.1–3.2 split + leakage test |
| **10 Nov Review 2** | problem definition + H1–H3 + success criteria, dataset choice + audit + split plan, literature gap table (draft), UG-MARL design diagram, protocol draft |
| 10–11 Nov | 3.3–3.5 EDA + **protocol freeze**; Phase 4 env + Fixed-Time / Max-Pressure |
| 11–13 Nov | Phases 5–6: DQN, IDQN, UG-MARL + validation tuning |
| 13–15 Nov | Phase 7 runs (overnight on native Ubuntu), Phase 8 analysis |
| 14–16 Nov | Phase 9 report/poster/demo; Phase 2 literature written alongside throughout |
| 16 Nov | Phase 10 viva prep |
Tick `[x]` as tasks complete; record decisions in the Decision Log at the bottom.
Full what/why/how of each piece of work lives in `worklog/` (index: `worklog/README.md`).

Rubric rows referenced as **R1–R8** (see `PRD.md` §7).

Guideline steps referenced as **G1–G12** (`ProjectGuidelines.pdf`; PS16 rubric wins on conflict,
see `PRD.md` §7).

| Guideline step | Covered by |
|---|---|
| G1 Problem & success metric | `PRD.md` §1a–1c |
| G2 Literature & gap | Phase 2 |
| G3 Data collection & audit | Phase 1, Phase 3 |
| G4 Splitting & leakage prevention | 1.4, 3.1–3.2 |
| G5 EDA | 3.3 |
| G6 Preprocessing & features | 3.4 |
| G7 Baselines | Phases 4–5 |
| G8 Train & tune | Phases 5–6 |
| G9 Evaluation metrics | 4.2, `PRD.md` §5 |
| G10 Final testing & fair comparison | Phase 7 |
| G11 Error analysis, explainability, robustness | 7.3–7.6, Phase 8 |
| G12 Reproducibility, ethics, documentation | 0.2–0.3, Phase 9 |

---

## Phase 0 — Setup & decisions (27 Oct – 10 Nov) → G12

- [x] 0.1 **Choose simulator** → CityFlow (see Decision Log); docs updated
- [x] 0.1b Build CityFlow (pybind11 bump for Py 3.12) on WSL + native Ubuntu; time one Jinan
      episode with Fixed-Time & Max-Pressure; write install steps to README
      - [x] WSL: built (pybind11 v2.13.6), Jinan episode FT 2.7 s / MP 1.8 s, README written → worklog/007
      - [x] native Ubuntu (11 Nov): built (CMake 3.28.3 from pip, `--no-build-isolation`, no sudo),
            FT 1.13 s / MP 1.14 s, metrics identical to WSL; Phase 5 val runs reproduced
            **bit-identically** (FT, MP, DQN and IDQN, all 5 seeds) → worklog/018
- [x] 0.2 Create `.venv`, `requirements.txt`, repo skeleton (`src/trafficrl`, `configs`, `scripts`, `tests`, `notebooks`)
- [x] 0.3 Seeding utility + YAML config loader (`utils/seed.py`, `utils/config.py`)

## Phase 1 — Dataset survey & selection (27 Oct – 10 Nov) — **deadline 10 Nov** (was 31 Oct) → G3, G4

Look beyond Jinan/Hangzhou before committing.

- [x] 1.1 Survey candidate datasets and fill the comparison table below
- [x] 1.2 Check each candidate against criteria: real-world origin, simulator compatibility, size
      feasible on a laptop, multiple flow files / time ranges (for a clean split), grid vs.
      irregular topology, usage in prior papers (for literature comparison)
- [x] 1.3 Select **primary dataset** (train/val/test) and **held-out generalization dataset**
- [x] 1.4 Finalize **split plan** (which file / time range → train, val, test) and write it to
      `.claude/reference/database.md`
- [x] 1.5 Download selected datasets to `data/raw/`, record sources/versions **plus license,
      citation and collection period**
- [x] 1.6 **Dataset card + data dictionary** for each selected dataset: origin, license,
      citation, collection period, unit of observation (one vehicle trip), every roadnet/flow
      field with type and units, sizes (vehicles, intersections, lanes, duration) →
      `docs/dataset_card.md`

| Candidate | Type | Network | Simulator | Notes |
|---|---|---|---|---|
| Jinan | real flows | 3×4 grid, 12 int. | CityFlow (SUMO via LibSignal) | 3 flow files → natural train/val/test |
| Hangzhou | real flows | 4×4 grid, 16 int. | CityFlow (SUMO via LibSignal) | 3 flow files (`real`, `real_5734`, `real_5816`); good held-out city |
| New York | real flows | 16×3 / 28×7 | CityFlow | 48 / 196 int. — likely too large for laptop |
| RESCO Cologne (1/3/8) | real-world calibrated | irregular | SUMO | standard benchmark; small variants usable |
| RESCO Ingolstadt (1/7/21) | real-world calibrated | irregular | SUMO | tests non-grid generalization |
| RESCO grid4x4 / arterial4x4 | synthetic | grid / arterial | SUMO | sanity checks only |
| LuST (Luxembourg) / MoST (Monaco) | real-world calibrated | city-scale | SUMO | probably too large; sub-area only |
| PeMS / UTD19 | real loop-detector counts | — | not directly simulatable | optional: realistic demand profiles / EDA |

Simulator is CityFlow (27 Oct): SUMO-only candidates (RESCO, LuST/MoST) would need
conversion — prefer CityFlow-native sets (Jinan, Hangzhou, New York, synthetic grids).

**Selected (10 Nov):** **Jinan** = primary (`real` 6295 veh → train, `real_2000` 4365 → val,
`real_2500` 5494 → test); **Hangzhou** = held-out generalization city (all 3 flow files, final
evaluation only). Split: `data/processed/splits.json`; card: `docs/dataset_card.md`. → worklog/011

## Phase 2 — Literature review (27 Oct – 16 Nov; gap table draft by 10 Nov) → R2, G2

- [x] 2.1 Conventional: Fixed-Time (Webster), SOTL, Max-Pressure (Varaiya 2013) → `docs/literature/01`
- [x] 2.2 Single-agent RL: IntelliLight, DQN-TSC → `docs/literature/02`, `03`
- [x] 2.3 Multi-agent RL: IDQN, PressLight, CoLight, MPLight → `docs/literature/02`, `03`
- [x] 2.4 Uncertainty & safety in RL: Bootstrapped DQN, Q-ensembles, shielded RL → `docs/literature/04`
- [x] 2.5 Agentic / LLM traffic agents (LLMLight) — related work only → `docs/literature/05`
- [ ] 2.6 **Gap table**: method × uncertainty × coordination × safety × explainability → motivates UG-MARL
      - [x] draft for Review 2 → `docs/literature/06` (worklog/010)
      - [x] UG-MARL row and action items A1–A9 brought in line with the built system (13 Nov,
            worklog/021): stability stays ◐ (no queue trigger), A8 moved to 8.2
      - [ ] final: check ✔/◐ entries and *(unverified)* numbers against PDFs; re-run novelty
            search — at the start of Phase 9, before the related-work chapter, by 15 Nov

### ✅ Review 2 — 10 Nov (revised 9 Nov)
Problem definition + H1–H3 + success criteria · dataset choice + audit + split plan ·
literature gap table (draft) · UG-MARL design diagram · evaluation protocol draft.

---

## Phase 3 — Data, EDA & preprocessing (9–11 Nov) → R3, G3–G6

- [x] 3.0 **Data audit** (structural checks only — nothing is fitted): missing fields,
      duplicate vehicle IDs, impossible values (invalid / zero-length routes, start times outside
      the episode, bad speed or length values), outliers (arrival bursts), privacy (IDs
      anonymised, no personal data). Audit summary with counts and plots → `docs/data_audit.md`
      — **done 10 Nov: 0 defects in 6 flow files + 2 roadnets** (`python scripts/audit_data.py`) → worklog/013
- [x] 3.1 **Split first**, per the Phase 1 plan; saved as one hash-pinned `data/processed/splits.json`
      (whole files, so nothing to materialise) → worklog/011
- [x] 3.2 Leakage test: no overlap between splits; held-out dataset never loaded in train/tune
      (`tests/data/test_splits.py`; held-out only reachable via `flow_files(..., "heldout")`)
- [x] 3.3 **EDA notebook — on the train split only** → `notebooks/01_eda_train.ipynb`,
      figures in `docs/figures/eda/`, traces from `scripts/collect_traces.py` (worklog/013).
      Minimum evidence:
      - sample counts (vehicles, trips per hour, intersections, lanes) and summary statistics
      - "target" distribution: vehicle arrival rate over time and, under Fixed-Time, the
        travel-time distribution (RL has no labels; these stand in for the guidelines' target
        histogram)
      - missing-value summary (from the 3.0 audit)
      - feature distributions: lane queue, lane vehicle count, pressure; turning ratios; lane
        load imbalance; peaks
      - network plot
      - **a short interpretation under every figure**, plus a list of insights that shaped later
        choices (e.g. Δt, normalisation, reward)
      - Held-out city: 3.0 audit only (counts and checks), no distribution plots before final
        testing
- [x] 3.4 Feature engineering: lane queue, vehicle count, movement pressure, one-hot phase
      (44 values per intersection, `trafficrl.envs.features`). Normalisation = **lane-capacity
      scaling, nothing fitted** (replaces "normalization stats from train only" — Decision Log
      10 Nov) → worklog/013
- [x] 3.5 **Freeze the evaluation protocol** (must be done before Phase 5 training starts):
      primary + held-out datasets, split definition, final metric set, seeds, episode length,
      decision interval Δt, success criteria (`PRD.md` §1c). Write `docs/evaluation_protocol.md`,
      add a dated Decision Log row, and commit. **After freezing, changes are allowed only with a
      dated Decision Log entry giving the reason.**
      - Note: the held-out city is a **group-wise (location) split**, as the guidelines require
        for repeated locations.
      — **frozen 10 Nov: Δt = 15 s, reward = −|pressure|** → `docs/evaluation_protocol.md`, worklog/014

## Phase 4 — Environment & conventional baselines (10–11 Nov) → R4, G7, G9

- [x] 4.1 Env wrapper (shared interface): multi-intersection obs, neighbor map, phase control, yellow/min-green
- [x] 4.2 Metrics: travel time, waiting time, queue, throughput, max wait, decision latency
- [x] 4.3 Robustness hooks: sensor noise, detector dropout, demand-surge multiplier
- [x] 4.4 Fixed-Time agent
- [x] 4.5 Max-Pressure agent
- [x] 4.6 `scripts/evaluate.py`: any agent × N seeds → metrics CSV
      — **done 10 Nov** (worklog/015): validation (Jinan `real_2000`, seeds 0–4, deterministic):
      **Fixed-Time 434.84 s, Max-Pressure 326.11 s** average travel time; max per-vehicle wait
      1124 s vs 1334 s. Train sanity check: MP 349.67 s / FT 516.79 s (EDA 344.13 / 515.21;
      gap = EDA trace's 14 s first green, reproduced exactly).

### ✅ Review 1 — 3 Nov
Problem & motivation · lit gaps · dataset survey + selection + split · EDA plots ·
Fixed-Time & Max-Pressure validation numbers · UG-MARL design diagram.

---

## Phase 5 — RL baselines (11–13 Nov) → R4, G7, G8

- [x] 5.0 **Gate:** evaluation protocol frozen (3.5) — no RL training before this
- [x] 5.1 DQN core: replay buffer, target network, ε-greedy — `agents/core/` (Double DQN, SMDP
      target, grouped replay, K-head axis), 11 Nov
- [x] 5.2 **Single-agent baseline**: one DQN, parameters shared across intersections, local obs only
      — val (dev, WSL, 5 seeds) **299.8 ± 1.0 s** (MP 326.1, FT 434.8)
- [x] 5.3 **Multi-agent baseline (IDQN)**: independent network + buffer per intersection
      — val (dev, WSL, 5 seeds) **312.7 ± 2.5 s**; not transferable zero-shot to Hangzhou (7.4)
- [x] 5.4 `scripts/train.py`: checkpoint selection on **validation** flows, CSV/TensorBoard
      logging; **every run appends one row to `docs/experiment_log.csv`** (Run ID | Split / seed |
      Model | Hyperparameters | Validation metric | Runtime | Git commit) and saves software
      versions + hardware (Python/PyTorch/CUDA/CityFlow versions, GPU/CPU) next to its results
- [x] 5.5 Light tuning on validation (lr, γ, batch size, ε schedule) — **search space written in
      `configs/` before the first tuning run**; grid or random search; test flows never inspected
      — random search, 8 trials/agent (`configs/search/rl_baselines.yaml`, committed 71b9b48
      before trials); both pick t6 (lr 1e-4, γ 0.8, batch 64, ε-decay 0.3). → worklog/016

## Phase 6 — Proposed UG-MARL (11–13 Nov) → R4, R5, G8

- [x] 6.1 Neighbor-attention encoder (variable neighbor count) — `NeighborhoodEncoder` + 1-head
      GAT layer in `UGMARLNet`, absent neighbours masked (12 Nov, worklog/017)
- [x] 6.2 Q-ensemble (K heads, bootstrap masks) → μ(a), σ(s) — Bernoulli(0.8) masks in replay,
      per-head Double-DQN target, masked loss; DQN path bit-identical
- [x] 6.3 Uncertainty gate: σ > τ → Max-Pressure, else argmax μ; τ = percentile of σ on validation
      — evaluation only; τ = 0 reproduces Max-Pressure (326.11 s); `scripts/calibrate_gate.py`
      → p* = 80, `gate.json` per seed-model (`docs/results/gate_calibration*.md`)
- [x] 6.4 Explanation logging: attention weights, σ, gate firings — `DecisionRecorder`,
      `evaluate.py --log-decisions`; σ validity check `docs/figures/ugmarl/sigma_validity.png`
- [x] 6.5 Ablation flags: `use_attention`, `use_ensemble`, `use_gate` — −attention / −ensemble
      trained on seeds 0–4, −gate = same checkpoints (`docs/results/val_phase6.md`)
- [x] 6.6 Math write-up: MDP, ensemble uncertainty, attention equations, gate stability argument
      — `docs/methodology.md`
- [x] 6.7 Tune K, τ percentile, attention dim on validation — **predefined search space in
      `configs/`**, every run logged in `docs/experiment_log.csv` — grid K × d (9 runs) → K = 3,
      d = 128 locked; p from {80, 90, 95, 99} → 80

- [x] 6.8 **Validation robustness references (added 12 Nov, user decision after U14)** — evaluate
      the locked **DQN, IDQN, Max-Pressure and Fixed-Time** on the validation flow under the same
      four detector perturbations used for the gate calibration (noise 0.1 / 0.2, dropout
      10 / 20 %; surge not used), seeds 0–4, existing checkpoints, **evaluation only**. Purpose:
      the gate's robustness result is so far compared only with −gate; this adds the baselines.
      Rules fixed before the runs:
      - nothing is selected or retuned from these runs — checkpoints, K, d, p\* = 80 and every
        `gate.json` stay locked; validation flow only, test / held-out not opened;
      - command: `python scripts/robustness.py --spec configs/robustness_val.yaml` (spec
        committed before the runs; plan `.claude/plans/val-robustness-references.md`) — one
        `scripts/evaluate.py` job per model × condition; UG-MARL and −gate and the clean flow
        are re-run through the same script (must reproduce `val_phase6.md` and
        `gate_calibration.md`); every run logged in `docs/experiment_log.csv`;
      - output: `docs/results/val_robustness.{csv,md}` — one table, model × condition, travel
        time (mean ± std, 95 % CI) and max wait, with the existing UG-MARL / −gate rows;
      - reading, written in advance: **(a)** DQN / IDQN degrade under noise like −gate
        (599 / 909 s) and Max-Pressure under noise is not better than gated UG-MARL
        (376 / 423 s) → robustness claim holds against the baselines, go to Phase 7 as locked;
        **(b)** DQN is robust without a gate, or Max-Pressure under noise beats gated UG-MARL →
        reported as a finding (not tuned away); the next step (Phase 7 as is, or one time-boxed,
        pre-registered improvement attempt) is the user's decision.

      - **done 12 Nov** → `docs/results/val_robustness.md` (150 val episodes; clean column and
        UG-MARL / −gate rows reproduce `val_phase6.md` and `gate_calibration.md` exactly).
        Outcome **mixed**: DQN / IDQN collapse under noise like −gate (553 / 855 s and
        529 / 767 s at noise 0.1 / 0.2) — reading (a) holds for the learned baselines;
        Max-Pressure under noise (372 / 415 s) is level with gated UG-MARL (376 / 423 s,
        CIs overlap, Max-Pressure nominally 5–9 s lower) — the (b) element. Next step is the
        user's decision. → worklog/019

## Phase 7 — Experiments (13–15 Nov) → R5, R6, G10, G11

All model choices (checkpoints, hyperparameters, τ) are **locked on validation before Phase 7
starts**. Test flows are run **once** per agent / configuration · ≥ 3 seeds · mean ± std **and
95% CI** · long runs overnight. If a bug forces a re-run, log it with the reason in the Decision
Log.

- [x] 7.1 Main comparison: 5 agents on primary test set **(must-have)**, reported as **Model |
      Validation score | Test score | Training time | Model size** (`PRD.md` §5)
- [x] 7.2 Ablation: full vs. −gate / −attention / −ensemble **(must-have)**
- [x] 7.3 Robustness: noise {0, 0.1, 0.2}, dropout {0, 10, 20 %}, surge ×1.5
- [x] 7.4 Generalization: zero-shot on held-out dataset
- [x] 7.5 Efficiency: inference latency, training time, params, GPU memory (not used: CPU)
- [x] 7.6 Safety & explainability: max wait, % fallback, attention heatmaps, gate-firing analysis
      (tables done; the heatmap / gate figures are drawn in 8.2 from the saved logs)

**Phase 7 result (12 Nov, native Ubuntu, commit `94c1bbb`, 240 test + 105 held-out episodes,
each run once)** → `docs/results/test_{main,ablation,robustness,efficiency,safety_explain}.md`,
`docs/results/heldout_generalisation.md`, worklog/020:

| Model | Val (s) | **Test (s)** | Test, noise 0.2 (s) | Test, surge ×1.5 (s) | Hangzhou, 3 flows (s) |
|---|---|---|---|---|---|
| Fixed-Time | 434.8 | 462.2 | 462.2 | 661.4 | 667.5 |
| Max-Pressure | 326.1 | 320.8 | 413.7 | 435.2 | 431.8 |
| DQN (shared) | 299.8 | **292.2 ± 0.6** | 828.2 | 373.0 | 410.4 ± 7.1 |
| IDQN | 312.7 | 307.9 ± 1.4 | 758.9 | 455.3 | not transferable |
| UG-MARL | 302.3 | 295.4 ± 1.1 | 420.5 | 375.0 | 410.0 ± 2.9 |
| −gate | 301.8 | 295.0 ± 1.0 | 870.2 | 376.4 | 420.1 ± 14.6 |
| −attention | 301.9 | 294.2 ± 1.1 | 419.3 | 376.3 | 403.8 ± 1.2 |
| −ensemble | 303.7 | 297.3 ± 1.5 | 864.9 | 395.1 | 423.7 ± 14.9 |

- **H1 supported** (UG-MARL's CI [294.1, 296.8] below Max-Pressure 320.8 and IDQN [306.1, 309.7]).
- **H2 supported by the pre-fixed rule**: max wait at noise 0.2 1493 vs 3306 s and travel-time
  increase +125 vs +575 s are clear (paired CIs exclude 0); the clean max-wait difference
  (896 vs 910 s) is inside seed noise (paired −15 ± 296 s) and must be described as "no
  difference on clean data".
- **H3 not supported** (−attention 294.2 vs full 295.4 s, CIs overlap) — negative result.
- **Minimum outcome met** in every condition (UG-MARL max wait ≤ Max-Pressure's).
- **UG-MARL is worse than DQN on clean data** (+3.2 s, CIs separated) — not part of H1, reported.
- **Smallest worst-case regret: UG-MARL 6.8 s** (Max-Pressure 62 s, Fixed-Time 288 s, IDQN
  345 s, DQN 415 s). It does not beat Max-Pressure under noise (+3.5 / +6.8 s, CIs overlap).

**Phase 7 set-up (fixed 12 Nov, before any test / held-out flow is opened; plan
`.claude/plans/phase7-final-experiments.md`):**

- **Models as locked in Phase 6** — no retraining, retuning or re-calibration.
  `models/manifest.json` (SHA-256 of every `best.pt`, `gate.json`, training config) is
  committed before the runs; `scripts/robustness.py --final` refuses to start if a file differs.
- **Two studies, every episode run once, seeds 0–4, `--workers 1` on an idle machine (native
  Ubuntu):**
  - `configs/final_test.yaml` — Jinan test, 8 models (Fixed-Time, Max-Pressure, DQN, IDQN,
    UG-MARL, −gate, −attention, −ensemble) × 6 conditions (clean, noise 0.1 / 0.2, dropout
    10 / 20 %, surge ×1.5) = 240 episodes → 7.1, 7.2, 7.3, 7.5, 7.6;
  - `configs/final_heldout.yaml` — Hangzhou, 7 models (IDQN is not transferable: one network
    per Jinan intersection) × 3 flows, clean, τ from Jinan validation = 105 episodes → 7.4.
  - Commands: `python scripts/robustness.py --spec configs/final_{test,heldout}.yaml --final
    --workers 1 --results-dir results/final`, then `python scripts/final_tables.py --study all`.
  - Every final episode writes per-step and per-vehicle logs (UG-MARL variants also
    per-decision logs), so Phase 8 needs no second run on the test flow.
- **Readings, written in advance** (computed by `trafficrl.eval.hypotheses`, tested before the
  runs; test flow, 95 % CI = Student t over 5 seeds):
  - **H1 (clean):** supported iff the upper end of UG-MARL's travel-time CI is below
    Max-Pressure's value **and** below the lower end of IDQN's CI.
  - **H3 (clean):** supported iff the upper end of UG-MARL's CI is below the lower end of
    −attention's CI.
  - **H2 (gate vs −gate, same checkpoints; per-seed paired differences with 95 % CI also
    reported):** (i) mean max per-vehicle wait on the clean flow lower with the gate; (ii)
    mean max per-vehicle wait at noise 0.2 lower with the gate; (iii) travel-time increase
    from clean to noise 0.2 smaller with the gate. **Supported** = (i), (ii), (iii);
    **partly supported** = (ii) and (iii) only; **not supported** = anything else.
  - **Minimum acceptable outcome:** UG-MARL's mean max wait ≤ Max-Pressure's on the clean
    flow (the comparison is also listed for every condition).
  - **DQN** is not part of H1; reported as better / level / worse by CI overlap.
  - **Robustness summary (descriptive, no threshold):** regret = a model's mean travel time −
    the lowest mean among the five agents in that condition; reported as each model's largest
    and mean regret over the six conditions. Claim written from 6.8: *UG-MARL has the
    smallest worst-case regret; it does not beat Max-Pressure under noise.* If the test table
    says otherwise, that is what is reported.
  - **Surge and Hangzhou:** no hypothesis; described as measured.
  - Nothing is selected from any Phase 7 table.
- **Optional extension (not started; needs a separate go-ahead and plan):** detector
  smoothing inside UG-MARL, chosen on validation only, with Max-Pressure + the same smoothing
  as a control, reported as an extra row labelled "added after the main evaluation". The
  locked model stays the headline.

## Phase 8 — Analysis (14–15 Nov) → R6, G11

- [x] 8.1 Comparison tables + bar/line charts — `notebooks/02_analysis.ipynb` §1–2, figures
      `docs/figures/results/{main_comparison,ablation,wait_tail,generalisation}`
- [x] 8.2 Robustness degradation plot, ablation table, explainability figures — notebook §3, §8;
      `robustness_degradation`, `regret`, `gate_firing`, `attention_map`,
      `attention_faithfulness` (check: consistent under replacement, influence small →
      `docs/results/attention_faithfulness.md`)
      - **Attention faithfulness check** (gap analysis A8): on validation flows, perturb the
        highest-α vs. the lowest-α neighbour's observation and compare the change in the
        chosen action / Q; reported as an association check, no test-flow runs
- [x] 8.3 Written interpretation (why UG-MARL wins/loses where it does; limitations) →
      `docs/analysis.md` —
      **attention weights, SHAP and gate statistics are described as associations, not causes**
- [x] 8.4 **Error analysis** → notebook §4–7, `docs/results/error_analysis.md` (+ CSVs),
      `docs/results/tau_sensitivity.md`; cases `docs/figures/results/case_{win,loss,gate}`
      - Per-intersection and per-time-window breakdown of travel time / queue / wait for
        UG-MARL vs. the best baseline — where does UG-MARL lose?
      - 2–3 **representative cases**: a window where UG-MARL clearly wins, one where it loses,
        one where the gate fired (show σ(s), action taken, queues)
      - **Unstable decisions:** phase flip-flopping; spread across seeds (per-seed results, not
        just the mean)
      - **Sensitivity:** to seed, to τ (small sweep around the chosen value, on validation flows
        only), and to distribution shift (held-out city from 7.4)

## Phase 9 — Report, poster, demo (14–16 Nov) → R8, G12

Format changed 12 Nov (Decision Log): **poster replaces the 10 slides; report is 12–15 pages**
in the structure of `updated Sections of a Machine Learning Project Report.pdf`. Claude drafts
the report and poster; the student reviews and edits before submission.

- [ ] 9.1 **Report — LaTeX in `report/`, one PDF, 12–15 pages including all sections.** Each
      chapter starts on a new page as a numbered chapter; every figure, table, equation and
      algorithm is numbered and referenced in the text; proper citations (`references.bib`),
      no plagiarism. Rough page budget: preliminaries ~3, Ch 1 ~1.5, Ch 2 ~1.5, Ch 3 ~3.5,
      Ch 4 ~3, Ch 5 ~1, references ~1, appendix ≤ 1.
      - **Preliminary pages:** title page (author name, roll number, e-mail, department as
        **placeholders** — the student fills them in) · abstract · table of contents · list of
        figures and tables
      - **1 Introduction:** background and motivation · problem statement (`PRD.md` §1a–1c) ·
        objectives (H1–H3, success criteria) · scope of the work · **contributions (3–5)**
      - **2 Literature Review:** existing approaches (conventional TSC, RL-TSC, uncertainty /
        safe RL; Phase 2) · research gaps (gap table)
      - **3 Proposed Methodology:** dataset description (dataset card, audit) · data
        preprocessing (split protocol, leakage controls, lane-capacity scaling) · EDA (train
        split only) · proposed model architecture (diagram 9.5; MDP, Q-ensemble uncertainty,
        neighbour attention, gate + stability argument) · algorithm (9.6) · experimental setup
        (baselines and why each is included, seeds, tuning / search spaces, hardware, software)
      - **4 Results and Discussion:** performance evaluation metrics · experimental results
        (7.1 table, mean ± std / 95 % CI) · comparison with existing models (Fixed-Time,
        Max-Pressure, DQN, IDQN; published Jinan / Hangzhou numbers only where the setup is
        comparable) · graphical analysis · discussion of findings (error analysis 8.3, ablation
        7.2, robustness 7.3, generalisation 7.4, explainability 8.4)
      - **5 Conclusion and Future Scope:** key findings · limitations (incl. ethics, 9.8) ·
        future research directions
      - **End matter:** references · appendix (reproducibility notes: commands, seeds, versions,
        hyperparameters; extra figures if needed)

      **Writing rule:** keep clearly separate what was *planned*, what was *implemented*, what
      the *evidence shows*, and what *remains uncertain*.
- [ ] 9.2 **Poster — `poster/`, PPTX on the `updated poster sample.pptx` template** (one 16:9
      slide, 13.33 × 7.5 in; keep its header band, IIT Patna + IMPACT logos, colours and box
      layout), plus a PDF export:
      - **Header:** title · author name, roll number, e-mail · department, institute
        (**placeholders** `[Author Name]`, `[Roll No.]`, `[E-mail]`, `[Department]` — the student
        fills them in)
      - **Left column:** Abstract · Introduction (3 cited bullets) · Problem Statement
      - **Middle column:** Key Contributions (3 bullets) · Methodology (Figure 1: UG-MARL
        architecture, 9.5) · Algorithm 1 (compact version of 9.6)
      - **Right column:** Result and Discussion (Table 1: UG-MARL vs. baselines on test; one
        key plot, e.g. ablation or robustness) · Conclusion
      - **Bottom strip:** numbered references

      **Every claim on the poster is backed by an experiment or a citation**; numbers match the
      report exactly.
- [ ] 9.3 Live demo: Fixed-Time vs. UG-MARL side by side; **recorded backup video**
- [ ] 9.4 README: dataset access (download commands, license), install, **exact train/evaluate
      commands for every table and figure**, and **expected outputs** (numbers ± tolerance, file
      paths). No hidden manual steps.
- [ ] 9.5 **Architecture diagram** (guidelines p.22): data-flow order — raw flows → split →
      normalisation (*fit on train only*) → CityFlow env → attention encoder → Q-ensemble → gate
      (σ > τ ? Max-Pressure : argmax μ) → phase action → metrics; **separate training path**
      (replay, ensemble updates, τ chosen on validation) from **inference path**; caption. Used
      in report §3 (proposed model architecture) and as poster Figure 1.
- [ ] 9.6 **Algorithm pseudocode** (guidelines p.23): "Algorithm 1: UG-MARL training, tuning and
      evaluation" — Input (roadnet, train/val/test flows, search space H), Output (trained
      ensemble Q*, τ*, test metrics), numbered steps. **Checked line by line against the final
      code before submission.**
- [ ] 9.7 **Fresh-clone reproduction test:** clone into a new directory, follow only the README,
      regenerate the main comparison table (7.1) and one figure; results within seed variance of
      the reported ones. Log the outcome in the worklog.
- [ ] 9.8 **Ethics & limitations write-up** (report Ch 5, limitations):
      - dataset licenses / terms of use
      - privacy (anonymised trajectories, no personal data)
      - fairness (can minor approaches be starved? max wait per approach)
      - misuse / deployment risks (sim-to-real gap, not safety-certified)
      - limitations (simulator fidelity, grid networks only, number of seeds)

## Phase 10 — Viva prep (16 Nov) → R7 (30 marks)

- [ ] 10.1 Q&A sheet: MDP, DQN/Bellman update, target net, attention, ensemble uncertainty,
      Max-Pressure & its stability, leakage prevention, ablation results
- [ ] 10.2 Mock viva + demo rehearsal

### ✅ Review 3 — 17 Nov — final submission

---

## Single-agent → multi-agent: effort estimate

The env wrapper is multi-intersection from the start (Phase 4), so the multi-agent step is
incremental:

| Step | Change | Effort |
|---|---|---|
| Shared-param DQN → IDQN | one network/optimizer/buffer per intersection instead of one shared; same training loop | **Low** (~2–4 h) |
| IDQN → neighbor attention (coordination) | encoder takes neighbor obs, needs neighbor map from env, batched variable-size neighbor sets | **Medium** (~1 day) |
| + Q-ensemble + gate | K output heads + bootstrap masks; gate is a few lines calling Max-Pressure | **Low–Medium** (~0.5 day) |
| Training cost | more networks → slower; mitigate with parameter sharing + agent-id embedding | compute, not code |

Main risk is training time/stability, not implementation difficulty.

## Final gate — guidelines completion checklist

Tick all before submission on 17 Nov.

- [ ] Problem and metrics defined → `PRD.md` §1a–1c, §5
- [ ] Literature and gap documented → Phase 2
- [x] Dataset source and license recorded → 1.5, 1.6
- [x] Data audit and EDA completed → 3.0, 3.3
- [x] Leak-free split and pipeline verified → 3.1, 3.2, 3.5
- [x] Simple baseline established → 4.4, 4.5
- [x] Models tuned using validation only → 5.5, 6.7
- [x] Held-out evaluation completed → 7.1–7.4
- [x] Results and failures interpreted → 8.3, 8.4
- [ ] Seeds, versions and commands recorded → 5.4, 9.4
- [ ] Code, report and poster organized → 9.1, 9.2, 9.7
- [ ] Limitations and references included → 9.1, 9.8

## Solo-project scope controls

- **Never cut:** 7.1 main comparison, 7.2 ablation, report, poster, demo.
- **Shrink first if late:** 7.3 (fewer noise levels), 7.4 (one held-out set), seeds 3 → 2 for
  robustness runs only, episode length 3600 s → 1800 s.
- **Optional:** IPPO baseline, extra datasets beyond primary + held-out, SHAP.

## Risks

| Risk | Mitigation |
|---|---|
| Simulator slow on laptop | CityFlow (fast); shorter episodes, overnight runs on native Ubuntu |
| CityFlow build fails (old pybind11, unmaintained) | bump pybind11 submodule; fall back to a Python 3.10 venv |
| UG-MARL doesn't beat strong baselines | frame contribution as reliability/robustness/safety; honest ablations still score |
| RL instability | Max-Pressure as sanity floor, reward normalization, grad clipping |
| Time overrun (solo) | follow scope controls above |

## Decision Log

| Date | Decision |
|---|---|
| 27 Oct | Approach: UG-MARL. Env: venv + requirements.txt. Git, notebooks for EDA, seeded runs, no data leakage. → worklog/001, 002 |
| 27 Oct | Single-agent baseline = shared-parameter DQN, no coordination. → worklog/003 |
| 27 Oct | Simulator: **CityFlow** over SUMO — faster (needed for ensemble + ablations × seeds), native Jinan/Hangzhou data, results comparable with CoLight/PressLight/MPLight. Trade-offs accepted: simpler vehicle dynamics, basic replay viewer (mitigated by a gate-decision animation). → worklog/004 |
| 27 Oct | Platforms: native Ubuntu 24.04 for training & final numbers; WSL2 Ubuntu 24.04 for dev/plots/demo on Windows. Repo cloned into each Linux filesystem, synced via git; separate `.venv` per environment. → worklog/004 |
| 27 Oct | Work log: one numbered file per piece of work in `worklog/`. → worklog/006 |
| 28 Oct | Python **3.12** (Ubuntu 24.04 system Python) over 3.11/3.10; CityFlow pinned at `81ee0f4` with pybind11 submodule replaced by upstream **v2.13.6**. One simulator thread per run (faster than 4 on Jinan); parallelise across runs. → worklog/007 |
| 9 Nov | Packaging: `pyproject.toml` + `pip install -e .`; `requirements.txt` = exact `pip freeze` pins (torch 2.15.0+cu132). Seeding is **strictly deterministic** (torch deterministic algorithms, cuDNN deterministic, `CUBLAS_WORKSPACE_CONFIG`). Configs: YAML with `base:` inheritance and a required `seed`. → worklog/009 |
| 9 Nov | Adopted `ProjectGuidelines.pdf` (IIT Patna IMPACT Lab, 5 Nov) as second governing document; PS16 rubric wins on conflict → **10 slides, ≤ 8 pages**. Title: *Uncertainty-Gated Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control* (short: **UG-MARL**). Added problem definition, hypotheses H1–H3 and success criteria (`PRD.md` §1a–1c); dataset and metric set **not frozen yet** — frozen in 3.5 before training. → worklog/008 |
| 9 Nov | **Schedule re-baselined** from today: only 0.1 and 0.1b (WSL) were done; all phase dates moved, Review 2 content revised to problem definition / dataset / audit / split / gap table / design diagram / protocol draft. → worklog/008 |
| 9 Nov | Literature review scope: depth over paper count; conventional TSC, RL foundations, RL-TSC, uncertainty & safe RL in full; LLM agents + explainability as short notes. Metadata verified online, summaries from abstracts/knowledge; unverified numbers flagged. **Only 2016+ papers reviewed**; pre-2016 originals of implemented methods (Webster, Varaiya, Watkins, Mnih, Sha) cited as background only. Output: `docs/literature/` (one file per topic) + `report/references.bib`. → worklog/010 |
| 10 Nov | **Datasets: Jinan (primary) + Hangzhou (held-out)** from CoLight repo `04440b7`. Criteria (1.2): real-world, CityFlow-native, laptop-sized, ≥ 3 flow files, used in CoLight/PressLight/MPLight. Rejected: New York (too large), RESCO / LuST / MoST (SUMO-only), PeMS / UTD19 (no routes). **Split = whole flow files:** Jinan `real` → train, `real_2000` → val, `real_2500` → test; Hangzhou all 3 files (`real`, `real_5734`, `real_5816`; the third was not in the original plan) → held-out, zero-shot, final evaluation only. Files SHA-256-pinned in `data/processed/splits.json` (one file, not one per dataset). Leakage check is per *file* (path + hash) because CityFlow vehicle IDs are generated per file; cross-file content overlap measured at 2–5 % (chance level). No license in source repo → academic use only, not redistributed, cite CoLight. Limitation accepted: grid topologies only. → worklog/011 |
| 10 Nov | **Data audit (3.0): data used as released.** 0 defects in all 6 flow files and both roadnets (17 flow checks, 12 roadnet checks). Three properties are documented, not repaired: 1–2 % of trips end inside the network (valid in CityFlow), 3–4 % of routes pass an intersection twice, and the Jinan train flow has a five-minute low-demand period. The "route ends at a boundary node" check was reclassified from defect to information after inspection (the plan listed it as a defect). The audit never edits hash-pinned files. → worklog/013 |
| 10 Nov | **Normalisation = lane-capacity scaling; nothing is fitted.** Lane value / ⌊lane length / 7.5 m⌋ (53, 80, 106 vehicles), clipped to [0, 1]. Chosen over z-scores fitted on train rollouts (policy-dependent, drift during training, may not transfer to Hangzhou) and over no normalisation. Checked on train traces: maximum occupancy 0.96 under Fixed-Time and random control, so clipping never acts. Supersedes the wording "normalization stats from train only" in 3.4. → worklog/013 |
| 10 Nov | **Observation = 44 values per intersection:** queue (12) ‖ vehicle count (12) ‖ movement pressure (12) ‖ one-hot green phase (8), lanes in a canonical order (incoming roads E, N, W, S; lane left / through / right). Groups are selectable in `configs/base.yaml` (`features.groups`). Provisional until the 3.5 freeze. → worklog/013 |
| 10 Nov | **Seeds (for the 3.5 freeze): 5 for the main comparison and ablations (7.1, 7.2), 3 for robustness and generalisation (7.3, 7.4).** Reason: with 3 seeds the 95 % CI uses t = 4.30, with 5 seeds t = 2.78. EDA finding to carry into the protocol: Fixed-Time and Max-Pressure are deterministic across seeds, so only learned agents have seed variance. → worklog/013 |
| 10 Nov | **Δt and reward stay provisional (Δt = 10 s, reward = −pressure) until 3.5.** EDA on the train flow: Max-Pressure averages 451 / 375 / 344 / 347 s travel time at Δt = 5 / 10 / 15 / 20 s, so 15 s is 8 % better than 10 s for that controller; queue and \|pressure\| correlate at 0.65–0.82 but pressure is near zero when upstream and downstream lanes are both full. **Open for the freeze:** keep Δt = 10 s (literature), use 15 s, or tune Max-Pressure's interval on validation; −queue remains the documented reward alternative. → worklog/013 |
| 10 Nov | **Evaluation protocol frozen (3.5)** → `docs/evaluation_protocol.md`. **Δt = 15 s** (Max-Pressure 344 s vs 375 s at 10 s on the train flow, −8 %; 20 s no better; chosen on train only, same Δt for all controllers; trade-off: absolute numbers not directly comparable with 10 s papers). **Reward = −\|Pᵢ\|** (same objective as the Max-Pressure fallback; −queue rejected; the "both sides full" blind spot is reported as a limitation). Also frozen: Jinan train/val/test + Hangzhou held-out (hash-pinned), 3600 s episodes, 5 s yellow, 44-value state, metric set and definitions, seeds 0–4 (7.1, 7.2) / 0–2 (7.3, 7.4), robustness perturbation definitions, reporting rules. Rule-based baselines are run on every seed and reported with std = 0 ("deterministic"); H1 then needs the learned agent's whole CI below the baseline value. Supersedes the open point of the row above. → worklog/014 |
| 10 Nov | **Env timing = per-intersection clocks** (Phase 4): each intersection decides every Δt = 15 s of green; a change adds 5 s yellow before the new green; intersections drift out of sync; `step()` returns when ≥ 1 is ready. Only option that implements the frozen protocol §2 literally (yellow on top of Δt, min green = Δt), keeps Fixed-Time at exactly 30 s and matches the EDA setting in which Δt was chosen (Max-Pressure reproduces the EDA's 344.13 s exactly once the trace's 14 s first green is copied). Rejected: synchronous steps with yellow outside Δt (non-changers get 20 s green, FT not 30 s), yellow inside Δt (contradicts protocol). Consequence: RL experience is per intersection (`Transition`). Protocol clarification: average queue sampled at every env step (≈ 5 s). → worklog/015 |
| 10 Nov | **Agent interface: `act(obs, info, explore)`** (was `act(obs, explore)`): Max-Pressure and the UG-MARL gate need observed raw lane counts, Fixed-Time needs green elapsed time — neither is in the 44-vector. Agents return a phase for every intersection (decision latency = batch of all). Robustness noise/dropout perturb the lane readings every controller sees; rewards and metrics use the true state. Test / held-out need `make_env(allow_final=True)` (code argument, never config), set only by `evaluate.py --final`. Phase 4 ran train (sanity) and val only. → worklog/015 |
| 11 Nov | **RL baselines design (Phase 5, plan `phase5-rl-baselines.md`):** Double DQN + Huber loss + grad clipping, target network, uniform replay, ε-greedy (no dueling / PER / n-step, so the UG-MARL comparison isolates attention / ensemble / gate). One configurable Q-learner core shared by DQN, IDQN and later UG-MARL (heads axis K, grouped replay, pluggable encoder). DQN = one MLP 44→128→128→8 shared by all intersections, one buffer; IDQN = 12 independent MLPs (batched) + 12 buffers; both local obs only; network size fixed, not tuned. Training target: SMDP discount γ^(duration_s/Δt); episode end treated as time-limit truncation (always bootstrap); reward multiplied by a fixed constant 0.01 for training only (optimal policy unchanged; metrics use raw −\|P\|). 100 episodes per run, greedy val episode every 5, best checkpoint by val avg travel time. Tuning: random search, 8 trials per agent on seed 0, same space for both (lr, γ, batch, ε-decay fraction); chosen config then trained on seeds 0–4. → worklog/016 |
| 11 Nov | **Phase 5 runs on WSL are *dev* numbers.** Locked configs are retrained and evaluated on native Ubuntu in Phase 7 for the reported results (protocol §7 unchanged); ROADMAP 0.1b must be done before Phase 7. → worklog/016 |
| 11 Nov | **UG-MARL design (Phase 6, plan `phase6-ugmarl.md`, decisions U1–U15):** one network **shared** by all intersections (zero-shot on Hangzhou; shared DQN beat IDQN on val). **Attention:** one CoLight-style GAT layer, one attention head, over self + ≤ 4 neighbours with a slot embedding. **Ensemble:** shared encoder + K independent MLP heads, Bernoulli(0.8) bootstrap masks in replay, per-head Double-DQN target. **σ(s)** = std over heads of Q at the greedy action of the mean Q. **Gate at evaluation only** (training and checkpoint selection are ungated, ε-greedy on mean Q as the baselines), so −gate = the same checkpoint with the gate off. **τ** = p-th percentile of each seed-model's σ on a clean val episode; one p for all seeds, from {80, 90, 95, 99}, chosen by mean val travel time over clean + perturbed validation (noise 0.1 / 0.2, dropout 10 / 20 %; surge not used) — clean val alone would select "never fire" because DQN already beats Max-Pressure. **Tuning:** full grid K ∈ {3, 5, 10} × d ∈ {32, 64, 128}, seed 0; lr / γ / batch / ε-decay inherited from baseline trial t6; best config **locked whatever it scores** (no extra tuning; next step decided afterwards). **Ablations:** −attention and −ensemble (K = 1, no gate) trained on seeds 0–4 with the full model's (K, d). Extra: σ validity check on validation. Math write-up → `docs/methodology.md`; the stability claim is rigorous only for "gate always fires = Max-Pressure", the mixed policy gets a sketch. **Platform:** long runs on native Ubuntu if 0.1b is done (numbers then reportable, baselines' finals re-run there), else WSL dev; never mixed in one table. Rejected alternatives: plan § Justification. → worklog/017 |
| 12 Nov | **Final-submission format changed** (course instructions, 12 Nov): a **poster** on `updated poster sample.pptx` **replaces the 10-slide presentation**; the **report is 12–15 pages (was ≤ 8)**, one PDF, in the chapter structure of `updated Sections of a Machine Learning Project Report.pdf` (preliminaries; 1 Introduction, 2 Literature Review, 3 Proposed Methodology, 4 Results and Discussion, 5 Conclusion and Future Scope; references, appendix). These two documents win over the PS16 rubric and the guidelines for report/poster format; demo, viva, 17 Nov date and marks unchanged. Tooling: LaTeX report in `report/`, PPTX poster in `poster/`. Claude drafts, student reviews. Ethics folded into Ch 5 limitations, reproducibility notes into the appendix. → worklog/019 |
| 12 Nov | **Phase 6 executed (native Ubuntu; numbers are final-val).** Grid (`configs/search/ugmarl.yaml`, committed first): best 301.08 s (K 10, d 128); K 10 / d 64 and **K 3 / d 128** within the 1 s tolerance → smallest K wins, **locked (U14)** though σ over 3 heads is a noisy estimate. Gate calibration (`configs/search/ugmarl_gate.yaml`): **p\* = 80** (80 and 90 tie within 1 s → smaller p). −attention gets its own τ; its calibration also selected 80 (the plan said "same p as full" — identical here). Calibration episodes use full tracking (plan: light) so max wait is reported. Validation (5 seeds): UG-MARL 302.3 ± 1.1 s, −gate 301.8, −attention 301.9, −ensemble 303.7, DQN 299.8, IDQN 312.7 — On val UG-MARL beats IDQN and Max-Pressure with separated CIs (the H1 pattern) but **not** DQN, and attention shows no gain over −attention (H3 pattern not supported); test decides in Phase 7. H2 is mixed on val: on clean val the gate slightly raises max wait (761 vs 691 s, CIs overlap), under detector noise 0.1 / 0.2 it keeps travel time at 376 / 423 s where the ungated policy degrades to 599 / 909 s (max wait 1238 vs 3307 s at 0.2). **Next step after this result is the user's decision (U14).** Also fixed: training rows were always labelled `dev-WSL`. → worklog/017 |
| 12 Nov | **After Phase 6 (U14): no retuning to beat DQN; add validation robustness references first (6.8).** UG-MARL is 2.5 s behind DQN on clean val with touching CIs, and H1 names IDQN and Max-Pressure, not DQN; another tuning round aimed at one baseline would break the locked-on-validation rule. The claim to test is two-sided — about equal to the learned baselines on clean data and better under degraded detectors; better than Max-Pressure on clean data — and only the −gate half is measured. 6.8 runs DQN, IDQN, Max-Pressure and Fixed-Time under the gate-calibration perturbations on validation (evaluation only, nothing selected, readings (a) / (b) written before the runs). Phase 7 starts after 6.8. Evaluation protocol unchanged. → worklog/019 |
| 12 Nov | **6.8 result (validation, evaluation only, nothing selected): the gate makes UG-MARL track the better controller in each condition, but it does not beat Max-Pressure under noise.** Travel time, 5 seeds: DQN 300 → 553 → 855 s, IDQN 313 → 529 → 767 s, −gate 302 → 599 → 909 s, Max-Pressure 326 → 372 → 415 s, UG-MARL 302 → 376 → 423 s (clean → noise 0.1 → 0.2; fallback 21 → 81 → 96 %). Under dropout 10 / 20 % UG-MARL (306 / 311 s) stays with DQN (305 / 309 s) and ahead of Max-Pressure (323 / 322 s). Max wait at noise 0.2: DQN 3407 s, Max-Pressure 1277 s, UG-MARL 1238 s. Mean over the five conditions: UG-MARL 344 s, Max-Pressure 351 s, DQN 464 s, Fixed-Time 435 s (optimistic for UG-MARL: p\* was chosen on this score). Pre-written reading: (a) holds against DQN / IDQN; against Max-Pressure under noise the result is a tie with overlapping CIs (the (b) element), which is the ceiling of a gate that falls back to Max-Pressure. **Not tuned away; whether to go to Phase 7 as locked or make one pre-registered improvement attempt is the user's decision.** `scripts/robustness.py` is reused for 7.3. → worklog/019 |
| 12 Nov | **After 6.8: Phase 7 runs with the models as locked (user decision).** No improvement attempt before the test flow is opened: the tie with Max-Pressure under noise is the ceiling of a gate that falls back to Max-Pressure (81–96 % fallback there), H1–H3 do not require beating it, and another round of changes would reopen selection 5 days before submission. **"Run the test, then improve if UG-MARL loses" is rejected:** a change chosen because of a test result uses the test flow for selection. A follow-up is possible only as the pre-specified optional extension (detector smoothing; validation-only selection; extra labelled row; locked model stays the headline). → worklog/020 |
| 12 Nov | **Protocol amendment (before any final run): seeds 0–4 for 7.3 and 7.4** (was 0–2). Five trained seeds exist and an episode takes ~6 s; the 95 % CI uses t = 2.78 instead of 4.30; same seeds as the validation table 6.8; the clean column of 7.3 is then the same runs as 7.1 / 7.2, so every test episode runs exactly once. `docs/evaluation_protocol.md` §6, §9. → worklog/020 |
| 12 Nov | **Phase 7 readings fixed before the runs** (ROADMAP Phase 7 set-up; `PRD.md` §1c; protocol §9): H1 / H3 by CI rule on the clean test flow; **H2 in three levels** — (i) max wait clean, (ii) max wait at noise 0.2, (iii) travel-time increase clean → noise 0.2, gate vs −gate; supported = all three, partly supported = (ii) + (iii) — because PRD §1c did not say in which condition max wait is compared and validation differs between clean (761 vs 691 s) and noise 0.2 (1238 vs 3307 s). Added descriptive statistic: per-condition **regret** against the best of the five agents (largest and mean over conditions). Verdicts are computed by code written before the test flow is opened. → worklog/020 |
| 12 Nov | **Phase 7 execution design:** one runner (`scripts/robustness.py`) for 7.1–7.4, two specs (`final_test`, `final_heldout`), each episode run once with `--workers 1` (valid decision latency), all Phase 8 inputs logged in the same pass (per-step, per-vehicle, per-decision), lock manifest `models/manifest.json` verified by `--final`. IDQN is reported as "not transferable" on Hangzhou. No combined perturbations and no perturbed Hangzhou (not in the protocol). The planned `scripts/run_ablation.py` is dropped (ablations are rows of the same grid). SHAP (optional) not done. → worklog/020 |
| 12 Nov | **Phase 7 executed (native Ubuntu, commit `94c1bbb`; models verified against `models/manifest.json`; every test / held-out episode run once, no re-run).** Jinan test, clean, 5 seeds: DQN 292.2 ± 0.6, UG-MARL 295.4 ± 1.1, IDQN 307.9 ± 1.4, Max-Pressure 320.8, Fixed-Time 462.2 s. **H1 supported; H3 not supported** (−attention 294.2 ± 1.1 s, 61 % fewer parameters); **H2 supported by the pre-fixed three-part rule**, with the caveat that part (i) (clean max wait 896 vs 910 s) is inside seed noise while (ii) max wait at noise 0.2 (1493 vs 3306 s) and (iii) travel-time increase (+125 vs +575 s) are clear; minimum outcome met in all six conditions. **UG-MARL is 3.2 s worse than DQN on clean data (CIs separated)** and level with Max-Pressure under noise (370 / 421 s vs 367 / 414 s); DQN / IDQN / −gate / −ensemble collapse under noise (759–870 s at 0.2). **Worst-case regret over six conditions: UG-MARL 6.8 s, Max-Pressure 62 s, DQN 415 s** — the pre-written claim holds. Surge ×1.5 (untuned): UG-MARL 375 s ≈ DQN 373 s, Max-Pressure 435 s; the gate fires on 55 % of decisions and max wait rises to 2162 s (DQN 1161 s, Max-Pressure 3260 s). Hangzhou zero-shot (3 flows): UG-MARL 410.0 ± 2.9, DQN 410.4 ± 7.1, −attention 403.8 ± 1.2, Max-Pressure 431.8, Fixed-Time 667.5 s; fallback 34 %; UG-MARL's max wait there (2262 s) is below Max-Pressure's (2734 s) but above DQN's (1660 s). Latency 0.81 ms per joint decision (DQN 0.22, Max-Pressure 0.14). Nothing selected; the optional smoothing extension is not started. → worklog/020 |
| 13 Nov | **Phase 8 set-up (plan `.claude/plans/phase8-analysis.md`, user decisions A1–A4, A12).** (1) The error analysis compares UG-MARL with **both DQN and Max-Pressure, on both the clean test flow and noise 0.2**, with −gate as a paired row — the best baseline changes with the condition. (2) **τ sensitivity = new validation runs at p = 50, 60, 70** (5 seeds × 5 conditions, evaluation only), joined with the existing p = 80–99, ungated and Max-Pressure rows; `gate.json`, p\* = 80 and `models/manifest.json` are not touched. (3) **The Jinan test and Hangzhou flow files may be read, read-only, to join each vehicle's route onto the saved per-vehicle logs** (route-group breakdown); no simulation, nothing selected. (4) Full 8.1–8.4 with a cut order. (5) **The reader analysis lives in `notebooks/02_analysis.ipynb`** (one `nbconvert` command); the rule "notebooks only for EDA" is relaxed for it — all logic stays in tested functions under `src/trafficrl/eval/`, the notebook only calls and displays; the scripts that launch validation runs (τ sweep, faithfulness) stay scripts. No test / held-out episode is run in Phase 8; everything on those flows is read from `results/final/`. **Nothing is changed because of a Phase 8 finding** (locked model, test tables and verdicts stay). → worklog/022 |
| 13 Nov | **Phase 8 readings fixed before the new validation runs.** *τ sweep* (score = mean travel time over seeds × the five validation conditions): every p in 50–90 within 5 s of p\* = 80 → "insensitive"; some p < 80 better than p\* by more than the 1 s tie tolerance → reported as "the calibration grid's lower end was binding", p\* stays 80 and the test flow is not re-run; lower p worse → reported as the clean-data cost of more fallback. *Attention faithfulness* (validation, offline: one neighbour's observation replaced by the same slot's observation from another decision of that intersection): "consistent with the weights" iff the per-seed paired difference in mean \|Δμ\| (highest-α − lowest-α neighbour) is > 0 with a 95 % CI excluding 0; otherwise "the weights do not rank the neighbours' influence on Q" and the attention maps carry that caveat. Representative cases are picked by rule (largest / smallest waiting-vehicle-seconds difference to DQN per intersection × 5-minute window on the median seed; most gate-changed decisions). → worklog/022 |
| 13 Nov | **Phase 8 executed (no test / held-out episode run; 75 + 5 new validation episodes, evaluation only; `lock_models.py --verify` passes).** All 305 final episodes are reproduced from the saved logs (largest difference 0.00005 s) once travel time is counted from the scheduled start in the flow file — **the per-vehicle logs alone miss vehicles blocked outside the network**, and the waiting-time metrics (incl. max wait) count waiting inside the network only. **Findings:** (1) the 3.2 s deficit to DQN on clean data is spread over all 12 intersections (clear at 4), all windows and all route lengths, is not caused by the gate (gate on vs off level) and is shared by all four UG-MARL variants → attributed to the network family / inherited hyperparameters, not isolated further. (2) Against Max-Pressure on clean data UG-MARL has 17–40 % less waiting at every intersection. (3) At noise 0.2 the ungated learned policies leave 584–795 of 5494 vehicles outside the network (UG-MARL and Max-Pressure: 0), so their reported max wait understates the worst case; UG-MARL is level with Max-Pressure at 11 of 12 intersections. (4) **The gate costs waiting time under the demand surge** (fallback 38 → 68 % over the hour, max wait 2162 vs 1430 s without the gate) and on the most congested Hangzhou flow (max wait ≈ Max-Pressure's). (5) **τ sweep:** p = 50 / 60 / 70 score 343.2 / 343.8 / 342.3 s vs 343.7 s at p\* = 80 → insensitive within 5 s, and by the pre-fixed rule the grid's lower end was binding (p = 70 is 1.5 s better); **p\* stays 80, test not re-run.** (6) **Attention faithfulness:** consistent with the weights under replacement (\|Δμ\| 0.0061 vs 0.0022, paired CI [0.0011, 0.0066]) but the own observation gives 0.0415; knock-out variant not consistent. (7) Flip-flop rate 21.5 % (DQN 20.2 %, Max-Pressure 20.3 %). **Deviations from the plan:** no on-disk cache (not needed, the notebook runs in 2 min); surge has no route join (its flow is generated), so its breakdown uses time inside the network; of the 75 sweep rows 10 carry commit `57dd6ea`, 19 `57dd6ea-dirty` and 46 the next commit `4ab9ac9`, because unrelated new files were created and committed while the sweep ran — the code under run was `57dd6ea` throughout. Nothing was selected or changed. → `docs/analysis.md`, worklog/022 |
