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
| 14–16 Nov | Phase 9 report/slides/demo; Phase 2 literature written alongside throughout |
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
- [ ] 0.1b Build CityFlow (pybind11 bump for Py 3.12) on WSL + native Ubuntu; time one Jinan
      episode with Fixed-Time & Max-Pressure; write install steps to README
      - [x] WSL: built (pybind11 v2.13.6), Jinan episode FT 2.7 s / MP 1.8 s, README written → worklog/007
      - [ ] native Ubuntu: follow README, run `scripts/bench_cityflow.py`, record numbers
- [x] 0.2 Create `.venv`, `requirements.txt`, repo skeleton (`src/trafficrl`, `configs`, `scripts`, `tests`, `notebooks`)
- [ ] 0.3 Seeding utility + YAML config loader (`utils/seed.py`, `utils/config.py`)

## Phase 1 — Dataset survey & selection (27 Oct – 10 Nov) — **deadline 10 Nov** (was 31 Oct) → G3, G4

Look beyond Jinan/Hangzhou before committing.

- [ ] 1.1 Survey candidate datasets and fill the comparison table below
- [ ] 1.2 Check each candidate against criteria: real-world origin, simulator compatibility, size
      feasible on a laptop, multiple flow files / time ranges (for a clean split), grid vs.
      irregular topology, usage in prior papers (for literature comparison)
- [ ] 1.3 Select **primary dataset** (train/val/test) and **held-out generalization dataset**
- [ ] 1.4 Finalize **split plan** (which file / time range → train, val, test) and write it to
      `.claude/reference/database.md`
- [ ] 1.5 Download selected datasets to `data/raw/`, record sources/versions **plus license,
      citation and collection period**
- [ ] 1.6 **Dataset card + data dictionary** for each selected dataset: origin, license,
      citation, collection period, unit of observation (one vehicle trip), every roadnet/flow
      field with type and units, sizes (vehicles, intersections, lanes, duration) →
      `docs/dataset_card.md`

| Candidate | Type | Network | Simulator | Notes |
|---|---|---|---|---|
| Jinan | real flows | 3×4 grid, 12 int. | CityFlow (SUMO via LibSignal) | 3 flow files → natural train/val/test |
| Hangzhou | real flows | 4×4 grid, 16 int. | CityFlow (SUMO via LibSignal) | 2 flow files; good held-out city |
| New York | real flows | 16×3 / 28×7 | CityFlow | 48 / 196 int. — likely too large for laptop |
| RESCO Cologne (1/3/8) | real-world calibrated | irregular | SUMO | standard benchmark; small variants usable |
| RESCO Ingolstadt (1/7/21) | real-world calibrated | irregular | SUMO | tests non-grid generalization |
| RESCO grid4x4 / arterial4x4 | synthetic | grid / arterial | SUMO | sanity checks only |
| LuST (Luxembourg) / MoST (Monaco) | real-world calibrated | city-scale | SUMO | probably too large; sub-area only |
| PeMS / UTD19 | real loop-detector counts | — | not directly simulatable | optional: realistic demand profiles / EDA |

Simulator is CityFlow (27 Oct): SUMO-only candidates (RESCO, LuST/MoST) would need
conversion — prefer CityFlow-native sets (Jinan, Hangzhou, New York, synthetic grids).

Default proposal (pending survey): **Jinan** = primary (`real` → train, `real_2000` → val,
`real_2500` → test); **Hangzhou** = held-out generalization city.

## Phase 2 — Literature review (27 Oct – 16 Nov; gap table draft by 10 Nov) → R2, G2

- [ ] 2.1 Conventional: Fixed-Time (Webster), SOTL, Max-Pressure (Varaiya 2013)
- [ ] 2.2 Single-agent RL: IntelliLight, DQN-TSC
- [ ] 2.3 Multi-agent RL: IDQN, PressLight, CoLight, MPLight
- [ ] 2.4 Uncertainty & safety in RL: Bootstrapped DQN, Q-ensembles, shielded RL
- [ ] 2.5 Agentic / LLM traffic agents (LLMLight) — related work only
- [ ] 2.6 **Gap table**: method × uncertainty × coordination × safety × explainability → motivates UG-MARL

### ✅ Review 2 — 10 Nov (revised 9 Nov)
Problem definition + H1–H3 + success criteria · dataset choice + audit + split plan ·
literature gap table (draft) · UG-MARL design diagram · evaluation protocol draft.

---

## Phase 3 — Data, EDA & preprocessing (9–11 Nov) → R3, G3–G6

- [ ] 3.0 **Data audit** (structural checks only — nothing is fitted): missing fields,
      duplicate vehicle IDs, impossible values (invalid / zero-length routes, start times outside
      the episode, bad speed or length values), outliers (arrival bursts), privacy (IDs
      anonymised, no personal data). Audit summary with counts and plots → `docs/data_audit.md`
- [ ] 3.1 **Split first**, per the Phase 1 plan; save `data/processed/<dataset>/splits.json`
- [ ] 3.2 Leakage test: no overlap between splits; held-out dataset never loaded in train/tune
- [ ] 3.3 **EDA notebook — on the train split only.** Minimum evidence:
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
- [ ] 3.4 Feature engineering: lane queue, pressure, one-hot phase; normalization stats from **train only**
- [ ] 3.5 **Freeze the evaluation protocol** (must be done before Phase 5 training starts):
      primary + held-out datasets, split definition, final metric set, seeds, episode length,
      decision interval Δt, success criteria (`PRD.md` §1c). Write `docs/evaluation_protocol.md`,
      add a dated Decision Log row, and commit. **After freezing, changes are allowed only with a
      dated Decision Log entry giving the reason.**
      - Note: the held-out city is a **group-wise (location) split**, as the guidelines require
        for repeated locations.

## Phase 4 — Environment & conventional baselines (10–11 Nov) → R4, G7, G9

- [ ] 4.1 Env wrapper (shared interface): multi-intersection obs, neighbor map, phase control, yellow/min-green
- [ ] 4.2 Metrics: travel time, waiting time, queue, throughput, max wait, decision latency
- [ ] 4.3 Robustness hooks: sensor noise, detector dropout, demand-surge multiplier
- [ ] 4.4 Fixed-Time agent
- [ ] 4.5 Max-Pressure agent
- [ ] 4.6 `scripts/evaluate.py`: any agent × N seeds → metrics CSV

### ✅ Review 1 — 3 Nov
Problem & motivation · lit gaps · dataset survey + selection + split · EDA plots ·
Fixed-Time & Max-Pressure validation numbers · UG-MARL design diagram.

---

## Phase 5 — RL baselines (11–13 Nov) → R4, G7, G8

- [ ] 5.0 **Gate:** evaluation protocol frozen (3.5) — no RL training before this
- [ ] 5.1 DQN core: replay buffer, target network, ε-greedy
- [ ] 5.2 **Single-agent baseline**: one DQN, parameters shared across intersections, local obs only
- [ ] 5.3 **Multi-agent baseline (IDQN)**: independent network + buffer per intersection
- [ ] 5.4 `scripts/train.py`: checkpoint selection on **validation** flows, CSV/TensorBoard
      logging; **every run appends one row to `docs/experiment_log.csv`** (Run ID | Split / seed |
      Model | Hyperparameters | Validation metric | Runtime | Git commit) and saves software
      versions + hardware (Python/PyTorch/CUDA/CityFlow versions, GPU/CPU) next to its results
- [ ] 5.5 Light tuning on validation (lr, γ, batch size, ε schedule) — **search space written in
      `configs/` before the first tuning run**; grid or random search; test flows never inspected

## Phase 6 — Proposed UG-MARL (11–13 Nov) → R4, R5, G8

- [ ] 6.1 Neighbor-attention encoder (variable neighbor count)
- [ ] 6.2 Q-ensemble (K heads, bootstrap masks) → μ(a), σ(s)
- [ ] 6.3 Uncertainty gate: σ > τ → Max-Pressure, else argmax μ; τ = percentile of σ on validation
- [ ] 6.4 Explanation logging: attention weights, σ, gate firings
- [ ] 6.5 Ablation flags: `use_attention`, `use_ensemble`, `use_gate`
- [ ] 6.6 Math write-up: MDP, ensemble uncertainty, attention equations, gate stability argument
- [ ] 6.7 Tune K, τ percentile, attention dim on validation — **predefined search space in
      `configs/`**, every run logged in `docs/experiment_log.csv`

## Phase 7 — Experiments (13–15 Nov) → R5, R6, G10, G11

All model choices (checkpoints, hyperparameters, τ) are **locked on validation before Phase 7
starts**. Test flows are run **once** per agent / configuration · ≥ 3 seeds · mean ± std **and
95% CI** · long runs overnight. If a bug forces a re-run, log it with the reason in the Decision
Log.

- [ ] 7.1 Main comparison: 5 agents on primary test set **(must-have)**, reported as **Model |
      Validation score | Test score | Training time | Model size** (`PRD.md` §5)
- [ ] 7.2 Ablation: full vs. −gate / −attention / −ensemble **(must-have)**
- [ ] 7.3 Robustness: noise {0, 0.1, 0.2}, dropout {0, 10, 20 %}, surge ×1.5
- [ ] 7.4 Generalization: zero-shot on held-out dataset
- [ ] 7.5 Efficiency: inference latency, training time, params, GPU memory
- [ ] 7.6 Safety & explainability: max wait, % fallback, attention heatmaps, gate-firing analysis

## Phase 8 — Analysis (14–15 Nov) → R6, G11

- [ ] 8.1 Comparison tables + bar/line charts
- [ ] 8.2 Robustness degradation plot, ablation table, explainability figures
- [ ] 8.3 Written interpretation (why UG-MARL wins/loses where it does; limitations) —
      **attention weights, SHAP and gate statistics are described as associations, not causes**
- [ ] 8.4 **Error analysis**
      - Per-intersection and per-time-window breakdown of travel time / queue / wait for
        UG-MARL vs. the best baseline — where does UG-MARL lose?
      - 2–3 **representative cases**: a window where UG-MARL clearly wins, one where it loses,
        one where the gate fired (show σ(s), action taken, queues)
      - **Unstable decisions:** phase flip-flopping; spread across seeds (per-seed results, not
        just the mean)
      - **Sensitivity:** to seed, to τ (small sweep around the chosen value, on validation flows
        only), and to distribution shift (held-out city from 7.4)

## Phase 9 — Report, slides, demo (14–16 Nov) → R8, G12

- [ ] 9.1 Report ≤ 8 pages, guidelines' 10 sections:
      1. Title, abstract, contributions
      2. Introduction & problem statement (`PRD.md` §1a–1c)
      3. Related work & gap (gap table)
      4. Dataset & quality audit
      5. Split protocol & methodology (MDP, ensemble uncertainty, attention, gate + stability
         argument)
      6. Experimental setup & baselines (why each baseline is included)
      7. Results (tables + plots, mean ± std / 95% CI)
      8. Error analysis & ablations
      9. Limitations, ethics & conclusion
      10. References & reproducibility notes

      **Writing rule:** keep clearly separate what was *planned*, what was *implemented*, what
      the *evidence shows*, and what *remains uncertain*.
- [ ] 9.2 10 slides (guidelines order, compressed):
      1. Title, motivation, problem & objectives
      2. Literature review & gap
      3. Dataset, audit, splits & leakage controls
      4. Baselines & experiment protocol
      5. UG-MARL architecture diagram
      6. Math: MDP, uncertainty, gate & stability
      7. Main results & comparison
      8. Error analysis & explainability
      9. Ablation, robustness & limitations
      10. Conclusion, future work, references / Q&A

      **Every claim on a slide is backed by an experiment or a citation.**
- [ ] 9.3 Live demo: Fixed-Time vs. UG-MARL side by side; **recorded backup video**
- [ ] 9.4 README: dataset access (download commands, license), install, **exact train/evaluate
      commands for every table and figure**, and **expected outputs** (numbers ± tolerance, file
      paths). No hidden manual steps.
- [ ] 9.5 **Architecture diagram** (guidelines p.22): data-flow order — raw flows → split →
      normalisation (*fit on train only*) → CityFlow env → attention encoder → Q-ensemble → gate
      (σ > τ ? Max-Pressure : argmax μ) → phase action → metrics; **separate training path**
      (replay, ensemble updates, τ chosen on validation) from **inference path**; caption. Used
      in report §5 and slide 5.
- [ ] 9.6 **Algorithm pseudocode** (guidelines p.23): "Algorithm 1: UG-MARL training, tuning and
      evaluation" — Input (roadnet, train/val/test flows, search space H), Output (trained
      ensemble Q*, τ*, test metrics), numbered steps. **Checked line by line against the final
      code before submission.**
- [ ] 9.7 **Fresh-clone reproduction test:** clone into a new directory, follow only the README,
      regenerate the main comparison table (7.1) and one figure; results within seed variance of
      the reported ones. Log the outcome in the worklog.
- [ ] 9.8 **Ethics & limitations write-up** (report §9):
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
- [ ] Dataset source and license recorded → 1.5, 1.6
- [ ] Data audit and EDA completed → 3.0, 3.3
- [ ] Leak-free split and pipeline verified → 3.1, 3.2, 3.5
- [ ] Simple baseline established → 4.4, 4.5
- [ ] Models tuned using validation only → 5.5, 6.7
- [ ] Held-out evaluation completed → 7.1–7.4
- [ ] Results and failures interpreted → 8.3, 8.4
- [ ] Seeds, versions and commands recorded → 5.4, 9.4
- [ ] Code, report and slides organized → 9.1, 9.2, 9.7
- [ ] Limitations and references included → 9.1, 9.8

## Solo-project scope controls

- **Never cut:** 7.1 main comparison, 7.2 ablation, report, slides, demo.
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
| 9 Nov | Adopted `ProjectGuidelines.pdf` (IIT Patna IMPACT Lab, 5 Nov) as second governing document; PS16 rubric wins on conflict → **10 slides, ≤ 8 pages**. Title: *Uncertainty-Gated Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control* (short: **UG-MARL**). Added problem definition, hypotheses H1–H3 and success criteria (`PRD.md` §1a–1c); dataset and metric set **not frozen yet** — frozen in 3.5 before training. → worklog/008 |
| 9 Nov | **Schedule re-baselined** from today: only 0.1 and 0.1b (WSL) were done; all phase dates moved, Review 2 content revised to problem definition / dataset / audit / split / gap table / design diagram / protocol draft. → worklog/008 |
| — | Primary / held-out datasets & split: **pending, due 10 Nov** (Phase 1) |
