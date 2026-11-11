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
      - [ ] final: check ✔/◐ entries and *(unverified)* numbers against PDFs; re-run novelty search before 16 Nov

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
- [x] Dataset source and license recorded → 1.5, 1.6
- [x] Data audit and EDA completed → 3.0, 3.3
- [ ] Leak-free split and pipeline verified → 3.1, 3.2, 3.5
- [x] Simple baseline established → 4.4, 4.5
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
