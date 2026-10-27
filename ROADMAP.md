# ROADMAP — UG-MARL Traffic Signal Control (PS16)

Solo project. Timeline: **27 Oct → 17 Nov 2026**. Reviews: **3 Nov, 10 Nov, 17 Nov**.
Tick `[x]` as tasks complete; record decisions in the Decision Log at the bottom.

Rubric rows referenced as **R1–R8** (see `PRD.md` §7).

---

## Phase 0 — Setup & decisions (27–29 Oct)

- [ ] 0.1 Trial-install SUMO (+ `sumo-rl`) and CityFlow (WSL/Docker); **choose simulator** → update `CLAUDE.md`
- [ ] 0.2 Create `.venv`, `requirements.txt`, repo skeleton (`src/trafficrl`, `configs`, `scripts`, `tests`, `notebooks`)
- [ ] 0.3 Seeding utility + YAML config loader (`utils/seed.py`, `utils/config.py`)

## Phase 1 — Dataset survey & selection (27–31 Oct) — **deadline 31 Oct**

Look beyond Jinan/Hangzhou before committing.

- [ ] 1.1 Survey candidate datasets and fill the comparison table below
- [ ] 1.2 Check each candidate against criteria: real-world origin, simulator compatibility, size
      feasible on a laptop, multiple flow files / time ranges (for a clean split), grid vs.
      irregular topology, usage in prior papers (for literature comparison)
- [ ] 1.3 Select **primary dataset** (train/val/test) and **held-out generalization dataset**
- [ ] 1.4 Finalize **split plan** (which file / time range → train, val, test) and write it to
      `.claude/reference/database.md`
- [ ] 1.5 Download selected datasets to `data/raw/`, record sources/versions

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

Default proposal (pending survey): **Jinan** = primary (`real` → train, `real_2000` → val,
`real_2500` → test); **Hangzhou** = held-out generalization city.

## Phase 2 — Literature review (27 Oct – 2 Nov) → R2

- [ ] 2.1 Conventional: Fixed-Time (Webster), SOTL, Max-Pressure (Varaiya 2013)
- [ ] 2.2 Single-agent RL: IntelliLight, DQN-TSC
- [ ] 2.3 Multi-agent RL: IDQN, PressLight, CoLight, MPLight
- [ ] 2.4 Uncertainty & safety in RL: Bootstrapped DQN, Q-ensembles, shielded RL
- [ ] 2.5 Agentic / LLM traffic agents (LLMLight) — related work only
- [ ] 2.6 **Gap table**: method × uncertainty × coordination × safety × explainability → motivates UG-MARL

## Phase 3 — Data, EDA & preprocessing (31 Oct – 2 Nov) → R3

- [ ] 3.1 **Split first**, per the Phase 1 plan; save `data/processed/<dataset>/splits.json`
- [ ] 3.2 Leakage test: no overlap between splits; held-out dataset never loaded in train/tune
- [ ] 3.3 EDA notebook: arrival rate over time, turning ratios, lane load imbalance, peaks, network plot
- [ ] 3.4 Feature engineering: lane queue, pressure, one-hot phase; normalization stats from **train only**

## Phase 4 — Environment & conventional baselines (30 Oct – 2 Nov) → R4

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

## Phase 5 — RL baselines (3–6 Nov) → R4

- [ ] 5.1 DQN core: replay buffer, target network, ε-greedy
- [ ] 5.2 **Single-agent baseline**: one DQN, parameters shared across intersections, local obs only
- [ ] 5.3 **Multi-agent baseline (IDQN)**: independent network + buffer per intersection
- [ ] 5.4 `scripts/train.py`: checkpoint selection on **validation** flows, CSV/TensorBoard logging
- [ ] 5.5 Light tuning on validation (lr, γ, batch size, ε schedule)

## Phase 6 — Proposed UG-MARL (5–9 Nov) → R4, R5

- [ ] 6.1 Neighbor-attention encoder (variable neighbor count)
- [ ] 6.2 Q-ensemble (K heads, bootstrap masks) → μ(a), σ(s)
- [ ] 6.3 Uncertainty gate: σ > τ → Max-Pressure, else argmax μ; τ = percentile of σ on validation
- [ ] 6.4 Explanation logging: attention weights, σ, gate firings
- [ ] 6.5 Ablation flags: `use_attention`, `use_ensemble`, `use_gate`
- [ ] 6.6 Math write-up: MDP, ensemble uncertainty, attention equations, gate stability argument
- [ ] 6.7 Tune K, τ percentile, attention dim on validation

### ✅ Review 2 — 10 Nov
All 5 agents trained · early validation comparison · math formulation · experiment plan.

---

## Phase 7 — Experiments (10–13 Nov) → R5, R6

Test flows only · 3 seeds · mean ± std · long runs overnight.

- [ ] 7.1 Main comparison: 5 agents on primary test set **(must-have)**
- [ ] 7.2 Ablation: full vs. −gate / −attention / −ensemble **(must-have)**
- [ ] 7.3 Robustness: noise {0, 0.1, 0.2}, dropout {0, 10, 20 %}, surge ×1.5
- [ ] 7.4 Generalization: zero-shot on held-out dataset
- [ ] 7.5 Efficiency: inference latency, training time, params, GPU memory
- [ ] 7.6 Safety & explainability: max wait, % fallback, attention heatmaps, gate-firing analysis

## Phase 8 — Analysis (12–14 Nov) → R6

- [ ] 8.1 Comparison tables + bar/line charts
- [ ] 8.2 Robustness degradation plot, ablation table, explainability figures
- [ ] 8.3 Written interpretation (why UG-MARL wins/loses where it does; limitations)

## Phase 9 — Report, slides, demo (13–16 Nov) → R8

- [ ] 9.1 Report ≤ 8 pages: Intro & Contribution → Lit Review → Dataset → Methodology & Math →
      Implementation → Results & Ablation → Discussion & Limitations → Conclusion
- [ ] 9.2 10 slides (one per rubric section + architecture diagram + key result)
- [ ] 9.3 Live demo: Fixed-Time vs. UG-MARL side by side; **recorded backup video**
- [ ] 9.4 README with reproduction commands

## Phase 10 — Viva prep (15–16 Nov) → R7 (30 marks)

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

## Solo-project scope controls

- **Never cut:** 7.1 main comparison, 7.2 ablation, report, slides, demo.
- **Shrink first if late:** 7.3 (fewer noise levels), 7.4 (one held-out set), seeds 3 → 2 for
  robustness runs only, episode length 3600 s → 1800 s.
- **Optional:** IPPO baseline, extra datasets beyond primary + held-out, SHAP.

## Risks

| Risk | Mitigation |
|---|---|
| Simulator slow on laptop | shorter episodes, libsumo, overnight runs |
| UG-MARL doesn't beat strong baselines | frame contribution as reliability/robustness/safety; honest ablations still score |
| RL instability | Max-Pressure as sanity floor, reward normalization, grad clipping |
| Time overrun (solo) | follow scope controls above |

## Decision Log

| Date | Decision |
|---|---|
| 27 Oct | Approach: UG-MARL. Env: venv + requirements.txt. Git, notebooks for EDA, seeded runs, no data leakage. |
| 27 Oct | Single-agent baseline = shared-parameter DQN, no coordination. |
| — | Simulator: **pending** (Phase 0.1) |
| — | Primary / held-out datasets & split: **pending, due 31 Oct** (Phase 1) |
