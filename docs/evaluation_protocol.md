# Evaluation protocol — UG-MARL (frozen)

**Status: FROZEN on 10 Nov 2026** (ROADMAP 3.5, Decision Log 10 Nov, worklog/014).
Written before any RL training (Phase 5). After this date a change is allowed only with a
dated Decision Log row in `ROADMAP.md` that gives the reason; the change is also noted at the
bottom of this file. Every value below is checked in by `configs/base.yaml` and
`data/processed/splits.json`.

## 1. Datasets and split

| Role | Dataset | Files (`data/processed/splits.json`, SHA-256 pinned) | Used for |
|---|---|---|---|
| Train | Jinan 3×4 (12 intersections) | `anon_3_4_jinan_real.json` | fitting network weights |
| Validation | Jinan 3×4 | `anon_3_4_jinan_real_2000.json` | hyperparameters, gate threshold τ, checkpoint / model selection |
| Test | Jinan 3×4 | `anon_3_4_jinan_real_2500.json` | final numbers only, run **once** after selection |
| Held-out city | Hangzhou 4×4 (16 intersections) | `anon_4_4_hangzhou_real{,_5734,_5816}.json` | zero-shot generalisation (7.4), run **once** |

- Whole flow files are assigned to splits; nothing is resampled. The held-out city is a
  **group-wise (location) split**: no Hangzhou flow is simulated before final evaluation.
- Files are loaded only through `trafficrl.data.splits.flow_files`, which checks the hashes.
- Nothing is fitted on data: observations use lane-capacity scaling from road lengths only
  (Decision Log 10 Nov), so there are no normalisation statistics to leak.

## 2. Simulation settings

| Item | Value | Reason |
|---|---|---|
| Simulator | CityFlow, pinned commit (worklog/007) | ROADMAP Decision Log 27 Oct |
| Episode length | 3600 simulated seconds; metrics read at t = 3600 (no extra drain time) | flow files span 0–3599 s |
| Decision interval Δt | **15 s** of green between decisions | EDA §10: Max-Pressure 375 s → 344 s (−8 %) from 10 s to 15 s on the train flow; 20 s gives no further gain (347 s) |
| Yellow / transition | 5 s of phase 0, inserted only when the phase changes (on top of Δt) | roadnet transition phase |
| Minimum green | = Δt (a chosen phase is held for at least 15 s) | follows from the decision loop |
| Simulator threads | 1 per run; runs parallelised across processes | worklog/007 |

Δt was chosen on the train flow only, before any validation or test flow was simulated. It is
the same for **every** controller (Fixed-Time excepted, see §4), including Max-Pressure when
it acts as the UG-MARL fallback. Published CityFlow results (CoLight, PressLight) mostly use
Δt = 10 s, so our absolute numbers are not directly comparable with those papers; comparisons
are made only among our own runs.

## 3. MDP (per intersection i)

- **State** sᵢ ∈ ℝ⁴⁴: queue (12) ‖ vehicle count (12) ‖ movement pressure (12) ‖ one-hot
  green phase (8), lane values divided by lane capacity ⌊length / 7.5 m⌋ and clipped to [0, 1]
  (`trafficrl.envs.features.FeatureBuilder`, `configs/base.yaml` `features:`). Coordinated
  agents also receive their neighbours' states.
- **Action** aᵢ ∈ {1, …, 8}: the next green phase.
- **Reward** rᵢ = **−|Pᵢ|**, Pᵢ = Σ vehicles on incoming lanes − Σ vehicles on outgoing lanes
  (`FeatureBuilder.intersection_pressure`), measured at the end of the Δt interval.
  Reason: it is the quantity Max-Pressure minimises, so the learned policy and the gate's
  fallback optimise the same objective; queue and |P| correlate at 0.65–0.82 on the train
  traces (EDA §9). Known weakness, reported as a limitation: |P| ≈ 0 when upstream and
  downstream lanes are both full. −Σ queue is **not** used.

## 4. Agents compared

All agents use the same env wrapper, flows, seeds, episode length and Δt.

| Agent | Settings fixed now |
|---|---|
| Fixed-Time | 30 s green per phase in order 1 → 8, 5 s yellow (cycle independent of Δt) |
| Max-Pressure | argmax over phases of Σ(x_in − x_out), ties → lowest id, every Δt |
| DQN (single-agent) | shared parameters, local state only, no coordination |
| IDQN (multi-agent) | one independent DQN per intersection |
| UG-MARL | neighbour attention + K-head Q-ensemble + gate (σ > τ → Max-Pressure) |
| Ablations | UG-MARL −gate, −attention, −ensemble |

Learned agents are trained on the train flow only. Hyperparameters (search spaces written in
`configs/` before the first tuning run), the checkpoint and τ are selected on the validation
flow by the primary metric. The number of training episodes is a hyperparameter, not part of
this protocol.

## 5. Metrics

**Primary:** average travel time (s) over all vehicles that entered the network; vehicles
still in the network at t = 3600 count with their time so far (CityFlow
`get_average_travel_time()`).

**Secondary:**

| Metric | Definition |
|---|---|
| Average waiting time (s) | mean over vehicles of the time spent at speed < 0.1 m/s |
| Average queue length (veh) | mean over decision steps and incoming lanes of waiting vehicles |
| Throughput (veh) | vehicles that left the network by t = 3600 |
| Maximum per-vehicle wait (s) | max over vehicles of total waiting time (safety / starvation) |
| % fallback actions | share of UG-MARL decisions taken by Max-Pressure (gate firing rate) |
| Decision latency (ms) | wall-clock time per joint decision, batch of all intersections |
| Training time, parameters, GPU memory | efficiency (7.5) |

## 6. Experiments, seeds and conditions

| Experiment | Flows | Seeds | Conditions |
|---|---|---|---|
| 7.1 Main comparison | Jinan test (selection on val) | 0–4 (5) | all agents, clean observations |
| 7.2 Ablation | Jinan test | 0–4 (5) | full vs −gate / −attention / −ensemble |
| 7.3 Robustness | Jinan test | 0–2 (3) | sensor noise σ ∈ {0, 0.1, 0.2}; detector dropout p ∈ {0, 10, 20 %}; demand surge ×1.5 |
| 7.4 Generalisation | Hangzhou, 3 flows | 0–2 (3) | zero-shot, trained on Jinan train, τ from Jinan val |

A seed fixes Python, NumPy, PyTorch and CityFlow (`trafficrl.utils.seed`) for both training
and evaluation of that run. Robustness perturbations apply to what **every** controller
observes (Max-Pressure reads the same detectors):
- *noise* — Gaussian with standard deviation σ added to each capacity-scaled lane value, then
  clipped to [0, 1];
- *dropout* — each lane reading set to 0 independently with probability p at each decision;
- *surge* — each vehicle of the flow duplicated with probability 0.5 (same route, start time
  + 1 s), drawn from the run seed.

The exact implementation (ROADMAP 4.3) must follow these definitions.

## 7. Reporting

- Every metric: **mean ± std and 95 % CI** (Student t, n − 1 degrees of freedom) over the
  seeds above; never a single run. Table format: Model | Validation score | Test score |
  Training time | Model size (params).
- **Rule-based baselines are deterministic** across seeds (EDA, worklog/013). They are still
  run on every seed; the table reports their value with std = 0 and the CI marked
  "deterministic". For H1, "non-overlapping CI" against a deterministic baseline means the
  learned agent's whole 95 % CI lies below the baseline's value.
- Test and held-out flows are run once, after selection on validation; numbers are reported
  whatever they are. Final numbers come from native Ubuntu runs.
- Every run appends a row to `docs/experiment_log.csv` and saves its resolved config and
  `env.json`.

## 8. Success criteria

Unchanged from `PRD.md` §1c (H1 performance, H2 safety, H3 coordination, minimum acceptable
outcome), evaluated on the test flows with the seeds of §6. A hypothesis that is not
supported is reported as a negative result.

## 9. Changes after the freeze

| Date | Change | Reason | Decision Log |
|---|---|---|---|
| — | none | — | — |
