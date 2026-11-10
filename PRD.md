# PRD — Uncertainty-Gated Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control (UG-MARL)

## 1. Problem

Conventional signal control (fixed-time, actuated) cannot adapt to dynamic traffic. Existing
RL / Agentic AI controllers suffer from (PS16): unreliable decisions under uncertainty and noisy
sensors, weak multi-agent coordination, error propagation, no safety guarantees, poor
explainability, and high computational cost.

### 1a. Problem definition

*Given* per-lane queue lengths, vehicle counts and the current signal phase at every
intersection of a real-world road network (primary dataset — selection pending, see ROADMAP
Phase 1), *choose* the next signal phase for each intersection every Δt seconds over a 3600 s
episode, *under* min-green and yellow-transition constraints and possibly noisy or missing
detector readings, *achieving* lower average vehicle travel time than the baselines while
keeping the worst-case per-vehicle wait bounded.

**Task type:** sequential decision-making / control (multi-agent reinforcement learning). It
is not one of the guidelines' four task types (regression, classification, clustering,
forecasting), because the model chooses actions that change its future inputs rather than
predicting a fixed label.

### 1b. Hypotheses

- **H1 (performance):** UG-MARL gives lower average travel time than the multi-agent RL
  baseline (IDQN) and the conventional baseline (Max-Pressure) on the test flows.
- **H2 (safety / self-correction):** the uncertainty gate lowers worst-case behaviour — the
  maximum per-vehicle wait and the drop in performance under sensor noise — compared with the
  same model without the gate (−gate ablation).
- **H3 (coordination):** neighbour attention lowers average travel time compared with the same
  model without attention (−attention ablation).

### 1c. Success criteria

Fixed before experiments; evaluated on test flows, ≥ 3 seeds, mean ± 95% CI.

- **H1 supported** if UG-MARL's average travel time is lower than both IDQN's and
  Max-Pressure's, with non-overlapping 95% confidence intervals.
- **H2 supported** if, versus −gate, the full model has a lower maximum per-vehicle wait **and**
  a smaller travel-time degradation as sensor noise goes from 0 to 0.2.
- **H3 supported** if the full model's average travel time is lower than −attention's, with
  non-overlapping 95% CIs.
- **Minimum acceptable outcome:** UG-MARL is no worse than Max-Pressure on maximum wait (the
  gate should guarantee this).
- A hypothesis that is not supported is reported as a negative result, not dropped.

## 2. Goal

1. Investigate the working principles and limitations of conventional, single-agent and
   multi-agent frameworks for traffic signal control on public real-world datasets.
2. Propose **UG-MARL**: neighbor-attention coordination + Q-ensemble uncertainty + gated
   fallback to Max-Pressure (self-correction with a stability argument) + explainability.
3. Validate experimentally against baselines, with ablations and robustness/generalization tests.

## 3. Scope

**In scope:** traffic signal control on grid networks (Jinan 3×4, Hangzhou 4×4; optional
others), simulation-based evaluation, offline analysis, demo via simulator.
**Out of scope:** real-world deployment, vehicle routing, pedestrian modeling, LLM agents.

## 4. Functional Requirements

| ID | Requirement |
|----|-------------|
| F1 | Simulator environment wrapper with configurable network, flow file, episode length, seed |
| F2 | EDA + preprocessing of flow datasets; train/val/test split with no leakage |
| F3 | Baselines: Fixed-Time, Max-Pressure, single-agent DQN, multi-agent IDQN (IPPO optional) |
| F4 | Proposed UG-MARL agent (attention, ensemble, uncertainty gate) |
| F5 | Hyperparameter tuning on validation flows only |
| F6 | Evaluation harness producing a metrics table + plots for every agent |
| F7 | Ablations: −gate, −attention, −ensemble |
| F8 | Robustness: sensor noise, detector dropout, demand surge |
| F9 | Generalization: train on one city, test on another without retraining |
| F10 | Explainability outputs: attention / SHAP plots, gate-firing statistics |
| F11 | Demo: side-by-side baseline vs. UG-MARL in the simulator |

## 5. Metrics (PS16 → traffic)

| PS16 metric | Measured as |
|---|---|
| Task completion rate | Throughput (completed trips) |
| Decision accuracy | Avg travel time, avg waiting time, avg queue length |
| Response time | Inference latency per decision (ms) |
| Resource utilization | Training time, GPU/CPU usage, parameter count |
| Robustness | Metric degradation under noise / dropout / surge |
| Generalization | Cross-city performance |
| Safety | Max per-vehicle wait (starvation), % fallback actions |
| Explainability | Attention/SHAP attributions, gate statistics |

The metric set above is **provisional** — not frozen yet; it may change until the evaluation
protocol is frozen (before training).

**Reporting rules**
- Every metric is reported as mean ± std **and** 95% CI over ≥ 3 seeds — never a single best run.
- Results table format: **Model | Validation score | Test score | Training time | Model size
  (params)**, with all agents run on identical flows, seeds and episode lengths.
- Fixed-Time and Max-Pressure have no training, so their training time is "—".

## 6. Non-Functional Requirements

- Reproducible: config-driven, seeded runs; results saved with resolved config.
- **No data leakage** between train / validation / test flows or the generalization city.
- Runs on a single laptop GPU (RTX 4050) within the project timeline.

## 7. Rubric Mapping

**Governing documents:** `Problem Statement.pdf` (PS16 + graded rubric) and
`ProjectGuidelines.pdf` (IIT Patna IMPACT Lab end-to-end ML guidelines, 5 Nov 2026). Where they
conflict, **the PS16 rubric wins** (it is what is graded):
- Presentation: **10 slides**, following the guidelines' slide order (p.19) compressed into 10.
- Report: **≤ 8 pages**, using the guidelines' 10-section structure (p.18).

| Rubric row (marks) | Covered by |
|---|---|
| Title, intro, contribution (5) | §1–2, UG-MARL contribution |
| Literature review (5) | Fixed-time, SOTL, Max-Pressure, IntelliLight, PressLight, CoLight, MPLight |
| Dataset & preprocessing (5) | F2 + dataset card, data audit, frozen split protocol (guidelines Steps 3–4) |
| Implementation (15) | F3, F4, F5 |
| Methodology, baseline, ablation (10) | F4, F7, math (MDP, uncertainty, stability argument) |
| Results & comparison (10) | F6, F8, F9 |
| Viva + demo (30) | F11, team understanding |
| Presentation & report (20) | 10 slides (guidelines order), ≤ 8 pages (guidelines' 10 sections) |

## 8. Milestones

Re-baselined 9 Nov (see ROADMAP Decision Log).

| Date | Deliverable |
|---|---|
| 10 Nov 2026 (Review 2) | Problem definition + H1–H3 + success criteria; dataset choice, audit and split plan; literature gap table (draft); UG-MARL design diagram; evaluation protocol draft |
| 11 Nov 2026 | EDA, **evaluation protocol frozen**, env + Fixed-Time & Max-Pressure results |
| 13 Nov 2026 | DQN, IDQN, UG-MARL implemented and tuned on validation |
| 15 Nov 2026 | Test runs, ablations, robustness, generalization, error analysis |
| 17 Nov 2026 (Review 3) | Report, slides, demo, reproduction test passed |

## 9. Open Questions

- ~~Simulator choice~~ — **resolved 27 Oct: CityFlow** (see ROADMAP Decision Log).
- Final reward definition (−pressure vs. −queue) and gate threshold τ selection method.
- Decision interval Δt: 10 s (literature convention) vs. 15 s — on the train flow Max-Pressure
  is 8 % better at 15 s (`notebooks/01_eda_train.ipynb` §10). To be settled at the protocol
  freeze (ROADMAP 3.5).
