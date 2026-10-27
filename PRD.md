# PRD — Uncertainty-Aware Coordinated Multi-Agent AI for Adaptive Traffic Signal Control

## 1. Problem

Conventional signal control (fixed-time, actuated) cannot adapt to dynamic traffic. Existing
RL / Agentic AI controllers suffer from (PS16): unreliable decisions under uncertainty and noisy
sensors, weak multi-agent coordination, error propagation, no safety guarantees, poor
explainability, and high computational cost.

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

## 6. Non-Functional Requirements

- Reproducible: config-driven, seeded runs; results saved with resolved config.
- **No data leakage** between train / validation / test flows or the generalization city.
- Runs on a single laptop GPU (RTX 4050) within the project timeline.

## 7. Rubric Mapping

| Rubric row (marks) | Covered by |
|---|---|
| Title, intro, contribution (5) | §1–2, UG-MARL contribution |
| Literature review (5) | Fixed-time, SOTL, Max-Pressure, IntelliLight, PressLight, CoLight, MPLight |
| Dataset & preprocessing (5) | F2 |
| Implementation (15) | F3, F4, F5 |
| Methodology, baseline, ablation (10) | F4, F7, math (MDP, uncertainty, stability argument) |
| Results & comparison (10) | F6, F8, F9 |
| Viva + demo (30) | F11, team understanding |
| Presentation & report (20) | 10 slides, ≤ 8 pages |

## 8. Milestones

| Date | Deliverable |
|---|---|
| 31 Oct 2026 | Dataset survey done; primary + held-out datasets and split plan selected |
| 3 Nov 2026 | Lit review + gaps, simulator chosen & set up, EDA, Fixed-Time & Max-Pressure results |
| 10 Nov 2026 | DQN, IDQN, UG-MARL implemented and tuned; math write-up |
| 17 Nov 2026 | Ablations, robustness, generalization, report, slides, demo |

## 9. Open Questions

- ~~Simulator choice~~ — **resolved 27 Oct: CityFlow** (see ROADMAP Decision Log).
- Final reward definition (−pressure vs. −queue) and gate threshold τ selection method.
