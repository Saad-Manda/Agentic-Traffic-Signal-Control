# Project Guidelines — condensed

Source: `ProjectGuidelines.pdf` (IIT Patna IMPACT Lab, "End-to-End Machine Learning Project
Guidelines", 5 Nov 2026, 24 slides). Read this instead of the PDF; open the PDF only for the
exact wording. **Where the guidelines conflict with the PS16 rubric (`Problem Statement.pdf`),
the rubric wins** — e.g. 10 slides (not 10–15), report ≤ 8 pages.

Core rule: *every methodological decision should be justified, recorded and verified.*
Goal: a scientifically sound, reproducible, clearly documented solution — not merely a high score.

## The 12 steps (G1–G12) and how this project meets them

| # | Guideline asks for | Here |
|---|---|---|
| G1 | Precise problem via template *Given [inputs], predict [target] for [population] under [constraints], achieving [metric/threshold]*; task type; success criteria **before** experiments; 1-page definition | `PRD.md` §1a–1c (task type = sequential control / MARL) |
| G2 | Literature: datasets, methods, protocols, metrics; comparison table *Study \| Dataset \| Method \| Evaluation \| Gap*; state the gap; no unsupported novelty; cite originals | ROADMAP Phase 2, gap table |
| G3 | Dataset card (origin, license, citation, collection period, unit of observation, feature descriptions + units, size) + data dictionary + audit (missing, duplicates, impossible values, outliers, bias, privacy) with counts and plots | ROADMAP 1.5, 1.6, 3.0 → `docs/dataset_card.md`, `docs/data_audit.md` |
| G4 | **Freeze the evaluation protocol before training**; train fits / validation selects / test once; group-wise splits for locations; chronological splits for time; no normalisation, augmentation or feature selection using test | ROADMAP 3.1, 3.2, 3.5 → `docs/evaluation_protocol.md`; held-out city = group-wise split |
| G5 | EDA on training data: counts, missingness, summary stats, target distribution, feature distributions, correlations / drift; **interpretation under each figure** | ROADMAP 3.3 |
| G6 | Reusable preprocessing pipeline; raw → split → fit on train → transform val/test; domain-knowledge features | ROADMAP 3.4 |
| G7 | Simple baseline first, then several reasonable models; identical splits and metrics; record training time and complexity; **explain why each model is included** | Fixed-Time, Max-Pressure, DQN, IDQN (Phases 4–5) |
| G8 | Fixed seeds; record software versions + hardware; predefined search spaces; no tuning from test; experiment log *Run ID \| Split/seed \| Model \| Hyperparameters \| CV metric \| Runtime* | ROADMAP 5.4, 5.5, 6.7 → `docs/experiment_log.csv` (+ Git commit column; validation metric instead of CV) |
| G9 | Metrics aligned to the goal; **variance / confidence intervals, not just the best run** | `PRD.md` §5: mean ± std and 95% CI, ≥ 3 seeds |
| G10 | Final pipeline chosen on train/val only; test **once**; same splits and metrics for all; table *Model \| Validation \| Test \| Training time \| Model size* | ROADMAP Phase 7 |
| G11 | Error analysis: correct vs. incorrect cases, error by subgroup, unstable predictions; importance / SHAP when justified; ablations; sensitivity to noise, shift, seed; **attribution ≠ causation** | ROADMAP 7.2–7.6, 8.3, 8.4 |
| G12 | Folder structure; README with dataset access, commands, **expected outputs**; save splits, configs, seeds, artifacts; `requirements.txt`; ethics (licensing, privacy, fairness, misuse, limitations); **fresh run reproduces results** | ROADMAP 9.4, 9.7, 9.8 |

## Common mistakes to avoid (slide 17)

1. Training, feature selection or preprocessing using test-set information.
2. Reporting only accuracy on an imbalanced dataset (here: only the mean travel time — also
   report max wait, throughput, queue).
3. Comparing methods on different splits or incompatible metrics.
4. Claiming novelty or improvements without adequate baselines.
5. Reporting a single favourable run without uncertainty or failure analysis.
6. Plots without labels, units, captions or interpretation.
7. Notebooks or slides without reproducible code and documentation.

## Report (slide 18) — 10 sections, ≤ 8 pages (rubric)

Title/abstract/contributions · Introduction & problem · Related work & gap · Dataset & quality
audit · Split protocol & methodology · Experimental setup & baselines · Results · Error
analysis & ablations · Limitations, ethics & conclusion · References & reproducibility notes.
**Separate what was planned, what was implemented, what the evidence shows, what remains
uncertain.** (Project mapping: ROADMAP 9.1.)

## Slides (slide 19) — rubric: 10 slides

Guideline order (compressed to 10 in ROADMAP 9.2): motivation & problem → literature & gap →
dataset, audit, splits, leakage controls → baselines & protocol → proposed method → results →
error analysis → ablation, robustness, limitations → conclusion & references.
**Every claim backed by an experiment or a citation.**

## Title (slide 21)

[Task] + [Approach] + [Application]; ~10–16 words; no "novel", "best", "100 %"; must reflect
the implemented work. Project title: *Uncertainty-Gated Multi-Agent Reinforcement Learning
for Adaptive Traffic Signal Control* (UG-MARL).

## Architecture diagram (slide 22)

Labelled blocks in data-flow order (inputs → preprocessing → model → output); **distinguish
training from inference; mark where fitting happens**; caption. (ROADMAP 9.5.)

## Algorithm (slide 23)

Descriptive name; **Input / Output**; numbered steps in execution order covering split,
preprocessing, training, tuning, evaluation; loops / selection rules; no code syntax.
**Must match the implementation and must not tune on test.** (ROADMAP 9.6; draft in
`architecture.md`.)

## Completion checklist (slide 20)

Mirrored as the "Final gate" section of `ROADMAP.md` — tick it there.
*Correct protocol + fair comparison + reproducible evidence + clear communication.*
