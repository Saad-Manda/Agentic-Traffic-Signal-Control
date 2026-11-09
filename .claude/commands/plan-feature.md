---
description: Write an implementation plan for a feature to .claude/plans/
argument-hint: <feature description>
---

Plan the feature: $ARGUMENTS

1. Read `CLAUDE.md`, `PRD.md` and the relevant `.claude/reference/` files (incl.
   `guidelines.md`). Inspect the
   existing code under `src/trafficrl/`, `scripts/`, `configs/`, `tests/` that the feature
   touches.
2. Write `.claude/plans/<kebab-case-feature-name>.md` with:
   - **Goal** — what it does and which PRD requirement(s) (F1–F11) / rubric row (R1–R8) /
     guideline step (G1–G12) it serves.
   - **Design** — modules, classes, functions; how it fits the shared agent interface
     (`reference/api.md`); any math (equations for losses, uncertainty, gate).
   - **Justification** — why this approach; alternatives considered and why each was rejected
     (guidelines: every methodological decision justified, recorded, verified).
   - **Files** — each file to create or modify, with a one-line purpose.
   - **Config** — new YAML keys and defaults; seed handling.
   - **Data-leakage check** — which data split(s) the feature reads, and why test flows are not
     used for training, normalization stats, or tuning.
   - **Tests** — what `pytest` tests will be added.
   - **Verification** — exact commands to run and expected outputs (e.g. a short training run
     whose reward improves, a metrics CSV produced).
   - **Experiment log** — for any run (training, tuning, evaluation): the rows it appends to
     `docs/experiment_log.csv`, and which split it touches. Tuning must use a search space
     written in `configs/` beforehand; test flows only after the protocol freeze (ROADMAP 3.5)
     and selection on validation.
   - **Risks / open questions.**
   - **Worklog** — note that executing this plan ends with a `worklog/` entry (see
     `worklog/README.md`).
3. Keep steps small and ordered so each can be run and checked on its own.
4. Show the plan summary and ask for confirmation before executing (unless pre-approved).
