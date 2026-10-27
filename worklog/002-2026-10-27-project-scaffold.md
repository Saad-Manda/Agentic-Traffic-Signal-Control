# 002 — Project scaffold (CLAUDE.md, PRD, `.claude/` workspace)
**Date:** 2026-10-27   **Tag:** setup   **ROADMAP:** —   **Status:** done

## What
Created the project's documentation and the Claude Code agent workspace, and initialised git.

## Why
A greenfield project needs one place that records the goal, stack, conventions and
constraints. Then every later session (human or Claude) starts from the same context.

## How
- Ran `/init`. The repo held only `Problem Statement.pdf`, so the details came from the
  conversation and from questions answered by the user:
  - **Approach:** UG-MARL.
  - **Environment:** `venv` + `requirements.txt`.
  - **Conventions:** git, notebooks for EDA and plots only, seeded and reproducible runs.
  - **Hard rule:** no data leakage.
  - **Simulator:** left as an open task at that point.
- Wrote:
  - `CLAUDE.md`
  - `PRD.md` (requirements F1–F11, metric mapping, rubric mapping, milestones)
  - `.claude/commands/` (`prime`, `plan-feature`, `execute`, `commit`)
  - `.claude/reference/` (`architecture`, `api`, `database`, `testing`)
  - `.claude/plans/`
  - `.gitignore`
- **No-leakage rules** written into `.claude/reference/database.md`:
  - Split the data before any processing.
  - Compute normalization statistics on train only.
  - Tune on validation only.
  - Use test flows and the held-out city only once, for final evaluation.

## Result
Workspace in place. Unknowns are marked TODO instead of guessed.

## Files / commits
`CLAUDE.md`, `PRD.md`, `.claude/**`, `.gitignore` · `e33fb36`

## Next
Write the phase-by-phase plan of action (003).
