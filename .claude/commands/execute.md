---
description: Execute a saved plan from .claude/plans/ with minimal, targeted edits
argument-hint: <plan file name>
---

Execute the plan `.claude/plans/$ARGUMENTS` (if no name given, list `.claude/plans/` and ask).

1. Read the plan, `CLAUDE.md`, and the reference files it mentions.
2. Implement the steps in order. For each step:
   - Make the smallest change that satisfies it; match existing style in `src/trafficrl/`.
   - Keep new agents conforming to the shared interface in `reference/api.md`.
   - Put experiment parameters in `configs/*.yaml`, not hard-coded; keep seeding intact.
   - Never let training, normalization statistics, or tuning read test flows or the held-out
     generalization city.
3. After each step run its verification: `pytest` (or the relevant subset), `ruff check
   src tests`, and any short smoke run the plan specifies. Fix failures before moving on;
   report honestly if something cannot be made to pass.
4. Tick off completed steps in the plan file and note deviations.
5. Afterwards update `CLAUDE.md` (Commands / Structure / Notes), the relevant
   `.claude/reference/*.md`, and `PRD.md` if requirements status changed.
6. Summarize what changed and what was verified. Do not commit unless asked — then follow
   `commit.md`.
