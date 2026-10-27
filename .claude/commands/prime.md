---
description: Load project context at the start of a session
---

Load context for the traffic-signal-control PBL project before doing any work.

1. Read `CLAUDE.md` (project constitution), `PRD.md` (requirements, milestones) and
   `ROADMAP.md` (phase checklist, decision log).
2. Read every file in `.claude/reference/`: `architecture.md`, `api.md`, `database.md`,
   `testing.md`.
3. List `.claude/plans/` and note any plan that is in progress.
4. Check state: `git status`, `git log --oneline -10`, and what exists under `src/`,
   `configs/`, `results/`.
5. Note today's date relative to the review dates (3, 10, 17 Nov 2026) and which PRD
   milestone is next.

Then give a short summary: current milestone, what's implemented, open tasks (e.g. simulator
choice still pending?), and the suggested next step. Do not start editing code during prime.
