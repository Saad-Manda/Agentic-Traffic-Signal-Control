# Work Log

A record of every meaningful piece of work on the project: **what** we did, **why**, **how**,
and the result. One file per piece of work, numbered in the order the work happened.

## Rules

- **File name:** `NNN-YYYY-MM-DD-short-slug.md`. `NNN` is a three-digit number, sequential and
  never reused. The date is the day the work started.
- **One file per piece of work.** If the work spans several days, update the same file and
  record the end date in its header.
- **Log:** setup and installs, decisions, code, experiments, non-trivial debugging, and
  report/slides/demo work. **Don't log:** pure Q&A where nothing changed.
- **Log failures too**, with status `failed` or `abandoned`. They feed the report's
  limitations section and the viva.
- Add a row to the index below for every new entry, and keep its status current.
- **Tags:** `setup` `decision` `data` `code` `experiment` `debug` `report`
- **Status:** `in-progress` `done` `failed` `abandoned`

## Template

```markdown
# NNN — Title
**Date:** YYYY-MM-DD (→ end date if multi-day)   **Tag:** …   **ROADMAP:** task IDs   **Status:** …

## What
What was done, in one or two sentences.

## Why
The reason: ROADMAP task, problem being solved, requirement.
Decision entries: alternatives considered and why each was rejected.

## How
Steps, commands, settings.
Experiment entries: config file, seed(s), dataset split, episode length.

## Result
Outcome, with numbers where possible. Experiment entries: key metrics.
If it failed: the error and the workaround.

## Files / commits
Files touched · commit hashes.

## Next
Follow-up tasks.
```

## Index

| # | Date | Title | Tag | ROADMAP | Status |
|---|------|-------|-----|---------|--------|
| [001](001-2026-10-27-problem-statement-analysis.md) | 2026-10-27 | Problem statement analysis and topic alignment | decision | — | done |
| [002](002-2026-10-27-project-scaffold.md) | 2026-10-27 | Project scaffold (CLAUDE.md, PRD, `.claude/` workspace) | setup | — | done |
| [003](003-2026-10-27-roadmap.md) | 2026-10-27 | Plan of action (ROADMAP.md) | decision | all | done |
| [004](004-2026-10-27-simulator-platform-decision.md) | 2026-10-27 | Simulator (CityFlow) and platform decision | decision | 0.1 | done |
| [005](005-2026-10-27-repo-conventions.md) | 2026-10-27 | Repo name, line endings, commit attribution | setup | — | done |
| [006](006-2026-10-27-worklog-rule.md) | 2026-10-27 | Work log rule | setup | — | done |
