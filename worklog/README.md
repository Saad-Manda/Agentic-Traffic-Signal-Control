# Work Log

A record of every meaningful piece of work on the project: **what** we did, **why**, **how**,
and the result. One file per piece of work, numbered in the order the work happened.

## Rules

- **File name:** `NNN-YYYY-MM-DD-short-slug.md`. `NNN` is a three-digit number, sequential and
  never reused. The date is the day the work started.
- **One file per piece of work.** If the work spans several days, update the same file and
  record the end date in its header.
- **Log:** setup and installs, decisions, code, experiments, non-trivial debugging, and
  report/poster/demo work. **Don't log:** pure Q&A where nothing changed.
- **Log failures too**, with status `failed` or `abandoned`. They feed the report's
  limitations section and the viva.
- **Files changed is mandatory and per file** (from entry 012 on). List every file created,
  modified, renamed or deleted, with: the action, *what* changed inside it (sections,
  functions, config keys, table rows — concrete, never just "updated"), and *why*. Cross-check
  with `git status` / `git diff --stat` before finishing. Trivially repetitive files (e.g.
  empty `__init__.py`) may share one row.
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

## Files changed
| File | Action | What changed | Why |
|------|--------|--------------|-----|
| `path/to/file` | created / modified / renamed / deleted | Concrete change: sections, functions, keys, rows added/removed/edited | Reason this file needed the change |

## Commits
Commit hashes (or "not committed yet").

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
| [007](007-2026-10-28-cityflow-build-wsl.md) | 2026-10-28 | Build CityFlow on WSL, time Jinan episode | setup | 0.1b | in-progress |
| [008](008-2026-11-09-adopt-project-guidelines.md) | 2026-11-09 | Adopt project guidelines; re-baseline schedule | decision | all | done |
| [009](009-2026-11-09-phase0-skeleton-seed-config.md) | 2026-11-09 | Repo skeleton, pinned env, seed + config utils | setup | 0.2, 0.3 | done |
| [010](010-2026-11-09-literature-review.md) | 2026-11-09 | Literature review (Phase 2) and draft gap table | report, decision | 2.1–2.6 | in-progress |
| [011](011-2026-11-10-dataset-selection-split.md) | 2026-11-10 | Dataset selection (Jinan / Hangzhou), split, dataset card | data, decision | 1.1–1.6, 3.1–3.2 | done |
| [012](012-2026-11-10-worklog-per-file-changes.md) | 2026-11-10 | Work log rule: per-file change records | setup | — | done |
| [012](012-2026-11-10-how-it-works-explainer.md) | 2026-11-10 | Explainer: data, simulator and training | docs | 9.x, G12 | done |
| [013](013-2026-11-10-phase3-audit-eda-features.md) | 2026-11-10 | Phase 3: data audit, train-split EDA, feature engineering | data, code, experiment, decision | 3.0, 3.3, 3.4 | done |
| [014](014-2026-11-10-evaluation-protocol-freeze.md) | 2026-11-10 | Evaluation protocol frozen (Δt 15 s, reward −pressure) | decision, docs | 3.5, 5.0 | done |
| [015](015-2026-11-10-phase4-env-baselines.md) | 2026-11-10 | Phase 4: env wrapper, metrics, robustness hooks, rule baselines, evaluation | code, experiment, decision | 4.1–4.6 | done |
| [016](016-2026-11-11-phase5-rl-baselines.md) | 2026-11-11 | Phase 5: RL baselines (DQN, IDQN), training and tuning | code, experiment, decision | 5.1–5.5 | done |
| [017](017-2026-11-11-phase6-ugmarl.md) | 2026-11-11 | Phase 6: proposed UG-MARL — design, code, grid (K 3, d 128), 5-seed finals + ablations, gate calibration (p* 80), σ check, methodology | decision, code, experiment | 6.1–6.7 | done |
| [018](018-2026-11-11-native-ubuntu-setup.md) | 2026-11-11 | Native Ubuntu setup; Phase 5 reproduced bit-identically | setup, experiment | 0.1b | done |
| [019](019-2026-11-12-val-robustness-references.md) | 2026-11-12 | 6.8: baselines under detector noise / dropout on validation — DQN / IDQN collapse, UG-MARL level with Max-Pressure under noise | decision, code, experiment | 6.8 | done |
| [020](020-2026-11-12-phase7-final-experiments.md) | 2026-11-12 | Phase 7: final test + Hangzhou — H1 and H2 supported, H3 not; UG-MARL 3.2 s behind DQN on clean test, smallest worst-case regret (6.8 s) | decision, code, experiment | 7.1–7.6 | done |
| [021](021-2026-11-13-gap-table-update.md) | 2026-11-13 | Gap table: UG-MARL row (stability stays ◐) and status of action items A1–A9; A8 moved to Phase 8 | docs | 2.6 (partial), 8.2 | done |
| [022](022-2026-11-13-phase8-analysis.md) | 2026-11-13 | Phase 8: analysis notebook, 18 figures, error analysis, τ sweep (insensitive; p = 70 1.5 s better, not changed), attention faithfulness (weights rank a small influence), `docs/analysis.md` | decision, code, experiment, report | 8.1–8.4 | done |
| [023](023-2026-11-14-gap-table-final.md) | 2026-11-14 | Gap table finalised: every row checked against sources, novelty search re-run (still novel), 2 rows added, external numbers not quoted | docs, decision | 2.6 | done |
| [024](024-2026-11-14-report.md) | 2026-11-14 | Report drafted: 15-page LaTeX report with generated tables and numbers, architecture diagram, Algorithm 1 checked against the code, ethics and limitations; student review open | report, code, decision | 9.1 (draft), 9.5, 9.6, 9.8 | done |
| [025](025-2026-11-15-report-revision.md) | 2026-11-15 | Report revised: one-paragraph abstract, contributions as changes over existing work, 38 → 10 references, gaps as prose, humanized text, no title page / chapter breaks — 12 pages | report, decision | 9.1 | done |
| [026](026-2026-11-15-poster.md) | 2026-11-15 | Poster built: one 16:9 slide from a `python-pptx` script, numbers / table / figure from `docs/results`, own design, no logos; user review open | poster, code, decision | 9.2 | built |
| [027](027-2026-11-17-contributions-rewrite.md) | 2026-11-17 | Contributions rewritten along the mentor's seven contribution categories: six items in report §1.5 (13 pages), four merged bullets on the poster | report, poster, decision | 9.1, 9.2 | done |
| [028](028-2026-11-17-appendix-submission.md) | 2026-11-17 | Report appendix (code / poster / data links, learning curves; 13 pages), humanized §1.5 Contributions, and `submission/results/results.xlsx` with every model and its Python files | report, submission, decision | 9.1 | done |
| [029](029-2026-11-17-report-humanize-pass.md) | 2026-11-17 | Targeted humanize pass on the report: honesty statements cut, Com-MP / Hoel comparison only in §2.2, paragraph-end summary lines and caveat follow-ups merged; 13 pages, checks ok | report | 9.1 | done |
| [030](030-2026-11-17-report-docx.md) | 2026-11-17 | Word version of the report (`report/report.docx`) via flattened LaTeX → pandoc → python-docx restyle; native equations, numbered captions, IEEE references | report | 9.1 | done |
