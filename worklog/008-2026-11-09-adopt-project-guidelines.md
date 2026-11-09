# 008 — Adopt project guidelines; re-baseline schedule
**Date:** 2026-11-09   **Tag:** decision   **ROADMAP:** all (new 1.6, 3.0, 3.5, 5.0, 8.4, 9.5–9.8, Final gate)   **Status:** done

## What
Read the new `ProjectGuidelines.pdf` (IIT Patna IMPACT Lab end-to-end ML guidelines, 5 Nov 2026,
24 slides) and aligned every rules/plan file with it: PRD, ROADMAP, CLAUDE.md, the
`.claude/reference/` files and the `.claude/commands/` files. Re-baselined the schedule from today.

## Why
The guidelines are a second governing document next to PS16 (12 steps, report/slide structure,
completion checklist). Gaps found: no measurable problem definition / hypotheses / success
criteria, no dataset card or data audit, no protocol freeze, no experiment log or recorded
versions/hardware, no CIs, no error analysis, no reproduction test, no ethics section.
Schedule: on 9 Nov only 0.1 and 0.1b (WSL) were done, every phase date had passed, Review 2 is
10 Nov — the old dates were no longer a usable plan.

Each change was confirmed with the user piece by piece (PRD and ROADMAP one change at a time;
the remaining CLAUDE.md / reference / command changes approved in bulk). Decisions:
- **Conflict rule:** PS16 rubric wins → 10 slides (guidelines say 10–15), report ≤ 8 pages.
- **Title:** *Uncertainty-Gated Multi-Agent Reinforcement Learning for Adaptive Traffic Signal
  Control*, short form **UG-MARL** (alternatives with "on Real-World Road Networks" / "with
  Max-Pressure Fallback" rejected by user in favour of a shorter title).
- **Not frozen yet (user):** dataset choice and the evaluation metric set — both frozen in
  ROADMAP 3.5 before RL training.
- **Success criteria:** non-overlapping 95% CIs over ≥ 3 seeds rather than a fixed % margin
  (Max-Pressure is a strong baseline).
- **Scope controls:** left unchanged for now (user), even though H2 depends on the noise runs in 7.3.
- Old Review 1 block left as-is (unknown what was presented on 3 Nov).

## How
- `PRD.md`: new title; §1a problem definition (guidelines template; task type = sequential
  control/MARL); §1b hypotheses H1–H3; §1c success criteria; §5 provisional-metrics note and
  reporting rules (mean ± std + 95% CI, Model | Val | Test | Training time | Model size);
  §7 governing documents + rubric rows; §8 milestones re-baselined.
- `ROADMAP.md`: G1–G12 mapping table and phase tags; 1.5 extended, 1.6 dataset card, 3.0 data
  audit, 3.3 EDA rewritten (train-only, minimum evidence), 3.5 protocol freeze, 5.0 gate,
  5.4/5.5/6.7 experiment log + predefined search spaces, Phase 7 test-once rule + 7.1 table,
  8.3 caveat + 8.4 error analysis, 9.1/9.2 report and slide outlines, 9.4 README with expected
  outputs, 9.5 diagram, 9.6 algorithm, 9.7 fresh-clone reproduction test, 9.8 ethics; Final
  gate checklist; re-baselined schedule table + phase dates; Review 2 revised and moved; two
  Decision Log rows.
- `CLAUDE.md`: title, governing documents, Key Files, dates 31 Oct → 10 Nov, structure
  (`docs/`, `models/`, `slides/`; notebooks EDA-only), "Guidelines rules" conventions.
- `.claude/reference/guidelines.md` (new, condensed PDF); `database.md` (dataset card, audit,
  split type, protocol freeze, test-once); `testing.md` (CIs, experiment log, reproducibility
  check, bash fence fix); `architecture.md` (diagram requirements, draft Algorithm 1).
- `.claude/commands/`: `plan-feature.md` (Justification, Experiment log, G-steps), `execute.md`
  (log runs, plot rule, freeze/test-once), `prime.md` (read guidelines.md, report Final gate).
- New title propagated to the `README.md` and `ROADMAP.md` headings; the repo name stays
  `Agentic-Traffic-Signal-Control` (user: keep the URL stable); CLAUDE.md clone path fixed from
  `~/pbl` to `~/Agentic-Traffic-Signal-Control`. Worklog 001/005 keep the old names (historical record).
- Renamed `ProjectGuidlines.pdf` → `ProjectGuidelines.pdf` (typo; was untracked).
- PDF text extracted with `pypdf` installed to a temporary directory (no poppler on WSL).

## Result
All rules and plan files are consistent with the guidelines and with each other. No code or
experiments changed. `README.md` intentionally untouched (reproduction section comes in 9.4).

## Files / commits
`PRD.md`, `ROADMAP.md`, `CLAUDE.md`, `ProjectGuidelines.pdf`, `.claude/reference/guidelines.md`,
`.claude/reference/database.md`, `.claude/reference/testing.md`,
`.claude/reference/architecture.md`, `.claude/commands/plan-feature.md`,
`.claude/commands/execute.md`, `.claude/commands/prime.md`, `worklog/README.md` · `5269765`, `5383020`, `8fcc57b`

## Next
- Phase 1 dataset selection + split plan, 0.2–0.3 skeleton, 3.0 audit — by 10 Nov (Review 2).
- Decide scope controls in light of H2 (noise runs) when time pressure is clearer.
- Mark or move the old Review 1 block once its outcome is recorded.
