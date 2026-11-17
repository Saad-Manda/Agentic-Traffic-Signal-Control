# 027 — Contributions rewritten along the mentor's contribution categories
**Date:** 2026-11-17   **Tag:** report, poster, decision   **ROADMAP:** 9.1, 9.2   **Status:** done; user review open

## What
Rewrote the Contributions section of the report (§1.5) and the Contributions block of the
poster so that they follow the general list of contribution types sent by the mentor
(problem formulation; preprocessing and feature engineering; mathematical modelling and
theory; novel methodology; technical advancement; experimental validation; generalisation).

## Why
The mentor's message says the contributions must be identified per project along those seven
categories. The old section listed five changes over prior work and did not name the
categories. User decisions of 17 Nov:
- Use only the categories that fit the project strongly (not all seven by force).
- Poster: four merged bullets in the same space; nothing else on the poster changes.
- Theory: state exactly what holds (Varaiya's theorem covers the fallback; the gated policy
  has a sketch and no proof).
- The report may grow; an appendix will be added later.

## How
- Report: six items. Category 2 (preprocessing) is light in this project (capacity scaling
  with nothing fitted, whole-file splits), so it is merged into item 1. The contrasts with
  Tan et al., Com-MP and Hoel et al. moved into item 3 (framework); the CoLight contrast into
  item 4 (technical advancement). Numbers are `\res` macros where a macro exists; typed
  numbers were already in `report/checks/claims.md`.
- Added `\label{sec:safety}` to §3.4.6 so that item 2 can point at the safety argument.
- Poster: first version with four full bullets overflowed the left column (the last "Data and
  protocol" bullet was cut off). The bullets were shortened until the column fitted again; the
  explicit contrasts with [3], [4], [5] are now only citation numbers.
- Rebuilt with `latexmk`, `scripts/check_report.py`, `scripts/build_poster.py --pdf`; the
  poster was rendered to PNG and looked at after each of the three builds.

## Result
- `report/report.pdf`: 13 pages (was 12), `check_report.py`: ok, no undefined references.
- `poster/poster.pdf`: left column fits, nothing clipped (LibreOffice render).
- `pytest`: 296 passed, 1 skipped. No simulation, no `--final`, no experiment-log row.
- Not claimed anywhere: a stability proof for the gated policy.

## Files changed
| File | Action | What changed | Why |
|---|---|---|---|
| `report/chapters/01-introduction.tex` | modified | §1.5 Contributions: five items replaced by six (formulation and data design; mathematical modelling and theory; proposed framework; technical advancement; experimental validation; generalisation) | follow the mentor's categories |
| `report/chapters/03-methodology.tex` | modified | `\label{sec:safety}` on the "Safety argument" subsection | cross-reference from contribution 2 |
| `report/report.pdf` | generated | rebuilt, 13 pages | deliverable |
| `poster/content.yaml` | modified | `contributions.items`: three bullets replaced by four (Formulation, data / Model, theory / Gain / Validation) | same categories on the poster, same space |
| `poster/poster.pptx`, `poster/poster.pdf` | generated | rebuilt | deliverables |
| `report/checks/claims.md` | modified | section tags of three typed numbers (25 s no longer in 1.5; 22 s and 0.81 ms now also in 1.5); two rows under "Statements about other work"; one poster row | every claim traceable |
| `CLAUDE.md` | modified | report note: 13 pages, pointer to this entry; structure line page count | keep the session primer correct |
| `worklog/027-2026-11-17-contributions-rewrite.md`, `worklog/README.md` | created / modified | this entry and its index row | work log rule |

## Commits
`e8487f1` docs(poster): regroup contributions; `a55a104` docs(report): rewrite contributions, add appendix, tighten prose

## Next
User review of the new wording; appendix for the report (user plans to add one).
