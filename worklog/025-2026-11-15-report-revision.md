# 025 — Report revision: one-paragraph abstract, contributions as changes, 10 references, prose gaps, humanized text, no blank space
**Date:** 2026-11-15   **Tag:** report, decision   **ROADMAP:** 9.1   **Status:** done (user review open)

## What
Revised the 15-page draft report (worklog 024) on the student's requests. The abstract is now
one paragraph. The contributions say what each one changes in existing work. The report cites
10 works instead of 38, and the research gaps are written as prose instead of a table. The prose
was rewritten with the humanizer skill. The separate title page and the per-chapter page breaks
are gone. The report is now 12 pages.

## Why
User requests of 15 Nov:
- An abstract that gives the whole project in one paragraph.
- Contributions framed as changes or upgrades over existing solutions.
- Reviewing 38 papers is not realistic, so cite only the important ones. The user chose
  "as few as possible", then 10 (asked "can we reduce more?").
- The research gap does not have to be a table.
- Text that reads as written by a person (`/humanizer:humanizer`).
- No empty space on pages, and no whole page for the title.

The user also answered three layout questions: title block plus abstract on page 1, 3–4
headline numbers in the abstract, and let the report shrink rather than pad it (the minimum is
12 pages).

**Choosing the 10 papers.** Each one is either a building block (Max-Pressure, DQN,
PressLight reward, CoLight attention/data, CityFlow, Bootstrapped DQN), the benchmark-fragility
motivation (Ault 2021), or a work that a contribution is compared against (Tan 2020, Com-MP,
Hoel 2020). Five more were considered and dropped (Webster, Double DQN, IDQN/Tampuu, deep
ensembles, Jain & Wallace). These are standard methods, so they are named without a citation;
the cost is a slightly thinner viva defence.

## How
- Plan: `.claude/plans/report-revision.md`.
- **Layout:** the `titlepage` was replaced by a compact title block. `\titleclass{\chapter}{straight}`
  removes the page break before each chapter. The ToC, LoF and LoT run on after the abstract,
  inside a group where `\clearpage` is disabled, and the page numbering is now Arabic throughout.
  The `\clearpage` before References was removed.
- **Rewritten chapters:** the abstract, Ch 1 (contributions recast as five "existing X does A, we
  do B" items with evidence pointers) and Ch 2 (only the 10 cited works; the gap table is
  replaced by three gap paragraphs plus a positioning paragraph).
- **Ch 3–5:** the first pass was sentence-level only. On the user's follow-up ("humanize the
  rest too, excluding figures, tables, algorithms") every prose paragraph of Ch 3, 4 and 5 was
  rewritten. Captions, tables, equations and Algorithm 1 were left as they were. A token check
  (numbers, `\res`, `\cref`, `\cite`, `\label`) against the pre-pass text of Ch 3 and Ch 4
  found nothing missing or added.
- **Voice:** the full pass had introduced "we"/"our". On the user's decision (single-author
  report, so no "we", and "I" is unprofessional), all first-person wording in Ch 2–5 was made
  impersonal (about 30 phrases). The subject is the study, the data or the method where that
  reads naturally, and the passive is used otherwise. "To our knowledge" in Ch 2 became "the
  search behind this review found no …" / "holds only as far as that search reached".
- **Unchanged:** all `\res` macros, equations, labels, table and figure content, and the section
  names that the structure test prescribes.
- `references.bib` was pruned from 66 entries to the 10 cited ones.
- Checks: built with `latexmk`, ran `scripts/check_report.py` and
  `pytest tests/scripts/test_report_sources.py tests/eval/test_latex.py`, and inspected the
  rendered pages.

## Result
- The report is 12 pages, which is exactly the lower limit. Any further cut needs added text
  elsewhere.
- `check_report`: ok (no undefined references or citations, no banned wording); 14 tests passed.
- The ~4-line gap on page 7 seen after the first pass disappeared after the full Ch 3–5
  rewrite, because the text reflowed. No page has blank space except the end of References.

## Files changed
| File | Action | What changed | Why |
|------|--------|--------------|-----|
| `report/main.tex` | modified | Title page → title block; chapters `straight` (no page break), spacing 12 pt; ToC/LoF/LoT run on with `\clearpage` disabled; roman numbering and `\clearpage` before Ch 1 and References removed | No whole title page, no blank space |
| `report/chapters/00-abstract.tex` | modified | Two paragraphs → one paragraph covering problem, method, protocol, 4 headline `\res` results, negative results; `\clearpage` removed | One-paragraph self-contained abstract |
| `report/chapters/01-introduction.tex` | modified | §1.1 citations reduced to varaiya/presslight/colight/ault; prose humanized; §1.6 rewritten as 5 changes over existing work (Tan 2020, Com-MP, Hoel, clean benchmarks, CoLight attention) | Contributions as upgrades; fewer references |
| `report/chapters/02-literature.tex` | modified | Rewritten with 10 citations; Table 2.1 (gap table) deleted; §2.2 as 3 gap paragraphs + position paragraph | Fewer references; gaps as prose |
| `report/chapters/03-methodology.tex` | modified | Citations of vanhasselt, pardo, velickovic, kendall, lakshminarayanan, webster, tampuu removed; every prose paragraph rewritten (floats, equations, algorithm unchanged) | 10-reference set; full humanizer pass |
| `report/chapters/04-results.tex` | modified | Every prose paragraph and error-analysis bullet rewritten; captions and tables unchanged | Full humanizer pass |
| `report/chapters/05-conclusion.tex` | modified | osband2018randomized citation removed; whole chapter rewritten, content unchanged | 10-reference set; full humanizer pass |
| `report/references.bib` | modified | 66 → 10 entries; header comment updated, section comment rules removed | Bibliography matches the cited works |
| `report/checks/claims.md` | modified | "Statements about other work" now lists the contribution sources and notes the dropped papers; `check` list reduced to the 6 remaining entries; title placeholder item removed | Claims register in step with the text |
| `report/report.pdf` | modified | Rebuilt: 12 pages | Deliverable |
| `.claude/plans/report-revision.md` | created | Plan of this revision with user decisions | Plan-before-execute rule |
| `CLAUDE.md` | modified | Notes "Report (9.1)": 12 pages, 10 references, layout | Keep project memory current |
| `worklog/025-2026-11-15-report-revision.md`, `worklog/README.md` | created / modified | This entry and its index row | Work log rule |

## Commits
`cdf1a4c` docs(report): add 12-page project report (draft 14 Nov, revised 15 Nov); tooling in `dd049a9`.

## Next
- User review of the rewritten text, especially the contributions and the gap prose.
- Verify the 6 bibliography entries still marked `check`.
- The poster (9.2) should use the same contribution wording.
