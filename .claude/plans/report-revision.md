# Plan — Report revision (15 Nov 2026)

Source: user request 15 Nov. Report = `report/` (LaTeX), currently 15 pages, 38 cited references,
66 entries in `references.bib`. Numbers stay `\res{}` macros (generated, never typed).
Allowed page range stays 12–15 (`scripts/check_report.py`).

## Current state (observed in `report/report.pdf`)
- p. i title page uses a whole page; p. ii abstract is two paragraphs and ~half a page;
  p. iii ToC/LoF/LoT ~half empty.
- Every `\chapter` starts a new page, so Ch 1 ends ~40 % empty, Ch 3 ends after Algorithm 1
  with a gap, Ch 5 ends ~50 % empty before References.
- Contributions (Ch 1.6) describe *what we did*, not *what we changed relative to existing work*.
- Research gaps (Ch 2.2) are a 7-column table (Table 2.1) + 3 gaps + positioning paragraph.

## Changes

### 1. Abstract → one paragraph (`chapters/00-abstract.tex`)
Single paragraph, ~200–230 words, readable alone: problem (RL signals lack a reliability
signal and fail under input shift; Max-Pressure is stable but does not learn) → what was
studied (Fixed-Time, Max-Pressure, shared DQN, IDQN on Jinan / Hangzhou, CityFlow) → method
(per-intersection agents, neighbour attention, Q-ensemble σ, gate to Max-Pressure above a
validation-chosen τ) → protocol (selection on val only, test and second city run once, 5 seeds)
→ headline results (`\res` macros: clean tt vs MP/IDQN/DQN, noise 0.2, worst-case regret)
→ negative results (attention no gain; gate raises max wait under surge). Keywords line kept.

### 2. Contributions = changes over existing solutions (`chapters/01-introduction.tex` §1.6)
Rewrite each item as "existing X does A; we change it to B", with the evidence pointer:
1. **Ensemble disagreement used at execution, not for exploration/voting.** Bootstrapped/
   ensemble DQN in signal control (Tan et al.) uses heads for exploration; we use σ(s) to decide
   each action.
2. **Learned policy selects, Max-Pressure takes over (reverse of Com-MP).** Com-MP keeps MP as
   selector and lets RL adapt it; here RL selects and MP is the fallback, so learned performance
   is kept on clean data and MP's behaviour returns under shift.
3. **Simplex / uncertainty-fallback pattern moved from a single vehicle (Hoel et al.) to a
   multi-agent network**, per intersection, with a fallback that has a stability result
   (stated with its limits).
4. **Evaluation changed from clean, single-condition benchmarks to a frozen protocol** with
   detector noise, dropout, surge and an unseen city, ablations, hypotheses fixed before test.
5. **Attention explanations tested, not just shown** (CoLight shows weights; we run a
   faithfulness check) — finding: weights rank neighbours but neighbours are ~1/7 of influence.

### 3. Fewer references; research gap as prose (`chapters/02-literature.tex`, others)
User decision (15 Nov): **as few as possible.** Cut from 38 to **10** (15 Nov, second decision: 10, not 15) — only what the method
is built from, the baselines, the simulator/data, and the closest works the contribution claims
depend on. Ch 2 keeps its structure (conventional / single-agent / multi-agent / uncertainty /
explainability) but becomes shorter: sentences that exist only to describe a dropped paper are
removed; generic statements (e.g. "fixed plans are common in practice", "epistemic vs aleatoric",
"time-limit states bootstrap") stay without a citation. No claim about a specific dropped paper
remains. Delete Table 2.1; write §2.2 Research Gaps as 3 short prose paragraphs (one per gap) +
the "position relative to closest work" paragraph. Prune `references.bib` to the 10 cited keys.

Keep (10) — and why:
| Key | Needed for |
|---|---|
| varaiya2013max | Max-Pressure baseline + fallback, stability argument |
| mnih2015human | DQN |
| wei2019presslight | pressure reward |
| wei2019colight | neighbour attention; Jinan/Hangzhou data |
| zhang2019cityflow | simulator |
| ault2021reinforcement | benchmark fragility, tuned IDQN strong (motivation) |
| osband2016deep | shared-trunk Q-heads |
| tan2020multiagent | ensembles in signal control used for exploration (contribution 1) |
| zhu2026commp | closest: MP + RL hybrid (contribution 2) |
| hoel2020tactical | closest: ensemble + threshold + fallback (contribution 3) |

Drop (28): webster1958traffic, vanhasselt2016double, tampuu2017multiagent,
lakshminarayanan2017simple, jain2019attention (standard methods named without citation), wei2019survey, genders2016using, wei2018intellilight, zheng2019learning,
chu2020multi, chen2020toward, oroojlooy2020attendlight, mei2024libsignal, wei2021recent,
zhang2022expression, nguyen2025robustness, kendall2017uncertainties, ovadia2019can,
kahn2017uncertainty, lutjens2019safe, sha2001using, du2023safelight, qi2022random,
wiegreffe2019attention, lai2025llmlight, pardo2018time, velickovic2018graph,
osband2018randomized. The intro's "RL degrades under input shift" is then backed by ault2021 and
by our own result in §4.3 (learned baselines lose 345–415 s to MP under noise).

### 4. Humanize all prose (`/humanizer:humanizer`)
Run the humanizer skill chapter by chapter (abstract, Ch 1–5, figure/table captions,
Algorithm 1 comments untouched). Rules: meaning, numbers, `\res` macros, labels, `\cref`,
citations and equations unchanged; vary sentence length; remove stock phrasing, forced
triads, "not X but Y", bold run-in labels where they are only decoration. Diff reviewed per
chapter against `report/checks/claims.md` so no claim changes.

### 5. Layout: no whole title page, no empty space (`report/main.tex`)
- Replace `titlepage` with a compact title block at the top of page i (course, PS16, title,
  name / roll / email, department, institute, date — ~1/3 page), abstract directly below.
- ToC + LoF + LoT on the next page (required by the 12 Nov format), kept compact.
- Chapters no longer force a page break (`\titleclass{\chapter}{straight}` or redefine
  `\chapter` without `\clearpage`); keep chapter headings visually strong.
- Re-place floats (`[t]`/`[h]`) so no float page leaves gaps; References follow Ch 5 directly.
- Check result is still 12–15 pages; expected ~12–13 (user: let it shrink). If it falls
  below 12, stop and ask (do not pad).

## Verification
- `python scripts/report_tables.py` (unchanged numbers), build with latexmk, then
  `python scripts/check_report.py` (pages, undefined refs/cites, leftovers), `pytest
  tests/scripts/test_report_sources.py tests/eval/test_latex.py`.
- Render pages to PNG and inspect for gaps / overfull boxes.
- Update `report/checks/claims.md` for any reworded claim; worklog entry 025 + README row;
  CLAUDE.md note "Report (9.1)" (page count, reference count).

## Order
5 (layout, so pages are visible) → 1 → 2 → 3 → 4 (humanize last, on final content) → verify.
