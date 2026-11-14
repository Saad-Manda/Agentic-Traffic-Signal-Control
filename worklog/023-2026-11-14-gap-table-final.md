# 023 — Gap table finalised: verification against sources and novelty re-run

**Date:** 2026-11-14   **Tag:** docs, decision   **ROADMAP:** 2.6   **Status:** done

## What
Closed ROADMAP 2.6 "final": checked every row of the gap table (`docs/literature/06`) against the
paper's abstract and, where reachable, arXiv HTML full text; re-ran the novelty search; resolved
the *(unverified)* numbers in literature notes 01 and 03; added a condensed 12-row × 6-column
table for the report and a verification log.

## Why
The gap table is the evidence for the novelty claim (rubric R2, guideline G2). Its marks had been
scored from abstracts and memory; a wrong mark is an easy viva target. ROADMAP required the check
before the related-work chapter.

## How
- Web search (standard + extended) with the 9 Nov queries plus 2026-specific ones; candidate
  abstracts read (2602.07784, 2607.09899, 2412.16225, 2603.15283, T2-BAS-DQN).
- Each table row checked on arXiv abstract / HTML pages; ACM DL returned 403, so IntelliLight
  and PressLight were checked at abstract level only.

## Result
- Novelty holds: no paper found using execution-time ensemble uncertainty to defer to Max-Pressure.
  Closest: BCT-APLight (Bayesian critique → policy fine-tune), T2-BAS-DQN (always-on fuzzy
  correction); both added as rows. Com-MP (2610.31037) confirmed as concurrent work.
- Corrections: RELight's ensemble votes at execution (no uncertainty use); Nguyen 2025 shows
  degradation for independent/decentralised RL (Max-Pressure not named) → §2 paragraph
  reworded; CoLight dataset sizes are counts (12/16/196); FRAP invariance is to symmetries;
  SafeLight's strongest variant is an action filter; Hoel 2020 threshold detail is abstract-silent (†).
- UG-MARL safety cell ✔ → ◐ (gate raised max wait under the surge, 2162 vs 1430 s).
- Decision: no external ATT numbers quoted as comparable (MP on "Jinan" is 361 s in CoLight vs
  246–274 s in Advanced-MP — different flows/settings).
- Open: `bi2026t2bas` bib entry needs the full author list and exact title before citing.

## Files changed
| File | Action | What changed | Why |
|---|---|---|---|
| `docs/literature/06-gap-analysis.md` | modified | §1 rows: RELight cell, SafeLight "(action filter)", Hoel †, UG-MARL safety ◐; new rows BCT-APLight, T2-BAS-DQN; table note rewritten; new §1b condensed report table, §1c verification log + external-numbers decision; §2 Nguyen sentence; §3 τ sweep p = 50–70 and novelty re-run; §5 re-run + verification level | finalise 2.6 |
| `docs/literature/01-conventional-control.md` | modified | Advanced-MP result: *(numbers unverified)* → source table + "not quoted" | close unverified number |
| `docs/literature/03-rl-for-traffic-signal-control.md` | modified | PressLight and CoLight *(numbers unverified)* → not quoted; CoLight datasets as counts; FRAP invariance wording | verification corrections |
| `report/references.bib` | modified | added `duan2024bct`, `bi2026t2bas` (latter flagged: check authors) | new table rows need keys |
| `ROADMAP.md` | modified | ticked 2.6, its "final" sub-item and final-gate "Literature and gap"; Decision Log row 14 Nov | progress + decision record |
| `worklog/023-2026-11-14-gap-table-final.md` | created | this entry | mandatory work log |
| `worklog/README.md` | modified | row 023 | index |

## Commits
See git log for 14 Nov `docs(lit): …`.

## Next
9.5 architecture diagram and 9.6 pseudocode; then report Ch 2 uses §1b + §2 paragraph.
