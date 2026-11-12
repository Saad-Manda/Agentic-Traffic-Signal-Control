# 021 — Gap table: UG-MARL row and action items brought in line with the built system

**Date:** 2026-11-13 · **ROADMAP:** 2.6 (partial), 8.2 · **Type:** docs

## What
Updated the Review-2 gap analysis (`docs/literature/06-gap-analysis.md`) so that it describes
UG-MARL as built and evaluated in Phases 6–7: corrected the UG-MARL row, the table note and
novelty claim 3, and added a status (done / partly / dropped / open) to action items A1–A9.
Moved the attention faithfulness check (A8) into Phase 8.

## Why
The draft promised things that were never built: a queue / Lyapunov trigger (A2) that would
have turned the stability cell into ✔, a risk–coverage curve and σ-AUROC (A6), and a
faithfulness check (A8). Left as is, these claims would reach the related-work chapter, the
poster or the viva. User decision (13 Nov): keep A8 as a Phase 8 experiment; check the PDFs and
re-run the novelty search at the start of Phase 9 (by 15 Nov).

## How
Checked each action item against the code, plans and results: `.claude/plans/phase6-ugmarl.md`
(Bernoulli masks chosen over randomized priors; σ > τ gate only), `docs/methodology.md` §6
(mixed policy = sketch), `src/trafficrl/agents/baselines/idqn.py` (independent nets),
`docs/results/{gate_calibration.md, sigma_validity.csv, test_*.md, heldout_generalisation.md}`,
and `PRD.md` §1c. No code, experiment or result changed.

## Result
- UG-MARL row: guarantee **◐** (Max-Pressure stability only if the gate always fires; mixed
  policy = sketch); explainability = gate log, σ, attention weights (faithfulness check: Phase 8).
- A1 partly, A2 dropped, A3 dropped, A4 done (IDQN independent by design), A5 dropped,
  A6 partly, A7 done, A8 open → 8.2, A9 done. Dropped items become report limitations.
- Still open in 2.6: PDF check of ✔/◐ cells and *(unverified)* numbers; novelty re-search.
- `PRD.md` §1c "minimum acceptable outcome" still reads "the gate should guarantee this". Not
  edited because §1c was fixed before the test runs; the report must state it as an empirical check.

## Files changed
| File | Action | What changed | Why |
|---|---|---|---|
| `docs/literature/06-gap-analysis.md` | modified | Header line (update note); UG-MARL row: guarantee and explainability cells; table note (no queue trigger, PDF check timing); §3 novelty claim 3 (τ-percentile sweep + Spearman σ check instead of risk–coverage / AUROC); §4 intro line + new "Status (13 Nov)" column for A1–A9 | Align the gap analysis with the built system so no unbuilt feature is claimed |
| `ROADMAP.md` | modified | 2.6: new ticked sub-item for this update; the "final" sub-item now says it happens at the start of Phase 9, by 15 Nov; 8.2: new sub-item "Attention faithfulness check (A8)" | Record progress and move A8 into Phase 8 |
| `worklog/021-2026-11-13-gap-table-update.md` | created | This entry | Work-log rule |
| `worklog/README.md` | modified | Row 021 | Index |

## Commits
`7409220` docs(lit): align gap table with the built UG-MARL, status of A1-A9

## Next
Phase 8 (incl. A8). At the start of Phase 9: PDF check + novelty re-search → tick 2.6.
