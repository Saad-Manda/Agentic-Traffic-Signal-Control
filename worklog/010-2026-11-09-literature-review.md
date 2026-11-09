# 010 — Literature review (Phase 2) and draft gap table
**Date:** 2026-11-09   **Tag:** report, decision   **ROADMAP:** 2.1–2.6   **Status:** in-progress

## What
Wrote the Phase 2 literature review as six topic files plus an index under `docs/literature/`,
and a BibTeX file (`report/references.bib`, 63 entries after the post-2015 rule) covering every citation. The review
includes a draft gap table for Review 2 (10 Nov).

## Why
ROADMAP Phase 2, rubric R2, guideline G2. Phase 2 was done before Phase 1 (datasets) at the user's request,
so that the literature shapes the dataset choice and design. Scope decisions (user, 9 Nov): depth over count;
conventional TSC, RL foundations, RL-TSC, and uncertainty & safe RL covered in full; LLM agents and
explainability as short notes; per paper: method + equations, setup/results, limitations → gap,
implementation tips; **metadata-only verification** (faster). Numbers not checked against PDFs
are flagged *(unverified)*. **Post-2015 rule (user, later on 9 Nov):** only 2016+ papers are reviewed. Five
pre-2016 originals of implemented methods (Webster, Varaiya, Watkins, Mnih, Sha) stay as one-line background
citations. Ten others (SCOOT, SCATS, SOTL, Tassiulas, Tan, Wiering, Abdulhai, MARLIN, García, Altman) were removed
and referred to through the 2019 survey or the 2024 safe-RL review. File 01 was restructured: background (§0),
post-2015 MP work (§1: Advanced-MP, Lyapunov BP-RL, Com-MP), our MP-guarantee analysis (§2).

## How
- Plan: `.claude/plans/literature-review.md`.
- About 35 web searches (9 Nov) to verify authors, venues, years and DOIs/arXiv IDs, and targeted
  novelty searches for "uncertainty / ensemble / safe fallback + traffic signal control".
- Legend in every file: ✓ verified online · † from memory, check later.
- Consistency check: every citation key in `docs/literature/*.md` has a BibTeX entry and vice
  versa (`comm` over the grep'd keys: no differences).

## Result
- **Novelty search:** no prior TSC paper found that uses ensemble uncertainty **at execution** to
  gate between MARL and Max-Pressure. Closest work: multi-agent bootstrapped DQN for TSC (CCTA 2020,
  exploration only), RELight (2022, ensemble DDQN), SafeLight (AAAI 2023, collision safety),
  Com-MP (arXiv Oct 2026, RL *inside* MP with a stability claim). The ensemble→threshold→fallback
  pattern exists in robotics and driving (Kahn 2017, Lütjens 2019, Hoel 2020) and must be cited.
- **Findings that affect the design** (06 §4, action items A1–A9, *not yet decided*):
  - MP's theorem bounds mean queues, not per-vehicle max wait, so the PRD §1c "minimum acceptable
    outcome" wording overclaims (A1).
  - An uncertainty-only gate does not inherit MP stability. Adding a queue/Lyapunov trigger gives a
    Simplex-style argument (A2).
  - Use randomized prior functions so σ grows off-distribution (A3). Use Double DQN and shared
    parameters in all learners (A4). Consider Advanced-MP as a stronger baseline (A5).
  - Add a τ risk–coverage curve and a σ-AUROC diagnostic to the evaluation protocol (A6, A7).
- Best external motivation: T-REX (Nguyen et al. 2025), where IDQN beat MP by 5.62% normally but
  degraded 57.41% / 23.2% under incidents.

## Files / commits
`docs/literature/README.md`, `01`–`06`; `report/references.bib`;
`.claude/plans/literature-review.md`; `ROADMAP.md` (2.1–2.5 ticked, 2.6 draft ticked, Decision Log
row) · commits: not yet committed.

## Next
- Decide action items A1–A5 (Decision Log) before freezing the protocol (ROADMAP 3.5).
- Before the final report: check † metadata and *(unverified)* numbers against PDFs, and re-run the
  novelty search (2.6 final).
- Phase 1 dataset selection (due 10 Nov). The review supports Jinan/Hangzhou (CoLight release).
