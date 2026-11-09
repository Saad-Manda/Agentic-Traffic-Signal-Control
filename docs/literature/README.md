# Literature review — UG-MARL

ROADMAP Phase 2 (2.1–2.6) · Rubric R2 · Guideline G2 · Started 9 Nov 2026 · Plan:
`.claude/plans/literature-review.md` · Worklog: `worklog/010-2026-11-09-literature-review.md`
BibTeX for every citation: [`report/references.bib`](../../report/references.bib) (keys in
backticks in each file, e.g. `varaiya2013max`).

## Files

| File | ROADMAP | Covers | Read it for |
|---|---|---|---|
| [01-conventional-control.md](01-conventional-control.md) | 2.1 | Background: Webster, **Max-Pressure** (cited only); reviewed: Advanced-MP, Lyapunov back-pressure RL, Com-MP | What the fallback guarantees and what it does not (§2) |
| [02-rl-foundations.md](02-rl-foundations.md) | 2.2–2.3 (theory) | Q-learning, DQN family, IQL, CTDE (VDN/QMIX/MADDPG/MAPPO), attention/graph communication | Baseline theory; limitations of conventional agentic (RL) frameworks |
| [03-rl-for-traffic-signal-control.md](03-rl-for-traffic-signal-control.md) | 2.2, 2.3 | Genders, IntelliLight, FRAP, MA2C, PressLight, CoLight, MPLight, AttendLight, Advanced-XLight, ensemble-TSC, SafeLight, Com-MP, RESCO, LibSignal, T-REX | The lineage we compare against, design defaults, closest prior art |
| [04-uncertainty-and-safe-rl.md](04-uncertainty-and-safe-rl.md) | 2.4 | Epistemic vs aleatoric, deep ensembles, Bootstrapped DQN, randomized priors, SUNRISE/REDQ, uncertainty under shift, selective prediction, safe RL, shielding, Simplex, uncertainty-triggered fallback | Justifying the Q-ensemble and the gate |
| [05-llm-agents-and-explainability.md](05-llm-agents-and-explainability.md) | 2.5 | LLMLight; attention-as-explanation; SHAP; XRL | Short positioning notes and viva answers |
| [06-gap-analysis.md](06-gap-analysis.md) | 2.6 | **Gap table**, novelty claim, threats, action items, search method | Review 2 (10 Nov) |

## Scope rule (user, 9 Nov)

Only papers from **2016 onward** are reviewed. Five pre-2016 originals of methods we implement are
cited once as **background** and not reviewed: Webster 1958 (Fixed-Time), Varaiya 2013 (Max-Pressure),
Watkins 1992 (Q-learning), Mnih 2015 (DQN), Sha 2001 (Simplex). Older conventional and tabular-RL work
(SCOOT, SCATS, SOTL, MARLIN, …) is referred to through the 2019 survey `wei2019survey`.

## Verification legend (used in every file)

- ✓ bibliographic metadata (authors, venue, year, ID) verified online on 9 Nov 2026.
- † metadata from memory, well-known paper; check against DOI before the final report.
- *(unverified)* a number not checked against the paper's PDF. Do not put it in the report as-is.

Summaries are written from abstracts and prior knowledge (the user chose metadata-only
verification). Re-read the PDF for any paper whose numbers or equations go into the report.

## Ten things this review taught us

1. **Max-Pressure is the right fallback**: decentralised, parameter-free, same observations as
   our agents, and queue-stable under stated assumptions (Varaiya 2013).
2. **Its guarantee is narrower than the PRD implies**: bounded *mean queues*, not bounded
   per-vehicle wait, and it holds only if MP acts every slot. An uncertainty-only gate does not
   inherit it. A queue/Lyapunov trigger (σ > τ **or** V > V_max) restores a Simplex-style
   argument (01 §2, 04 §5.4).
3. **UG-MARL is a Simplex architecture** (Sha 2001) with an epistemic-uncertainty decision
   module. This is the cleanest framing for the report and viva.
4. **Ensemble spread in a Q-ensemble ≈ epistemic uncertainty**, because each head already averages traffic
   randomness (04 §1).
5. **Use randomized prior functions**, or the heads may agree on unseen states, exactly where the gate
   must fire (04 §2.3).
6. **Ensembles are the most robust uncertainty method under shift, but can still be confidently
   wrong** (Ovadia 2019). Report gate-firing rates under shift and test σ directly (AUROC).
7. **The external motivation exists**: RL beat MP on average but degraded 23–57% under incidents in
   T-REX (Nguyen 2025).
8. **Ensembles already appear in RL-TSC, but only for exploration or stability**. Execution-time gating
   to a guaranteed controller is the gap. The ensemble→threshold→fallback pattern is borrowed
   from driving and robotics and must be cited.
9. **Strong baselines matter**: a tuned independent DQN often beats fancier methods (RESCO), and
   state design matters more than architecture (Advanced-XLight). Use Double DQN and shared parameters
   everywhere, and consider Advanced-MP.
10. **Attention weights and gate logs are associations**, not causes. A cheap perturbation check
    makes the attention claim defensible.

Action items arising from these are tracked in [06 §4](06-gap-analysis.md#4-action-items-this-review-creates-for-the-project).
