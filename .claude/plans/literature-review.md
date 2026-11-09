# Plan — Literature review (ROADMAP Phase 2)

## Goal
Write the Phase 2 literature review (ROADMAP 2.1–2.6), covering rubric R2 and guideline step G2. It must explain
how each line of prior work operates, its key equations, setup/results, limitations and implementation
tips, and end in a gap table that motivates UG-MARL. Draft gap table due for Review 2 (10 Nov).

## Decisions (user, 9 Nov)
- Depth over count: write what is relevant per paper, not fixed-length blurbs.
- Scope: conventional TSC, RL foundations (single + multi-agent), RL for TSC, uncertainty & safe
  RL. LLM agents and explainability get **short notes only**.
- Per key paper: method + key equations · results & setup · limitations → our gap ·
  implementation tips.
- Verification: **metadata verified online** (authors, venue, year, DOI/arXiv); summaries from
  existing knowledge. Numbers not checked against the PDF are marked *(unverified)*.
- Output: several topic files under `docs/literature/` + `report/references.bib`.
- **Only papers from 2016 onward are reviewed**; pre-2016 originals of implemented methods are background citations only.

## Files
| File | Purpose |
|---|---|
| `docs/literature/README.md` | Index, reading order, verification legend, key takeaways |
| `docs/literature/01-conventional-control.md` | Webster, SOTL, SCOOT/SCATS, back-pressure, Max-Pressure (2.1) |
| `docs/literature/02-rl-foundations.md` | Q-learning → DQN family; MARL (IQL, CTDE, VDN/QMIX, MADDPG, MAPPO); attention/graph comms |
| `docs/literature/03-rl-for-traffic-signal-control.md` | Single-agent (2.2) + multi-agent (2.3) RL-TSC, benchmarks |
| `docs/literature/04-uncertainty-and-safe-rl.md` | Ensembles, bootstrapping, priors, shift; safe RL, shielding, Simplex (2.4) |
| `docs/literature/05-llm-agents-and-explainability.md` | Short notes (2.5 + explainability) |
| `docs/literature/06-gap-analysis.md` | Gap table, closest prior work, defensible novelty claim (2.6) |
| `report/references.bib` | BibTeX for every cited work |

## Verification
- Every paper cited in the files has a BibTeX entry, and every BibTeX key is cited in at least one file
  (checked with a grep script).
- ROADMAP 2.1–2.6 ticked only where content exists; worklog entry 010 added.

## Risks
- Novelty: individual components exist (ensembles for TSC, MP + RL hybrids). Claim the
  *combination + gate + evaluation*, not the components.
- Max-Pressure's stability proof does not automatically transfer to an intermittently switched
  controller, so this is recorded as an open design point in 06 and 01.
