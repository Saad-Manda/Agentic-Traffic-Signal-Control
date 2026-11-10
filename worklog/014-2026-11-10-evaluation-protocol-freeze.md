# 014 — Evaluation protocol frozen (ROADMAP 3.5)
**Date:** 2026-11-10   **Tag:** decision, docs   **ROADMAP:** 3.5, 5.0   **Status:** done

## What
Froze the evaluation protocol in `docs/evaluation_protocol.md` (Step 8 of
`.claude/plans/phase3-data-audit-eda-features.md`), completing Phase 3. The user chose
**Δt = 15 s** and **reward = −|pressure|**. Config, ROADMAP, PRD, references and the explainer
now state the frozen values.

## Why
Guideline G4 / ROADMAP 3.5: the protocol must be fixed before any RL training (5.0 gate), so
that model selection cannot steer the metrics, seeds or splits.

| Decision | Chosen | Rejected, and why |
|---|---|---|
| Δt | 15 s | 10 s (literature convention, but Max-Pressure is 8 % worse on the train flow: 375 vs 344 s); 20 s (no gain, 347 s, fewer decisions); tuning Δt on validation (adds a hyperparameter shared by every agent) |
| Reward | −\|Pᵢ\| | −Σ queue (does not match the Max-Pressure fallback's objective; the gate and the policy would optimise different things) |
| Deterministic baselines | run on every seed, report std = 0, CI marked "deterministic"; H1 needs the learned agent's whole CI below the baseline value | dropping the CI requirement for them (inconsistent table); single run (guidelines forbid single runs) |

Known costs, recorded in the protocol: absolute numbers are not directly comparable with
10 s papers; |P| ≈ 0 when both sides of an intersection are full (limitation).

## How
Wrote the protocol from the values proposed in plan Step 8, with the two user choices; defined
every secondary metric and the robustness perturbations (noise, dropout, surge) precisely so
Phase 4.3 implements them as written. Checked that Δt and the reward were chosen from the
train flow only (EDA §9, §10). `pytest` (92 passed) and `ruff check src tests scripts` clean.

## Result
Protocol frozen 10 Nov 2026. ROADMAP 3.5 and the 5.0 training gate ticked. Phase 4 can start
(env wrapper at Δt = 15 s; Max-Pressure must reproduce ≈ 344 s on the train flow).

## Files changed
| File | Action | What changed | Why |
|------|--------|--------------|-----|
| `docs/evaluation_protocol.md` | created | §1 datasets/split, §2 simulation (3600 s, Δt 15 s, 5 s yellow, min-green = Δt), §3 MDP (44-value state, action, reward −\|P\|), §4 agents, §5 metric definitions, §6 experiments × seeds × robustness definitions, §7 reporting (incl. deterministic baselines), §8 success criteria, §9 change table | ROADMAP 3.5 deliverable |
| `configs/base.yaml` | modified | Header says frozen; `sim.decision_interval` 10 → 15; new `sim.reward: pressure`; PROVISIONAL markers removed | Config must match the protocol |
| `ROADMAP.md` | modified | 3.5 and 5.0 ticked; Decision Log row "Evaluation protocol frozen" | Progress and decision record |
| `PRD.md` | modified | §1a names Jinan / Hangzhou; §5 metric set marked frozen; §9 reward and Δt questions resolved, τ method left open | Remove stale "pending/provisional" text |
| `.claude/reference/architecture.md` | modified | State, reward, Δt marked frozen; diagram note "fit on train only" → capacity scaling | Sync with the protocol |
| `CLAUDE.md` | modified | MDP line (Δt 15 s, frozen); Notes entry replaced by the frozen summary; key file `docs/evaluation_protocol.md`; explainer sync note | Project constitution in sync |
| `docs/how_it_works.md` | modified | Provisional note rewritten; Δt 10 → 15 s in the loop box, glossary and experience count (240 decisions, ≈ 2,900 experiences) | Explainer in sync with the MDP |
| `.claude/plans/phase3-data-audit-eda-features.md` | modified | Status line and Step 8 heading: executed with the user's choices | Plan execution record |
| `worklog/014-2026-11-10-evaluation-protocol-freeze.md`, `worklog/README.md` | created / modified | This entry; index row | Work log rule |

## Commits
- `ac449b7` docs(eval): freeze evaluation protocol (Δt 15 s, reward −pressure)

## Next
- Phase 4: env wrapper (4.1) at Δt = 15 s, Fixed-Time / Max-Pressure agents, robustness hooks
  (4.3) exactly as defined in protocol §6.
- Renumber one of the two 012 worklog entries (confirm with the user).
