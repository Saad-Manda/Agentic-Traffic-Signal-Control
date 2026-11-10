# 012 — Explainer: data, simulator and training (`docs/how_it_works.md`)
**Date:** 2026-11-10   **Tag:** docs   **ROADMAP:** 9.x (report/viva prep), G12   **Status:** done

## What
Wrote `docs/how_it_works.md`: a plain-language explanation of what the dataset contains,
what CityFlow does with it, how RL training generates its own experience, the meaning of
train / val / test / held-out, and how the uncertainty gate works, plus a glossary.

## Why
The user asked, after Phase 1, for a clear explanation of how the data looks, how the simulator
uses it, and how a model is trained from data that has no labels, and then asked for it to
be documented properly. The same explanation is needed for the report's method section and
the viva.

## How
- Built from a chat explanation, rewritten as a structured doc with real examples: the
  first three records of the Jinan training file, decoded against the road network.
- Verified the road-ID convention (`road_<x>_<y>_<dir>`, 0 = E, 1 = N, 2 = W, 3 = S) and the
  grid layout (4 columns × 3 rows; 400 m E–W, 800 m N–S spacing) from `roadnet_3_4.json`.
- **Correction:** the chat explanation said the example car crosses two intersections. It
  crosses **four**, (1,2) → (2,2) → left at (3,2) → (3,3). The doc has the correct version.
- Δt, reward and episode count are marked *provisional* until the protocol freeze (3.5).
- Linked from `README.md` (Data section) and `CLAUDE.md` (Key Files).

## Result
`docs/how_it_works.md` (7 sections + glossary). No code changes.

## Files / commits
`docs/how_it_works.md`, `README.md`, `CLAUDE.md`, `worklog/README.md` · commit _pending_

## Next
Update the doc when the MDP (state, reward, Δt) is frozen in 3.5. Reuse sections 1, 4 and 6
for the report's method section and slides.
