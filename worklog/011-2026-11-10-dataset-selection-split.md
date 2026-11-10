# 011 — Phase 1: dataset selection, split plan, dataset card
**Date:** 2026-11-10   **Tag:** data, decision   **ROADMAP:** 1.1–1.6, 3.1, 3.2   **Status:** done

## What
Selected **Jinan** (3×4, 12 intersections) as the primary dataset and **Hangzhou** (4×4, 16
intersections) as the held-out generalisation city. Defined the split as whole flow files,
pinned by SHA-256 in `data/processed/splits.json`. Added a split loader with leakage and hash
checks, its tests, a stats script, and the dataset card with a full data dictionary.

## Why
Phase 1 deadline / Review 2 (10 Nov). Guidelines G3 (data collection, card, dictionary) and
G4 (split before processing, leakage prevention). The user approved the default proposal
(Jinan primary, Hangzhou held-out) after an explanation of the candidates.

Criteria (ROADMAP 1.2) and outcome:
- Real-world, CityFlow-native, laptop-sized, ≥ 3 flow files, used in CoLight / PressLight /
  MPLight → Jinan and Hangzhou are the only candidates meeting all of them.
- Rejected: New York (48 / 196 int., too slow for ensemble × ablations × seeds), RESCO
  Cologne / Ingolstadt and LuST / MoST (SUMO-only, conversion cost), PeMS / UTD19 (loop
  counts, no routes, not simulatable), synthetic grids (tests only).

Decisions made along the way:
- **Jinan split:** `real` (6295 veh, the largest) → train for the most training signal;
  `real_2000` (4365) → val; `real_2500` (5494) → test. This is the original ROADMAP proposal.
- **Hangzhou has 3 flow files, not 2** (`real_5734` was not in the ROADMAP table). All three
  are held out, which gives a broader zero-shot test: light (2983), medium (6538) and heavy
  (6984) demand.
- **One `splits.json`** for both datasets instead of one per dataset: one place to read the
  whole split. Nothing is materialised in `data/processed/` because the splits are whole files.
- **Leakage test is per file (path + SHA-256), not per vehicle ID.** CityFlow generates IDs
  per file, so ID overlap is meaningless. Hashing also catches a renamed copy of a file.
- **No license** in the CoLight repo → academic use only, not redistributed, cite CoLight.

## How
- Data was already downloaded on 28 Oct (worklog/007): sparse clone of
  `github.com/wingsweihua/colight` @ `04440b7` into gitignored `data/raw/colight_src/`.
- One-off survey script (scratchpad, not kept): sizes, field sets, route validity, records
  per file, and cross-file (startTime, route) overlap → 2–5 % coincidence between any two
  files of a city (chance level), so the files are independent scenarios.
  **Note for honesty:** that survey also printed 5-minute arrival counts for all six files,
  including test and Hangzhou. They were not used for any decision (the split equals the
  pre-registered ROADMAP proposal), and they are not reproduced in any document or plot.
- `src/trafficrl/data/splits.py`: `load_splits` (structure + disjointness), `verify_hashes`,
  `flow_files(dataset, split)` (the single entry point; held-out reachable only via
  `"heldout"`).
- `scripts/dataset_stats.py`: sizes and structure only (safe on the held-out city) →
  `docs/dataset_stats.json`; feeds the card.
- Inspected the phase definitions: 9 light phases = phase 0 (5 s transition, right turns
  only) + 8 green phases → agent action space will be the 8 green phases (Phase 4).

## Result
- `docs/dataset_card.md` complete (overview, source / commit / license / citation, sizes,
  split, data dictionary for flow and road-network fields with units, quality checks,
  limitations). Open item: collection dates are not in the release (**verify in the
  CoLight paper**).
- All 6 flow files: 0 invalid road references, 0 duplicate records, one vehicle profile,
  start times in [0, 3599] s.
- 39 tests pass (10 new in `tests/data/test_splits.py`); `ruff check` and `ruff format` clean.

## Files / commits
`data/processed/splits.json`, `src/trafficrl/data/{__init__,splits}.py`,
`tests/data/{__init__,test_splits}.py`, `scripts/dataset_stats.py`, `docs/dataset_card.md`,
`docs/dataset_stats.json`, `configs/base.yaml`, `.claude/reference/database.md`,
`ROADMAP.md`, `CLAUDE.md`, `README.md` · commit _pending_

## Next
- 3.0 data audit (`docs/data_audit.md`): Jinan train split with plots; Hangzhou structural
  counts only.
- 3.3 EDA notebook on the Jinan train split only.
- Verify the collection period in the CoLight paper and update card §2.
