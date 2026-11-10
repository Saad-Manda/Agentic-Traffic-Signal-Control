# Data & Storage

No database. Data = public traffic datasets + simulator files + experiment outputs on disk.

## Datasets (selected 10 Nov 2026 — ROADMAP Phase 1, worklog/011)

| Name | Network | Source | Use |
|---|---|---|---|
| **Jinan** | 3×4 grid, 12 signalised int. | CoLight repo `04440b7`, `data/Jinan/3_4` (CityFlow format) | **primary**: train / val / test |
| **Hangzhou** | 4×4 grid, 16 signalised int. | CoLight repo `04440b7`, `data/Hangzhou/4_4` | **held-out generalisation city** (final evaluation only) |

Not selected: New York (48 / 196 int., too large for the laptop budget), RESCO Cologne /
Ingolstadt and LuST / MoST (SUMO-only, conversion cost), PeMS / UTD19 (detector counts, no
routes — not simulatable), synthetic grids (unit/sanity tests only, never reported).

Flow files: per-vehicle records (start time, route). Road-net files: intersections, lanes,
phases. Full description, sizes and data dictionary: `docs/dataset_card.md`.

## Split plan (ROADMAP 1.4) — `data/processed/splits.json`

| Split | Files | Vehicles | Used for |
|---|---|---|---|
| train | `Jinan/3_4/anon_3_4_jinan_real.json` | 6295 | RL training, normalisation stats, EDA (3.3) |
| val | `Jinan/3_4/anon_3_4_jinan_real_2000.json` | 4365 | hyperparameters, gate threshold τ, checkpoint selection |
| test | `Jinan/3_4/anon_3_4_jinan_real_2500.json` | 5494 | final evaluation, run once |
| heldout | `Hangzhou/4_4/anon_4_4_hangzhou_{real,real_5734,real_5816}.json` | 2983 / 6538 / 6984 | final generalisation evaluation, run once; zero-shot (no retraining) |

- Whole files per split (no resampling); each file pinned by SHA-256.
- All code gets flow files through `trafficrl.data.splits.flow_files(dataset, split)`, which
  checks the hashes; `tests/data/test_splits.py` asserts no file (by path or content hash)
  is in two splits and that Hangzhou has no train/val/test split.
- CityFlow generates vehicle IDs per file, so "no overlapping vehicle sets" is enforced as
  "no overlapping files"; content overlap between files was checked at selection
  (2–5 % coincidental (startTime, route) matches — independent scenarios; dataset card §5).
- Train = the largest Jinan file (most training signal); val / test = the other two.

Dataset choice and split are frozen together with the evaluation protocol (ROADMAP 3.5).

## Dataset card, data dictionary, audit (guidelines G3)

`docs/dataset_card.md` — per selected dataset:
- origin / URL / commit, **license or terms of use**, citation, collection period
- unit of observation (one vehicle trip), network size (intersections, lanes), duration,
  number of vehicles per flow file
- data dictionary: every roadnet / flow field with type and units (e.g. `startTime` s,
  `vehicle.maxSpeed` m/s, `length` m, `route` list of road IDs)

`docs/data_audit.md` (ROADMAP 3.0) — structural checks only, nothing fitted, counts + plots:
- missing fields; duplicate vehicle IDs
- impossible values: invalid / zero-length routes, start times outside the episode, bad speed
  or length values, routes using roads absent from the roadnet
- outliers: arrival bursts
- privacy: IDs anonymised, no personal data
- held-out city: audit only (no distribution plots before final testing)

## Layout

```
data/raw/<dataset>/         # untouched downloads
data/processed/<dataset>/   # split flow files + derived stats
results/<experiment>/       # metrics.csv, resolved config.yaml, env.json (versions + hardware), plots, checkpoints
models/                     # final selected checkpoints only
docs/                       # dataset_card.md, data_audit.md, evaluation_protocol.md, experiment_log.csv (tracked)
```

## No-data-leakage rules (mandatory)

1. **Split first.** Split each dataset's flows into train / validation / test (e.g. by
   time window or by separate flow files) *before* EDA-driven preprocessing decisions that
   produce learned parameters.
2. **Fit on train only.** Observation normalization statistics, any learned feature
   transforms, and replay data come from training flows only.
3. **Tune on validation only.** Hyperparameters, gate threshold τ, early stopping and
   checkpoint selection use validation flows; never test flows.
4. **Test once.** Test flows and the held-out generalization city are used only for final
   evaluation, identically for all agents.
5. **Same conditions for all agents.** Identical flows, seeds, episode lengths.
6. Split definitions live in `data/processed/splits.json` (hash-pinned);
   `tests/data/test_splits.py` asserts no file is shared between splits.
7. **Split type.** Separate flow files (or time windows) per split within the primary city; the
   held-out city is a **group-wise (location) split** — no data from it is used for training,
   tuning, normalisation or EDA plots.
8. **Freeze the protocol before RL training** (ROADMAP 3.5) in `docs/evaluation_protocol.md`:
   datasets, split definition, final metric set, seeds, episode length, Δt, success criteria
   (`PRD.md` §1c). Later changes only with a dated Decision Log entry giving the reason.
9. **EDA on the train split only** (ROADMAP 3.3); data-dependent modelling choices (Δt,
   normalisation, reward) come from training data.
10. **Test once.** Test flows are run once per agent / configuration after all choices are
    locked on validation; a bug-forced re-run is logged with its reason.
