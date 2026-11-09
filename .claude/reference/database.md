# Data & Storage

No database. Data = public traffic datasets + simulator files + experiment outputs on disk.

## Datasets (TODO: download and record exact sources/versions)

| Name | Network | Source | Use |
|---|---|---|---|
| Jinan | 3×4 grid, 12 intersections | CoLight / PressLight releases (CityFlow format; SUMO conversions via LibSignal) | train/val/test |
| Hangzhou | 4×4 grid, 16 intersections | same | train/val/test and/or held-out generalization city |
| New York / RESCO Cologne, Ingolstadt | optional | — | optional extra eval |

Flow files: per-vehicle records (start time, route). Road-net files: intersections, lanes, phases.

Dataset selection is **not frozen yet** (due 10 Nov, ROADMAP Phase 1); it is frozen with the
evaluation protocol (ROADMAP 3.5).

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
6. Record split definitions in `data/processed/<dataset>/splits.json` and add a test that
   asserts train/val/test vehicle sets do not overlap.
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
