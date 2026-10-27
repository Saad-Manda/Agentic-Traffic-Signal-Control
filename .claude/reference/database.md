# Data & Storage

No database. Data = public traffic datasets + simulator files + experiment outputs on disk.

## Datasets (TODO: download and record exact sources/versions)

| Name | Network | Source | Use |
|---|---|---|---|
| Jinan | 3×4 grid, 12 intersections | CoLight / PressLight releases (CityFlow format; SUMO conversions via LibSignal) | train/val/test |
| Hangzhou | 4×4 grid, 16 intersections | same | train/val/test and/or held-out generalization city |
| New York / RESCO Cologne, Ingolstadt | optional | — | optional extra eval |

Flow files: per-vehicle records (start time, route). Road-net files: intersections, lanes, phases.

## Layout

```
data/raw/<dataset>/         # untouched downloads
data/processed/<dataset>/   # split flow files + derived stats
results/<experiment>/       # metrics.csv, resolved config.yaml, plots, checkpoints
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
