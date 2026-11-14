# Algorithm 1 — line-by-line check against the code (ROADMAP 9.6)

Checked 14 Nov 2026 against the code at the commit of this work. Source of the algorithm:
`report/figures/algorithm.tex`. "Line" = line number printed in the report.

| Line | Statement | Code | Matches |
|---|---|---|---|
| 1 | Whole flow files → train / val / test; Hangzhou unopened; capacities from road lengths, nothing fitted | `data/processed/splits.json`, `trafficrl.data.splits.flow_files` (hash check; test / held-out need `allow_final`); `envs/features.py` `FeatureBuilder` (`vehicle_space_m: 7.5`) | yes |
| 2 | `Train(K, d_e, s)` runs with the gate off | `agents/ugmarl/agent.py` `act(explore=True)`: ε-greedy on μ, gate never applied; validation episodes inside training are ungated (τ unknown → `gate_active` false) | yes |
| 3 | θ initialised with seed s, target copy, replay of 100 000 | `agents/core/qlearner.py` (`_clone`, `buffer_size: 100000` in `configs/base.yaml`) | yes |
| 4 | 100 episodes on the train flow | `train/loop.py` `train` (`train.episodes: 100`) | yes |
| 5 | ε linear 1.0 → 0.05 over the first 30 episodes | `LinearSchedule(eps_start, eps_end, eps_decay_frac * episodes)`, `eps.value(ep - 1)`; `eps_decay_frac: 0.3` (`configs/ugmarl.yaml`) | yes (ε = 0.05 from episode 31) |
| 6–7 | each ready intersection: attention → z, K heads → μ | `agents/ugmarl/network.py` `UGMARLNet`; `gate.py` `sigma_from_q` | yes |
| 8 | random phase w.p. ε, else argmax μ | `agent.py` `act`, lines "if self.rng.random() < self.epsilon" | yes |
| 9 | store transition with mask m_k ~ Bernoulli(0.8) | `qlearner.py` `observe` (`mask_prob: 0.8`, own RNG stream) | yes |
| 10 | one gradient step per 4 transitions after the first 2000, loss Eq. (3.5) | `observe` / `update` (`learning_starts: 2000`, `train_every: 4`), `masked_ensemble_loss` | yes |
| 11 | target copy every 1000 gradient steps | `_gradient_step` (`target_sync: 1000`) | yes |
| 13–15 | every 5th episode a greedy validation episode; keep the checkpoint with the lowest travel time | `train/loop.py`: `if ep % val_every == 0 or ep == episodes`, `best.pt` | yes (episode 100 is a multiple of 5) |
| 19 | grid over (K, d_e) on seed 0; smallest K, then d_e, within 1 s of the best | `configs/search/ugmarl.yaml` (`tie_tolerance_s: 1.0`, `tie_break`), `train/tune.py` `select_best` | yes — result K = 3, d_e = 128 |
| 20 | train the selected setting for every seed | `scripts/tune.py --config configs/ugmarl.yaml --seeds 0 1 2 3 4` | yes |
| 21 | σ of every decision of one clean, ungated validation episode; τ_s(p) = p-th percentile | `scripts/calibrate_gate.py` (reference pass), `gate.py` `percentile_thresholds` | yes |
| 22 | p* = smallest p within 1 s of the lowest mean travel time over seeds × five validation conditions | `configs/search/ugmarl_gate.yaml`, `gate.py` `select_percentile` | yes — p* = 80 |
| 23 | lock weights and thresholds by hash | `scripts/lock_models.py`, `models/manifest.json` (verified by `--final`) | yes |
| 24–26 | every test condition and held-out flow once, gated policy Eq. (3.7) | `scripts/robustness.py --final` with `configs/final_test.yaml`, `final_heldout.yaml`; `agent.py` `act` | yes |
| 27 | mean ± std and 95 % CI (Student t) over seeds | `utils/stats.py` `summarise`, `scripts/final_tables.py` | yes |

Not shown in the algorithm (stated in the report text instead): the learning rate, γ, batch
size and ε-decay were tuned for DQN / IDQN by random search and inherited (Section 3.6);
the baselines use the same `Train` procedure with K = 1 and no attention.

Difference from the earlier draft in `.claude/reference/architecture.md`: the draft's step 2
("build observations") is merged into line 1, and the lock step (23) was added.
