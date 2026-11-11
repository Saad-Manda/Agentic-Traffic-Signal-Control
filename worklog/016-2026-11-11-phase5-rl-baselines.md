# 016 — Phase 5: RL baselines (DQN, IDQN), training and tuning
**Date:** 2026-11-11   **Tag:** code, experiment, decision   **ROADMAP:** 5.1–5.5   **Status:** done

## What
Executed `.claude/plans/phase5-rl-baselines.md`. The work added:
- a Q-learning core shared by every learned agent (Double DQN, SMDP target, grouped replay,
  networks with a K-head axis);
- the two RL baselines PS16 requires: a **shared-parameter DQN** (single-agent, no
  coordination) and **IDQN** (multi-agent, independent);
- a training loop that selects checkpoints on validation, and random-search tuning;
- 5-seed final runs, the validation table and the learning-curve figure.

All numbers are **dev numbers on WSL**. The reported numbers come from native Ubuntu in
Phase 7 (Decision Log, 11 Nov).

## Why
ROADMAP Phase 5 (R4, G7, G8). PS16 asks for conventional, single-agent and multi-agent
baselines before the proposed system. UG-MARL (Phase 6) reuses the same core, so the
comparison isolates attention, ensemble and gate.

Decisions (user, 11 Nov; Decision Log rows U1–U12, plan section "User decisions"):

| Decision | Chosen | Rejected, and why |
|---|---|---|
| Learner | Double DQN + Huber + grad clip 10, hard target sync | dueling / PER / n-step (extra mechanisms would blur the UG-MARL ablation) |
| Discount | SMDP γ^(duration/Δt); always bootstrap (episode end = time limit) | plain γ (15 s and 20 s transitions discounted alike); cutting the target at the episode end (biased: traffic does not end) |
| Reward scale | fixed 0.01, training only | normalising by running statistics (fitted; changes during training) |
| Budget | 100 episodes, greedy val every 5, best checkpoint on val | early stopping (one more knob) |
| Tuning | random search, 8 trials per agent, seed 0, same space for both; then seeds 0–4 | grid (36 points × 2 agents, too slow); tuning on 5 seeds (5× cost) |
| Network | MLP 44→128→128→8, fixed | tuning the size (not in the PS16 baseline scope) |
| Device | CPU (benchmark, step 6) | CUDA: slower for these small MLPs, see Result |
| Where | WSL now (dev), native Ubuntu for reported numbers (Phase 7) | reporting WSL numbers (platform not yet validated, 0.1b) |

## How
Plan steps 1–13; one commit per step group (see Commits).

- **Core** (`agents/core/`). `ReplayBuffer` holds G groups (1 for DQN, 12 for IDQN).
  `MLPQNet` / `StackedMLPQNet` output (G, B, K, A); `StackedMLPQNet` evaluates the 12 IDQN
  networks with one `baddbmm`. `QLearner` does ε-greedy act, observe, update every 4
  transitions after 2000 warm-up transitions, target sync every 1000 gradient steps, and
  save/load. Target: y = 0.01·r + γ^(d/15) · Q_target(s′, argmax_a Q_online(s′, a)).
- **Leakage guard.** `run_episode` calls `agent.observe`, so the validation episodes inside
  training would have pushed validation transitions into replay. `QLearner.observe` now
  stores only while `agent.training` is True; tested.
- **Training** (`train/loop.py`, `scripts/train.py`). Training uses the train flow only;
  `train.py` exits with code 2 on any other split. A greedy val episode runs every 5
  episodes and keeps `best.pt`. Outputs per run: CSV logs, TensorBoard, resolved config,
  `env.json`, `summary.json` and one experiment-log row.
- **Tuning** (`train/tune.py`, `scripts/tune.py`). The search space
  (`configs/search/rl_baselines.yaml`: lr {1e-4, 3e-4, 1e-3}, γ {0.8, 0.9, 0.95}, batch
  {64, 128}, ε-decay fraction {0.3, 0.5}; sampler seed 2026; rule: lowest best-val travel
  time, ties within 1 s → lower lr) was committed in 71b9b48, before the first trial. Trials
  run as parallel processes with 1 thread each.

Runs:
```bash
python scripts/train.py --config configs/dqn.yaml  --run-name dqn_sanity_s0   # defaults
python scripts/train.py --config configs/idqn.yaml --run-name idqn_sanity_s0
python scripts/tune.py --search configs/search/rl_baselines.yaml --agent dqn  --workers 4
python scripts/tune.py --search configs/search/rl_baselines.yaml --agent idqn --workers 4
python scripts/tune.py --config configs/dqn.yaml  --seeds 0 1 2 3 4 --workers 2 --tag final
python scripts/tune.py --config configs/idqn.yaml --seeds 0 1 2 3 4 --workers 4 --tag final
python scripts/evaluate.py --config configs/eval_rl_baselines.yaml --agent dqn  --split val \
    --checkpoint 'results/train/dqn_final_s{seed}/checkpoints/best.pt'      # same for idqn
python scripts/val_table.py && python scripts/plot_training.py
```

## Result
- `pytest` 159 passed (fast); `ruff check src tests scripts` clean.
- **Device benchmark** (3 train episodes + 1 val, seed 0). CPU: DQN 25.4 s, IDQN 24.8 s.
  CUDA (RTX 4050): DQN 28.6 s, IDQN 36.2 s. CityFlow runs on the CPU only, and the networks
  are 23k parameters with batch 64, so GPU launch and transfer overhead outweighs the compute.
  5 parallel CPU runs take 36–43 s each (≈ 3.4× throughput). Same seed → identical weights.
  CPU and CUDA give different IDQN numbers, so determinism is per device. Re-benchmark for
  UG-MARL (K heads + attention) in Phase 6.
- **Sanity runs** (seed 0, defaults, 100 episodes; dev runs, not tuning trials). Best val:
  DQN 302.4 s @ ep 100, IDQN 320.5 s @ ep 90. Both beat Fixed-Time (434.84 s) by ep 25–30
  and Max-Pressure (326.11 s) by the end.
- **Tuning** (seed 0, best val avg travel time, s):

| Trial | lr | γ | batch | ε-decay | DQN | IDQN |
|---|---|---|---|---|---|---|
| t0 | 1e-3 | 0.8 | 64 | 0.3 | 308.52 | 335.56 |
| t1 | 1e-4 | 0.9 | 64 | 0.5 | 301.25 | 314.55 |
| t2 | 3e-4 | 0.8 | 64 | 0.5 | 302.49 | 320.04 |
| t3 | 1e-4 | 0.8 | 128 | 0.3 | 300.64 | 315.04 |
| t4 | 3e-4 | 0.95 | 64 | 0.3 | 303.17 | 324.65 |
| t5 | 3e-4 | 0.8 | 64 | 0.3 | 303.89 | 320.96 |
| **t6** | **1e-4** | **0.8** | **64** | **0.3** | **298.86** | **310.42** |
| t7 | 3e-4 | 0.8 | 128 | 0.5 | 304.70 | 325.97 |

  The same trial wins for both agents. The DQN spread (≈ 10 s) is small, and seed noise
  could account for a few seconds of it (the final 5-seed std is ≈ 1 s). The IDQN spread is
  larger (25 s), and lr 1e-3 is clearly worst for both. Each IDQN trial took 28–43 min
  (DQN 13–16 min).
- **Validation, 5 seeds** (Jinan `real_2000`, 3600 s; seed s evaluated with the checkpoint
  trained with seed s; table `docs/results/val_phase5.md`):

| Model | Avg travel time (s) | Avg waiting (s) | Max per-vehicle wait (s) | Throughput (veh) | Latency (ms) | Params |
|---|---|---|---|---|---|---|
| Fixed-Time | 434.8 (det.) | 187.8 | 1124 | 3991 | < 0.01 | 0 |
| Max-Pressure | 326.1 (det.) | 83.4 | 1334 | 4106 | 0.13 | 0 |
| DQN (shared) | **299.8 ± 1.0** [298.5, 301.1] | 55.6 ± 1.2 | 637 ± 47 | 4129 ± 4 | 1.27 ± 0.24 | 23,304 |
| IDQN | 312.7 ± 2.5 [309.7, 315.8] | 68.2 ± 2.5 | 717 ± 135 | 4117 ± 9 | 0.62 ± 0.07 | 279,648 |

  Per-seed best val (training) = evaluation value, to the printed precision: DQN 298.86 /
  299.43 / 299.51 / 301.58 / 299.46 s; IDQN 310.42 / 313.01 / 314.81 / 315.37 / 310.02 s.
  So evaluation reproduces training exactly (deterministic CPU).

  **Reading.**
  - Both RL baselines have lower average travel time than Max-Pressure on val: DQN by
    8.1 % and IDQN by 4.1 %. Neither 95 % CI contains 326.1 s. Both beat Fixed-Time by
    28–31 %. These are validation scores after selection on validation, so the gap to
    Max-Pressure may shrink on test. No claim until Phase 7.
  - DQN beats IDQN by 12.9 s, and the CIs do not overlap. A plausible cause (not tested) is
    data per parameter. Within the same 100 episodes, the shared network learns from all
    12 intersections' transitions, while each IDQN network sees 1/12 of them and the agent
    has 12× the parameters. IDQN is also less stable across seeds (std 2.5 vs 1.0 s).
  - Worst case: both RL agents roughly halve the max per-vehicle wait of Max-Pressure (637 /
    717 vs 1334 s), and are also below Fixed-Time (1124 s). Max-Pressure starves
    low-pressure approaches (worklog/015); the learned policies do so less on this flow. The
    seed spread is wide (IDQN ± 135 s). Relevant to H2 / fairness (9.8).
  - The latency difference is not meaningful. DQN was evaluated while 4 IDQN trainings were
    running, and IDQN on an idle machine. Both are ≈ 1 ms against a 15 s decision interval.
  - For H1 (PRD §1b), UG-MARL must beat IDQN (312.7 s) and Max-Pressure. DQN (299.8 s) is
    the stronger RL baseline and the more demanding comparison.

- **Figure** `docs/figures/training/learning_curves.png` (caption in `.md` next to it).
  Mean over 5 seeds with a 95 % CI band; the right panel zooms on episodes ≥ 30.
  DQN crosses Max-Pressure at about episode 25–30 and IDQN at
  about episode 40. IDQN starts worse: its 12 networks each get 1/12 of the early
  experience. DQN keeps improving slowly to episode 100, with its best mean at the last
  evaluation, so the fixed budget may cut it short. IDQN flattens after about episode 70
  (best mean 313.6 s at ep 85, last 315.6 s). The seed bands narrow to a few seconds after
  episode 65.
- **Caveats.** Validation chose both the hyperparameters and the checkpoint, so these
  scores are optimistic; the test flow (Phase 7) is the unbiased number. Each run trains on
  one 1-hour flow.
- Rows added to `docs/experiment_log.csv`: 2 smoke, 2 sanity, 16 tuning trials, 6 tune-smoke
  rows (annotated "SMOKE TEST … not a tuning trial"), 10 final runs, 10 evaluation rows
  (5 per agent). No test or held-out flow was opened: `make_env` is never called with
  `allow_final` outside `evaluate.py --final`.
- **Failures / dead ends.**
  - The first DQN finals launch failed with exit 127. `source activate && (A) & (B) &`
    activated the venv only for job A; relaunched with `.venv/bin/python`.
  - The tune smoke test found a `train.py` print crash for results dirs outside the repo
    (fixed).
  - A background status of "failed" earlier was only the `tail` exit code on several files;
    the runs had completed.

## Files changed
| File | Action | What changed | Why |
|------|--------|--------------|-----|
| `src/trafficrl/agents/core/__init__.py` | created | exports `ReplayBuffer`, `MLPQNet`, `StackedMLPQNet`, `n_params`, `LocalEncoder`, `LinearSchedule`, `QLearner` | one import path for DQN, IDQN, UG-MARL |
| `src/trafficrl/agents/core/replay.py` | created | `ReplayBuffer(groups, capacity, state_shape, rng)`: per-group ring buffers, `add`, seeded `sample` → (G, B, …), stores `duration_s` | IDQN needs one buffer per intersection; SMDP target needs durations (D3) |
| `src/trafficrl/agents/core/networks.py` | created | `MLPQNet` (heads axis K), `StackedMLPQNet` (G independent MLPs via `baddbmm`, `load_from`), `n_params` | shared vs independent nets with one code path; K heads for Phase 6 (D4) |
| `src/trafficrl/agents/core/encoders.py` | created | `LocalEncoder.encode_all` / `encode_transition` (local 44-value state) | pluggable encoder; attention replaces it in Phase 6 |
| `src/trafficrl/agents/core/schedules.py` | created | `LinearSchedule(start, end, duration).value(t)` | ε schedule |
| `src/trafficrl/agents/core/qlearner.py` | created | `QLearner`: `DEFAULTS`, ε-greedy `act`, `observe` (only while training), `update` (train_every / learning_starts), `td_target` (Double DQN, γ^(d/Δt), 0.01 reward scale, always bootstrap), Huber + grad clip, target sync, `save` / `load`, seeded RNGs | D5 |
| `src/trafficrl/agents/baselines/dqn.py` | created | `DQNAgent` (name `dqn`, one shared MLP, 1 buffer group) | ROADMAP 5.2 |
| `src/trafficrl/agents/baselines/idqn.py` | created | `IDQNAgent` (name `idqn`, `StackedMLPQNet` with 12 nets, 12 buffer groups) | ROADMAP 5.3 |
| `src/trafficrl/agents/__init__.py` | modified | `AGENTS` + `dqn`, `idqn`; `LEARNED_AGENTS`; `make_agent` merges `cfg["train"]` and passes `seed` | one construction path for train and evaluate |
| `src/trafficrl/eval/runner.py` | modified | `resolve_checkpoint` (`{seed}` template); checkpoint path in the logged hyperparameters | seed s evaluated with the checkpoint of seed s |
| `scripts/evaluate.py` | modified | `--checkpoint` is a string template with help text | as above |
| `src/trafficrl/train/__init__.py`, `train/loop.py` | created | `TrainSplitError`, `train_episode`, `train` (val checkpointing, CSV / TB / summary / experiment-log row) | ROADMAP 5.4 (D7) |
| `src/trafficrl/train/tune.py` | created | `load_search`, `sample_trials`, `overrides`, `select_best`, `run_parallel` (1 thread per process), `train_argv`, `collect` | ROADMAP 5.5 (D8) |
| `scripts/train.py`, `scripts/tune.py` | created | CLIs for one run / a study / a seed set (`trials.csv`, `best.json`, `seeds.csv`); exit 2 on non-train split | every result comes from a script |
| `src/trafficrl/utils/config.py` | modified | public `read_yaml(path)` | load the search file without the private parser |
| `src/trafficrl/utils/plotting.py` | modified | `CONTROLLER_COLOR` / `LABEL` for `dqn` (aqua), `idqn` (yellow) + comment on the validated order | consistent colours across result figures |
| `src/trafficrl/eval/report.py` | created | `latest_eval_run`, `training_summaries`, `val_curves`, `fmt`, `metric_stats` | shared by the table and figure scripts |
| `scripts/val_table.py` | created | writes `docs/results/val_phase5.{csv,md}` (mean ± std, 95 % CI, training time with a load caveat, params) | plan D10 |
| `scripts/plot_training.py` | created | two-panel learning-curve figure (full + zoom ≥ ep 30) + caption `.md` | plan D10 |
| `configs/base.yaml` | modified | `agents.dqn/idqn.hidden`; `train:` block (episodes, val_every, device cpu + benchmark comment, torch_threads, reward_scale, lr, γ, batch, buffer, learning_starts, train_every, target_sync, grad_clip, ε schedule) | all training knobs in config |
| `configs/dqn.yaml`, `configs/idqn.yaml` | created / modified | per-agent configs; tuned lr 1e-4, γ 0.8, batch 64, ε-decay 0.3 (trial t6) with comments | ROADMAP 5.2, 5.3, 5.5 |
| `configs/search/rl_baselines.yaml` | created | search space, sampler seed, 8 trials, selection rule (committed before trials) | guidelines: search space before tuning |
| `configs/eval_rl_baselines.yaml` | created | val evaluation, seeds 0–4, track full | step 12 |
| `.gitignore` | modified | `results/` → `/results/` (comment line above) | the pattern also ignored `docs/results/` |
| `docs/results/val_phase5.{csv,md}` | created (generated) | Phase 5 validation table | review deliverable |
| `docs/figures/training/learning_curves.{png,md}` | created (generated) | learning curves + caption | review deliverable |
| `tests/agents/test_core.py` | created | replay wrap / per-group / seeded sampling, MLP shapes and params, stacked = loop of MLPs, schedule, Double-DQN SMDP target by hand, observe only while training, bootstrap at episode end, target sync, ε actions | verifies D3–D5 |
| `tests/agents/test_dqn_idqn.py` | created | sharing vs independence (gradient isolation), per-intersection buffers, save / load round trip, determinism, `make_agent` reads `train:` | verifies D6 |
| `tests/agents/test_interface.py` | modified | `update() == {}` only for rule agents; unknown agent `ppo` | learned agents return losses |
| `tests/eval/test_evaluate.py` | modified | `{seed}` checkpoint template test | `resolve_checkpoint` |
| `tests/train/__init__.py`, `tests/train/test_loop.py` | created | split refusal (+ exit code 2), outputs and log row, no val experience in the buffer, same seed → same weights | verifies D7 + leakage guard |
| `tests/train/test_tune.py` | created | sampler deterministic / unique / in-space, selection rule and tie rule | verifies D8 |
| `tests/eval/test_report.py` | created | report helpers on hand-made run dirs | verifies D10 helpers |
| `docs/experiment_log.csv` | modified | rows listed in Result | guidelines: every run logged |
| `docs/evaluation_protocol.md` | modified | §9 clarification: reward scale 0.01, SMDP discount, truncation bootstrap (no frozen value changed) | protocol record |
| `docs/how_it_works.md` | modified | RL training section: Double-DQN SMDP target, 100 episodes, val checkpointing, DQN vs IDQN table | keep the explainer in sync |
| `ROADMAP.md` | modified | 2 Decision Log rows (RL design; WSL = dev numbers); 5.1–5.5 ticked with results | progress + decisions |
| `.claude/plans/phase5-rl-baselines.md` | created / modified | plan; steps ticked; "Deviations during execution" | plan record |
| `.claude/reference/api.md` | modified | core, training, reporting and script sections | reference must match the code |
| `.claude/reference/architecture.md` | modified | agents table, QLearner paragraph, IDQN-not-transferable note | architecture record |
| `.claude/reference/testing.md` | modified | core / agent / train / tune / report test bullets | test inventory |
| `CLAUDE.md` | modified | Phase 5 commands; structure lines for `core/`, `train/`, `report.py`, scripts | project instructions current |
| `README.md` | modified | new §8 "RL baselines" (commands, outputs, timings); Tests → §9 | reproducibility (G12) |
| `worklog/016-…`, `worklog/README.md` | created / modified | this entry and its index row | work-log rule |

## Commits
- `28b77a4` docs: plan Phase 5 RL baselines and log design decisions
- `3cb0b55` feat(baselines): add Q-learning core, shared DQN and IDQN agents
- `701a7e5` feat(train): add training loop and scripts/train.py
- `71b9b48` feat(train): add random-search tuning and the Phase 5 search space
- `129fdbe` exp(baselines): select DQN hyperparameters on validation
- `49454d2` exp(baselines): select IDQN hyperparameters on validation
- `882822b` feat(eval): add Phase 5 validation table and learning-curve scripts
- `debd084` exp(baselines): DQN and IDQN 5-seed finals on validation
- `bc0454a` docs: record Phase 5 RL baselines
- docs(worklog): complete entry 016 for Phase 5 (this entry)

## Next
- Phase 6 (UG-MARL): attention encoder + K-head ensemble + gate on the same `QLearner`,
  trainer and tuning; re-run the CPU/CUDA benchmark for the larger model.
- Phase 7.4: **IDQN cannot run zero-shot on Hangzhou.** It has one network per Jinan
  intersection, and Hangzhou has 16. Decide how to handle this (e.g. report IDQN as N/A, or
  use a shared network for transfer) and record it in the Decision Log before Phase 7.
- ROADMAP 0.1b (native Ubuntu) before Phase 7; retrain the locked configs there.
- DQN's best checkpoint often comes at episode 95–100, so DQN may still be improving at the
  fixed 100-episode budget. The budget stays fixed (same for all agents); note it as a
  limitation.
