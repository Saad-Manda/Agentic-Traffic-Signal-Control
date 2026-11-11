# Plan — Phase 5: RL baselines (DQN, IDQN), training, tuning (ROADMAP 5.1–5.5)

Status: **executed 11 Nov 2026** — DQN 299.8 ± 1.0 s, IDQN 312.7 ± 2.5 s on val (dev, WSL) (worklog/016; decisions taken with the user the same day). Tick steps
`[x]` during execution and note deviations under each step.

## Goal

Build the learned baselines that PS16 requires and the Q-learning core that UG-MARL (Phase 6)
will reuse. Tune them on the **validation** flow and lock their configs before Phase 7.

| Task | Output | Serves |
|---|---|---|
| 5.1 DQN core | `trafficrl.agents.core`: replay buffer, Q-networks with a heads axis, Double-DQN + Huber loss with SMDP discount, ε schedule, target network | F3, R4, G7 |
| 5.2 Single-agent baseline | `DQNAgent`: **one** Q-network shared by all 12 intersections, local 44-value obs only, one replay buffer | F3, R4, G7 |
| 5.3 Multi-agent baseline | `IDQNAgent`: **12 independent** Q-networks (batched) and 12 replay buffers, local obs only | F3, R4, G7 |
| 5.4 Training script | `scripts/train.py`: train on Jinan *train*, greedy val evaluation every 5 episodes, best checkpoint by val travel time, CSV + TensorBoard logs, one `experiment_log.csv` row per run, `env.json` | F5, F6, R4, G8 |
| 5.5 Light tuning | search space in `configs/search/rl_baselines.yaml` (written and committed **before** the first trial); `scripts/tune.py`: 8 random trials per agent on seed 0, then the chosen config on seeds 0–4 | R4, G8 |
| Outputs for Review 3 | dev validation table (FT, MP, DQN, IDQN; val; 5 seeds) + learning-curve figure, both made by scripts | R6, G9, G11 |

Out of scope: UG-MARL modules (attention, ensemble, gate — Phase 6; the core is built so they
plug in), any test / held-out run (Phase 7), native-Ubuntu final runs (Phase 7, needs 0.1b).

## Decisions taken (user, 11 Nov)

| # | Topic | Decision | Reason |
|---|---|---|---|
| U1 | Algorithm | **Double DQN + Huber loss + gradient clipping**, target network, uniform replay, ε-greedy. No dueling, no PER, no n-step. | Standard, stable; avoids a strawman vanilla DQN; keeps the UG-MARL comparison about attention / ensemble / gate only. |
| U2 | Platform | **Develop and tune on WSL** (numbers labelled *dev*). Locked configs are retrained / evaluated on **native Ubuntu in Phase 7** for reported numbers (protocol §7). | Fast dev loop now; protocol unchanged. ROADMAP 0.1b must be done before Phase 7. |
| U3 | Seeds | Each tuning trial = 1 training run, **seed 0**, scored on val. The chosen config per agent is then trained on **seeds 0–4**, each run picking its own best val checkpoint → "Validation score" = mean ± std, 95 % CI over 5 seeds. | Cheap tuning, proper multi-seed validation numbers. |
| U4 | Budget | **100 training episodes per run** (≈ 15–40 min). | CoLight-style budget; ~26 runs total fit in about a day with parallel processes. |
| U5 | Discount | **SMDP**: γ^(duration_s / Δt) per transition (15 s → γ, 20 s → γ^(4/3)). | Transitions are 15 s or 20 s (yellow); discount follows elapsed time. |
| U6 | Reward scale | Training uses r̃ = **c · r** with a **fixed constant c = 0.01** (not fitted). Metrics and logs use the raw −\|Pᵢ\|. | A positive constant does not change the optimal policy, so the frozen reward is unchanged; keeps Q-values O(1–10). Recorded as a *clarification* (no frozen value changes). |
| U7 | Checkpoint | **Greedy val episode every 5 training episodes**; keep `best.pt` (lowest val avg travel time) and `last.pt`. | Selection on validation is what the protocol allows. |
| U8 | Tuning | **Random search, 8 trials per agent**, same search space for DQN and IDQN, fixed sampler seed. | ≈ 16 runs; fair to both baselines. |
| U9 | Code structure | **One configurable Q-learner core** with pluggable network / state encoder; DQN, IDQN (and Phase 6 UG-MARL + ablations) share the same trainer and loss. | Fair ablations: one code path, no implementation confounds. |
| U10 | Network | **MLP 44 → 128 → 128 → 8, ReLU, fixed (not tuned)**; ≈ 23k params per net (IDQN: 12 × ≈ 23k). | Model-size column compares like with like; UG-MARL per-head trunk matches. |
| U11 | Execution | `scripts/tune.py` runs `train.py` trials as **parallel processes** (default 4, 1 sim thread and 1 torch thread each); device from a benchmark step (D9). | Every table from a script (guidelines); uses the 12 cores. |
| U12 | Outputs | Dev **validation table** and **learning-curve figure** produced by scripts in Phase 5. | Concrete RL-baseline numbers for Review 3 (17 Nov). |

## Design

### D1. Package layout

```
src/trafficrl/agents/core/        # shared by DQN, IDQN, UG-MARL (Phase 6)
  replay.py        ReplayBuffer (grouped)
  networks.py      MLPQNet, StackedMLPQNet  (output (G, B, K, A); K = 1 here)
  encoders.py      LocalEncoder  (Phase 6 adds NeighborhoodEncoder)
  schedules.py     LinearSchedule (ε)
  qlearner.py      QLearner: act / observe / update / save / load (implements Agent)
src/trafficrl/agents/baselines/
  dqn.py           DQNAgent  = QLearner(shared net, 1 buffer group)
  idqn.py          IDQNAgent = QLearner(stacked nets, n buffer groups)
src/trafficrl/train/
  loop.py          train_episode, train(cfg, agent_name) -> run dir
  tune.py          sample_trials(search_cfg, n, seed), run_parallel(...)
```

### D2. State encoding and action mapping

- `LocalEncoder.encode(obs: dict[iid → (44,)]) -> np.ndarray (n, 44)` in `env.intersection_ids`
  order; for a `Transition` it keeps only row `iid` (state = `obs[iid]`, next state =
  `next_obs[iid]`). Phase 6 adds a `NeighborhoodEncoder` (own + 4 neighbour rows + mask, from
  `env.neighbor_index()`), so the buffer stores whatever the encoder returns.
- Actions: phase ids `env.phase_ids` (1..8) ↔ indices 0..7 (`phase_ids[a]`). Jinan has 8
  green phases at every intersection (9 light phases incl. yellow 0), so no action masking.

### D3. Replay buffer — `core/replay.py`

```python
ReplayBuffer(groups: int, capacity: int, state_shape: tuple, rng: np.random.Generator)
  add(group: int, state, action: int, reward: float, next_state, duration_s: int, done: bool)
  sample(batch: int) -> dict of tensors shaped (G, B, ...)   # B per group, uniform
  __len__ -> min fill over groups
```
- DQN: `groups = 1`, every intersection's transitions go to group 0, capacity 100 000
  (≈ 35 episodes × 2.9k transitions).
- IDQN: `groups = n` (12), group = intersection index, capacity 100 000 / n per group (same
  total memory, same ≈ 35-episode horizon per intersection).
- Ring buffer of NumPy arrays; sampling uses the run's own `Generator` (seeded) — no global
  RNG. Phase 6 adds a `(K,)` bootstrap mask column; the buffer API reserves an `extras` dict.

### D4. Q-networks — `core/networks.py`

- `MLPQNet(in_dim=44, hidden=(128, 128), n_actions=8, heads=1)` — one parameter set; input
  `(G, B, 44)` with G broadcast → output `(G, B, K, A)`.
- `StackedMLPQNet(groups=n, …)` — weights shaped `(n, in, out)`, forward with `torch.einsum`
  / `bmm`, so 12 independent networks run in one call. Test: equals a loop of 12 `MLPQNet`
  with the same weights.
- Init: PyTorch default (Kaiming-uniform), seeded by `seed_everything`. Heads axis `K` is
  fixed to 1 in Phase 5; UG-MARL sets K > 1.

### D5. Loss and update — `core/qlearner.py`

For a sampled transition (s, a, r, s′, d_s):

```
r̃      = c · r                                   (c = 0.01, U6)
a*     = argmax_a′ Q_online(s′, a′)               (Double DQN)
y      = r̃ + γ^(d_s / Δt) · Q_target(s′, a*)      (SMDP discount, U5)
loss   = Huber_δ=1( Q_online(s, a) − y )           mean over (G, B)
```
- **Episode end is truncation, not termination** (`Transition.done` = t reached 3600 s, the
  traffic does not stop): the target **always bootstraps** (Pardo et al., 2018). `done` is
  stored only for logging. Justification goes in the report's MDP section.
- Adam, gradient-norm clip 10, hard target sync every 1000 gradient steps.
- Update schedule: after `learning_starts` = 2000 transitions, one gradient step per
  `train_every` = 4 new transitions (≈ 720 steps / episode, ≈ 72k per run). One IDQN step
  updates all 12 networks, each on a batch from its own buffer (so each IDQN network takes
  as many steps as the shared DQN on 1/12 of the data — noted in the report).
- `act(obs, info, explore)`: encode all intersections → one forward pass → argmax over the
  mean of the K heads (K = 1 now); with `explore=True` each intersection independently takes a
  uniform random phase with probability ε (agent's own seeded `Generator`). Returns a phase for
  every intersection (interface rule); greedy at evaluation.
- `observe(transitions)`: encode each `Transition` and `add` it to the buffer; counts new
  transitions for `train_every`.
- `update()`: runs the due gradient steps; returns `{loss, q_mean, td_abs, grad_norm, eps}`.
- `save(path)` / `load(path)`: `torch.save` of network + target + optimiser state + step
  counters + the agent's resolved hyperparameters; `load` restores weights for evaluation.
  Checkpoints are device-independent (`map_location`).

### D6. Agents and registry

- `DQNAgent(env, cfg)` and `IDQNAgent(env, cfg)` are thin constructors over `QLearner`
  (encoder = local; net = `MLPQNet` vs `StackedMLPQNet`; buffer groups 1 vs n).
- `make_agent` gains `"dqn"`, `"idqn"`; `AGENTS` updated. Params from `cfg["agents"][name]`
  plus `cfg["train"]`. `scripts/evaluate.py --agent dqn --checkpoint …` then works unchanged.
- `evaluate.py --checkpoint` accepts a `{seed}` template
  (`results/train/<study>/seed{seed}/checkpoints/best.pt`) so each evaluation seed loads the
  model trained with that seed (protocol §6: a seed fixes training **and** evaluation).

### D7. Training loop — `train/loop.py` + `scripts/train.py`

```
scripts/train.py --config configs/<dqn|idqn>.yaml [--run-name NAME] [overrides]
```
1. Refuse any `data.split` other than `train` (exit 2); the val env is built internally with
   `split = val`; `allow_final` is never passed (test / held-out unreachable).
2. `seed_everything(seed)`; `train_env = make_env(cfg, seed, track="light")`,
   `val_env = make_env(cfg_val, seed, track="light")`; agent from `make_agent`.
3. For episode e = 1..100: `train_episode` (reset → act(explore) → step → observe → update,
   until done); ε linear from 1.0 to 0.05 over `eps_decay_frac` × 100 episodes.
   Log per episode: train avg travel time, mean raw reward, ε, mean loss / q / grad norm,
   wall time → `train_log.csv` + TensorBoard.
4. Every 5 episodes (and the last): greedy val episode via `run_episode` → `val_log.csv`;
   if val avg travel time improves → `checkpoints/best.pt`. Always write `checkpoints/last.pt`.
5. End: `summary.json` (best val travel time and episode, total time, params, peak GPU memory
   if CUDA), resolved `config.yaml`, `env.json`; **one row** in `docs/experiment_log.csv`
   (run_id, split = `train`, seed, model, hyperparameters JSON incl. device, metric =
   best val `avg_travel_time_s`, `validation_metric` = same, runtime, git commit, note
   `dev-WSL` + checkpoint episode).

Output: `results/train/<run_name>/` (gitignored). Run names:
`<agent>_<study>_t<trial>_s<seed>` for tuning, `<agent>_final_s<seed>` for the 5-seed runs.

### D8. Tuning — `configs/search/rl_baselines.yaml` + `scripts/tune.py`

Search space (committed before the first trial; sampler seed 2026):

| Hyperparameter | Values | Fixed (not tuned) |
|---|---|---|
| learning rate | {1e-4, 3e-4, 1e-3} | hidden (128, 128), Adam, Huber δ = 1, grad clip 10 |
| γ (per 15 s) | {0.8, 0.9, 0.95} | buffer 100k, learning_starts 2000, train_every 4 |
| batch size | {64, 128} | target sync 1000 steps, ε 1.0 → 0.05 |
| ε decay fraction | {0.3, 0.5} of the episodes | 100 episodes, val every 5, reward scale 0.01 |

```
scripts/tune.py --search configs/search/rl_baselines.yaml --agent dqn --trials 8 --workers 4
scripts/tune.py --config configs/dqn.yaml --seeds 0 1 2 3 4 --workers 5   # chosen config
```
- Trials sampled without replacement (36 combinations, 8 drawn), identical trial list for
  DQN and IDQN; each trial's resolved config written to `results/tune/<study>/configs/` before
  launch; `trials.csv` (trial, hyperparameters, best val travel time, best episode, runtime).
- **Selection rule (fixed now):** lowest best-val avg travel time on seed 0; tie (< 1 s) →
  lower lr. The winner is copied into `configs/dqn.yaml` / `configs/idqn.yaml` with a comment
  pointing to the study, committed, then run on seeds 0–4.
- No test / held-out access anywhere in tuning (same guard as D7).

### D9. Device benchmark (one step, recorded)

Time 2 training episodes of DQN and IDQN on `cpu` (torch 1 thread) vs `cuda`, with 1 and 4
parallel processes. Choose the faster setting for 4 parallel runs and set `train.device` in
`configs/base.yaml`. Expectation: CPU wins for 23k-param MLPs, but this is measured, not
assumed. Determinism is checked on the chosen device (same seed → identical `last.pt`
weights after 2 episodes).

### D10. Phase 5 outputs (scripts, not notebooks)

- `scripts/evaluate.py --config configs/eval_rl_baselines.yaml --agent dqn --split val
  --checkpoint results/train/dqn_final_s{seed}/checkpoints/best.pt` (and idqn), full tracking,
  5 seeds → `results/eval/…` and log rows.
- `scripts/val_table.py` → `docs/results/val_phase5.{csv,md}`: Model | Val avg travel time
  (mean ± std, 95 % CI) | max wait | throughput | training time | params; FT / MP from the
  Phase 4 val runs (deterministic, std = 0). Labelled *dev (WSL)*.
- `scripts/plot_training.py` → `docs/figures/training/learning_curves.png`: val avg travel
  time (s) vs training episode, mean ± 95 % CI band over seeds 0–4 for DQN and IDQN, MP and
  FT val values as horizontal reference lines; caption + interpretation in the worklog and a
  `.md` sidecar.

## Justification (for the report / viva)

- **Shared-parameter DQN = single-agent baseline** (Decision Log 27 Oct): one learner, one
  policy applied to each intersection's local state; no coordination signal.
- **IDQN = multi-agent baseline**: the classic independent-learners MARL baseline
  (Tan 1993; used as baseline by CoLight / PressLight); non-stationarity from other learners
  is exactly the limitation UG-MARL's attention targets.
- **Double DQN** reduces max-operator overestimation (van Hasselt et al., 2016) — relevant
  because UG-MARL's σ is computed on Q-values; overestimated baselines would bias the story.
- **SMDP discount and truncation bootstrapping** are the mathematically correct targets for
  this env (variable-length intervals, time-limit episode end); both go in report §MDP.
- **Same core for all learned agents** → differences in results come from the components
  under study, not from code.

## Files

| File | Action | Content |
|---|---|---|
| `src/trafficrl/agents/core/{__init__,replay,networks,encoders,schedules,qlearner}.py` | create | D2–D5 |
| `src/trafficrl/agents/baselines/dqn.py`, `idqn.py` | create | D6 |
| `src/trafficrl/agents/__init__.py` | modify | registry `dqn`, `idqn` |
| `src/trafficrl/train/{__init__,loop,tune}.py` | create | D7, D8 |
| `scripts/train.py`, `scripts/tune.py` | create | CLIs (D7, D8) |
| `scripts/evaluate.py`, `src/trafficrl/eval/runner.py` | modify | `{seed}` checkpoint template (D6) |
| `scripts/val_table.py`, `scripts/plot_training.py` | create | D10 |
| `configs/base.yaml` | modify | `train:` defaults (device, episodes, val_every, reward_scale, …), `agents.dqn` / `agents.idqn` blocks |
| `configs/dqn.yaml`, `configs/idqn.yaml` | create | per-agent training config (chosen values after tuning) |
| `configs/search/rl_baselines.yaml` | create | search space + sampler seed + selection rule (D8) |
| `configs/eval_rl_baselines.yaml` | create | val evaluation of trained checkpoints |
| `tests/agents/test_core.py`, `test_dqn_idqn.py`, `tests/train/test_loop.py`, `test_tune.py` | create | see Tests |
| `.claude/reference/api.md`, `architecture.md`, `testing.md` | modify | implemented sections |
| `CLAUDE.md`, `README.md` | modify | Phase 5 commands |
| `docs/how_it_works.md` | modify | training section (SMDP, Double DQN, checkpoint on val) |
| `docs/evaluation_protocol.md` §9 | modify | clarification row (U5 / U6 / truncation — training details, no frozen value changed) |
| `ROADMAP.md` | modify | tick 5.1–5.5; Decision Log rows (U1–U12 summary; dev-on-WSL / retrain natively) |
| `docs/experiment_log.csv` | append | one row per training run + evaluate rows |
| `docs/results/val_phase5.*`, `docs/figures/training/*` | create | D10 |
| `worklog/016-2026-11-11-phase5-rl-baselines.md`, `worklog/README.md` | create / modify | work log |

## Config (sketch, `configs/base.yaml` additions)

```yaml
train:                    # scripts/train.py (Phase 5); UG-MARL reuses it
  episodes: 100
  val_every: 5
  device: cpu             # set by the D9 benchmark
  torch_threads: 1
  reward_scale: 0.01      # constant, not fitted (U6)
  lr: 3.0e-4
  gamma: 0.9              # per Δt = 15 s; SMDP: gamma ** (duration_s / 15)
  batch_size: 64
  buffer_size: 100000
  learning_starts: 2000
  train_every: 4
  target_sync: 1000
  grad_clip: 10.0
  eps_start: 1.0
  eps_end: 0.05
  eps_decay_frac: 0.5

agents:
  dqn:  {hidden: [128, 128]}
  idqn: {hidden: [128, 128]}
```

## Data-leakage check

- Training env: `split = train` only (train.py refuses anything else, exit 2).
- Checkpoint selection + tuning: `split = val` only, built inside the trainer.
- `allow_final` is never passed by `train.py` / `tune.py`; test and held-out raise
  `SplitError`. Test: calling `train(...)` with `data.split=test` or a `val_split=test`
  override fails.
- Nothing is fitted on data (capacity scaling, constant reward scale), so no statistics leak.
- The validation flow is used for both checkpoint and hyperparameter selection → val scores
  are optimistic; that is why test (Phase 7) is separate. Stated in the report.

## Tests (`pytest`; simulator tests marked `slow`, fast ones on `tests/envs/fake_engine.py`)

- **Replay**: wrap-around at capacity, per-group fill, sample shapes `(G, B, …)`, same seed →
  same batch.
- **Networks**: output `(G, B, K, A)`; `StackedMLPQNet` = loop of 12 `MLPQNet` with copied
  weights; param counts (≈ 23k / 12 × ≈ 23k).
- **Loss**: hand-computed Double-DQN target with SMDP discount (15 s vs 20 s), reward scale,
  bootstrap through `done`; Huber value on a known residual.
- **ε schedule**: endpoints and midpoint.
- **Interface conformance**: DQN / IDQN satisfy `Agent`, `act` returns a valid phase id for
  every intersection, greedy `act` is deterministic, `explore=True` at ε = 1 gives every phase.
- **IDQN independence**: an update with data only in group A leaves group B's weights
  unchanged.
- **Save / load**: round trip gives identical Q-values; loads on CPU from a CUDA checkpoint
  (skipped without CUDA).
- **Determinism**: two 1-episode trainings (fake engine) with the same seed → identical weights
  and logs; different seed → different.
- **Trainer guards**: refuses non-train split; never opens test / held-out; writes
  `train_log.csv`, `val_log.csv`, `best.pt`, `last.pt`, `summary.json`, one log row.
- **Tuning**: sampler deterministic for a fixed seed, no duplicate trials, values inside the
  search space; selection rule picks the expected trial from a hand-made `trials.csv`.
- **Evaluate**: `{seed}` checkpoint template resolves per seed.

## Steps (each can be run and checked on its own)

1. [x] Decision Log rows in `ROADMAP.md` (U1–U12) + protocol §9 clarification row; worklog/016
   started (`in-progress`).
2. [x] Core: `replay.py`, `networks.py`, `encoders.py`, `schedules.py` + unit tests.
3. [x] `qlearner.py` (loss, act / observe / update / save / load) + loss / determinism tests.
4. [x] `DQNAgent`, `IDQNAgent`, registry, `{seed}` checkpoint template + conformance tests.
5. [x] `train/loop.py` + `scripts/train.py` + `configs/dqn.yaml`, `idqn.yaml`, `base.yaml`
   `train:` block; smoke test: 2 episodes, val every episode, on Jinan train/val.
6. [x] **D9 device benchmark** → set `train.device`; determinism check on that device; log
   timings in worklog.
7. [x] **Learning sanity run** (seed 0, default config, 100 episodes, DQN and IDQN): val travel
   time must drop clearly below the first evaluation and below Fixed-Time (434.84 s). If not,
   debug (reward sign, action mapping, target sync, discount) **before** tuning — logged as a
   dead end if it happens. This run is a dev run, not a tuning trial.
8. [x] Write and **commit** `configs/search/rl_baselines.yaml` (search space + selection rule)
   — before any tuning trial.
9. [x] `train/tune.py` + `scripts/tune.py` + tests; run DQN study (8 trials) and IDQN study
   (8 trials), 4 workers.
10. [x] Apply the selection rule; copy winners into `configs/dqn.yaml` / `idqn.yaml`, commit.
11. [x] Final 5-seed runs for each agent (`tune.py --seeds 0 1 2 3 4`).
12. [x] Evaluate best checkpoints on val with full tracking (`evaluate.py … {seed}`);
    `scripts/val_table.py`, `scripts/plot_training.py`.
13. [x] Docs: api.md, architecture.md, testing.md, how_it_works.md, CLAUDE.md, README; tick
    ROADMAP 5.1–5.5; finish worklog/016 (files table cross-checked with `git status`);
    commits per `.claude/commands/commit.md`.

### Deviations during execution (11 Nov)

- **Step 3 — leakage guard added:** `run_episode` calls `agent.observe`, so the trainer's
  validation episodes would have pushed **validation transitions into the replay buffer**.
  `QLearner.observe` now stores experience only while `agent.training` is True (set by
  `train_episode` only); tested (`test_validation_episodes_never_enter_the_buffer`).
- **Step 4 — interface test:** `update() == {}` is asserted for rule agents only (learned
  agents return losses); the "unknown agent" test now uses `"ppo"`.
- **Step 4 — IDQN independence test** checks gradient isolation directly (a loss on network
  0 gives exactly zero gradient to the others) instead of comparing two trainings, which
  differed only by initialisation.
- **Step 5 — run naming:** smoke runs `smoke_{dqn,idqn}` (2 episodes) are logged in
  `docs/experiment_log.csv` with note "smoke".
- **Step 6 — benchmark runs** logged to a scratch file, not the experiment log (timing, not
  an experiment). Result: CPU 25.4 / 24.8 s vs CUDA 28.6 / 36.2 s (DQN / IDQN, 3 episodes +
  1 val); 5 parallel CPU processes 36–43 s each (≈ 3.4× throughput); same seed → identical
  weights (single and parallel). `train.device: cpu`. CUDA and CPU give different IDQN
  numbers (floating point), so determinism is per device.
- **Step 7 — sanity criterion met by episode 30** (DQN best 322.6 s @ 25, IDQN 351.8 s @ 30,
  both < Fixed-Time 434.84 s), so the search space was committed (71b9b48) and the DQN study
  started while the sanity runs finished their 100 episodes.
- **Step 9 — tune.py smoke tests** (1-episode trials into a scratch results dir) appended 6
  rows to the experiment log; they are annotated "SMOKE TEST … not a tuning trial". A
  `train.py` print bug for results outside the repo was found and fixed by this smoke test.
- **Step 9 — wall time:** DQN trials took 13–16 min, IDQN trials 28–43 min each with 4–6
  runs sharing the CPU (12 per-network forward/backward passes per update), more than the
  compute estimate below.
- **Step 10 — same winner for both agents:** trial t6 (lr 1e-4, γ 0.8, batch 64, ε-decay 0.3);
  DQN 298.86 s, IDQN 310.42 s best val. DQN trials span 298.9–308.5 s, IDQN 310.4–335.6 s.
- **Step 11 — launch bug:** the first DQN finals launch failed (`python: command not found`,
  exit 127): `source .venv/bin/activate && (A) & (B) &` activates the venv only for the first
  background job. Relaunched with `.venv/bin/python`; no run was lost. IDQN finals used 4
  workers (RAM ≈ 2.4 GB free while the last DQN seed ran).
- **Step 12 — `.gitignore`:** `results/` also ignored `docs/results/`, so the generated val
  table could not be committed; the pattern is now `/results/` (repo-root run outputs only).
- **Step 12 — learning-curve figure** has a second, zoomed panel (episodes ≥ 20, y axis cut):
  the first evaluation (~980 s) squeezed the 300–330 s range where the agents and
  Max-Pressure differ.
- `utils.config.read_yaml` added (public loader for the search file) instead of importing
  the private `_parse`.
- Memory: ≈ 0.9 GB RSS per training process (torch + buffer) on a 7.7 GB WSL VM → at most
  ~6 concurrent runs.

## Verification

- `pytest` (incl. `-m slow`) and `ruff check src tests scripts` pass.
- Smoke run produces every output file listed in D7; `experiment_log.csv` gains one row.
- Same seed twice → identical `last.pt` (step 6).
- Sanity run: learning curve decreases; both agents beat Fixed-Time on val.
- `trials.csv` has 8 rows per agent and the search-space file's commit predates the first
  trial's timestamp.
- Val table: DQN / IDQN mean ± std and 95 % CI over 5 seeds; FT 434.84 s and MP 326.11 s
  reproduced (deterministic).
- `git grep -n "allow_final" src scripts` → only `evaluate.py --final` path and `make_env`.

## Compute estimate (WSL, 12 cores, ~7 GB RAM)

Per run: 100 train episodes (≈ 5–15 s each with NN cost) + 21 val episodes (≈ 3 s) ≈
10–30 min. Sanity 2 runs + tuning 16 runs + finals 10 runs + 10 val evaluations ≈ 28 runs →
about 2–4 h wall time at 4–5 parallel workers. Buffer memory ≈ 40 MB per run (local states),
so RAM is not the limit.

## Experiment log

Every `train.py` run → one row (split `train`, metric = best val avg travel time, note
`dev-WSL`, trial / study id). Every `evaluate.py` run → one row per seed (existing behaviour).
Tuning trials are runs and are logged like any other.

## Risks / open questions

| Risk | Mitigation |
|---|---|
| DQN/IDQN fail to beat Max-Pressure on val | Expected in the literature for simple DQN; reported as is (it motivates the gate). Not a reason to tune beyond the fixed search space. |
| Training on a single 1-hour train flow overfits | Checkpoint chosen on val; noted as a limitation; no augmentation added in Phase 5. |
| Random actions during exploration cause many yellows → slow early learning | Expected; ε decays to 0.05; the sanity run (step 7) checks it. |
| IDQN networks see 1/12 of the data each | Same buffer horizon and update count per network; documented. |
| WSL numbers differ from native Ubuntu | Deterministic setup; Phase 7 retrains natively and reports only those (U2). |
| Phase 6 overlaps (11–13 Nov) | Core has a heads axis, grouped buffer with `extras`, and pluggable encoder, so Phase 6 adds modules without changing the trainer. |
| CUDA determinism / speed on WSL | D9 benchmark decides; CPU is the fallback. |

## Worklog

`worklog/016-2026-11-11-phase5-rl-baselines.md` (code, experiment, decision; ROADMAP 5.1–5.5):
decisions U1–U12, benchmark timings, sanity-run result, tuning tables, final val table and
figure with interpretation, every file changed, commits.
