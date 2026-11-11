# Interfaces

No HTTP API — this is a research codebase. "API" = the internal Python interfaces.
Sections marked *implemented* match the code; the rest are planned.

## Environment (`src/trafficrl/envs/traffic_env.py`) — implemented (ROADMAP 4.1)

```python
make_env(cfg, seed, *, allow_final=False, track="full") -> TrafficEnv
    # resolves cfg["data"] via flow_files (hash check); test / heldout raise SplitError
    # unless allow_final=True (only scripts/evaluate.py --final); robustness.surge != 1 writes a
    # surged flow copy into a temp workdir (hash in env.data_info)
class TrafficEnv:
    intersection_ids: list[str]; phase_ids: tuple[int, ...]; obs_dim: int   # 44
    delta_t, yellow_s, episode_s: int; features: FeatureBuilder; data_info: dict
    def neighbors(self, iid) -> list[str]                   # signalised, order E, N, W, S
    def neighbor_index(self) -> (index (n, 4) int64, mask (n, 4) bool)   # batched attention
    def reset(self) -> (obs, info)                          # t = 0, every intersection ready
    def step(self, actions: dict[str, int]) -> (obs, rewards, done, info)
    def metrics(self) -> dict[str, float | None]            # after done; see metrics.py
@dataclass(frozen=True)
class Transition: iid, obs (ALL intersections), action, reward, next_obs (ALL), done, duration_s
```
- **Timing — per-intersection clocks.** Each intersection holds its green Δt (15 s), then is
  *ready*; same phase → another Δt, new phase → 5 s yellow (phase 0) + Δt. `step` applies the
  actions of `info["ready"]` only (others ignored; a missing ready action raises `KeyError`)
  and returns as soon as ≥ 1 intersection is ready again or t = episode end. Decisions fall
  on a 5 s grid. At t = 0 the first green starts without a yellow.
- `obs`: all intersections, from the *observed* readings (after `SensorModel`); during yellow
  the one-hot shows the pending green. `rewards`: `{iid: −|P_i|}` for intersections whose
  interval just ended, from *true* counts. `info`: `t`, `ready`, `phase`, `in_yellow`,
  `green_elapsed`, `lane_count` (observed, vehicles), `transitions` (list of `Transition`).
- `track="light"` skips per-vehicle reads (training); `"full"` for evaluation.

```python
# metrics.py
EpisodeTracker(in_lanes, track).on_second(speeds) / on_step(lane_waiting) / on_switch() /
    result(avg_travel_time_s, in_network) -> {avg_travel_time_s, avg_queue_veh, phase_switches,
    in_network_at_end, avg_waiting_time_s, max_waiting_time_s, throughput_veh, vehicles_entered}
# perturb.py (protocol §6)
SensorModel(capacity, noise_std, dropout_p, rng).observe(lane_count, lane_waiting) -> (count, wait)
surge_flow(roadnet_path, flow_path, out_dir, surge, rng) -> (roadnet copy, flow, n added)
# engine.py
make_engine(roadnet_path, flow_path, seed, threads, workdir) -> cityflow.Engine
```

## Agent (`src/trafficrl/agents/`) — implemented for rule agents and DQN / IDQN (ROADMAP 4.4–4.5, 5.1–5.3)

```python
class Agent(Protocol):                                     # agents/base.py
    name: str
    def act(self, obs: dict[str, np.ndarray], info: dict, explore: bool = False) -> dict[str, int]
        # returns a phase for EVERY intersection (latency = batch of all); env uses info["ready"]
    def observe(self, transitions: list[Transition]) -> None   # no-op for rule agents
    def update(self) -> dict[str, float]                   # training losses; {} for rule agents
    def save(self, path) -> None ; def load(self, path) -> None
    def last_explanations(self) -> dict                    # {} by default; UG-MARL: σ, attention,
                                                           #   {"fallback": {iid: bool}} (-> fallback rate)
class RuleAgent  # no-op base
FixedTimeAgent(phase_ids, delta_t, green_seconds=30)       # keep while green_elapsed < 30 s
MaxPressureAgent(features).choose(iid, lane_count) -> int  # observed counts; ties -> lowest id
make_agent(name, env, cfg) -> Agent                        # registry AGENTS; params cfg["agents"][name]
                                                           # learned (LEARNED_AGENTS): + cfg["train"], cfg["seed"]
DQNAgent(env, params, seed)    # shared MLP 44-128-128-8, one replay group, local obs (5.2)
IDQNAgent(env, params, seed)   # StackedMLPQNet: one MLP + one replay group per intersection (5.3)
```

### Q-learning core (`src/trafficrl/agents/core/`) — implemented (ROADMAP 5.1)

```python
ReplayBuffer(groups, capacity_per_group, state_shape, rng)
    add(group, state, action_idx, reward_raw, next_state, duration_s, done) ; sample(B) -> (G, B, ...)
MLPQNet(in_dim, n_actions, hidden=(128,128), heads=1)          # (G, B, in) -> (G, B, K, A), shared
StackedMLPQNet(groups, in_dim, n_actions, hidden, heads)       # G independent nets, one bmm
LocalEncoder(intersection_ids, obs_dim).encode_all(obs) / encode_transition(tr)
LinearSchedule(start, end, duration).value(t)
QLearner(net, encoder, intersection_ids, phase_ids, delta_t, *, shared, params, seed)
    act(obs, info, explore)      # argmax of mean over heads; ε per intersection (own RNG)
    observe(transitions)         # stores ONLY while .training is True (val/eval never enter replay)
    update() -> {loss, q_mean, td_abs, grad_norm} | {}   # 1 step per train_every new transitions
    td_target(batch)             # c·r + γ^(d/Δt)·Q_target(s′, argmax Q_online(s′)); always bootstraps
    save(path) / load(path)      # net, target, optimiser, counters, params; checks agent + intersections
    .epsilon, .training, .grad_steps, .buffer, .net
DEFAULTS                         # = configs/base.yaml `train:` learning keys
```
Phase 6 reuses `QLearner` with K > 1 heads and a neighbourhood encoder.

## Training (`src/trafficrl/train/`) — implemented (ROADMAP 5.4, 5.5)

```python
train_episode(env, agent) -> {train_avg_travel_time_s, mean_reward, transitions, loss, ...}
train(cfg, *, run_name=None, log_path=EXPERIMENT_LOG, note="", verbose=True) -> run dir
    # data.split must be "train" (TrainSplitError); val env built internally; never allow_final
    # -> results/train/<run>/{train_log,val_log}.csv, checkpoints/{best,last}.pt, tb/,
    #    config.yaml, env.json, summary.json + one experiment-log row
# tune.py
load_search(path) ; sample_trials(space, n, seed) ; select_best(df, metric, tie_tolerance)
run_parallel(jobs, workers, log_dir) ; train_argv(...) ; collect(results_dir, runs)
```

## Reporting (`src/trafficrl/eval/report.py`)

```python
latest_eval_run(eval_dir, agent, dataset, split) ; training_summaries(train_dir, agent, tag, seeds)
val_curves(train_dir, agent, tag, seeds) -> per-episode mean / std / 95 % CI ; fmt(stats) ; metric_stats(csv, metric)
```
`info` was added to `act` (10 Nov, plan D5): Max-Pressure (and the UG-MARL gate) need raw
observed lane counts and Fixed-Time needs the green elapsed time; neither is in the 44-vector.

## Evaluation (`src/trafficrl/eval/runner.py`) — implemented (ROADMAP 4.6)

```python
resolve_checkpoint(checkpoint, seed) -> Path | None          # "{seed}" -> seed
run_episode(env, agent, *, explore=False) -> metrics + decisions, fallback_rate,
                                             decision_latency_ms (mean), decision_latency_ms_p95
evaluate(cfg, agent_name, seeds, *, final=False, checkpoint=None, track="full", note="",
         log_path=EXPERIMENT_LOG) -> (out_dir, metrics DataFrame)
condition_label(cfg) -> "clean" | "noise0.1-drop0.1-surge1.5" ; summary_table(metrics) -> DataFrame
# utils/stats.py
summarise(values) -> {mean, std, ci95_low, ci95_high, n, deterministic} ; t975(df)
```

## Data layer (`src/trafficrl/data/`) — implemented (ROADMAP 3.0–3.4)

```python
# roadnet.py — reads the roadnet only (never a flow file)
load_roadnet(path) -> Roadnet ; parse_roadnet(raw: dict) -> Roadnet ; RoadnetError
Roadnet: intersections {id: Intersection} (signalised only), virtual_ids, road_start / road_end,
         road_links {(start road, end road)}, lane_length_m, lane_max_speed, green_phase_ids
Intersection: in_roads / out_roads (E, N, W, S), in_lanes / out_lanes (12 each, canonical order),
              movements (12 × Movement(kind, start_road, end_road, in_lanes, out_lanes)),
              green_phases {phase id 1..8: movement indices}, neighbors (per direction, None = boundary)
lane_capacity(roadnet, vehicle_space_m) -> {lane: int}       # floor(length / 7.5 m)
YELLOW_PHASE = 0

# audit.py — counts only, never edits data
audit_flow(flows, roadnet, episode_seconds, bin_seconds=60) -> {check: count}
audit_roadnet(raw) -> {check: count} ; defects(counts) -> non-zero defect counts
arrival_counts(flows, episode_seconds, bin_seconds) -> np.ndarray

# trace.py — train split only (EDA rollouts with rule controllers)
train_flow(cfg) -> (roadnet path, flow path)      # SplitError unless primary dataset / train
make_engine(roadnet_path, flow_path, seed, threads, workdir) -> cityflow.Engine
make_chooser("fixed_time" | "max_pressure" | "random", features, seed) -> chooser
collect_trace(engine, roadnet, features, chooser, *, green_seconds, yellow_seconds,
              episode_seconds, record_every, start_times) -> (steps df, vehicles df, summary)

# eda.py — train split only: entry_demand, movement_demand, turning_ratios, intersection_load,
#          route_lengths, signalised_crossings, load_run, load_summaries, feature_arrays,
#          network_series
```

## Features (`src/trafficrl/envs/features.py`) — implemented (ROADMAP 3.4)

```python
FeatureSpec(groups=("queue", "count", "pressure", "phase"), vehicle_space_m=7.5, clip=True,
            normalisation="lane_capacity") ; FeatureSpec.from_config(cfg)   # `features:` block
class FeatureBuilder:
    def __init__(self, roadnet: Roadnet, spec: FeatureSpec | None = None)   # roadnet only
    dim -> int                                    # 44 with the default groups
    def slices(self) -> dict[str, slice]          # position of each group in the vector
    def observation(self, iid, lane_count, lane_waiting, green_phase) -> np.ndarray  # float32
    def observations(self, lane_count, lane_waiting, green_phases) -> dict[str, np.ndarray]
    def phase_pressure(self, iid, lane_count) -> dict[int, float]    # Max-Pressure rule
    def intersection_pressure(self, iid, lane_count) -> float        # reward = −|P_i|
```
`TrafficEnv.reset()/step()` return `FeatureBuilder.observations(...)`.
`green_phase` is a green phase id (1..8); during a transition the env passes the phase being
switched to, never 0. `lane_count` / `lane_waiting` are the engine's
`get_lane_vehicle_count()` / `get_lane_waiting_vehicle_count()` dictionaries.

## Utilities (`src/trafficrl/utils/`) — implemented (ROADMAP 0.3, 3.x)

```python
# seed.py — sets CUBLAS_WORKSPACE_CONFIG at import
seed_everything(seed: int, deterministic: bool = True) -> None  # random, NumPy, torch(+CUDA)
seed_cityflow(engine, seed: int) -> None                        # engine.set_random_seed
make_rng(seed: int) -> np.random.Generator                      # local RNG, no global state

# config.py
read_yaml(path) -> dict                                    # plain mapping (search-space files)
load_config(path, overrides: Iterable[str] = ()) -> dict  # resolves `base:`, applies
                                                           # "a.b=val" overrides, requires `seed`
save_config(cfg, path) -> Path                              # resolved YAML next to results
apply_overrides(cfg, overrides) -> dict ; deep_merge(a, b) -> dict ; ConfigError

# runlog.py — every run logs itself (guidelines G8)
append_experiment_row(*, run_id, split, seed, model, hyperparameters, metric_name,
                      metric_value, runtime_s, validation_metric=None, notes="", path=...)
                      # one row in docs/experiment_log.csv (header written if new)
env_info() -> dict ; save_env_info(path) -> Path      # versions + hardware -> env.json
git_commit() -> str                                   # short hash, "-dirty" if uncommitted

# plotting.py — shared figure style
apply_style() ; save_figure(fig, path) ; CONTROLLER_COLOR / CONTROLLER_LABEL ; BLUE_RAMP
```
YAML is parsed with a SafeLoader that reads `1e-3` as a float (plain PyYAML gives a string).
Every experiment config starts with `base: base.yaml` (`configs/base.yaml`).

## Scripts

- `scripts/audit_data.py [--config configs/eda.yaml]` — data audit of all splits →
  `docs/data_audit.{md,json}`, `docs/figures/audit/` (train only); exit 1 on any defect
- `scripts/collect_traces.py --config configs/eda.yaml [overrides]` — train-split rule-controller
  rollouts → `results/eda/<run>/`, rows in `docs/experiment_log.csv`
- `scripts/dataset_stats.py`, `scripts/bench_cityflow.py` — dataset sizes; install benchmark
- `scripts/train.py --config configs/{dqn,idqn}.yaml [--run-name N] [overrides]` — train on
  Jinan train, checkpoint on val → `results/train/<run>/` + one log row; exit 2 on non-train split
- `scripts/tune.py --search configs/search/rl_baselines.yaml --agent dqn [--workers 4]` — random
  search (trials in parallel) → `results/tune/<study>_<agent>/{trials.csv,best.json,configs/,logs/}`;
  `scripts/tune.py --config configs/dqn.yaml --seeds 0 1 2 3 4 [--tag final]` — multi-seed runs
- `scripts/val_table.py` → `docs/results/val_phase5.{csv,md}`; `scripts/plot_training.py` →
  `docs/figures/training/learning_curves.{png,md}`
- `scripts/evaluate.py --config configs/eval_baselines.yaml --agent <name> --split val [--seeds …]
  [--checkpoint PATH (may contain {seed})] [--final] [overrides]` — any agent × seeds → `results/eval/<run>/`
  (`metrics.csv`, `summary.csv`, config, env.json, data.json) + one log row per seed; exit 2 on
  test / heldout without `--final`
- `scripts/run_ablation.py --config <yaml>` — TODO
