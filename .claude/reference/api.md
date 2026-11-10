# Interfaces

No HTTP API — this is a research codebase. "API" = the internal Python interfaces.
All signatures below are **planned**; update when implemented.

## Environment (`src/trafficrl/envs/`)

```python
class TrafficEnv:
    def __init__(self, config: dict, seed: int): ...
    def reset(self) -> dict[str, np.ndarray]: ...          # obs per intersection id
    def step(self, actions: dict[str, int]) -> tuple[obs, rewards, done, info]: ...
    @property
    def intersection_ids(self) -> list[str]: ...
    def neighbors(self, iid: str) -> list[str]: ...
    def metrics(self) -> dict[str, float]: ...             # travel time, wait, throughput, queue
```
Observation noise / detector dropout for robustness tests applied via config in the wrapper.

## Agent (`src/trafficrl/agents/`)

```python
class Agent(Protocol):
    def act(self, obs: dict[str, np.ndarray], explore: bool = False) -> dict[str, int]: ...
    def observe(self, transition) -> None: ...             # no-op for non-learning baselines
    def update(self) -> dict[str, float]: ...              # training losses; {} for baselines
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```
UG-MARL additionally exposes `last_explanations() -> dict` (attention weights, σ, gate flag).

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
`TrafficEnv.reset()/step()` (Phase 4) must return `FeatureBuilder.observations(...)`.
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
- `scripts/train.py --config <yaml>` — TODO
- `scripts/evaluate.py --config <yaml> --checkpoint <path>` — TODO
- `scripts/run_ablation.py --config <yaml>` — TODO
