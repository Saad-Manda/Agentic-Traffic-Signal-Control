# Plan — Phase 4: environment, metrics, robustness hooks, rule baselines, evaluation harness (ROADMAP 4.1–4.6)

Status: **draft, awaiting sign-off (10 Nov 2026)**. Tick steps `[x]` during execution and note
deviations under each step.

## Goal

Build the shared simulation and evaluation layer that every agent (rule-based, DQN, IDQN,
UG-MARL, ablations) will use, and produce the first baseline numbers on the **validation** flow.

| Task | Output | Serves |
|---|---|---|
| 4.1 Env wrapper | `trafficrl.envs.traffic_env.TrafficEnv` — per-intersection obs, neighbour map, phase control with yellow / min-green, per-intersection transitions | F1, R4, G7 |
| 4.2 Metrics | `trafficrl.envs.metrics.EpisodeTracker` + `trafficrl.utils.stats` (mean ± std, 95 % CI) | F6, R5, G9 |
| 4.3 Robustness hooks | `trafficrl.envs.perturb` — sensor noise, detector dropout, demand surge (protocol §6 definitions) | F8, R5, G11 |
| 4.4 Fixed-Time agent | `trafficrl.agents.baselines.fixed_time.FixedTimeAgent` | F3, R4, G7 |
| 4.5 Max-Pressure agent | `trafficrl.agents.baselines.max_pressure.MaxPressureAgent` (also the UG-MARL fallback) | F3, F4, R4, G7 |
| 4.6 Evaluation script | `scripts/evaluate.py`: any agent × N seeds × split → `metrics.csv`, `summary.csv`, experiment-log rows | F6, R5, G9, G10 |

Out of scope: RL agents and `scripts/train.py` (Phase 5), native-Ubuntu build (ROADMAP 0.1b stays
a separate open task), any test / held-out run (Phase 7).

## Decisions already taken (user, 10 Nov)

| Topic | Decision |
|---|---|
| Decision timing | **Per-intersection clocks** (event-driven). Each intersection holds its green for Δt = 15 s, then decides; a change inserts 5 s of yellow (phase 0) before the new green. Intersections drift out of sync after their first switch. Matches the frozen protocol §2 literally and the EDA traces that chose Δt. |
| Splits run in Phase 4 | **Train** (sanity check against the EDA traces) and **val** (the Fixed-Time / Max-Pressure validation numbers). **Test and held-out only in Phase 7**, through an explicit `--final` flag, together with every other agent. |
| ROADMAP 0.1b (native Ubuntu) | Not part of this plan. Phase 4 is developed and verified on WSL. |

## Design

### D1. Engine factory — `src/trafficrl/envs/engine.py` (moved, not rewritten)

`make_engine(roadnet_path, flow_path, seed, threads, workdir)` moves from
`trafficrl.data.trace` to `trafficrl.envs.engine`; `trace.py` re-imports it so the EDA script
and its tests keep working unchanged. Reason: the env must not depend on an EDA module.

### D2. Environment — `src/trafficrl/envs/traffic_env.py`

```python
@dataclass(frozen=True)
class Transition:                     # one per intersection per completed decision interval
    iid: str
    obs: dict[str, np.ndarray]        # observations of ALL intersections at i's decision time
    action: int                       # green phase chosen (1..8)
    reward: float                     # −|P_i| at i's next decision time (true counts)
    next_obs: dict[str, np.ndarray]   # ALL observations at i's next decision time
    done: bool                        # episode ended (t = 3600) before i decided again
    duration_s: int                   # 15 (kept phase) or 20 (yellow 5 + green 15)

class TrafficEnv:
    def __init__(self, cfg: dict, seed: int, *, allow_final: bool = False,
                 workdir: Path | None = None, track: str = "full"): ...
    # properties
    intersection_ids: list[str]               # sorted, fixed order (= index order below)
    obs_dim: int                              # 44
    phase_ids: tuple[int, ...]                # (1..8)
    def neighbors(self, iid) -> list[str]     # signalised neighbours (boundary dropped)
    def neighbor_index(self) -> tuple[np.ndarray, np.ndarray]
                                              # (n_int × 4) int indices, (n_int × 4) bool mask,
                                              # direction order E, N, W, S — for batched attention
    # loop
    def reset(self) -> tuple[dict[str, np.ndarray], dict]           # obs, info (all ready at t=0)
    def step(self, actions: dict[str, int]) -> tuple[obs, rewards, done, info]
    def metrics(self) -> dict[str, float]                           # episode metrics (D3)
```

**Timing (per-intersection clocks).** The engine runs at 1 s. Each intersection i has
`(green, seconds_left, pending)`. When its green has run for Δt it is *ready*:
- `step(actions)` applies the action of every **ready** intersection (actions for others are
  ignored, so agents may always return a full dict). Same phase → another Δt of green, no
  yellow. New phase → `YELLOW_PHASE` for 5 s, then the new green for Δt.
- The engine then advances second by second until at least one intersection is ready again
  or t = 3600, and returns. Because Δt = 15 and yellow = 5, every decision falls on a 5 s grid.
- Minimum green = Δt (a green is never cut short) — protocol §2.
- At `reset()` (t = 0) every intersection is ready; there is no initial phase forced by the env.
- Off-by-one note: the EDA trace (`collect_trace`) gives the **first** green 14 s, not 15 s (its
  counter is decremented before the first engine step; later greens are a correct 15 s). The env
  does not copy this; numbers on the train flow are therefore expected to be close to, but not
  bit-identical with, the EDA traces (Verification V3).

**Returned values.**
- `obs`: `FeatureBuilder.observations(lane_count, lane_waiting, green_phases)` for **all**
  intersections, computed from the *observed* (possibly perturbed, D4) lane readings. During
  yellow the phase one-hot is the pending green (api.md rule).
- `rewards`: `{iid: −|P_i|}` for the intersections whose interval just ended, from the **true**
  lane counts (`FeatureBuilder.intersection_pressure`). Rewards are a training signal, not a
  sensor, so perturbations never touch them.
- `info`: `t` (s), `ready` (sorted list of iids that must act now), `phase` (current green or
  pending green per iid), `green_elapsed` (s of the current green, 0 during yellow),
  `lane_count` (observed counts, vehicles — what Max-Pressure reads), `transitions`
  (list of `Transition` completed by this step).
- `done` = t ≥ 3600. Unfinished intervals at the end produce a transition with `done = True`.

**Split guard.** The env resolves `cfg["data"]` through `flow_files(dataset, split)` (hash
check) and raises `SplitError` for `test` / `heldout` unless constructed with
`allow_final=True`. `allow_final` is a constructor argument, never a config key, so a config
override cannot reach the test flows; only `scripts/evaluate.py --final` passes it (Phase 7).

**Seeding.** `seed` → `make_engine(seed)` + `seed_cityflow`; a separate `make_rng(seed + 10_000)`
stream drives perturbations so noise draws do not depend on how many agent RNG draws happened.

### D3. Metrics — `src/trafficrl/envs/metrics.py`

`EpisodeTracker` is fed by the env after every engine second and every env step.

| Metric (protocol §5) | Computation |
|---|---|
| `avg_travel_time_s` (**primary**) | `engine.get_average_travel_time()` at t = 3600 (includes vehicles still inside) |
| `avg_waiting_time_s` | per vehicle, seconds with speed < 0.1 m/s (`get_vehicle_speed()` each second); mean over vehicles that entered |
| `max_waiting_time_s` | max over vehicles of total waiting seconds (safety / starvation) |
| `avg_queue_veh` | mean over env steps (decision steps) and over incoming lanes of all intersections of waiting vehicles (true counts) |
| `throughput_veh` | vehicles that left the network by t = 3600 (vehicle-set difference each second) |
| `vehicles_entered`, `in_network_at_end` | bookkeeping / sanity |
| `phase_switches` | number of yellow insertions (instability indicator, Phase 8.4) |
| `fallback_rate` | share of decisions flagged `fallback` by the agent (0 for rule agents; UG-MARL fills it) |
| `decision_latency_ms` | measured in `evaluate.py`: wall time of `agent.act()` per call, which always computes all intersections (protocol: "batch of all intersections"); mean and p95 |

Tracking levels: `track="full"` (all of the above; evaluation) and `track="light"` (travel time,
queue, throughput only — skips per-vehicle speed reads; for RL training in Phase 5). The cost
of `full` is measured in Step 3. Optional per-step trace (`steps.csv.gz`: t, iid, phase, queue,
pressure) for Phase 8 error analysis, off by default.

`src/trafficrl/utils/stats.py`: `summarise(values) -> {mean, std, ci95_low, ci95_high, n,
deterministic}` — sample std (ddof = 1), 95 % CI = mean ± t₀.₉₇₅,n−1 · std/√n with a small
built-in Student-t table (n − 1 = 1…30; scipy is not a dependency). If all values are equal
`deterministic = True` and the CI collapses to the value (protocol §7).

### D4. Robustness hooks — `src/trafficrl/envs/perturb.py`

Follows protocol §6 exactly; config block `robustness:` (all off by default).

```python
class SensorModel:                                    # applied to every observed reading
    def __init__(self, capacity: dict[str, int], noise_std: float, dropout_p: float, rng)
    def observe(self, lane_count, lane_waiting) -> tuple[dict, dict]
```
- Noise: per lane, x̃ = x / C_l, x̃′ = clip(x̃ + N(0, σ²), 0, 1), reported as x̃′·C_l vehicles
  (so Max-Pressure, which works in vehicles, sees the same corrupted detector as the RL
  features). Count and waiting readings get independent draws.
- Dropout: per lane and per reading, with probability p the detector reports 0 for both count
  and waiting (a dead detector loses both).
- Readings are perturbed once per env step over **all** lanes, so a lane that is outgoing for
  one intersection and incoming for its neighbour shows the same value to both.
- `surge_flow(flow_path, out_dir, rate=0.5, rng) -> Path`: each vehicle duplicated with
  probability 0.5 (same route, `startTime`/`endTime` + 1 s); writes the new flow file and a
  copy of the roadnet into the run's workdir (CityFlow needs both in one directory) and returns
  the path; the derived file's SHA-256 is saved in the run outputs. The source file still comes
  through `flow_files` (hash check); `splits.json` is never touched.
- Surge factor in config is `surge: 1.5` (expected demand multiplier) → rate = surge − 1.

Phase 4 only implements and tests the hooks (plus a smoke run on train/val); robustness numbers
on test are Phase 7.3.

### D5. Agent interface and rule agents — `src/trafficrl/agents/`

The planned `act(obs, explore)` signature is extended with `info`, because Max-Pressure (and the
UG-MARL gate) need observed raw lane counts and Fixed-Time needs the green elapsed time; neither
is in the 44-value feature vector. Updated in `.claude/reference/api.md`.

```python
# agents/base.py
class Agent(Protocol):
    name: str
    def act(self, obs: dict[str, np.ndarray], info: dict, explore: bool = False) -> dict[str, int]
    def observe(self, transitions: list[Transition]) -> None     # no-op for rule agents
    def update(self) -> dict[str, float]                         # {} for rule agents
    def save(self, path) -> None ; def load(self, path) -> None
    def last_explanations(self) -> dict                          # {} by default; UG-MARL: σ, α, fallback
class RuleAgent:  # base with the no-op methods

# agents/baselines/fixed_time.py
class FixedTimeAgent(RuleAgent):   # green_seconds = 30 (protocol §4), order 1 → 8
    # at a ready intersection: keep the phase if green_elapsed < green_seconds else next phase;
    # at t = 0: phase 1. Requires green_seconds to be a multiple of Δt (else ValueError).

# agents/baselines/max_pressure.py
class MaxPressureAgent(RuleAgent):  # needs the FeatureBuilder (roadnet only)
    def choose(self, iid, lane_count) -> int   # argmax_phase Σ(x_in − x_out), ties → lowest id
    # reused by UG-MARL as the fallback (Phase 6)

# agents/__init__.py
make_agent(name: str, env: TrafficEnv, cfg: dict) -> Agent    # registry: fixed_time, max_pressure
```

Fixed-Time timing check: with Δt = 15 the agent keeps its phase at the 15 s decision and
switches at 30 s, giving 30 s green + 5 s yellow per phase — the frozen protocol value.

### D6. Evaluation — `src/trafficrl/eval/runner.py` + `scripts/evaluate.py`

```python
def run_episode(env, agent, *, explore=False) -> dict      # metrics + latency + fallback rate
def evaluate(cfg, agent_name, seeds, split, *, checkpoint=None, final=False, out_dir) -> pd.DataFrame
```
```bash
python scripts/evaluate.py --config configs/eval_baselines.yaml --agent max_pressure \
       --split val --seeds 0 1 2 3 4 [--checkpoint PATH] [--final] [--trace] [overrides ...]
```
- Per (agent, split, seed): fresh env + agent, `seed_everything(seed)`, one episode.
- Outputs `results/eval/<run_id>/`: `metrics.csv` (one row per seed), `summary.csv`
  (per metric: mean, std, CI, n, deterministic), resolved `config.yaml`, `env.json`, and for
  surge runs the derived flow hash. `run_id = <agent>_<split>_<condition>_<timestamp>`.
- Appends **one row per seed** to `docs/experiment_log.csv`: split, seed, model, hyperparameters
  (Δt, yellow, robustness settings, agent params), `metric_name = avg_travel_time_s`,
  `validation_metric` filled when split = val, runtime, git commit.
- `--split test|heldout` without `--final` → exits with an error before any simulation.
  `--final` additionally prints a warning and records `final=true` in the notes column.
- Rule agents load no checkpoint; `--checkpoint` is passed to `agent.load` for RL agents later.

## Justification

| Decision | Alternatives rejected | Reason |
|---|---|---|
| Per-intersection clocks | Synchronous step with yellow outside Δt (CoLight/LibSignal style); synchronous with yellow inside Δt | Only option that implements the frozen protocol (§2: yellow on top of Δt, min green = Δt) without a Decision Log change, keeps Fixed-Time at exactly 30 s, and matches the setting in which Δt = 15 s was chosen (EDA). Cost (per-intersection transitions) is carried by `Transition`. |
| Own wrapper, no Gym/PettingZoo | PettingZoo `ParallelEnv`, LibSignal | Asynchronous agents fit PettingZoo's AEC API poorly; LibSignal brings its own configs/agents and a second simulator layer. Our interface is small and already specified in api.md. |
| `info` passed to `act` | Max-Pressure from the normalised pressure features; agents calling the env | Protocol §4 defines Max-Pressure on raw vehicle counts; agents never touch the engine (architecture.md). |
| Noise applied to lane readings, not to the feature vector | Noise on the 44-vector | Protocol §6: perturbations affect what *every* controller observes, Max-Pressure included; one sensor model feeds both. |
| Reward from true counts | Reward from perturbed counts | Rewards are only used in training (clean); robustness is an evaluation condition. |
| Hand-written t-table | scipy dependency | One function needs it; avoids a new pinned dependency. Tested against tabulated values. |
| Test/held-out guard as constructor arg | Config key | Config overrides are free text on the CLI; the guard must need a deliberate code path. |

## Files

| File | Action | Purpose |
|---|---|---|
| `src/trafficrl/envs/engine.py` | create | `make_engine` (moved from `data/trace.py`) |
| `src/trafficrl/data/trace.py` | modify | import `make_engine` from `envs.engine` (no behaviour change) |
| `src/trafficrl/envs/traffic_env.py` | create | `TrafficEnv`, `Transition` (4.1) |
| `src/trafficrl/envs/metrics.py` | create | `EpisodeTracker` (4.2) |
| `src/trafficrl/envs/perturb.py` | create | `SensorModel`, `surge_flow` (4.3) |
| `src/trafficrl/envs/__init__.py` | modify | export `TrafficEnv`, `Transition` |
| `src/trafficrl/utils/stats.py` | create | `summarise`, Student-t table (4.2) |
| `src/trafficrl/agents/base.py` | create | `Agent` protocol, `RuleAgent` |
| `src/trafficrl/agents/baselines/fixed_time.py` | create | 4.4 |
| `src/trafficrl/agents/baselines/max_pressure.py` | create | 4.5 |
| `src/trafficrl/agents/__init__.py` | modify | `make_agent` registry |
| `src/trafficrl/eval/__init__.py`, `eval/runner.py` | create | `run_episode`, `evaluate` (4.6) |
| `scripts/evaluate.py` | create | CLI (4.6) |
| `configs/eval_baselines.yaml` | create | baseline evaluation config (below) |
| `configs/base.yaml` | modify | add `robustness:` (all off) and `agents:` blocks — new keys only, frozen keys untouched |
| `tests/envs/test_traffic_env.py`, `test_metrics.py`, `test_perturb.py` | create | see Tests |
| `tests/agents/__init__.py`, `test_rule_agents.py`, `test_interface.py` | create | see Tests |
| `tests/utils/test_stats.py`, `tests/eval/__init__.py`, `tests/eval/test_evaluate.py` | create | see Tests |
| `tests/data/toy.py` | modify (if needed) | a toy flow file for the toy roadnet, for fast simulator tests |
| `.claude/reference/api.md` | modify | implemented env / agent / eval signatures (`act(obs, info, explore)`) |
| `.claude/reference/testing.md`, `architecture.md` | modify | new tests; timing model; metrics |
| `CLAUDE.md`, `README.md` | modify | `evaluate.py` command moves from "planned" to real; expected val numbers |
| `docs/how_it_works.md` | modify | env timing paragraph (keep in sync with protocol) |
| `docs/experiment_log.csv` | append | rows from Steps 7–8 |
| `ROADMAP.md` | modify | tick 4.1–4.6, Decision Log rows (timing model, API change, Phase-4 split use) |
| `worklog/015-2026-11-10-phase4-env-baselines.md`, `worklog/README.md` | create / modify | work log |

## Config

`configs/base.yaml` (new keys only; `sim`, `features`, `data` stay frozen):
```yaml
robustness:            # protocol §6 — evaluation conditions; all off = clean
  noise_std: 0.0       # σ on capacity-scaled lane values  {0, 0.1, 0.2} in 7.3
  dropout_p: 0.0       # per-lane, per-reading detector loss {0, 0.1, 0.2}
  surge: 1.0           # expected demand multiplier (1.5 in 7.3)

agents:
  fixed_time:   {green_seconds: 30}   # protocol §4
  max_pressure: {}                    # every Δt; ties → lowest phase id

eval:
  track: full          # full | light
  trace: false         # write steps.csv.gz for error analysis
```
`configs/eval_baselines.yaml`: `base: base.yaml`, `seed: 0`, `data: {dataset: jinan, split: val}`,
`eval: {seeds: [0, 1, 2, 3, 4]}`, `output: {results_dir: results/eval}`.
Seeds: one run per seed, `seed_everything(seed)` + CityFlow seed + perturbation stream
`seed + 10000`. No tuning in Phase 4, so no search space is needed.

## Data-leakage check

- Reads: Jinan **train** (Step 7 sanity check) and Jinan **val** (Step 8 baseline numbers), each
  through `flow_files` (hash-checked). Roadnet for features/capacities as before (no flow data).
- Test and Hangzhou are unreachable: `TrafficEnv` raises unless `allow_final=True`, which only
  `evaluate.py --final` sets; `--final` is not used in this phase (tested in `test_evaluate.py`).
- Nothing is fitted (rule agents, capacity scaling). The surge hook writes a derived file to the
  run's workdir; it is never added to `splits.json`.
- Validation numbers of rule baselines are not used to choose anything in Phase 4; they become
  the "Validation score" column of 7.1.

## Tests (`pytest`; simulator tests marked `slow` and run on the toy network)

- **Env timing** (`test_traffic_env.py`): keep-phase → next decision after 15 s, no yellow;
  change → yellow 5 s then green, next decision after 20 s; decisions on the 5 s grid; all
  ready at t = 0; non-ready actions ignored; episode ends at exactly 3600 (toy: shorter
  `episode_seconds`); `Transition.duration_s` ∈ {15, 20}; final transitions `done=True`.
- **Env outputs**: obs dim 44 for every intersection; phase one-hot = pending green during
  yellow; reward = −|P_i| from true counts even with noise on; neighbour map and
  `neighbor_index` match `Intersection.neighbors`.
- **Split guard**: `test` / `heldout` raise `SplitError` without `allow_final`; train/val work.
- **Determinism (slow)**: same seed → identical metrics and transitions for Max-Pressure; with
  noise on, same seed → identical, different seed → different observations.
- **Metrics** (`test_metrics.py`): tracker on a hand-fed sequence of vehicle sets/speeds gives
  exact travel/wait/throughput/max-wait/queue values.
- **Perturb** (`test_perturb.py`): σ = 0 and p = 0 → identity; values stay within [0, C_l];
  p = 1 → all zero; empirical dropout rate ≈ p (tolerance); surge rate 0.5 → ≈ 1.5× vehicles,
  duplicates have start + 1 s and identical routes; deterministic given the RNG.
- **Rule agents** (`test_rule_agents.py`): Max-Pressure picks the max-pressure phase on
  hand-built counts, ties → lowest id; Fixed-Time 1 → 2 → … → 8 → 1 with 30 s greens, keeps at
  15 s; rejects green_seconds not a multiple of Δt.
- **Interface** (`tests/agents/test_interface.py`): every registered agent implements the
  protocol and returns a valid phase for every intersection.
- **Stats** (`test_stats.py`): t-values vs table (n = 3: 4.303, n = 5: 2.776); known-sample
  CI; constant sample → `deterministic`, std 0.
- **Evaluate** (`test_evaluate.py`, slow, toy net): writes `metrics.csv`/`summary.csv`, appends
  one log row per seed (to a temp log path); `--split test` without `--final` exits non-zero.

## Steps (each can be run and checked on its own)

1. [ ] **Engine move + agent base.** Create `envs/engine.py`, re-import in `trace.py`,
   `agents/base.py`. Check: `pytest tests/data` still green.
2. [ ] **Env wrapper (4.1)** with the timing model, split guard, transitions; toy-network tests.
   Check: `pytest tests/envs/test_traffic_env.py`.
3. [ ] **Metrics (4.2)** — `EpisodeTracker`, `utils/stats.py`; tests. Measure the wall-time cost
   of `track=full` vs `light` on one Jinan train episode (record in worklog).
4. [ ] **Robustness hooks (4.3)** — `SensorModel`, `surge_flow`; tests.
5. [ ] **Rule agents (4.4, 4.5)** + registry; tests.
6. [ ] **Evaluation harness (4.6)** — `eval/runner.py`, `scripts/evaluate.py`,
   `configs/eval_baselines.yaml`, `base.yaml` new keys; tests.
7. [ ] **Train sanity run**: Fixed-Time and Max-Pressure, seed 0, on the train flow; compare with
   the EDA traces (V3). Also one smoke run each with `robustness.noise_std=0.1`,
   `dropout_p=0.1`, `surge=1.5` on train (checks the hooks end to end; not reported).
8. [ ] **Validation baseline numbers**: Fixed-Time and Max-Pressure × seeds 0–4 on val.
9. [ ] **Docs**: api.md, architecture.md, testing.md, how_it_works.md, CLAUDE.md, README
   (exact commands + expected val numbers), ROADMAP ticks + Decision Log rows; worklog 015.
   Commit per `.claude/commands/commit.md` (no Claude co-author).

## Verification

| # | Command | Expected |
|---|---|---|
| V1 | `pytest -m "not slow"` then `pytest` | all green |
| V2 | `ruff check src tests scripts` | clean |
| V3 | `python scripts/evaluate.py --config configs/eval_baselines.yaml --agent max_pressure --split train --seeds 0` (and `--agent fixed_time`) | MP avg travel time ≈ **344 s** (EDA 344.13), FT ≈ **515 s** (EDA 515.21), within ±3 % (first-green off-by-one in the EDA trace, D2). FT `never_entered`/throughput of the same order as EDA. A larger gap is a bug to explain before Step 8. |
| V4 | same, `--split val --seeds 0 1 2 3 4` | `results/eval/<run>/metrics.csv` with 5 rows; identical values across seeds → `summary.csv` marks `deterministic`; MP < FT on avg travel time; 10 new rows in `docs/experiment_log.csv` with `validation_metric` filled |
| V5 | `python scripts/evaluate.py ... --split test` | exits non-zero with a "use --final in Phase 7" message; no simulation, no log row |
| V6 | train smoke runs with noise / dropout / surge | run completes; surge shows ≈ 1.5× `vehicles_entered`; noisy MP travel time ≥ clean (expected, not required) |
| V7 | Decision latency | reported per agent in `metrics.csv`; rule agents ≪ 1 ms per joint decision |

## Experiment log

| Step | Runs | Split | Rows appended |
|---|---|---|---|
| 7 | FT, MP × seed 0 (clean) | train | 2 (note: "env sanity check vs EDA") |
| 7 | MP × seed 0 × {noise 0.1, dropout 0.1, surge 1.5} | train | 3 (note: "hook smoke test; not reported") |
| 8 | FT, MP × seeds 0–4 | val | 10 (`validation_metric` = avg travel time) |

No tuning, no test or held-out flow. Phase 4 is after the protocol freeze (3.5, 10 Nov).

## Risks / open questions

| Risk | Mitigation |
|---|---|
| `track=full` per-second `get_vehicle_speed()` slows episodes | Measured in Step 3; training uses `light`; evaluation episodes are few. |
| Train numbers differ from EDA by more than 3 % | Compare per-interval timings with a trace on the toy net; document the cause before Step 8. |
| "Mean over decision steps" for queue is ambiguous with asynchronous decisions | Interpreted as "every env step" (every time ≥ 1 intersection decides, ≈ every 5 s). Clarification noted in the protocol's change table as a dated, non-substantive clarification with a Decision Log row — confirm at sign-off. |
| API change (`act` gets `info`) affects Phase 5/6 code | Done now, before any RL agent exists. |
| Async transitions complicate batched attention in Phase 6 | `Transition` stores all intersections' obs, and `neighbor_index` gives padded indices; batching is per sampled transition, not per time step. |
| Surge duplicates at start + 1 s may collide at entry lanes (CityFlow queues them) | Expected and realistic; `never_entered` is reported. |

## Worklog

Executing this plan ends with `worklog/015-2026-11-10-phase4-env-baselines.md` (What / Why /
How / Result / per-file Files changed table / Commits / Next) and its row in
`worklog/README.md`, including the V3 comparison, the `full` vs `light` timing, the validation
numbers and any deviation from this plan.
