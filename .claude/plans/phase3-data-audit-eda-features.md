# Plan — Phase 3: data audit, EDA and feature engineering (ROADMAP 3.0, 3.3, 3.4; 3.5 gated)

Status: **Steps 1–7 executed 10 Nov 2026** (worklog/013); **Step 8 (ROADMAP 3.5) deferred by the user** — to be run after sign-off.
Tick steps `[x]` during execution and note deviations under each step.

## Goal

Finish ROADMAP Phase 3 so that Phase 4 (env + conventional baselines) can start on a verified,
documented, leak-free data layer.

| Task | Output | Serves |
|---|---|---|
| 3.0 Data audit | `docs/data_audit.md` (+ `.json`, figures) | F2, R3 (dataset & preprocessing), G3 |
| 3.3 EDA on the train split | `notebooks/01_eda_train.ipynb` + `docs/figures/eda/` | F2, R3, G5 |
| 3.4 Feature engineering | `trafficrl.data.roadnet`, `trafficrl.envs.features` (reused by the Phase 4 env) | F1, F2, R3, G6 |
| 3.5 Protocol freeze (**gated, optional in this plan**) | `docs/evaluation_protocol.md` | G4, R3, R5 |

3.1 (split) and 3.2 (leakage test) are already done (worklog/011) and are not redone.

## Decisions already taken (user, 10 Nov)

| Topic | Decision |
|---|---|
| Scope | 3.0, 3.3, 3.4 for certain. 3.5 is written as a separate final step (Step 8) that runs only after explicit sign-off; it can be dropped from this plan without touching Steps 1–7. |
| Normalisation | **Lane-capacity scaling** — nothing is fitted. |
| Seeds (for the freeze) | **5** for the main comparison and ablations (7.1, 7.2); **3** for robustness and generalisation (7.3, 7.4). |
| Δt and reward | Keep the defaults **Δt = 10 s**, **reward = −pressure**; EDA sanity-checks them on the train flow and they change only if the evidence is clearly against them. |

## Design

### D1. Road-network model — `src/trafficrl/data/roadnet.py`

One parser used by the audit, the features, the EDA and (later) the env, replacing the ad-hoc
parsing in `scripts/bench_cityflow.py` and `scripts/dataset_stats.py` for new code (those two
scripts are left as they are).

```python
@dataclass(frozen=True)
class Movement:            # one roadLink
    kind: str              # go_straight | turn_left | turn_right
    in_lanes: tuple[str, ...]    # "<road>_<laneIndex>"
    out_lanes: tuple[str, ...]

@dataclass(frozen=True)
class Intersection:
    id: str
    in_lanes: tuple[str, ...]    # 12, canonical order (below)
    out_lanes: tuple[str, ...]   # 12, canonical order
    movements: tuple[Movement, ...]          # 12, roadnet order
    green_phases: dict[int, tuple[int, ...]] # engine phase id 1..8 -> movement indices
    neighbors: tuple[str | None, ...]        # signalised neighbour per direction E, N, W, S

@dataclass(frozen=True)
class Roadnet:
    intersections: dict[str, Intersection]   # signalised only, sorted by id
    lane_length_m: dict[str, float]
    road_ids: frozenset[str]
    virtual_ids: frozenset[str]

def load_roadnet(path: Path) -> Roadnet
def lane_capacity(roadnet: Roadnet, vehicle_space_m: float) -> dict[str, int]
```

- **Canonical lane order** (so one shared network can serve all intersections): incoming
  roads sorted by travel direction suffix (`_0` E, `_1` N, `_2` W, `_3` S), then lane index
  0 (left), 1 (through), 2 (right) — verified on `intersection_1_1`: lane 0 → `turn_left`,
  1 → `go_straight`, 2 → `turn_right`.
- `YELLOW_PHASE = 0` (5 s, right turns only); action space = green phases 1–8.
- Fails loudly (`ValueError`) if a signalised intersection does not have 4 incoming roads,
  12 movements and 9 light phases — both cities satisfy this today.

### D2. Audit checks — `src/trafficrl/data/audit.py` (pure functions, counts only)

`audit_flow(flows, roadnet, episode_seconds) -> dict[str, int]` and
`audit_roadnet(raw) -> dict[str, int]`. Nothing is fitted.

| Group | Checks (each returns a count; expected 0 unless stated) |
|---|---|
| Missing | records missing `startTime` / `endTime` / `interval` / `route` / `vehicle` or any of the 9 vehicle fields; null or wrong-type values |
| Duplicates | exact duplicate records; note that vehicle IDs do not exist in the files (CityFlow generates them), so "duplicate ID" is checked as duplicate record |
| Impossible values | `startTime` < 0, ≥ episode, non-integer; `endTime` < `startTime`; route length < 2; road not in roadnet; **consecutive roads not connected** (no `roadLink` for the pair); route not starting / ending at a virtual node; road repeated within a route (loop); non-positive vehicle length, speed, acceleration, gap, headway; `maxSpeed` above the lane limit |
| Outliers | arrivals per 60 s bin: number of bins above Q3 + 1.5·IQR (reported, not removed); records with the same `startTime` on the same entry lane (cannot be inserted at once — reported as information) |
| Bias / coverage | (train only) share of demand per entry road and per movement type; signalised intersections or movements with zero demand |
| Privacy | field whitelist check (only the 5 documented keys), no ID / plate / date-like strings |
| Roadnet | 4 incoming roads, 12 `roadLinks`, 9 phases per signalised intersection; phase 0 = 5 s; road length > 0; lane width and `maxSpeed` > 0; every `availableRoadLinks` index valid; every road's end points exist |

**Policy:** the audit reports; it never edits or filters data (flow files are hash-pinned). If a
check finds a real defect, execution stops and the handling is decided with the user and
recorded in the Decision Log.

**Per split:** all six flow files get the count table. Plots (arrival-rate curve with the
flagged bins, route-length histogram, demand share per entry road) are drawn for the **train**
file only — `PLOT_SPLITS = ("train",)` in the script. Val, test and held-out get counts only.

### D3. Features — `src/trafficrl/envs/features.py`

For signalised intersection *i* with incoming lanes *l* ∈ L_in(i) (12), movements *m* (12) and
lane capacity C_l:

- Capacity: **C_l = ⌊ length_l / (vehicle length + minGap) ⌋ = ⌊ length_l / 7.5 m ⌋** →
  53 (400 m), 80 (600 m), 106 (800 m) vehicles.
- Vehicle count x_l (`get_lane_vehicle_count`), queue q_l (`get_lane_waiting_vehicle_count`,
  speed < 0.1 m/s).
- Normalised occupancy x̃_l = min(x_l / C_l, 1), queue q̃_l = min(q_l / C_l, 1).
- Movement pressure p_m = x̃_in(m) − mean over out-lanes of x̃_out ∈ [−1, 1].
- Phase: one-hot of the current green phase (8).
- **Observation o_i = [ q̃ (12) ‖ x̃ (12) ‖ p (12) ‖ one-hot phase (8) ] ∈ ℝ⁴⁴**, groups
  selectable by config.
- Raw intersection pressure (for the reward and Max-Pressure, defined here, used in Phase 4):
  P_i = Σ_{l∈L_in} x_l − Σ_{l∈L_out} x_l ; phase pressure = Σ over the phase's movements of
  (x_in − x_out). Reward (provisional until 3.5): r_i = −|P_i|.

```python
@dataclass(frozen=True)
class FeatureSpec:
    groups: tuple[str, ...] = ("queue", "count", "pressure", "phase")
    vehicle_space_m: float = 7.5
    clip: bool = True
    @property
    def dim(self) -> int

class FeatureBuilder:
    def __init__(self, roadnet: Roadnet, spec: FeatureSpec)   # reads the roadnet only
    def observation(self, iid, lane_count, lane_waiting, green_phase) -> np.ndarray  # float32
    def observations(self, lane_count, lane_waiting, green_phases) -> dict[str, np.ndarray]
    def phase_pressure(self, iid, lane_count) -> dict[int, float]   # for Max-Pressure
    def intersection_pressure(self, iid, lane_count) -> float       # for the reward
```

Fit with `reference/api.md`: `TrafficEnv.reset()/step()` (Phase 4) will return exactly
`FeatureBuilder.observations(...)`; agents never see raw engine dictionaries. No agent code is
written in this plan.

### D4. Train-split traces for EDA — `src/trafficrl/data/trace.py` + `scripts/collect_traces.py`

The flow file holds only demand; queues, pressure and travel times exist only once a
controller runs. EDA therefore needs short rollouts on the **train** flow:

| Controller | Why |
|---|---|
| Fixed-Time (30 s green, cyclic) | the "target" stand-in required by ROADMAP 3.3 (travel-time distribution) |
| Max-Pressure (Δt = 10 s) | a well-controlled regime; also the Δt sweep {5, 10, 15, 20 s} |
| Uniform-random phase (Δt = 10 s) | the badly-controlled regime an exploring RL agent visits first — shows the upper range of the features |

- `trace.py`: `make_engine(roadnet_path, flow_path, seed, threads, workdir)` (moved logic from
  the bench script; Phase 4 reuses it), three small phase-selection rules, and
  `collect_trace(...)` which records every Δt: time, intersection, per-lane count and queue,
  green phase, phase pressure; and per vehicle: entry time, exit time (from
  `engine.get_vehicles()` set differences) → travel time of completed trips.
- The rules here are plain functions used for data description only. Phase 4.4 / 4.5 implement
  the real `Agent`-interface baselines and must reproduce these numbers (cross-check).
- Output (gitignored, regenerable): `results/eda/<controller>[_dt<k>]_seed<s>/`
  `steps.csv.gz`, `vehicles.csv.gz`, `summary.json`, resolved `config.yaml`, `env.json`.
- **Seed check:** each controller is run with seeds 0, 1, 2. If CityFlow gives identical
  metrics for the rule-based controllers (expected: `laneChange` off, deterministic arrivals),
  that is recorded — it means evaluation variance comes only from training and action
  randomness, which matters for how 3.5 words the confidence intervals.

### D5. EDA notebook — `notebooks/01_eda_train.ipynb`

Thin notebook: loads the train flow via `flow_files("jinan", "train")` and the traces, calls
functions in `src/trafficrl/data/eda.py` (aggregation) and plots. Executed headless; figures
also saved to `docs/figures/eda/` (tracked, for the report and slides). Every figure has axis
labels with units, a caption, and a **2–4 sentence interpretation cell directly below it**.

| # | Figure / table | ROADMAP 3.3 item |
|---|---|---|
| 1 | Table: vehicles, trips per hour, intersections, lanes, routes, route-length stats | sample counts, summary stats |
| 2 | Network plot (signalised vs virtual nodes, road lengths, entry-demand arrows) | network plot |
| 3 | Arrival rate per minute over the hour, with peak marked | target distribution (demand), peaks |
| 4 | Travel-time histogram under Fixed-Time (completed trips; uncompleted count stated), with Max-Pressure overlaid | target distribution (outcome) |
| 5 | Missing-value / audit summary table (from `docs/data_audit.json`) | missing-value summary |
| 6 | Distributions of lane queue, lane vehicle count and movement pressure, raw and capacity-normalised, per controller | feature distributions; verifies the scaling |
| 7 | Turning ratios per approach (left / through / right) | turning ratios |
| 8 | Lane-load imbalance: demand per entry road and heat-map of mean queue per intersection | lane load imbalance |
| 9 | Queue vs. pressure scatter + correlation; total queue over time per controller | reward sanity check |
| 10 | Max-Pressure Δt sweep: average travel time and mean queue vs Δt ∈ {5, 10, 15, 20 s} | Δt sanity check |

Final cell: **"Insights that shaped later choices"** — Δt, normalisation (max observed x/C,
share of clipped values), reward, which intersections are loaded, expected episode backlog.
Held-out Hangzhou and Jinan val / test are **not** loaded anywhere in the notebook.

### D6. Experiment log — `src/trafficrl/utils/runlog.py`

Trace rollouts are simulator runs, so they follow the logging rule now rather than waiting for
5.4: `env_info() -> dict` (Python, PyTorch, CUDA, CityFlow commit if known, CPU, GPU, platform)
and `append_experiment_row(...)` writing to `docs/experiment_log.csv` with the header
`run_id, date, split, seed, model, hyperparameters, validation_metric, runtime_s, git_commit, notes`.
Phase 5.4 reuses both.

## Justification

| Decision | Why | Alternatives rejected |
|---|---|---|
| Lane-capacity scaling | Physical, bounded in [0, 1], independent of any controller or flow file, so it cannot leak; adapts automatically to Hangzhou's 600 m roads; no saved statistics to version | **Z-score fitted on train rollouts** — statistics depend on the behaviour policy, drift as the RL policy improves, and may not transfer to another city. **Both with a switch** — one more thing to tune for little gain. **No normalisation** — counts up to ~100 next to one-hot 0/1 inputs slow DQN training |
| Queue + count + pressure + phase (44-dim) | The three quantities in ROADMAP 3.4; count and queue differ (moving vs stopped vehicles); pressure is the Max-Pressure / PressLight signal, giving the network the fallback's own input | **Queue only** (IntelliLight-style subset) — loses approaching vehicles. **Lane segments / images** — larger state, no evidence it helps on these grids (LIT, PressLight findings). Groups stay config-selectable so a smaller state can be tried on validation |
| EDA traces from three rule controllers | Feature distributions do not exist without a controller; these three bracket good and bad control and need no learning | **Fixed-Time only** — under-represents congested states an exploring agent reaches. **Wait for Phase 4 agents** — blocks 3.3 and the Review evidence |
| Audit reports, never repairs | Files are hash-pinned and shared with published benchmarks; silent filtering would break comparability | **Auto-clean** — changes the benchmark without a recorded decision |
| Plots on train only; counts for other splits | Guidelines G5 and `database.md` rules 7 and 9; structural counts cannot inform modelling choices | **Plots for all splits** — test / held-out distributions would be seen before final evaluation |
| Keep Δt = 10 s, reward = −pressure, check in EDA | Literature convention (PressLight, CoLight, MPLight), keeps numbers comparable; the check uses the train flow only | **Sweep decides** — risks an unusual Δt chosen on one hour. **No check** — a guideline decision without evidence |
| Notebook for EDA, scripts for audit and traces | `CLAUDE.md` convention: notebooks for EDA only, importing from `src/`; anything reused lives in `src/` and is tested | **Scripts only** — ROADMAP 3.3 asks for a notebook with interpretation under each figure |

## Files

| File | Action | Purpose |
|---|---|---|
| `src/trafficrl/data/roadnet.py` | create | Roadnet parser, canonical lane order, phases, neighbours, lane capacity (D1) |
| `src/trafficrl/data/audit.py` | create | Structural audit checks returning counts (D2) |
| `src/trafficrl/data/trace.py` | create | Engine factory, rule controllers, trace collector (D4) |
| `src/trafficrl/data/eda.py` | create | Aggregations used by the notebook (arrival bins, turning ratios, load per road) |
| `src/trafficrl/envs/features.py` | create | `FeatureSpec`, `FeatureBuilder`, pressure definitions (D3) |
| `src/trafficrl/utils/runlog.py` | create | `env_info`, `append_experiment_row` (D6) |
| `scripts/audit_data.py` | create | Runs the audit on all splits → `docs/data_audit.{md,json}` + train-only figures |
| `scripts/collect_traces.py` | create | Train-split rollouts → `results/eda/…`, experiment-log rows |
| `configs/base.yaml` | modify | Add the `features:` block |
| `configs/eda.yaml` | create | Controllers, Δt sweep, seeds, bin width for the EDA traces |
| `notebooks/01_eda_train.ipynb` | create | EDA with interpretation under every figure (D5) |
| `docs/data_audit.md`, `docs/data_audit.json` | create (generated) | Audit summary with counts; machine-readable copy |
| `docs/figures/audit/*.png`, `docs/figures/eda/*.png` | create (generated) | Tracked figures for report / slides |
| `docs/experiment_log.csv` | create | Header + EDA rollout rows |
| `tests/data/test_roadnet.py`, `tests/data/test_audit.py`, `tests/data/test_trace.py`, `tests/envs/__init__.py`, `tests/envs/test_features.py`, `tests/utils/test_runlog.py` | create | See Tests |
| `docs/dataset_card.md` | modify | §5 points to the finished audit; correct anything the audit contradicts |
| `docs/how_it_works.md` | modify | State / normalisation paragraph now concrete (44-dim, capacity scaling) |
| `.claude/reference/architecture.md`, `api.md`, `database.md`, `testing.md` | modify | MDP state definition; `Roadnet` / `FeatureBuilder` signatures; rule 2 reworded ("no fitted statistics; capacity scaling from the roadnet"); new test list |
| `README.md`, `CLAUDE.md` | modify | New commands (`audit_data.py`, `collect_traces.py`, notebook execution); structure |
| `ROADMAP.md` | modify | Tick 3.0, 3.3, 3.4; Decision Log rows (normalisation, state vector, seeds, Δt / reward check result); Final-gate "Data audit and EDA completed" |
| `worklog/013-2026-11-10-phase3-audit-eda-features.md`, `worklog/README.md` | create / modify | Work log entry and index row |
| `docs/evaluation_protocol.md`, `PRD.md` | create / modify | **Step 8 only**, after sign-off |

## Config

`configs/base.yaml` (new block):

```yaml
features:
  groups: [queue, count, pressure, phase]   # observation = concatenation in this order (44-dim)
  normalisation: lane_capacity              # only supported value; nothing is fitted
  vehicle_space_m: 7.5                      # vehicle.length 5.0 m + minGap 2.5 m (dataset card §4.1)
  clip: true                                # clip x/C and q/C to [0, 1]
```

`configs/eda.yaml`:

```yaml
base: base.yaml
seed: 0
data: {dataset: jinan, split: train}        # the script rejects any other split
eda:
  seeds: [0, 1, 2]
  controllers:
    fixed_time:   {green_seconds: 30}
    max_pressure: {decision_interval: 10}
    random:       {decision_interval: 10}
  dt_sweep: {controller: max_pressure, values: [5, 10, 15, 20]}
  arrival_bin_seconds: 60
output: {results_dir: results/eda}
```

Seed handling: `seed_everything(seed)` + `seed_cityflow(engine, seed)`; the random controller
draws from `make_rng(seed)`. The resolved config is saved next to each trace.

## Data-leakage check

| Component | Splits read | Why this is safe |
|---|---|---|
| `audit_data.py` counts | train, val, test, held-out | Structural validity only (counts of defects); no distribution is shown and nothing is fitted. Allowed by `database.md` ("held-out city: audit only") |
| `audit_data.py` plots | **train only** | `PLOT_SPLITS = ("train",)`; test asserts it |
| `collect_traces.py` | **train only** | Calls `flow_files("jinan", "train")`; raises `SplitError` for any other `data.split` or dataset |
| EDA notebook | **train only** | Loads the train flow and `results/eda/` traces; a test scans the notebook source for other split names / file names |
| `FeatureBuilder` | none (roadnet only) | Capacity comes from road lengths and a constant vehicle spacing; it never opens a flow file. For Hangzhou it reads Hangzhou's roadnet geometry — a property of the map, not of the demand — so zero-shot evaluation stays zero-shot |
| Δt sweep, reward check | **train only** | Decisions are made before val / test are simulated at all |

No normalisation statistics exist, so there is nothing that could be computed from test data.
Val, test and held-out flows are not simulated anywhere in this plan.

## Tests

| File | Tests |
|---|---|
| `tests/data/test_roadnet.py` | hand-built 1-intersection roadnet: canonical lane order, movement → lanes, phases 1–8, yellow = 0; malformed intersection raises. On the real files (skipped if data absent, same pattern as `test_splits.py`): Jinan 12 / Hangzhou 16 intersections, 12 in-lanes each, neighbour counts (corner 2, edge 3, interior 4), capacities {53, 106} and {80, 106} |
| `tests/data/test_audit.py` | one synthetic defect per check (missing key, duplicate, bad start time, unknown road, disconnected route, looped route, bad vehicle value, extra key) → count 1; clean synthetic flow → all zeros; burst detector flags an injected spike |
| `tests/envs/test_features.py` | hand-built lane counts → exact expected 44-vector; values in [0, 1] (pressure in [−1, 1]); one-hot sums to 1; group selection changes `dim`; clipping at capacity; phase pressure picks the right phase on a toy state; `FeatureBuilder` built from a roadnet with **no flow file present** (leakage guard) |
| `tests/data/test_trace.py` | config with `split: val` / `test` / `heldout` or `dataset: hangzhou` → `SplitError`; notebook source contains no val / test / Hangzhou file names; `@pytest.mark.slow`: 300-step trace twice with the same seed → identical CSVs; vehicle travel times non-negative |
| `tests/utils/test_runlog.py` | header created once; rows appended, not overwritten; `env_info` has the required keys |

## Steps (each can be run and checked on its own)

- [x] **1. Roadnet model** — `data/roadnet.py` + tests. Check: `pytest tests/data/test_roadnet.py`.
- [x] **2. Audit** — `data/audit.py`, `scripts/audit_data.py`, tests; generate
      `docs/data_audit.{md,json}` and train-only figures. **Stop and report if any defect
      count is non-zero** (decide handling before continuing).
- [x] **3. Features** — `envs/features.py`, `features:` block in `base.yaml`, tests.
- [x] **4. Run log** — `utils/runlog.py`, `docs/experiment_log.csv` header, tests.
- [x] **5. Traces** — `data/trace.py`, `configs/eda.yaml`, `scripts/collect_traces.py`, tests;
      run it (3 controllers × 3 seeds + Δt sweep, ≈ 13 episodes, under a minute of simulation).
      Cross-check Fixed-Time / Max-Pressure against worklog/007 (515.2 s / 374.9 s).
- [x] **6. EDA** — `data/eda.py`, notebook, execute headless, save figures, write the
      interpretation cells and the insights list.
- [x] **7. Docs** — reference files, dataset card, how-it-works, README, CLAUDE.md; ROADMAP
      ticks and Decision Log; worklog 013. Full `pytest` + `ruff check src tests scripts`.
- [ ] **8. (Gated) Protocol freeze, ROADMAP 3.5** — only after the user has read the audit and
      EDA and confirmed the table below. Write `docs/evaluation_protocol.md`, remove
      "PROVISIONAL" from `configs/base.yaml` and `PRD.md` §5, dated Decision Log row, commit.

### Deviations during execution (10 Nov)

- **Step 2:** the check "route ends at a boundary node" fired on 1–2 % of every file. On
  inspection these are trips that end inside the network, which CityFlow simulates normally,
  so the check was reclassified from defect to information (`route_ends_inside_network`) and
  execution continued instead of stopping. Added two information checks not in the plan:
  `route_revisits_intersection` and `arrival_low_bins` (lower Tukey fence).
- **Step 4:** the experiment-log header has two extra columns, `metric_name` and
  `metric_value`, because the EDA runs are on the train split and have no validation metric.
- **Step 5:** 12 runs, not 13 (Max-Pressure Δt = 10 s, seed 0 is shared between the seed runs
  and the Δt check). Runs take ~10–15 s each because every vehicle is tracked. A first
  single-seed smoke run wrote 3 log rows; they were deleted before the real run.
- **Step 6:** 8 saved figures + 2 tables (items 1 and 5 are tables). The notebook was
  generated once from a cell list and is now the source.
- **Added files not in the plan:** `src/trafficrl/utils/plotting.py` (shared figure style and
  colours), `tests/data/toy.py` (hand-built roadnet for tests).
- **Findings that affect Step 8:** Max-Pressure is 8 % better at Δt = 15 s than at 10 s;
  −|pressure| misses states where upstream and downstream are both full; rule controllers
  are deterministic across seeds.

### Step 8 — values proposed for the freeze (to be confirmed, not yet decided)

| Item | Proposed value | Source |
|---|---|---|
| Datasets and split | Jinan train / val / test, Hangzhou held-out — `splits.json` hashes | worklog/011 |
| Episode length | 3600 s | flow files span 0–3599 s |
| Decision interval Δt | 10 s (unless Step 6 fig. 10 contradicts) | user decision + EDA |
| Yellow / transition | 5 s phase 0 on every phase change | roadnet |
| State | 44-dim capacity-normalised vector (D3) | this plan |
| Reward | r_i = −\|P_i\| (unless Step 6 fig. 9 contradicts) | user decision + EDA |
| Primary metric | average travel time (s) | `PRD.md` §1c |
| Secondary metrics | average waiting time, mean queue, throughput, max per-vehicle wait, % fallback actions, decision latency | `PRD.md` §5 |
| Seeds | 5 for 7.1 and 7.2 (0–4); 3 for 7.3 and 7.4 (0–2) | user decision |
| Reporting | mean ± std and 95% CI (Student t); test and held-out run once after selection on validation | guidelines G9, G10 |
| Success criteria | H1–H3 as in `PRD.md` §1c, unchanged | `PRD.md` |

## Verification

```bash
source .venv/bin/activate
pytest -m "not slow"                      # all green, incl. the new roadnet/audit/features/runlog tests
pytest -m slow tests/data/test_trace.py   # deterministic 300-step trace
ruff check src tests scripts              # no findings

python scripts/audit_data.py              # writes docs/data_audit.md, docs/data_audit.json,
                                          #   docs/figures/audit/*.png (train only); exit code 0
python scripts/collect_traces.py --config configs/eda.yaml
                                          # results/eda/<run>/{steps.csv.gz,vehicles.csv.gz,summary.json,
                                          #   config.yaml,env.json}; 13 new rows in docs/experiment_log.csv
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_train.ipynb
                                          # runs top to bottom without error; docs/figures/eda/*.png written
```

Expected:
- Audit: 6 flow files + 2 roadnets, every "impossible value" count 0 (consistent with the
  selection-time checks in the dataset card); outlier and same-lane-same-second counts reported.
- Traces: Fixed-Time ≈ 515 s and Max-Pressure ≈ 375 s average travel time on the train flow
  (worklog/007, ± small differences if the collector's state machine differs — any difference
  explained in the worklog); random clearly worse than both.
- Features: on all train traces, max x/C and max q/C reported; share of clipped values
  expected ≈ 0 (if not, the capacity formula is revisited before 3.4 is ticked).
- Notebook: 10 figures / tables, each followed by an interpretation cell.

## Experiment log

Rows appended to `docs/experiment_log.csv` by Step 5 — **split = train for every row**:

| Runs | Model | Seeds | Hyperparameters | Metric column |
|---|---|---|---|---|
| 3 | `fixed_time` (EDA trace) | 0, 1, 2 | green 30 s, yellow 5 s | avg travel time on **train** (validation column left empty, noted) |
| 3 | `max_pressure` (EDA trace) | 0, 1, 2 | Δt 10 s | same |
| 3 | `random` (EDA trace) | 0, 1, 2 | Δt 10 s | same |
| 4 | `max_pressure` Δt check | 0 | Δt ∈ {5, 10, 15, 20} s (10 s reuses the row above if identical) | same |

No training, no tuning, no validation / test / held-out run. The Δt values are written in
`configs/eda.yaml` before the first run. These are descriptive rule-based rollouts, not the
Phase 4 baseline results.

## Risks / open questions

| Risk / question | Handling |
|---|---|
| The audit finds a real defect (e.g. disconnected routes) | Stop at Step 2; options (keep and document / exclude with a Decision Log entry) are put to the user |
| CityFlow ignores the seed for rule-based controllers (identical runs) | Recorded in the EDA insights; 3.5 then states that seed variance comes from training and exploration only |
| Capacity under-estimates the real maximum (vehicles packed tighter than 7.5 m) | Checked in Step 6 fig. 6; `clip: true` bounds the input; formula revisited if clipping is not rare |
| Collector numbers differ from the bench script | Same state-machine logic is reused; any difference investigated before the EDA is written |
| Yellow phase in the one-hot | The feature takes the *green* phase (during transition: the pending one); the env in Phase 4 owns that logic — noted in `api.md` |
| Duplicate rule logic (bench script, `trace.py`, Phase 4 agents) | Accepted for now; Phase 4 agents become the single implementation and `trace.py` is pointed at them then |
| Time: Review 2 is today (10 Nov) | Steps 1–2 first (audit is review evidence); Step 6 figures 1–4 before the rest if time is short |
| Collection period of the datasets still "TODO verify" in the dataset card | Out of scope here; stays an open item for Phase 2.6 / report §4 |
| Two worklog entries are numbered 012 | Not fixed by this plan; the new entry is 013. Renumbering one of them is a separate small clean-up to confirm with the user |

## Worklog

Executing this plan ends with `worklog/013-2026-11-10-phase3-audit-eda-features.md` (What / Why
/ How / Result / Files changed per file / Commits / Next) and a new row in `worklog/README.md`,
per the template there. If Step 8 is run later, it gets its own entry (decision tag).
