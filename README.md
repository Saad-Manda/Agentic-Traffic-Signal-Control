# Uncertainty-Gated Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control (UG-MARL)

Built on the CityFlow simulator. Repository name: `Agentic-Traffic-Signal-Control`.

## Install (Ubuntu 24.04: native or WSL2)

Tested on WSL2 Ubuntu 24.04, Python 3.12.3, g++ 13.3, CMake 3.28.3 (28 Oct 2026).
Clone the repo into the Linux filesystem (e.g. `~/Agentic-Traffic-Signal-Control`), not `/mnt/...`.

### 1. System packages

```bash
sudo apt update && sudo apt install -y build-essential cmake python3-venv python3-dev git
```

### 2. Virtual environment

```bash
cd ~/Agentic-Traffic-Signal-Control
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt   # exact pins (torch with bundled CUDA, ~3 GB download)
pip install -e .                  # makes `src/trafficrl` importable (`import trafficrl`)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 3. CityFlow (built from source, pybind11 bumped for Python 3.12)

CityFlow pins an old pybind11 fork that does not build on Python 3.12, so we replace the
submodule with upstream pybind11 **v2.13.6**.

```bash
mkdir -p ~/src && cd ~/src
git clone https://github.com/cityflow-project/CityFlow
cd CityFlow
git checkout 81ee0f47659ca66177a71f81676691c58ee89184   # pinned commit (Aug 2025)
git submodule update --init extern/rapidjson
rm -rf extern/pybind11
git clone --depth 1 --branch v2.13.6 https://github.com/pybind/pybind11 extern/pybind11
pip install .            # with the project .venv active
python -c "import cityflow; print(cityflow.__file__)"
```

### 4. Data (Jinan 3×4, Hangzhou 4×4)

From the CoLight repo (commit `04440b7`), into the gitignored `data/raw/`:

```bash
cd ~/Agentic-Traffic-Signal-Control
mkdir -p data/raw && cd data/raw
git clone --depth 1 --filter=blob:none --sparse https://github.com/wingsweihua/colight colight_src
cd colight_src && git sparse-checkout set data/Jinan data/Hangzhou
```

Then verify the files against the pinned hashes and print their sizes (fails on any mismatch):

```bash
cd ~/Agentic-Traffic-Signal-Control
python scripts/dataset_stats.py --json docs/dataset_stats.json
```

The data has no license file. Use it for academic purposes only and cite CoLight (Wei et al.,
CIKM 2019). Split and data dictionary: `docs/dataset_card.md`.
New to the project? `docs/how_it_works.md` explains in plain language what the data is,
what the simulator does with it, and how the agents are trained.

### 5. Smoke test + timing

```bash
cd ~/Agentic-Traffic-Signal-Control
python scripts/bench_cityflow.py \
    --data-dir data/raw/colight_src/data/Jinan/3_4 \
    --roadnet roadnet_3_4.json --flow anon_3_4_jinan_real.json \
    --out results/bench/jinan_$(hostname).json
```

Reference (WSL2, 1 thread, one 3600 s episode, 12 intersections, 6295 vehicles, seed 0):

| Controller | Wall time | Steps/s | Avg travel time | Mean queue (veh) |
|---|---|---|---|---|
| Fixed-Time (30 s green, 5 s transition) | 2.7 s | 1327 | 515.2 s | 430.5 |
| Max-Pressure (10 s decisions, 5 s transition) | 1.8 s | 1983 | 374.9 s | 247.4 |

`--threads 4` gives identical metrics but is slower (~4.5 s) on this small network, so use 1
thread per simulator and parallelise across seeds/runs instead.

### 6. Data audit, EDA traces and EDA notebook (ROADMAP Phase 3)

```bash
cd ~/Agentic-Traffic-Signal-Control
python scripts/audit_data.py                                 # all splits, counts only; exit 1 on a defect
python scripts/collect_traces.py --config configs/eda.yaml   # train flow only, ~2.5 min
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_train.ipynb
```

| Command | Output | Expected |
|---|---|---|
| `audit_data.py` | `docs/data_audit.md`, `docs/data_audit.json`, `docs/figures/audit/*.png` (train split only) | every defect count 0 for 6 flow files and 2 road networks |
| `collect_traces.py` | `results/eda/<run>/{steps.csv.gz, vehicles.csv.gz, summary.json, config.yaml, env.json}`; 12 rows appended to `docs/experiment_log.csv` | average travel time on the train flow: Fixed-Time 515.2 s, Max-Pressure 374.9 s (Δt = 10 s), random 646–657 s; Max-Pressure at Δt = 5 / 15 / 20 s: 451.1 / 344.1 / 347.3 s |
| notebook | `docs/figures/eda/*.png` (8 figures), tables inside the notebook | runs top to bottom; reads the train split only |

The simulator is deterministic for a fixed flow file and a deterministic controller, so the
Fixed-Time and Max-Pressure numbers reproduce exactly. These rollouts describe the data; they
are not the baseline results (Phase 4).

### 7. Environment and rule baselines (ROADMAP Phase 4)

```bash
cd ~/Agentic-Traffic-Signal-Control
python scripts/evaluate.py --agent fixed_time   --split val   # 5 seeds, ~25 s
python scripts/evaluate.py --agent max_pressure --split val   # 5 seeds, ~25 s
python scripts/evaluate.py --agent max_pressure --split train --seeds 0   # sanity check vs EDA
```

Each run writes `results/eval/<agent>_<dataset>-<split>_<condition>_<time>/` with
`metrics.csv` (one row per seed), `summary.csv` (mean, std, 95 % CI, deterministic flag),
`config.yaml`, `env.json`, `data.json`, and appends one row per seed to
`docs/experiment_log.csv`. `--split test` / `heldout` exits with an error unless `--final`
is given (Phase 7 only). Robustness conditions are config overrides, e.g.
`robustness.noise_std=0.1`, `robustness.dropout_p=0.1`, `robustness.surge=1.5`.

Expected (Jinan, 3600 s, Δt = 15 s, 5 s yellow; deterministic, identical for every seed):

| Agent | Split | Avg travel time | Avg waiting time | Throughput | Max per-vehicle wait |
|---|---|---|---|---|---|
| Fixed-Time | val | 434.84 s | 187.78 s | 3991 veh | 1124 s |
| Max-Pressure | val | 326.11 s | 83.42 s | 4106 veh | 1334 s |
| Fixed-Time | train | 516.79 s | 256.10 s | 4971 veh | 1725 s |
| Max-Pressure | train | 349.67 s | 114.77 s | 5602 veh | 1897 s |

The train values differ slightly from the EDA traces (515.21 / 344.13 s) because the EDA trace
gives the first green 14 s instead of 15 s; copying that off-by-one into the env reproduces
Max-Pressure's 344.13 s exactly (worklog/015).

### 8. Tests and lint

```bash
pytest            # all tests; `pytest -m "not slow"` skips simulator-heavy ones
ruff check src tests scripts
```

## Repository layout

```
configs/          YAML experiment configs; every config inherits `base.yaml` and sets `seed`
src/trafficrl/    data/ (splits, roadnet, audit, trace, eda), envs/ (traffic_env, features,
                  metrics, perturb, engine), agents/ (base, baselines/, ugmarl/), eval/ (runner),
                  utils/ (seed, config, runlog, plotting, stats)
scripts/          entry points (dataset_stats, bench_cityflow, audit_data, collect_traces,
                  evaluate; train to come)
tests/            pytest suite mirroring src/
notebooks/        EDA only (train split): 01_eda_train.ipynb
docs/             dataset_card.md, data_audit.md, experiment_log.csv, figures/, literature/
```
