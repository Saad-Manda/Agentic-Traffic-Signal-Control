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

### 6. Tests and lint

```bash
pytest            # all tests; `pytest -m "not slow"` skips simulator-heavy ones
ruff check src tests
```

## Repository layout

```
configs/          YAML experiment configs; every config inherits `base.yaml` and sets `seed`
src/trafficrl/    envs/, agents/{baselines,ugmarl}/, utils/ (seed.py, config.py)
scripts/          entry points (bench_cityflow.py; train/evaluate to come)
tests/            pytest suite mirroring src/
notebooks/        EDA only (train split)
```
