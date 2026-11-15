# Uncertainty-Gated Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control (UG-MARL)

Built on the CityFlow simulator. Repository name: `Agentic-Traffic-Signal-Control`.

## Install (Ubuntu 24.04: native or WSL2)

Tested on WSL2 Ubuntu 24.04, Python 3.12.3, g++ 13.3, CMake 3.28.3 (28 Oct 2026), and on
native Ubuntu 24.04.3 (kernel 7.0, i5-13420H, 15 GB RAM; 11 Nov 2026).
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

No sudo / no system `cmake` (native Ubuntu, 11 Nov): install CMake into the venv and build
without isolation, so the build sees the venv's `cmake`:

```bash
pip install cmake==3.28.3
pip install --no-build-isolation .
```

### 4. Data (Jinan 3×4, Hangzhou 4×4)

From the CoLight repo (commit `04440b7`), into the gitignored `data/raw/`:

```bash
cd ~/Agentic-Traffic-Signal-Control
mkdir -p data/raw && cd data/raw
git clone --depth 1 --filter=blob:none --sparse https://github.com/wingsweihua/colight colight_src
cd colight_src && git sparse-checkout set data/Jinan data/Hangzhou
```

The files are pinned by SHA-256 in `data/processed/splits.json`; every load through
`trafficrl.data.splits.flow_files` checks the hash and fails on a mismatch.

The data has no license file. Use it for academic purposes only and cite CoLight (Wei et al.,
CIKM 2019). Split and data dictionary: `docs/dataset_card.md`.

### 5. EDA notebook

`notebooks/01_eda_train.ipynb` describes the Jinan training flow (train split only): arrivals,
routes and movements, plus queues and travel times under three rule controllers. Its outputs
are saved in the notebook; the data-audit counts are in `docs/data_audit.json`.

Reference timing (WSL2, 1 thread, one 3600 s episode on Jinan train, 12 intersections, 6295
vehicles): Fixed-Time 2.7 s, Max-Pressure 1.8 s wall time. One thread per simulator is fastest
on this network, so runs are parallelised across seeds instead.

### 6. Environment and rule baselines (ROADMAP Phase 4)

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

### 7. RL baselines: training, tuning, validation (ROADMAP Phase 5)

```bash
# one run: 100 episodes on Jinan train, greedy val episode every 5, best checkpoint on val
python scripts/train.py --config configs/dqn.yaml  --run-name dqn_try   # ~12-30 min (CPU, load-dependent)
python scripts/train.py --config configs/idqn.yaml --run-name idqn_try
# tuning: 8 random trials from the committed search space, 4 parallel processes
python scripts/tune.py --search configs/search/rl_baselines.yaml --agent dqn  --workers 4
python scripts/tune.py --search configs/search/rl_baselines.yaml --agent idqn --workers 4
# chosen configs on seeds 0-4, then full-tracking val evaluation (seed s loads checkpoint s)
python scripts/tune.py --config configs/dqn.yaml  --seeds 0 1 2 3 4 --workers 5 --tag final
python scripts/tune.py --config configs/idqn.yaml --seeds 0 1 2 3 4 --workers 5 --tag final
python scripts/evaluate.py --config configs/eval_rl_baselines.yaml --agent dqn --split val \
    --checkpoint 'results/train/dqn_final_s{seed}/checkpoints/best.pt'
python scripts/evaluate.py --config configs/eval_rl_baselines.yaml --agent idqn --split val \
    --checkpoint 'results/train/idqn_final_s{seed}/checkpoints/best.pt'
```

Training runs write `results/train/<run>/` (train / val logs, `checkpoints/{best,last}.pt`,
TensorBoard `tb/`, `summary.json`, config, `env.json`) and one row in
`docs/experiment_log.csv`. `train.py` accepts only `data.split=train`. Device: CPU (faster
than CUDA for these small MLPs, worklog/016). Phase 5 numbers are WSL *dev* numbers.
Each run uses ~0.9 GB RAM; on a 7.7 GB WSL VM run at most 4–5 at once.

Expected (WSL dev, Jinan val, 5 seeds; tuned trial t6 = lr 1e-4, γ 0.8, batch 64, ε-decay
0.3 for both): DQN **299.8 ± 1.0 s**, IDQN **312.7 ± 2.5 s**, vs Max-Pressure 326.1 s and
Fixed-Time 434.8 s. The validation table and learning curves of all models are built in
`notebooks/02_results.ipynb` (step 9).

### 8. Proposed UG-MARL: grid, finals, gate calibration, validation (ROADMAP Phase 6)

```bash
# smoke: 2 episodes; the gate with tau = 0 must reproduce Max-Pressure (326.11 s on val)
python scripts/train.py --config configs/ugmarl.yaml --run-name smoke_ugmarl train.episodes=2 train.val_every=1
python scripts/evaluate.py --config configs/eval_ugmarl.yaml --agent ugmarl --split val --seeds 0 \
    --checkpoint results/train/smoke_ugmarl/checkpoints/last.pt agents.ugmarl.gate.tau=0.0
# grid K {3,5,10} x d {32,64,128} on seed 0 (search file committed first), winner -> configs/ugmarl.yaml
python scripts/tune.py --search configs/search/ugmarl.yaml --agent ugmarl --workers 5
# finals on seeds 0-4: full model and the two trained ablations
python scripts/tune.py --config configs/ugmarl.yaml        --seeds 0 1 2 3 4 --workers 5
python scripts/tune.py --config configs/ugmarl_noattn.yaml --seeds 0 1 2 3 4 --workers 5
python scripts/tune.py --config configs/ugmarl_noens.yaml  --seeds 0 1 2 3 4 --workers 5
# gate threshold per seed-model (validation only) -> results/train/<variant>_final_s<seed>/gate.json
python scripts/calibrate_gate.py --search configs/search/ugmarl_gate.yaml --config configs/eval_ugmarl.yaml --tag final --workers 5
python scripts/calibrate_gate.py --search configs/search/ugmarl_gate.yaml --config configs/eval_ugmarl.yaml --tag final --workers 5 \
    agents.ugmarl.use_attention=false
# 6.8: every baseline + UG-MARL / −gate under detector noise / dropout on validation (evaluation only)
python scripts/robustness.py --spec configs/robustness_val.yaml --workers 5   # docs/results/val_robustness.{csv,md}
# validation evaluation of every variant (decision logs for the explanation figures)
C=configs/eval_ugmarl.yaml
python scripts/evaluate.py --config $C --agent ugmarl --split val --log-decisions \
    --checkpoint 'results/train/ugmarl_final_s{seed}/checkpoints/best.pt'
python scripts/evaluate.py --config $C --agent ugmarl --split val --log-decisions \
    --checkpoint 'results/train/ugmarl_final_s{seed}/checkpoints/best.pt' agents.ugmarl.use_gate=false
python scripts/evaluate.py --config $C --agent ugmarl --split val --log-decisions \
    --checkpoint 'results/train/ugmarl-noattn_final_s{seed}/checkpoints/best.pt' agents.ugmarl.use_attention=false
python scripts/evaluate.py --config $C --agent ugmarl --split val --log-decisions \
    --checkpoint 'results/train/ugmarl-noens_final_s{seed}/checkpoints/best.pt' agents.ugmarl.use_ensemble=false
```

The gate acts at evaluation only; training and checkpoint selection are ungated. Equations
and the stability argument: report Ch. 3.

### 9. Final evaluation: test flow and held-out city (ROADMAP Phase 7)

Run once, after everything above is fixed. `--final` is refused unless `models/manifest.json`
matches the checkpoints, and it stops on a started-but-unfinished job (`--rerun-failed` after
logging the reason in the ROADMAP Decision Log).

```bash
python scripts/lock_models.py --write      # models/manifest.json: hashes of best.pt / gate.json / config (commit it)
python scripts/lock_models.py --verify
# Jinan test: 8 models x 6 conditions x seeds 0-4 = 240 episodes (~30 min, one process -> valid latency)
python scripts/robustness.py --spec configs/final_test.yaml --final --workers 1 --results-dir results/final
# Hangzhou, zero-shot: 7 models x 3 flows x seeds 0-4 = 105 episodes (~15 min)
python scripts/robustness.py --spec configs/final_heldout.yaml --final --workers 1 --results-dir results/final
# tables (reads the runs, repeatable): learning curves, docs/results/val_phase6.csv,
#   test_{main,ablation,robustness,efficiency,safety_explain}.csv, heldout_generalisation{,_all}.csv
jupyter nbconvert --to notebook --execute --inplace notebooks/02_results.ipynb
```

Every final episode also writes `trace_s<seed>.csv` (per step × intersection),
`vehicles_s<seed>.csv` (per vehicle) and, for UG-MARL variants, `decisions_s<seed>.csv` under
`results/final/<study>/eval/<model>__<condition>/` — the inputs of the Phase 8 figures.

Expected output (deterministic on the same machine; learned agents mean ± std over 5 seeds),
average travel time on the clean test flow: Fixed-Time 462.2 s, Max-Pressure 320.8 s,
DQN 292.2 ± 0.6 s, IDQN 307.9 ± 1.4 s, UG-MARL 295.4 ± 1.1 s. Hypothesis verdicts printed by
`02_results.ipynb`: H1 supported, H2 supported, H3 not supported. Results: worklog/020.

### 10. Analysis: figures, error analysis, sensitivity (ROADMAP Phase 8)

No test or held-out episode is run here: everything on those flows is read from the logs of
step 9 (`results/final/`). The τ sweep and notebook 04 run on the **validation** flow only and
select nothing (`gate.json` and p\* = 80 stay as locked).

```bash
# sensitivity to the gate threshold: p = 50, 60, 70 on validation (75 episodes, ~25 min with 5 workers)
python scripts/tau_sweep.py --search configs/search/tau_sensitivity.yaml --workers 5
#   -> docs/results/tau_sensitivity.{csv,md};  --tables-only rebuilds the table without running
python scripts/lock_models.py --verify     # nothing locked has changed
# error analysis (no simulation, ~1 min): 17 figures + error-analysis tables + readings
jupyter nbconvert --to notebook --execute --inplace notebooks/03_error_analysis.ipynb
#   -> docs/figures/results/*.{png,pdf}, docs/results/error_analysis_*.csv
# σ check + attention faithfulness: 10 validation episodes (~1-3 min)
jupyter nbconvert --to notebook --execute --inplace notebooks/04_gate_attention.ipynb
#   -> docs/results/{sigma_validity,sigma_validity_td,attention_faithfulness}.csv + 2 figures
```

The analysis and plotting functions are defined in the notebook itself (settings:
`configs/analysis.yaml`). Its
first section recomputes travel time and queue of all 305 final episodes from the logs and
stops if any differs from `metrics.csv` by more than 0.001. Notebook 04 asserts the validation
split and checks that each episode reproduces the validation-table travel time.

Expected output: reproduction check passes with a largest difference of 0.00005 s; τ sweep
scores 343.2 / 343.8 / 342.3 s for p = 50 / 60 / 70 (343.7 s at the locked p\* = 80);
faithfulness, replacement: |Δμ| 0.0061 (highest-α neighbour) vs 0.0022 (lowest-α) vs 0.0415
(own observation). Interpretation: report Ch. 4.

### 11. Tests and lint

```bash
pytest            # data split / leakage checks
ruff check src tests scripts
```

## Repository layout

```
configs/          YAML experiment configs; every config inherits `base.yaml` and sets `seed`
src/trafficrl/    data/ (splits, roadnet), envs/ (traffic_env, features,
                  metrics, perturb, engine), agents/ (base, baselines/, ugmarl/), eval/ (runner,
                  explain, lock, report, hypotheses), utils/ (seed, config, runlog, plotting, stats)
scripts/          train, tune, evaluate, calibrate_gate, robustness, lock_models, tau_sweep
tests/            data split / leakage checks
notebooks/        01_eda_train (EDA, train split only), 02_results (learning curves + all result
                  tables), 03_error_analysis (figures + error analysis), 04_gate_attention
                  (σ check + attention faithfulness, validation)
docs/             dataset_card.md, data_audit.json, experiment_log.csv (one row per run),
                  results/ (result tables as CSV), figures/
```
