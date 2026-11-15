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

### 8. RL baselines: training, tuning, validation (ROADMAP Phase 5)

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
python scripts/val_table.py        # docs/results/val_phase5.{csv,md}
python scripts/plot_training.py    # docs/figures/training/learning_curves.{png,md}
```

Training runs write `results/train/<run>/` (train / val logs, `checkpoints/{best,last}.pt`,
TensorBoard `tb/`, `summary.json`, config, `env.json`) and one row in
`docs/experiment_log.csv`. `train.py` accepts only `data.split=train`. Device: CPU (faster
than CUDA for these small MLPs, worklog/016). Phase 5 numbers are WSL *dev* numbers.
Each run uses ~0.9 GB RAM; on a 7.7 GB WSL VM run at most 4–5 at once.

Expected (WSL dev, Jinan val, 5 seeds; tuned trial t6 = lr 1e-4, γ 0.8, batch 64, ε-decay
0.3 for both): DQN **299.8 ± 1.0 s**, IDQN **312.7 ± 2.5 s**, vs Max-Pressure 326.1 s and
Fixed-Time 434.8 s (table `docs/results/val_phase5.md`).

### 9. Proposed UG-MARL: grid, finals, gate calibration, validation (ROADMAP Phase 6)

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
python scripts/sigma_check.py --tag final          # docs/figures/ugmarl/sigma_validity.{png,md}
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
python scripts/val_table.py --agents fixed_time max_pressure dqn idqn ugmarl ugmarl-nogate ugmarl-noattn \
    ugmarl-noens --out docs/results/val_phase6 --title "Phase 6 — validation results (native Ubuntu)" --intro "…"
python scripts/plot_training.py --agents dqn idqn ugmarl ugmarl-noattn ugmarl-noens \
    --out docs/figures/training/learning_curves_phase6.png --platform native \
    --title "UG-MARL and RL baselines: greedy validation travel time during training"
```

The gate acts at evaluation only; training and checkpoint selection are ungated. Equations
and the stability argument: `docs/methodology.md`. Results: worklog/017.

### 10. Final evaluation: test flow and held-out city (ROADMAP Phase 7)

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
# tables (pure reader, repeatable): docs/results/test_{main,ablation,robustness,efficiency,safety_explain}.{csv,md}
#                                   docs/results/heldout_generalisation.{csv,md}
python scripts/final_tables.py --study all
```

Every final episode also writes `trace_s<seed>.csv` (per step × intersection),
`vehicles_s<seed>.csv` (per vehicle) and, for UG-MARL variants, `decisions_s<seed>.csv` under
`results/final/<study>/eval/<model>__<condition>/` — the inputs of the Phase 8 figures.

Expected output (deterministic on the same machine; learned agents mean ± std over 5 seeds),
average travel time on the clean test flow: Fixed-Time 462.2 s, Max-Pressure 320.8 s,
DQN 292.2 ± 0.6 s, IDQN 307.9 ± 1.4 s, UG-MARL 295.4 ± 1.1 s. Hypothesis verdicts printed by
`final_tables.py`: H1 supported, H2 supported, H3 not supported. Results: worklog/020.

### 11. Analysis: figures, error analysis, sensitivity (ROADMAP Phase 8)

No test or held-out episode is run here: everything on those flows is read from the logs of
step 10 (`results/final/`). The two scripts run on the **validation** flow only and select
nothing (`gate.json` and p\* = 80 stay as locked).

```bash
# sensitivity to the gate threshold: p = 50, 60, 70 on validation (75 episodes, ~25 min with 5 workers)
python scripts/tau_sweep.py --search configs/search/tau_sensitivity.yaml --workers 5
#   -> docs/results/tau_sensitivity.{csv,md};  --tables-only rebuilds the table without running
# attention faithfulness check: 5 validation episodes + offline slot perturbation (~1 min)
python scripts/attention_faithfulness.py --tag final
#   -> docs/results/attention_faithfulness.{csv,md}
python scripts/lock_models.py --verify     # nothing locked has changed
# reader notebook (no simulation, ~2 min): 18 figures + error-analysis tables + readings
jupyter nbconvert --to notebook --execute --inplace notebooks/02_analysis.ipynb
#   -> docs/figures/results/*.{png,pdf,md}, docs/results/error_analysis.md (+ 13 CSVs)
```

The notebook holds no logic: every number and figure is a tested function in
`trafficrl.eval.analysis` / `figures` / `faithfulness` (settings: `configs/analysis.yaml`). Its
first section recomputes travel time and queue of all 305 final episodes from the logs and
stops if any differs from `metrics.csv` by more than 0.001. Both scripts refuse to run twice
(`results/analysis/<study>` exists) and refuse any split other than `val`.

Expected output: reproduction check passes with a largest difference of 0.00005 s; τ sweep
scores 343.2 / 343.8 / 342.3 s for p = 50 / 60 / 70 (343.7 s at the locked p\* = 80);
faithfulness, replacement: |Δμ| 0.0061 (highest-α neighbour) vs 0.0022 (lowest-α) vs 0.0415
(own observation). Written interpretation: `docs/analysis.md`. Results: worklog/022.

### 12. Report (ROADMAP 9.1)

The report is LaTeX in `report/`; nothing is simulated to build it.

```bash
sudo apt install -y latexmk texlive-latex-recommended texlive-latex-extra \
    texlive-science texlive-pictures texlive-publishers texlive-bibtex-extra poppler-utils
python scripts/final_tables.py --study all   # docs/results/*.csv (incl. heldout_generalisation_all.csv)
python scripts/report_tables.py              # report/tables/*.tex, report/generated/numbers.tex
cd report
latexmk -pdf -outdir=build figures/architecture.tex && cp build/architecture.pdf figures/ \
    && pdftoppm -r 300 -png -singlefile build/architecture.pdf figures/architecture   # Figure 3.2
latexmk -pdf main.tex && cp build/main.pdf report.pdf
cd .. && python scripts/check_report.py      # pages 12-15, references, citations, leftovers
```

Every table and every result number in the text (`\res{metric}{model}{condition}`) is
generated from `docs/results/*.csv`; the result figures are the PDFs of step 11. Expected
output: `pages: 12`, `report check: ok`. `report/checks/claims.md` lists the source of every
claim, `report/checks/algorithm_check.md` the check of Algorithm 1 against the code.

### 13. Poster (ROADMAP 9.2)

One 16:9 slide (13.33 × 7.5 in), built by a script; nothing is simulated. Needs LibreOffice
(`soffice`) for the PDF export.

```bash
python scripts/build_poster.py --pdf   # poster/poster.pptx, poster/poster.pdf, poster/figures/
```

The text is `poster/content.yaml` (edit it and rebuild). Numbers are written there as
`{metric@model@condition}` and filled from `docs/results/*.csv` with the same keys and
formatting as the report's `\res` macros; Table 1 and Figure 2 are built from the same files,
and Figure 1 is drawn with native PowerPoint shapes, so everything stays editable. Hand-typed
numbers are listed in `report/checks/claims.md` ("Poster").

### 14. Tests and lint

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
notebooks/        01_eda_train.ipynb (EDA, train split only); 02_analysis.ipynb (Phase 8 reader:
                  calls tested functions in src/, simulates nothing)
docs/             dataset_card.md, data_audit.md, experiment_log.csv, figures/, literature/,
                  results/ (tables), analysis.md (interpretation of the results)
report/           LaTeX report: main.tex, chapters/, figures/ (architecture diagram, Algorithm 1),
                  tables/ + generated/ (written by scripts/report_tables.py), checks/, report.pdf
```
