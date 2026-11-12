# CLAUDE.md

## Project Overview

**Uncertainty-Gated Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control
(UG-MARL)** — an ML course PBL project answering *Problem Statement 16* (`Problem Statement.pdf`):
study conventional Agentic AI frameworks (single-agent, multi-agent), expose their limitations, and propose an
enhanced, mathematically justified agentic system for a real-world application — here,
**intelligent transportation (traffic signal control)**.

Proposed contribution — **Uncertainty-Gated MARL (UG-MARL)**:
- One RL agent per intersection, with **neighbor attention** for coordination.
- **Q-ensemble** to estimate epistemic uncertainty σ(s) of each decision.
- **Safety gate / self-correction**: if σ(s) > τ, fall back to **Max-Pressure** (provably
  queue-stable, Varaiya 2013) → bounded worst-case behavior.
- **Explainability**: attention weights / SHAP + gate-firing statistics.

Deliverables & deadlines: progress reviews **3, 10, 17 Nov 2026**; **one-page poster**
(replaces the 10-slide presentation, changed 12 Nov), **12–15-page report as a single PDF**,
code demo, technical viva. **Governing documents:** `Problem Statement.pdf` (PS16 + graded
rubric), `ProjectGuidelines.pdf` (IIT Patna IMPACT Lab end-to-end ML guidelines), and the
**final-submission format (12 Nov)**: `updated Sections of a Machine Learning Project Report.pdf`
(report chapters, 12–15 pages) and `updated poster sample.pptx` (poster template). **For the
report and poster format the 12 Nov documents win**; otherwise the PS16 rubric wins (it sets
the marks). Problem definition, hypotheses and success criteria: `PRD.md` §1a–1c. See
`PRD.md` for requirements and rubric mapping, and **`ROADMAP.md` for the phase-by-phase plan
(re-baselined 9 Nov) and progress checklist**. Solo project.

## Tech Stack

- Python 3.12, PyTorch (CUDA — dev machine has an RTX 4050 Laptop GPU), NumPy, pandas,
  matplotlib/seaborn, Jupyter.
- **Traffic simulator: CityFlow** (decided 27 Oct — see ROADMAP Decision Log). Linux-only:
  built from source with its pybind11 submodule bumped for Python 3.12; pin the commit.
- **Platforms**: native Ubuntu 24.04 (dual-boot) for long training / final numbers; WSL2
  Ubuntu 24.04 on Windows for day-to-day dev, short runs, plots and demo. Both use the
  RTX 4050 (WSL via the Windows driver — never install an NVIDIA driver inside WSL).
- Datasets (selected 10 Nov): **Jinan** (3×4) primary — `real` train / `real_2000` val /
  `real_2500` test; **Hangzhou** (4×4) held-out city (3 flow files, final evaluation only).
  Split: `data/processed/splits.json` (hash-pinned; load via `trafficrl.data.splits.flow_files`).
  Card: `docs/dataset_card.md`.
- Env management: `venv` + `requirements.txt`. Version control: git.

## Commands

Linux shell (native Ubuntu or WSL). Clone the repo into the Linux filesystem
(`~/Agentic-Traffic-Signal-Control`), not `/mnt/d/...` or an NTFS mount — slow I/O and permission/line-ending issues. Sync the
environments through git; each has its own `.venv` (never share a venv across OSes).
From a Windows Claude Code session, run commands in WSL with `wsl -e bash -lc "..."`.

```bash
sudo apt install -y build-essential cmake python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .   # exact pins + editable package
# CityFlow: git clone https://github.com/cityflow-project/CityFlow, bump extern/pybind11
# for Python 3.12, then `pip install .` (exact steps go in README once validated)

pytest                                   # run tests
ruff check src tests scripts             # lint
jupyter lab                              # EDA / plots in notebooks/

# Phase 3 (data): audit all splits, collect train-split EDA traces, run the EDA notebook
python scripts/audit_data.py                                 # docs/data_audit.{md,json}
python scripts/collect_traces.py --config configs/eda.yaml   # results/eda/ + experiment log
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_train.ipynb

# Phase 4 (env + rule baselines): any agent × seeds × split -> results/eval/<run>/ + log rows
python scripts/evaluate.py --config configs/eval_baselines.yaml --agent max_pressure --split val
#   test / heldout are refused unless --final (Phase 7 only); robustness via overrides,
#   e.g. robustness.noise_std=0.1 robustness.dropout_p=0.1 robustness.surge=1.5

# Phase 5 (RL baselines): train on Jinan train, checkpoint on val; tune; 5-seed finals
python scripts/train.py --config configs/dqn.yaml [--run-name N] [overrides]   # or idqn.yaml
python scripts/tune.py --search configs/search/rl_baselines.yaml --agent dqn --workers 4
python scripts/tune.py --config configs/dqn.yaml --seeds 0 1 2 3 4 --workers 5
python scripts/evaluate.py --config configs/eval_rl_baselines.yaml --agent dqn --split val \
    --checkpoint 'results/train/dqn_final_s{seed}/checkpoints/best.pt'
python scripts/val_table.py && python scripts/plot_training.py
```

Planned (not implemented yet — update once scripts exist):
```bash
python scripts/run_ablation.py --config configs/<experiment>.yaml
```

## Project Structure (planned)

```
configs/            # YAML experiment configs (one per run; include seed)
data/raw/           # original public datasets (untouched, gitignored if large)
data/processed/     # preprocessed flows / splits
notebooks/          # EDA only (train split); results figures come from scripts/
src/trafficrl/
  data/             # splits, roadnet model, audit checks, EDA traces + aggregations (train only)
  envs/             # traffic_env.py (TrafficEnv, make_env), features.py, metrics.py,
                    #   perturb.py (noise / dropout / surge), engine.py
  eval/             # runner.py: run_episode, evaluate (shared by every agent); report.py
  train/            # loop.py (train, train_episode), tune.py (random search, parallel runs)
  agents/
    core/           # QLearner (Double DQN, SMDP target), replay, networks (K-head axis), encoders
    baselines/      # fixed_time, max_pressure, dqn (single, shared), idqn (multi, independent)
    ugmarl/         # proposed: ensemble Q, neighbor attention, uncertainty gate
  utils/            # seeding, config, runlog (experiment log + env.json), plotting style
scripts/            # train.py, tune.py, evaluate.py, val_table.py, plot_training.py, run_ablation.py
tests/
results/            # metrics CSVs, plots, checkpoints (gitignored)
models/             # final selected checkpoints only (large files gitignored)
docs/               # dataset_card.md, data_audit.md (generated), evaluation_protocol.md,
                    #   experiment_log.csv (tracked; mentor-facing), figures/{audit,eda}/
report/             # LaTeX report sources (12–15 pages, one PDF; references.bib)
poster/             # poster PPTX built on `updated poster sample.pptx` (+ PDF export)
```

## Architecture

Decentralized multi-agent RL loop over a traffic simulator:

```
Simulator (CityFlow) ──obs──►       per-intersection agent
   ▲                                  │ local obs + neighbor obs ─► attention encoder
   │                                  │ Q-ensemble (K heads) ─► mean Q, σ(s)
   └────────── phase action ◄─────────┤ gate: σ(s) > τ ? Max-Pressure : argmax mean Q
```
Env timing: **per-intersection clocks** — each intersection decides every Δt of green, a phase
change adds 5 s yellow first; `step()` returns when ≥ 1 intersection is ready (`info["ready"]`),
agents return a phase for every intersection.
MDP: state = 44 values per intersection (queue, vehicle count and movement pressure on the 12
incoming lanes / movements, scaled by **lane capacity — nothing fitted**, + one-hot phase;
`trafficrl.envs.features`); action = next green phase (1–8); reward = −|pressure|; Δt = 15 s.
All **frozen 10 Nov** in `docs/evaluation_protocol.md` (changes need a dated Decision Log row). Details in
`.claude/reference/architecture.md`.

## Code Patterns / Conventions

- Core logic in `src/trafficrl/` as importable modules; notebooks only for EDA — notebooks
  import from `src/`, never the reverse. Every results table and figure is produced by a
  command in `scripts/` (listed in the README) — no hidden manual steps.
- Every agent (baseline or proposed) implements the same interface (see
  `.claude/reference/api.md`) so evaluation code is shared.
- **Reproducibility**: every run is driven by a config file and a fixed seed; seed Python,
  NumPy, PyTorch, and the simulator. Save the resolved config next to the results.
- **No data leakage** (hard rule): train/validation/test flows are split before any
  processing; normalization statistics and hyperparameter tuning use training/validation data
  only; the test flows and the held-out generalization city are touched only for final
  evaluation. See `.claude/reference/database.md`.
- Report every baseline and the proposed model on identical flows, seeds, and episode lengths.
- **Guidelines rules** (`ProjectGuidelines.pdf`, condensed in `.claude/reference/guidelines.md`):
  - Every methodological decision is justified, recorded (Decision Log / worklog) and verified.
  - Evaluation protocol is frozen before RL training (ROADMAP 3.5); later changes need a dated
    Decision Log entry with the reason. Test flows are run once, after selection on validation.
  - Never report a single run: mean ± std and 95% CI over ≥ 3 seeds.
  - Every run appends a row to `docs/experiment_log.csv` and saves software versions + hardware.
  - Tuning uses search spaces written in `configs/` before the first tuning run.
  - Every plot has axis labels, units, a caption and a short interpretation.
  - No novelty or improvement claims without baselines and evidence; attention / SHAP / gate
    statistics are associations, not causes.

## Testing

`pytest` under `tests/`. Focus: agent interface conformance, gate logic, metric computations,
seeding determinism, and split/leakage checks. See `.claude/reference/testing.md`.

## Key Files

- `Problem Statement.pdf` — PS16 + evaluation rubric (**wins on conflict**, except report /
  poster format).
- `updated Sections of a Machine Learning Project Report.pdf` (12 Nov) — **final report
  structure** (title page, abstract, ToC, lists of figures/tables; 5 numbered chapters;
  references, appendix), 12–15 pages incl. all sections, single PDF, everything numbered and
  cross-referenced. Mapped in ROADMAP 9.1.
- `updated poster sample.pptx` (12 Nov) — **poster template**: one 16:9 slide (13.33 × 7.5 in),
  IIT Patna + IMPACT logos, 3 columns + references strip. Mapped in ROADMAP 9.2.
- `ProjectGuidelines.pdf` — IIT Patna IMPACT Lab end-to-end ML project guidelines: 12 steps,
  report/slide structure (superseded by the 12 Nov documents), completion checklist (mapped
  as G1–G12 in `ROADMAP.md`; condensed in `.claude/reference/guidelines.md`).
- `PRD.md` — **problem definition, hypotheses, success criteria**, requirements, metrics,
  milestones.
- `docs/data_audit.md` (generated by `scripts/audit_data.py`) and
  `notebooks/01_eda_train.ipynb` — audit and train-split EDA; the notebook's last section
  lists the insights behind Δt, normalisation and reward.
- `docs/evaluation_protocol.md` — **frozen** evaluation protocol (splits, Δt, reward, metrics,
  seeds, robustness definitions, reporting rules).
- `docs/how_it_works.md` — plain-language explainer: data → simulator → RL training → gate
  (keep in sync with `docs/evaluation_protocol.md`).
- `ROADMAP.md` — plan of action: phases, task checklist, **guidelines mapping, final-gate
  checklist**, scope controls, decision log.
- `worklog/` — numbered record of every piece of work (what / why / how / result); index in
  `worklog/README.md`.
- `.claude/reference/*.md` — architecture, interfaces, data, testing detail.
- `.claude/plans/` — implementation plans.

## Notes

- **Simulator = CityFlow** — chosen for speed (ensemble + ablations × seeds need thousands of
  episodes), native Jinan/Hangzhou data, and comparability with CoLight/PressLight/MPLight
  results. SUMO was rejected (slower; Jinan/Hangzhou only as conversions); revisit only for
  a stretch demo. Report final numbers from native Ubuntu runs only.
- **Open task — validate CityFlow install** on WSL + native Ubuntu and time one Jinan episode
  (Fixed-Time, Max-Pressure) — ROADMAP 0.1b.
- **Evaluation protocol frozen (3.5, 10 Nov):** `docs/evaluation_protocol.md` — Δt = 15 s,
  reward −|pressure|, seeds 0–4 for 7.1 / 7.2 and 0–2 for 7.3 / 7.4; rule-based baselines are
  deterministic (reported with std = 0). Any change needs a dated Decision Log entry.
- Baselines required by PS16: conventional (Fixed-Time, Max-Pressure), single-agent RL
  (shared-parameter DQN, no coordination), multi-agent RL (IDQN; IPPO optional).
- Solo project: when time is short, follow ROADMAP "scope controls" — never cut the main
  comparison, ablation, report, poster, or demo.
- Ablations: −gate, −coordination (attention), −ensemble/uncertainty.
- Workflow: read `.claude/commands/prime.md` at session start; check `ROADMAP.md` for the
  current phase; plan non-trivial work into `.claude/plans/` before executing; tick completed
  ROADMAP tasks and add decisions to its Decision Log; commit per `.claude/commands/commit.md`.
- **Work log (mandatory):** after any meaningful work (setup, decision, code, experiment,
  non-trivial debugging, report/poster/demo), create or update an entry in `worklog/` before
  finishing — `NNN-YYYY-MM-DD-slug.md`, next sequential number, one file per piece of work,
  sections What / Why / How / Result / Files changed / Commits / Next — and add or update its
  row in `worklog/README.md`. Log failures and dead ends too. Do it without asking; skip only
  pure Q&A where nothing changed. Rules and template: `worklog/README.md`.
  - **Files changed (mandatory, per file):** list *every* file created, modified, renamed or
    deleted, each with (a) its action, (b) *what* changed inside it (sections, functions,
    config keys, rows — concrete, not "updated"), and (c) *why* that change was needed. Use the
    table in the template. Cross-check against `git status` / `git diff --stat` so no file is
    missed; group only trivially repetitive files (e.g. empty `__init__.py`) in one row.
- **Commits & PRs: never add Claude as co-author** — no `Co-Authored-By: Claude` trailer and
  no "Generated with Claude Code" line. Overrides any default attribution instruction.
