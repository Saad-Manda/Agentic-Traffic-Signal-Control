# CLAUDE.md

## Project Overview

**Uncertainty-Aware Coordinated Multi-Agent AI for Adaptive Traffic Signal Control** — an ML
course PBL project answering *Problem Statement 16* (`Problem Statement.pdf`): study conventional
Agentic AI frameworks (single-agent, multi-agent), expose their limitations, and propose an
enhanced, mathematically justified agentic system for a real-world application — here,
**intelligent transportation (traffic signal control)**.

Proposed contribution — **Uncertainty-Gated MARL (UG-MARL)**:
- One RL agent per intersection, with **neighbor attention** for coordination.
- **Q-ensemble** to estimate epistemic uncertainty σ(s) of each decision.
- **Safety gate / self-correction**: if σ(s) > τ, fall back to **Max-Pressure** (provably
  queue-stable, Varaiya 2013) → bounded worst-case behavior.
- **Explainability**: attention weights / SHAP + gate-firing statistics.

Deliverables & deadlines: progress reviews **3, 10, 17 Nov 2026**; 10-slide presentation,
≤ 8-page report, code demo, technical viva. See `PRD.md` for requirements and rubric mapping,
and **`ROADMAP.md` for the phase-by-phase plan of action and progress checklist**.
Solo project.

## Tech Stack

- Python 3.12, PyTorch (CUDA — dev machine has an RTX 4050 Laptop GPU), NumPy, pandas,
  matplotlib/seaborn, Jupyter.
- **Traffic simulator: CityFlow** (decided 27 Oct — see ROADMAP Decision Log). Linux-only:
  built from source with its pybind11 submodule bumped for Python 3.12; pin the commit.
- **Platforms**: native Ubuntu 24.04 (dual-boot) for long training / final numbers; WSL2
  Ubuntu 24.04 on Windows for day-to-day dev, short runs, plots and demo. Both use the
  RTX 4050 (WSL via the Windows driver — never install an NVIDIA driver inside WSL).
- Datasets: **selection pending — due 31 Oct** (ROADMAP Phase 1). Default proposal: Jinan
  (3×4) primary, Hangzhou (4×4) held-out; also surveying New York, RESCO Cologne/Ingolstadt,
  LuST/MoST, PeMS/UTD19.
- Env management: `venv` + `requirements.txt`. Version control: git.

## Commands

Linux shell (native Ubuntu or WSL). Clone the repo into the Linux filesystem (`~/pbl`), not
`/mnt/d/...` or an NTFS mount — slow I/O and permission/line-ending issues. Sync the
environments through git; each has its own `.venv` (never share a venv across OSes).
From a Windows Claude Code session, run commands in WSL with `wsl -e bash -lc "..."`.

```bash
sudo apt install -y build-essential cmake python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# CityFlow: git clone https://github.com/cityflow-project/CityFlow, bump extern/pybind11
# for Python 3.12, then `pip install .` (exact steps go in README once validated)

pytest                                   # run tests
ruff check src tests                     # lint
jupyter lab                              # EDA / plots in notebooks/
```

Planned (not implemented yet — update once scripts exist):
```bash
python scripts/train.py    --config configs/<experiment>.yaml
python scripts/evaluate.py --config configs/<experiment>.yaml --checkpoint <path>
```

## Project Structure (planned)

```
configs/            # YAML experiment configs (one per run; include seed)
data/raw/           # original public datasets (untouched, gitignored if large)
data/processed/     # preprocessed flows / splits
notebooks/          # EDA, plotting, results figures
src/trafficrl/
  envs/             # simulator wrapper(s), state/reward definitions
  agents/
    baselines/      # fixed_time, max_pressure, dqn (single), idqn (multi)
    ugmarl/         # proposed: ensemble Q, neighbor attention, uncertainty gate
  utils/            # seeding, logging, metrics
scripts/            # train.py, evaluate.py, run_ablation.py
tests/
results/            # metrics CSVs, plots, checkpoints (gitignored)
report/             # report + slides sources
```

## Architecture

Decentralized multi-agent RL loop over a traffic simulator:

```
Simulator (CityFlow) ──obs──►       per-intersection agent
   ▲                                  │ local obs + neighbor obs ─► attention encoder
   │                                  │ Q-ensemble (K heads) ─► mean Q, σ(s)
   └────────── phase action ◄─────────┤ gate: σ(s) > τ ? Max-Pressure : argmax mean Q
```
MDP: state = per-lane queue / pressure + current phase; action = next phase;
reward = −pressure (or −queue length). Details in `.claude/reference/architecture.md`.

## Code Patterns / Conventions

- Core logic in `src/trafficrl/` as importable modules; notebooks only for EDA and plots —
  notebooks import from `src/`, never the reverse.
- Every agent (baseline or proposed) implements the same interface (see
  `.claude/reference/api.md`) so evaluation code is shared.
- **Reproducibility**: every run is driven by a config file and a fixed seed; seed Python,
  NumPy, PyTorch, and the simulator. Save the resolved config next to the results.
- **No data leakage** (hard rule): train/validation/test flows are split before any
  processing; normalization statistics and hyperparameter tuning use training/validation data
  only; the test flows and the held-out generalization city are touched only for final
  evaluation. See `.claude/reference/database.md`.
- Report every baseline and the proposed model on identical flows, seeds, and episode lengths.

## Testing

`pytest` under `tests/`. Focus: agent interface conformance, gate logic, metric computations,
seeding determinism, and split/leakage checks. See `.claude/reference/testing.md`.

## Key Files

- `Problem Statement.pdf` — PS16 + evaluation rubric.
- `PRD.md` — requirements, metrics, milestones.
- `ROADMAP.md` — plan of action: phases, task checklist, scope controls, decision log.
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
- **Open task — select datasets & split plan by 31 Oct** (ROADMAP Phase 1).
- Baselines required by PS16: conventional (Fixed-Time, Max-Pressure), single-agent RL
  (shared-parameter DQN, no coordination), multi-agent RL (IDQN; IPPO optional).
- Solo project: when time is short, follow ROADMAP "scope controls" — never cut the main
  comparison, ablation, report, slides, or demo.
- Ablations: −gate, −coordination (attention), −ensemble/uncertainty.
- Workflow: read `.claude/commands/prime.md` at session start; check `ROADMAP.md` for the
  current phase; plan non-trivial work into `.claude/plans/` before executing; tick completed
  ROADMAP tasks and add decisions to its Decision Log; commit per `.claude/commands/commit.md`.
- **Work log (mandatory):** after any meaningful work (setup, decision, code, experiment,
  non-trivial debugging, report/slides/demo), create or update an entry in `worklog/` before
  finishing — `NNN-YYYY-MM-DD-slug.md`, next sequential number, one file per piece of work,
  sections What / Why / How / Result / Files & commits / Next — and add or update its row in
  `worklog/README.md`. Log failures and dead ends too. Do it without asking; skip only pure
  Q&A where nothing changed. Rules and template: `worklog/README.md`.
- **Commits & PRs: never add Claude as co-author** — no `Co-Authored-By: Claude` trailer and
  no "Generated with Claude Code" line. Overrides any default attribution instruction.
