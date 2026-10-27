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
- **Traffic simulator: TBD** (open task — see Notes). Candidates: SUMO + `sumo-rl`,
  CityFlow (WSL/Docker), or both via LibSignal.
- Datasets: **selection pending — due 31 Oct** (ROADMAP Phase 1). Default proposal: Jinan
  (3×4) primary, Hangzhou (4×4) held-out; also surveying New York, RESCO Cologne/Ingolstadt,
  LuST/MoST, PeMS/UTD19.
- Env management: `venv` + `requirements.txt`. Version control: git.

## Commands

Windows (PowerShell); use `.venv/Scripts/python` from Git Bash.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

pytest                                   # run tests
ruff check src tests                     # lint
jupyter lab                              # EDA / plots in notebooks/
```

Planned (not implemented yet — update once scripts exist):
```powershell
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
Simulator (SUMO/CityFlow) ──obs──► per-intersection agent
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
- `.claude/reference/*.md` — architecture, interfaces, data, testing detail.
- `.claude/plans/` — implementation plans.

## Notes

- **Open task — choose simulator.** Decide between SUMO + `sumo-rl` (native Windows, GUI for
  demo), CityFlow via WSL/Docker (faster, native Jinan/Hangzhou data), or both via LibSignal.
  Update Tech Stack, Commands and `reference/architecture.md` once decided.
- **Open task — select datasets & split plan by 31 Oct** (ROADMAP Phase 1).
- Baselines required by PS16: conventional (Fixed-Time, Max-Pressure), single-agent RL
  (shared-parameter DQN, no coordination), multi-agent RL (IDQN; IPPO optional).
- Solo project: when time is short, follow ROADMAP "scope controls" — never cut the main
  comparison, ablation, report, slides, or demo.
- Ablations: −gate, −coordination (attention), −ensemble/uncertainty.
- Workflow: read `.claude/commands/prime.md` at session start; check `ROADMAP.md` for the
  current phase; plan non-trivial work into `.claude/plans/` before executing; tick completed
  ROADMAP tasks and add decisions to its Decision Log; commit per `.claude/commands/commit.md`.
