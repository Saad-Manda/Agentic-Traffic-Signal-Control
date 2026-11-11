# 018 — Native Ubuntu setup and Phase 5 reproduction
**Date:** 2026-11-11   **Tag:** setup, experiment   **ROADMAP:** 0.1b   **Status:** done

## What
Set up the native Ubuntu 24.04 machine (dual boot of the same laptop) from a fresh clone,
following the "Handoff" section of `.claude/plans/phase6-ugmarl.md`: `.venv` with the pinned
requirements, CityFlow built from source, Jinan / Hangzhou data downloaded and hash-checked,
test suite, CityFlow benchmark (closes ROADMAP 0.1b), and a re-run of every Phase 5
validation result so that Phase 6 has local checkpoints and baselines to compare against.

## Why
Nothing outside git exists on a new machine (`.venv`, CityFlow, `data/raw/`, `results/`).
Phase 6 step 8 needs the DQN / IDQN checkpoints and val evaluations on this machine; the
Decision Log (11 Nov) also asked for final numbers from native Ubuntu runs.

## How
- `python3 -m venv .venv`, `pip install -r requirements.txt`, `pip install -e .`.
- **No sudo** (password required) and no system `cmake`: installed `cmake==3.28.3` from PyPI
  into the venv. Plain `pip install .` of CityFlow then failed — the isolated build env
  cannot import the venv's `cmake` module (`ModuleNotFoundError: No module named 'cmake'`).
  `pip install --no-build-isolation .` built it. CityFlow `81ee0f4`, pybind11 v2.13.6 as on WSL.
- Data: CoLight sparse clone at `04440b7`; `scripts/dataset_stats.py` → all SHA-256 hashes
  match `data/processed/splits.json`, `docs/dataset_stats.json` unchanged.
- `pytest`: 158 passed, 1 skipped (CUDA determinism test — the NVIDIA driver is not loaded on
  native Ubuntu; training uses CPU, `train.device: cpu`). `ruff`: clean.
- Re-ran the handoff sequence (script in the session scratchpad, 5 parallel workers):
  FT / MP val evaluation → DQN finals seeds 0–4 → IDQN finals seeds 0–4 → val evaluation of
  both with `--checkpoint 'results/train/<agent>_final_s{seed}/checkpoints/best.pt'`.

## Result
- Hardware: i5-13420H (8 cores / 12 threads), 15 GB RAM, Ubuntu 24.04.3, kernel 7.0,
  Python 3.12.3, torch 2.15.0 (CPU).
- Benchmark (Jinan `real`, one 3600 s episode, idle machine): **Fixed-Time 1.13 s,
  Max-Pressure 1.14 s** (WSL: 2.7 / 1.8 s); metrics identical to WSL (515.21 / 374.94 s).
  A first measurement during the pip install gave 3.8 / 4.8 s (load, not the platform).
- Training: ≈ 1415 s per DQN run, ≈ 1330 s per IDQN run, 5 in parallel (WSL 1430–2020 s).
- **Every Phase 5 number is bit-identical to WSL:**

| Run | WSL | Native |
|---|---|---|
| Fixed-Time val | 434.84 s | 434.84 s |
| Max-Pressure val | 326.11 s | 326.11 s |
| DQN best val, seeds 0–4 | 298.8598 / 299.4257 / 299.5081 / 301.5840 / 299.4570 | identical |
| IDQN best val, seeds 0–4 | 310.4215 / 313.0053 / 314.8131 / 315.3743 / 310.0220 | identical |
| DQN val evaluation | 299.8 ± 1.0 s | 299.77 ± 1.05 s, 95 % CI [298.46, 301.07] |
| IDQN val evaluation | 312.7 ± 2.5 s | 312.73 ± 2.45 s, 95 % CI [309.68, 315.77] |

  Reason: same CPU (WSL runs on the same laptop), same pinned wheels, strictly deterministic
  seeding. The expectation in the Phase 6 plan ("close to, but not identical with") was too
  cautious for this hardware. **Open (needs the user's decision, then a Decision Log row):**
  whether WSL runs on this laptop may count as reportable, so Phase 7 need not retrain the
  baselines natively.
- Dead end: CityFlow build with build isolation + pip-installed cmake (see How).
- Note: `runlog` appends experiment-log rows with CRLF (csv default); git normalises them to LF
  on commit (`.gitattributes eol=lf`), same as on WSL — harmless.

## Files changed
| File | Action | What changed | Why |
|------|--------|--------------|-----|
| `README.md` | modified | Install: native Ubuntu added to "Tested on"; new paragraph for building CityFlow without sudo (`pip install cmake==3.28.3`, `pip install --no-build-isolation .`) | Exact, working install steps for this machine (ROADMAP 0.1b, 9.4) |
| `ROADMAP.md` | modified | 0.1b ticked; native sub-item with build route, benchmark times and the bit-identical reproduction | Track progress |
| `docs/experiment_log.csv` | modified | 30 rows appended: 10 training runs (`dqn_final_s0–4`, `idqn_final_s0–4`) and 20 val-evaluation rows (FT, MP, DQN, IDQN × seeds 0–4), all native Ubuntu, commit `0e986b9-dirty` | Every run appends a row (guidelines) |
| `worklog/018-2026-11-11-native-ubuntu-setup.md` | created | This entry | Work log rule |
| `worklog/README.md` | modified | Index row 018 | Work log rule |

Not tracked (gitignored): `.venv/`, `data/raw/colight_src/`, `results/{bench,train,tune,eval}/`,
`~/src/CityFlow`.

## Commits
See `git log -- worklog/018-2026-11-11-native-ubuntu-setup.md`.

## Next
- User decision on the platform rule (see Result), then a Decision Log row.
- Phase 6 plan, step 1, on this machine (continue worklog/017).
- Optional: load the NVIDIA driver on native Ubuntu if CUDA is ever needed (not for the
  current CPU training).
