# 007 — Build CityFlow on WSL and time one Jinan episode
**Date:** 2026-10-28   **Tag:** setup   **ROADMAP:** 0.1b   **Status:** in-progress (WSL done, native Ubuntu pending)

## What
Built CityFlow from source for Python 3.12 on WSL2 Ubuntu 24.04. Downloaded Jinan 3×4 and
Hangzhou 4×4. Timed one 3600 s Jinan episode with Fixed-Time and Max-Pressure. Wrote the
install steps to `README.md`.

## Why
ROADMAP 0.1b: confirm the chosen simulator works before building on it, and get episode
timings to size the training/ablation budget.

Python version: kept the system **3.12.3**. 3.11 would need the deadsnakes PPA on both
machines and brings no benefit once pybind11 is bumped. 3.10 was the fallback in the Risks
table and wasn't needed.

## How
- WSL had no compiler. Installed `build-essential cmake python3-venv python3-dev`
  (g++ 13.3, CMake 3.28.3).
- CityFlow at commit `81ee0f4`. Initialised `extern/rapidjson` and replaced the
  `extern/pybind11` fork (`a8ee79d`) with upstream pybind11 **v2.13.6** (`a2e59f0`).
  `pip install .` into `.venv` built it on the first try. `setup.py`'s `distutils` import
  works through the setuptools shim.
- Data: sparse clone of `wingsweihua/colight` at `04440b7` → `data/raw/colight_src/`
  (gitignored).
- `scripts/bench_cityflow.py`: standalone (stdlib + cityflow). Phase 0 is the 5 s transition
  phase. Fixed-Time cycles green phases 1–8, 30 s each. Max-Pressure re-decides every 10 s,
  pressure = Σ over movements of (vehicles on incoming lanes − vehicles on outgoing lanes).
  A phase change inserts 5 s of phase 0. Seed 0, `anon_3_4_jinan_real.json`.

## Result
Jinan 3×4: 12 intersections, 6295 vehicles, 3600 steps, WSL2, 1 thread:

| Controller | Wall | Steps/s | Avg travel time | Mean total queue | Vehicles left in network |
|---|---|---|---|---|---|
| Fixed-Time | 2.71 s | 1327 | 515.2 s | 430.5 | 1092 |
| Max-Pressure | 1.82 s | 1983 | 374.9 s | 247.4 | 849 |

- With 4 threads the metrics are identical (deterministic) but it is slower: 4.5 / 4.8 s.
  Use 1 thread per simulator and parallelise across runs.
- Max-Pressure beats Fixed-Time by 27 % on travel time, as expected. That ordering is a
  sanity check. These are not tuned baseline numbers; Phase 2 implements the real baselines.
- At ~2–3 s of simulation per episode, the simulator won't be the bottleneck. NN forward
  passes will be.

## Files / commits
`scripts/bench_cityflow.py`, `README.md` (rewritten: it was UTF-16/CRLF, now UTF-8/LF),
`ROADMAP.md` · commit: see git log

## Next
- Repeat on native Ubuntu following the README and add the numbers here; then close 0.1b.
- 0.2: `requirements.txt` (record the CityFlow build as a separate manual step), repo skeleton.
