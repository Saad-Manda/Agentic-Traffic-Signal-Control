# 004 — Simulator (CityFlow) and platform decision
**Date:** 2026-10-27   **Tag:** decision   **ROADMAP:** 0.1   **Status:** done

## What
Chose **CityFlow** as the traffic simulator. Training and final numbers run on **native
Ubuntu 24.04** (dual-boot). **WSL2 Ubuntu 24.04** is used for development, plots and the demo.

## Why
- **Speed.** The ensemble, ablations and multiple seeds need thousands of episodes.
  CityFlow is much faster than SUMO.
- **Data.** Jinan and Hangzhou are CityFlow's native formats.
- **Comparability.** Our baselines can be checked against published CoLight, PressLight and
  MPLight results.

Alternatives considered:
- **SUMO + `sumo-rl`.** Rejected: slower, and Jinan/Hangzhou exist only as conversions.
  Kept only as an option for a stretch demo.
- **Both, via LibSignal.** Rejected: too much setup for a solo 3-week project.

Trade-offs accepted:
- Simpler vehicle dynamics.
- A basic replay viewer. Mitigated by an in-repo animation of the gate's decisions.
- No ready-made multi-agent RL environment, so we write our own wrapper.
- CityFlow only builds on Linux. Building it with Python 3.12 needs a newer pybind11
  submodule.

## How
- Compared the options against the project's needs: training volume, datasets, a laptop
  with an RTX 4050, a Windows host and a demo.
- Updated `CLAUDE.md`, `PRD.md`, `ROADMAP.md` and `.claude/reference/architecture.md`.
  The architecture file now lists the CityFlow engine calls and the replay/demo plan.
- Platform rules:
  - Clone the repo into each Linux filesystem (not `/mnt/d`).
  - Sync between environments through git.
  - Use a separate `.venv` per environment.
  - Never install an NVIDIA driver inside WSL.
  - Report final numbers from native Ubuntu runs only.

## Result
Decision recorded. The install has not been checked on either platform yet.

## Files / commits
`CLAUDE.md`, `PRD.md`, `ROADMAP.md`, `.claude/reference/architecture.md` · `95ee94a`

## Next
ROADMAP 0.1b:
- Build CityFlow on WSL and on native Ubuntu.
- Time one Jinan episode with Fixed-Time and with Max-Pressure.
- Write the install steps to the README.
