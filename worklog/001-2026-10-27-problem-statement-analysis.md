# 001 — Problem statement analysis and topic alignment
**Date:** 2026-10-27   **Tag:** decision   **ROADMAP:** —   **Status:** done

## What
Checked whether our proposed topic ("Agentic AI-Based Intelligent Traffic Management System")
fits Problem Statement 16, revised it, and chose the proposed method.

## Why
PS16 doesn't just ask to apply agentic AI. It asks us to study conventional single- and
multi-agent frameworks, propose a new mathematical or algorithmic change, and compare it
against conventional, single-agent and multi-agent baselines. The rubric gives 25 marks to
methodology, ablation and comparison (rows 5–6). The first draft had no contribution, no
baselines, no ablation and no named datasets.

Alternatives considered:
- **Keep the generic topic.** Rejected: it would score poorly on rubric rows 5–6.
- **Add an LLM agent layer.** Rejected: more compute and API cost, and harder to justify
  mathematically in 3 weeks.

## How
- Read `Problem Statement.pdf` (PS16 text plus the evaluation rubric).
- Listed the gaps between our draft and PS16, then mapped PS16's metrics onto traffic:
  task completion → throughput, decision accuracy → travel/waiting time,
  response time → inference latency, and so on.
- Designed a contribution that addresses the weaknesses PS16 names.

## Result
- **Title:** *Uncertainty-Aware Coordinated Multi-Agent AI for Adaptive Traffic Signal Control*.
- **Method: UG-MARL.** One RL agent per intersection, neighbor attention for coordination,
  a Q-ensemble to estimate uncertainty, and a gate that falls back to Max-Pressure when
  uncertainty is high. Max-Pressure is provably queue-stable (Varaiya 2013).
- **Baselines:** Fixed-Time and Max-Pressure (conventional), DQN (single-agent),
  IDQN (multi-agent).
- **Ablations:** remove the gate, remove attention, remove the ensemble.
- Revised project description drafted. It is summarised in `PRD.md` and `CLAUDE.md`.

## Files / commits
None (analysis only). Captured in the scaffold, see [002](002-2026-10-27-project-scaffold.md).

## Next
Scaffold the project workspace (002).
