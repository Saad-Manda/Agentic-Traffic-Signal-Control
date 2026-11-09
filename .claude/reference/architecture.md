# Architecture

## Simulator layer — CityFlow

CityFlow (C++ engine, Python bindings via `cityflow.Engine(config.json)`), built from source
on Linux (native Ubuntu 24.04 + WSL2), commit pinned. Chosen over SUMO for speed, native
Jinan/Hangzhou roadnet/flow files, and comparability with published CityFlow results.

- Wrapped by `src/trafficrl/envs/` so agents never call the engine directly; the wrapper
  builds the per-intersection obs, neighbor map, and enforces yellow / min-green.
- Engine calls used: `get_lane_waiting_vehicle_count()` (queues), `get_lane_vehicle_count()`
  (pressure = incoming − outgoing), `set_tl_phase(intersection_id, phase)`, `next_step()`,
  `get_average_travel_time()`, `reset()` / `set_random_seed(seed)` for seeding.
- No built-in multi-agent RL env — the wrapper is written in-repo (no `sumo-rl` equivalent).
- Demo/visualization: CityFlow replay (`saveReplay` → `frontend/index.html`, opened in a
  Windows browser via `\\wsl$`) plus an in-repo matplotlib animation colouring each
  intersection by gate decision (RL vs. Max-Pressure) and σ(s). Replay output only for demo
  runs — it slows training.

## MDP formulation (per intersection i)

- **State** sᵢ: per incoming-lane queue length (and/or vehicle count), lane pressure
  (incoming − outgoing vehicles), one-hot current phase. Neighbor states s_j for j ∈ N(i).
- **Action** aᵢ: next signal phase (with fixed min-green and yellow transitions handled by env).
- **Reward** rᵢ: −pressureᵢ (alternative: −Σ queue). TODO: finalize.
- **Decision interval**: Δt seconds (TODO, typically 5–10 s).

## Agents

| Agent | Type | Notes |
|---|---|---|
| Fixed-Time | conventional | fixed cycle, equal splits |
| Max-Pressure | conventional | picks phase maximizing pressure; also UG-MARL fallback |
| DQN | single-agent RL | one centralized agent (or shared params, no coordination) |
| IDQN | multi-agent RL | independent DQN per intersection |
| UG-MARL | proposed | below |

## UG-MARL

```
local obs sᵢ ─┐
neighbor obs ─┴► attention encoder hᵢ = Σⱼ αᵢⱼ W s_j   (αᵢⱼ = softmax over j∈N(i)∪{i})
hᵢ ─► K Q-heads Q₁..Q_K(hᵢ, ·)
μ(a) = mean_k Q_k(a),  σ(sᵢ) = std_k Q_k(argmax_a μ(a))
aᵢ = argmax_a μ(a)        if σ(sᵢ) ≤ τ
     MaxPressure(sᵢ)      otherwise
```
- Ensemble heads trained with bootstrapped masks / different init (TODO: decide).
- τ chosen on validation flows (e.g. a percentile of σ).
- Stability argument: when the gate fires, behavior equals Max-Pressure, which is
  throughput-optimal / queue-stable under Varaiya (2013) assumptions → bounded worst case.
- Explainability: log αᵢⱼ, σ, gate decisions per step.

## Data flow

raw flows → split (train/val/test, held-out city) → preprocessing → env configs →
training (train flows) → tuning (val flows) → final eval (test flows, held-out city) →
`results/<experiment>/metrics.csv` + plots.

## Diagram requirements (guidelines, ROADMAP 9.5)

Report/slide diagram: labelled blocks in data-flow order — raw flows → split → normalisation
(**fit on train only**) → CityFlow env → attention encoder → Q-ensemble → gate → phase action
→ metrics. Show the **training path** (replay, ensemble updates, τ chosen on validation)
separately from the **inference path**, and add a caption.

## Algorithm 1 — draft (guidelines format, ROADMAP 9.6)

Must be re-checked line by line against the final code before submission.

**Algorithm 1: UG-MARL training, tuning and evaluation**
**Input:** roadnet G; flows F_train, F_val, F_test (primary city), F_heldout; search space H;
seeds S. **Output:** trained ensemble Q* per agent, threshold τ*, test metrics.

1. Split the primary city's flows into F_train, F_val, F_test; keep F_heldout untouched.
2. On F_train, compute observation normalisation statistics.
3. For each h ∈ H (and seed s ∈ S): train UG-MARL on F_train — for each step, every
   intersection i encodes (sᵢ, s_N(i)) with attention, the K Q-heads give μ and σ, the agent
   acts ε-greedily on μ, transitions go to replay, and each head is updated on its bootstrap
   mask with a target network.
4. Evaluate each trained model on F_val; record σ distribution; set τ(h) = chosen percentile
   of σ on F_val.
5. Select h*, τ* = argmax over validation score.
6. Run the gated policy (σ > τ* ? Max-Pressure : argmax μ) **once** on F_test and F_heldout
   for every seed; report mean ± std and 95% CI.

