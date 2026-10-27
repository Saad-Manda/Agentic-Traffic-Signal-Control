# Architecture

## Simulator layer — TODO (decision pending)

Options: SUMO + `sumo-rl` (native Windows, GUI) · CityFlow via WSL/Docker (fast, native
Jinan/Hangzhou) · both via LibSignal. Whichever is chosen is wrapped by
`src/trafficrl/envs/` so agents never call the simulator directly.

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
