# Interfaces

No HTTP API — this is a research codebase. "API" = the internal Python interfaces.
All signatures below are **planned**; update when implemented.

## Environment (`src/trafficrl/envs/`)

```python
class TrafficEnv:
    def __init__(self, config: dict, seed: int): ...
    def reset(self) -> dict[str, np.ndarray]: ...          # obs per intersection id
    def step(self, actions: dict[str, int]) -> tuple[obs, rewards, done, info]: ...
    @property
    def intersection_ids(self) -> list[str]: ...
    def neighbors(self, iid: str) -> list[str]: ...
    def metrics(self) -> dict[str, float]: ...             # travel time, wait, throughput, queue
```
Observation noise / detector dropout for robustness tests applied via config in the wrapper.

## Agent (`src/trafficrl/agents/`)

```python
class Agent(Protocol):
    def act(self, obs: dict[str, np.ndarray], explore: bool = False) -> dict[str, int]: ...
    def observe(self, transition) -> None: ...             # no-op for non-learning baselines
    def update(self) -> dict[str, float]: ...              # training losses; {} for baselines
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```
UG-MARL additionally exposes `last_explanations() -> dict` (attention weights, σ, gate flag).

## Scripts

- `scripts/train.py --config <yaml>` — TODO
- `scripts/evaluate.py --config <yaml> --checkpoint <path>` — TODO
- `scripts/run_ablation.py --config <yaml>` — TODO
