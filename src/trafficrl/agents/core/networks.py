"""Q-networks with a group axis and a heads axis (plan phase5-rl-baselines D4).

Both networks map states shaped ``(G, B, in_dim)`` to Q-values shaped ``(G, B, K, A)``:

* :class:`MLPQNet` — **one** parameter set, broadcast over the group axis (shared-parameter
  DQN: G = 1 at training time, the B axis holds the intersections when acting).
* :class:`StackedMLPQNet` — **G independent** parameter sets evaluated in one batched matmul
  (IDQN: G = number of intersections). Equivalent to a loop of G :class:`MLPQNet`.

K (heads) is 1 for DQN / IDQN; UG-MARL's Q-ensemble (Phase 6) uses K > 1 on the same trunk.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _dims(in_dim: int, hidden: Sequence[int], out_dim: int) -> list[tuple[int, int]]:
    sizes = [in_dim, *hidden, out_dim]
    return list(zip(sizes[:-1], sizes[1:], strict=True))


class MLPQNet(nn.Module):
    def __init__(
        self, in_dim: int, n_actions: int, hidden: Sequence[int] = (128, 128), heads: int = 1
    ) -> None:
        super().__init__()
        self.in_dim, self.n_actions, self.heads = in_dim, n_actions, heads
        layers: list[nn.Module] = []
        dims = _dims(in_dim, hidden, heads * n_actions)
        for k, (i, o) in enumerate(dims):
            layers.append(nn.Linear(i, o))
            if k < len(dims) - 1:
                layers.append(nn.ReLU())
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        return out.view(*x.shape[:-1], self.heads, self.n_actions)


class StackedMLPQNet(nn.Module):
    """G independent MLPs; layer weights stored as ``(G, in, out)``."""

    def __init__(
        self,
        groups: int,
        in_dim: int,
        n_actions: int,
        hidden: Sequence[int] = (128, 128),
        heads: int = 1,
    ) -> None:
        super().__init__()
        self.groups, self.in_dim, self.n_actions, self.heads = groups, in_dim, n_actions, heads
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i, o in _dims(in_dim, hidden, heads * n_actions):
            # initialise each group exactly like an nn.Linear (Kaiming-uniform), then stack
            lins = [nn.Linear(i, o) for _ in range(groups)]
            self.weights.append(nn.Parameter(torch.stack([lin.weight.data.T for lin in lins])))
            self.biases.append(nn.Parameter(torch.stack([lin.bias.data for lin in lins])))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != self.groups:
            raise ValueError(f"expected {self.groups} groups on axis 0, got {x.shape[0]}")
        h = x
        last = len(self.weights) - 1
        for k, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            h = torch.baddbmm(b.unsqueeze(1), h, w)
            if k < last:
                h = torch.relu(h)
        return h.view(*x.shape[:-1], self.heads, self.n_actions)

    def load_from(self, nets: Sequence[MLPQNet]) -> None:
        """Copy G :class:`MLPQNet` weights into this module (tests / conversions)."""
        linears = [[m for m in net.body if isinstance(m, nn.Linear)] for net in nets]
        with torch.no_grad():
            for k in range(len(self.weights)):
                self.weights[k].copy_(torch.stack([lins[k].weight.T for lins in linears]))
                self.biases[k].copy_(torch.stack([lins[k].bias for lins in linears]))


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
