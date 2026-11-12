"""UG-MARL Q-network: neighbour attention + K-head Q-ensemble (plan phase6-ugmarl D3).

Input ``(G, B, 5, obs_dim + 1)`` from :class:`~trafficrl.agents.core.NeighborhoodEncoder`
(slot 0 = self, 1–4 = neighbours E, N, W, S; last column = presence flag). Output
``(G, B, K, A)`` — the same contract as :class:`~trafficrl.agents.core.MLPQNet`, so the
Q-learner core is unchanged. One parameter set is shared by every intersection (G = 1).

    e_j  = ReLU(W_e s_j + b_e) + p_slot(j)                j ∈ {self, E, N, W, S}, e_j ∈ R^d
    α_ij = softmax_j ((W_q e_i)ᵀ (W_k e_j) / √d)          absent neighbours masked (−∞)
    c_i  = Σ_j α_ij W_v e_j
    z_i  = [e_i ‖ c_i] ∈ R^{2d}                            (use_attention=False: z_i = e_i)
    Q_k(s_i, ·) = W2_k ReLU(W1_k z_i + b1_k) + b2_k        k = 1..K, hidden ``head_hidden``

One GAT-style layer with one attention head (CoLight, Wei et al. 2019), so α_i is a single
5-vector per decision that can be plotted. Self is always present, so the softmax is never
over an empty set. The K heads are stored stacked ``(K, in, out)`` and evaluated with one
``einsum``; each is initialised like an independent ``nn.Linear``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

N_SLOTS = 5


class UGMARLNet(nn.Module):
    def __init__(
        self,
        obs_dim: int = 44,
        n_actions: int = 8,
        attn_dim: int = 64,
        head_hidden: int = 128,
        heads: int = 5,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be >= 1")
        self.obs_dim, self.n_actions, self.heads = obs_dim, n_actions, heads
        self.attn_dim, self.head_hidden, self.use_attention = attn_dim, head_hidden, use_attention
        self.embed = nn.Linear(obs_dim, attn_dim)
        self.slot = nn.Parameter(torch.zeros(N_SLOTS, attn_dim))
        nn.init.normal_(self.slot, std=0.1)
        if use_attention:
            self.query = nn.Linear(attn_dim, attn_dim)
            self.key = nn.Linear(attn_dim, attn_dim)
            self.value = nn.Linear(attn_dim, attn_dim)
        z_dim = 2 * attn_dim if use_attention else attn_dim
        self.w1, self.b1 = _stacked_linear(heads, z_dim, head_hidden)
        self.w2, self.b2 = _stacked_linear(heads, head_hidden, n_actions)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``z`` ``(..., z_dim)`` and attention weights ``(..., 5)`` (None without attention)."""
        e = torch.relu(self.embed(x[..., : self.obs_dim])) + self.slot  # (..., 5, d)
        if not self.use_attention:
            return e[..., 0, :], None
        present = x[..., -1] > 0.5  # (..., 5); slot 0 is always 1
        q = self.query(e[..., 0, :])  # (..., d)
        k, v = self.key(e), self.value(e)  # (..., 5, d)
        scores = torch.einsum("...d,...jd->...j", q, k) / math.sqrt(self.attn_dim)
        scores = scores.masked_fill(~present, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        c = torch.einsum("...j,...jd->...d", alpha, v)
        return torch.cat([e[..., 0, :], c], dim=-1), alpha

    def heads_forward(self, z: torch.Tensor) -> torch.Tensor:
        """``(..., z_dim)`` -> ``(..., K, A)``."""
        h = torch.relu(torch.einsum("...i,kih->...kh", z, self.w1) + self.b1)
        return torch.einsum("...kh,kha->...ka", h, self.w2) + self.b2

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        z, alpha = self.encode(x)
        q = self.heads_forward(z)
        return (q, alpha) if return_attention else q


def _stacked_linear(k: int, i: int, o: int) -> tuple[nn.Parameter, nn.Parameter]:
    lins = [nn.Linear(i, o) for _ in range(k)]  # independent Kaiming-uniform init per head
    w = nn.Parameter(torch.stack([lin.weight.data.T for lin in lins]))  # (K, in, out)
    b = nn.Parameter(torch.stack([lin.bias.data for lin in lins]))  # (K, out)
    return w, b


def expected_params(
    obs_dim: int, n_actions: int, attn_dim: int, head_hidden: int, heads: int, use_attention: bool
) -> int:
    """Parameter count formula (checked against ``n_params`` in the tests; reported in tables)."""
    d = attn_dim
    enc = obs_dim * d + d + 5 * d + (3 * (d * d + d) if use_attention else 0)
    z = 2 * d if use_attention else d
    return enc + heads * (z * head_hidden + head_hidden + head_hidden * n_actions + n_actions)
