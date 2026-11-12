"""Configurable Q-learner shared by every learned agent (ROADMAP 5.1; plan D5).

Double DQN with Huber loss, a hard-synced target network, uniform grouped replay and
ε-greedy exploration. The TD target for a transition (s, a, r, s′, d) is

    a*  = argmax_a′ Q_online(s′, a′)                         (Double DQN, per head)
    y   = c · r + γ^(d / Δt) · Q_target(s′, a*)              (SMDP discount)
    L   = Huber_δ=1(Q_online(s, a) − y)

* ``c`` = constant reward scale (0.01; protocol §9 clarification, 11 Nov) — the frozen reward
  −|P_i| is unchanged, the optimal policy is unchanged by a positive constant.
* ``d`` = interval length in seconds (15, or 20 after a phase change), Δt = 15 s.
* The episode end (t = 3600 s) is a time-limit **truncation**, not a terminal state, so the
  target always bootstraps (Pardo et al., 2018).

The network maps ``(G, B, in)`` to ``(G, B, K, A)``. Shared parameters (DQN) use one group;
independent learners (IDQN) use one group per intersection. K (``net.heads``) is 1 for DQN /
IDQN. UG-MARL's ensemble (Phase 6, plan phase6-ugmarl D4) uses K > 1: every stored transition
gets a bootstrap mask m ∈ {0, 1}^K, m_k ~ Bernoulli(``mask_prob``), and the loss is

    L = (1/K) Σ_k [ Σ_b m_bk · Huber(Q_k(s_b, a_b) − y_bk) / max(1, Σ_b m_bk) ]

with the per-head Double-DQN target y_k. With K = 1 there is no mask and L is the Phase 5
loss (same code path, same RNG streams).

Experience is stored only while ``self.training`` is True (set by the trainer for training
episodes). Validation / evaluation episodes therefore never enter the replay buffer.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from trafficrl.agents.core.replay import ReplayBuffer
from trafficrl.utils.seed import make_rng

# Defaults of the `train:` block in configs/base.yaml (kept here so an agent built from an
# empty config — e.g. in a test — is still valid).
DEFAULTS: dict[str, Any] = {
    "lr": 3.0e-4,
    "gamma": 0.9,
    "batch_size": 64,
    "buffer_size": 100_000,
    "learning_starts": 2000,
    "train_every": 4,
    "target_sync": 1000,
    "grad_clip": 10.0,
    "reward_scale": 0.01,
    "eps_end": 0.05,
    "mask_prob": 0.8,
    "device": "cpu",
}
ACT_SEED_OFFSET = 20_000  # exploration RNG stream = seed + offset
REPLAY_SEED_OFFSET = 30_000  # replay sampling RNG stream
MASK_SEED_OFFSET = 40_000  # bootstrap-mask RNG stream (K > 1 only)


class QLearner:
    name = "qlearner"

    def __init__(
        self,
        net: nn.Module,
        encoder: Any,
        intersection_ids: Sequence[str],
        phase_ids: Sequence[int],
        delta_t: float,
        *,
        shared: bool,
        params: Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        p = {**DEFAULTS, **dict(params or {})}
        self.params = p
        self.device = torch.device(p["device"])
        self.encoder = encoder
        self.intersection_ids = list(intersection_ids)
        self.phase_ids = tuple(int(x) for x in phase_ids)
        self._action_index = {ph: k for k, ph in enumerate(self.phase_ids)}
        self.delta_t = float(delta_t)
        self.shared = shared
        n = len(self.intersection_ids)
        self.groups = 1 if shared else n
        self.heads = int(getattr(net, "heads", 1))
        self._group = {iid: (0 if shared else k) for k, iid in enumerate(self.intersection_ids)}

        self.net = net.to(self.device)
        self.target = _clone(net).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.requires_grad_(False)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=float(p["lr"]))

        capacity = max(1, int(p["buffer_size"]) // self.groups)
        extras = {"mask": ((self.heads,), np.float32)} if self.heads > 1 else None
        self.buffer = ReplayBuffer(
            self.groups,
            capacity,
            encoder.state_shape,
            make_rng(seed + REPLAY_SEED_OFFSET),
            extras=extras,
        )
        self.mask_rng = make_rng(seed + MASK_SEED_OFFSET)
        self.rng = make_rng(seed + ACT_SEED_OFFSET)
        self.epsilon = float(p["eps_end"])
        self.training = False
        self.grad_steps = 0
        self.transitions_seen = 0
        self._pending_updates = 0

    # --- acting ------------------------------------------------------------------------

    def q_values(self, states: np.ndarray) -> torch.Tensor:
        """``(n_intersections, K, A)`` Q-values for encoded states ``(n, *state_shape)``."""
        x = torch.as_tensor(states, device=self.device)
        x = x.unsqueeze(0) if self.shared else x.unsqueeze(1)
        with torch.no_grad():
            q = self.net(x)
        return q[0] if self.shared else q[:, 0]

    def act(
        self, obs: Mapping[str, np.ndarray], info: Mapping[str, Any], explore: bool = False
    ) -> dict[str, int]:
        q = self.q_values(self.encoder.encode_all(obs)).mean(dim=1)  # mean over heads
        greedy = q.argmax(dim=-1).cpu().numpy()
        actions = {}
        for k, iid in enumerate(self.intersection_ids):
            a = int(greedy[k])
            if explore and self.rng.random() < self.epsilon:
                a = int(self.rng.integers(len(self.phase_ids)))
            actions[iid] = self.phase_ids[a]
        return actions

    # --- learning ----------------------------------------------------------------------

    def observe(self, transitions: Sequence[Any]) -> None:
        if not self.training:
            return
        prob = float(self.params["mask_prob"])
        for tr in transitions:
            s, s2 = self.encoder.encode_transition(tr)
            extras = {}
            if self.heads > 1:
                extras["mask"] = (self.mask_rng.random(self.heads) < prob).astype(np.float32)
            self.buffer.add(
                self._group[tr.iid],
                s,
                self._action_index[int(tr.action)],
                float(tr.reward),
                s2,
                float(tr.duration_s),
                bool(tr.done),
                **extras,
            )
            self.transitions_seen += 1
            if self.transitions_seen > int(self.params["learning_starts"]):
                self._pending_updates += 1

    def update(self) -> dict[str, float]:
        every = int(self.params["train_every"])
        steps, self._pending_updates = divmod(self._pending_updates, every)  # keep remainder
        if steps == 0 or len(self.buffer) == 0:
            return {}
        stats = [self._gradient_step() for _ in range(steps)]
        return {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}

    def td_target(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``(G, B, K)`` Double-DQN SMDP target (no gradient)."""
        p = self.params
        with torch.no_grad():
            next_online = self.net(batch["next_state"])  # (G, B, K, A)
            a_star = next_online.argmax(dim=-1, keepdim=True)
            next_target = self.target(batch["next_state"]).gather(-1, a_star).squeeze(-1)
            discount = float(p["gamma"]) ** (batch["duration"] / self.delta_t)  # (G, B)
            reward = float(p["reward_scale"]) * batch["reward"]
            return reward.unsqueeze(-1) + discount.unsqueeze(-1) * next_target

    def _gradient_step(self) -> dict[str, float]:
        p = self.params
        raw = self.buffer.sample(int(p["batch_size"]))
        batch = {k: torch.as_tensor(v, device=self.device) for k, v in raw.items()}
        q = self.net(batch["state"])  # (G, B, K, A)
        idx = batch["action"][..., None, None].expand(*q.shape[:-1], 1)
        q_sa = q.gather(-1, idx).squeeze(-1)  # (G, B, K)
        y = self.td_target(batch)
        if "mask" in batch:
            loss = masked_ensemble_loss(q_sa, y, batch["mask"])
        else:
            loss = F.smooth_l1_loss(q_sa, y)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), float(p["grad_clip"]))
        self.optim.step()
        self.grad_steps += 1
        if self.grad_steps % int(p["target_sync"]) == 0:
            self.target.load_state_dict(self.net.state_dict())
        stats = {
            "loss": float(loss.detach()),
            "q_mean": float(q_sa.detach().mean()),
            "td_abs": float((q_sa - y).detach().abs().mean()),
            "grad_norm": float(grad_norm),
        }
        if self.heads > 1:  # head disagreement at the taken action (population std)
            stats["sigma_mean"] = float(q_sa.detach().std(dim=-1, unbiased=False).mean())
        return stats

    # --- persistence -------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "name": self.name,
                "net": self.net.state_dict(),
                "target": self.target.state_dict(),
                "optim": self.optim.state_dict(),
                "grad_steps": self.grad_steps,
                "transitions_seen": self.transitions_seen,
                "params": {k: v for k, v in self.params.items() if k != "device"},
                "intersection_ids": self.intersection_ids,
                "phase_ids": list(self.phase_ids),
                "shared": self.shared,
                "state_shape": list(self.encoder.state_shape),
                **self.checkpoint_extra(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        if ckpt.get("name") != self.name:
            raise ValueError(f"checkpoint is for {ckpt.get('name')!r}, agent is {self.name!r}")
        if self.shared:
            # one network for every intersection: it runs on any network with the same
            # state layout and phase set (e.g. zero-shot on Hangzhou, protocol §6)
            if list(ckpt["phase_ids"]) != list(self.phase_ids):
                raise ValueError("checkpoint was trained with a different phase set")
            shape = ckpt.get("state_shape")
            if shape is not None and tuple(shape) != tuple(self.encoder.state_shape):
                raise ValueError(
                    f"checkpoint state shape {tuple(shape)} != {tuple(self.encoder.state_shape)}"
                )
        elif list(ckpt["intersection_ids"]) != self.intersection_ids:
            raise ValueError("checkpoint was trained on a different set of intersections")
        self.check_checkpoint(ckpt)
        self.net.load_state_dict(ckpt["net"])
        self.target.load_state_dict(ckpt["target"])
        self.optim.load_state_dict(ckpt["optim"])
        self.grad_steps = int(ckpt["grad_steps"])
        self.transitions_seen = int(ckpt["transitions_seen"])

    def checkpoint_extra(self) -> dict[str, Any]:
        """Extra entries saved in the checkpoint (subclasses: architecture description)."""
        return {}

    def check_checkpoint(self, ckpt: Mapping[str, Any]) -> None:
        """Raise if ``ckpt`` does not fit this agent (subclasses); called before loading."""
        return None

    def last_explanations(self) -> dict[str, Any]:
        return {}


def masked_ensemble_loss(q_sa: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Bootstrapped-ensemble Huber loss, all tensors ``(G, B, K)``.

    Per head: mean Huber loss over the transitions its mask selects (a head with an all-zero
    mask column contributes 0); then the mean over heads and groups (1/K keeps the shared
    encoder's gradient scale independent of K, Osband et al. 2016).
    """
    h = F.smooth_l1_loss(q_sa, y, reduction="none") * mask
    per_head = h.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (G, K)
    return per_head.mean()


def _clone(net: nn.Module) -> nn.Module:
    return copy.deepcopy(net)
