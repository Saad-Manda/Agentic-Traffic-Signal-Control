"""UG-MARL agent: neighbour attention + Q-ensemble + uncertainty gate (plan phase6-ugmarl D5).

One :class:`~trafficrl.agents.ugmarl.network.UGMARLNet` shared by every intersection, trained
by the common :class:`~trafficrl.agents.core.QLearner` (Double DQN, SMDP target, bootstrap
masks when K > 1). Acting:

* ``explore=True`` (training): ε-greedy on the mean Q μ — the **gate is never applied**;
* ``explore=False``: a* = argmax μ, and if the gate is active and σ(s_i) > τ the action is
  Max-Pressure's (same rule, Δt and observed detectors as the baseline, protocol §2).

The gate is active iff ``use_gate`` and ``use_ensemble`` and τ is known. ``tau = None`` is
the ungated policy (validation episodes inside ``train.py``, so checkpoints are selected
ungated); evaluation code resolves τ first and refuses a gated run without one
(:func:`trafficrl.eval.runner.evaluate`).

Ablation flags (``agents.ugmarl``): ``use_attention`` (False: no neighbour input),
``use_ensemble`` (False: K = 1, no σ, no gate), ``use_gate`` (False: same weights, gate off).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from trafficrl.agents.baselines.max_pressure import MaxPressureAgent
from trafficrl.agents.core import NeighborhoodEncoder, QLearner
from trafficrl.agents.ugmarl.gate import sigma_from_q
from trafficrl.agents.ugmarl.network import UGMARLNet

ARCH_KEYS = ("use_attention", "use_ensemble", "heads", "attn_dim", "head_hidden")
SLOTS = NeighborhoodEncoder.SLOTS


def variant_name(params: Mapping[str, Any] | None) -> str:
    """``ugmarl``, ``ugmarl-noattn``, ``ugmarl-noens``, ``ugmarl-nogate`` (or combinations)."""
    p = dict(params or {})
    parts = ["ugmarl"]
    if not p.get("use_attention", True):
        parts.append("noattn")
    if not p.get("use_ensemble", True):
        parts.append("noens")  # no σ -> no gate either
    elif not p.get("use_gate", True):
        parts.append("nogate")
    return "-".join(parts)


class UGMARLAgent(QLearner):
    name = "ugmarl"

    def __init__(self, env: Any, params: Mapping[str, Any] | None = None, seed: int = 0):
        p = dict(params or {})
        self.use_attention = bool(p.get("use_attention", True))
        self.use_ensemble = bool(p.get("use_ensemble", True))
        self.use_gate = bool(p.get("use_gate", True))
        heads = int(p.get("heads", 5)) if self.use_ensemble else 1
        self.arch = {
            "use_attention": self.use_attention,
            "use_ensemble": self.use_ensemble,
            "heads": heads,
            "attn_dim": int(p.get("attn_dim", 64)),
            "head_hidden": int(p.get("head_hidden", 128)),
        }
        net = UGMARLNet(
            env.obs_dim,
            len(env.phase_ids),
            attn_dim=self.arch["attn_dim"],
            head_hidden=self.arch["head_hidden"],
            heads=heads,
            use_attention=self.use_attention,
        )
        super().__init__(
            net,
            NeighborhoodEncoder.from_env(env),
            env.intersection_ids,
            env.phase_ids,
            env.delta_t,
            shared=True,
            params=p,
            seed=seed,
        )
        features = getattr(env, "features", None)
        self.fallback = MaxPressureAgent(features) if features is not None else None
        tau = (p.get("gate") or {}).get("tau")
        self.tau: float | None = None if tau is None else float(tau)
        self._explanations: dict[str, Any] = {}

    # --- gate --------------------------------------------------------------------------

    @property
    def gate_requested(self) -> bool:
        """The configuration asks for a gate (τ may still be unknown)."""
        return self.use_gate and self.use_ensemble and self.heads > 1

    @property
    def gate_active(self) -> bool:
        return self.gate_requested and self.tau is not None

    def variant(self) -> str:
        return variant_name({**self.arch, "use_gate": self.use_gate})

    # --- acting ------------------------------------------------------------------------

    def forward_all(self, states: np.ndarray) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(n, K, A)`` Q-values and ``(n, 5)`` attention weights (None without attention)."""
        x = torch.as_tensor(states, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q, alpha = self.net(x, return_attention=True)
        return q[0], (None if alpha is None else alpha[0])

    def act(
        self, obs: Mapping[str, np.ndarray], info: Mapping[str, Any], explore: bool = False
    ) -> dict[str, int]:
        q, alpha = self.forward_all(self.encoder.encode_all(obs))
        mu, a_star, sigma = sigma_from_q(q)
        greedy = a_star.cpu().numpy()
        actions: dict[str, int] = {}
        if explore:  # training: ε-greedy on μ, identical to the baselines; no gate
            for k, iid in enumerate(self.intersection_ids):
                a = int(greedy[k])
                if self.rng.random() < self.epsilon:
                    a = int(self.rng.integers(len(self.phase_ids)))
                actions[iid] = self.phase_ids[a]
            self._explanations = {}
            return actions

        sig = sigma.cpu().numpy()
        gate = self.gate_active
        if gate and self.fallback is None:
            raise RuntimeError("gate active but the env has no feature builder for Max-Pressure")
        counts = info.get("lane_count")
        mp = (
            {iid: self.fallback.choose(iid, counts) for iid in self.intersection_ids}
            if self.fallback is not None and counts is not None
            else {}
        )
        fired: dict[str, bool] = {}
        for k, iid in enumerate(self.intersection_ids):
            fired[iid] = bool(gate and sig[k] > self.tau)
            actions[iid] = mp[iid] if fired[iid] else self.phase_ids[int(greedy[k])]
        mu_np = mu.cpu().numpy()
        att = None if alpha is None else alpha.cpu().numpy()
        self._explanations = {
            "fallback": fired,
            "sigma": {iid: float(sig[k]) for k, iid in enumerate(self.intersection_ids)},
            "tau": self.tau if gate else None,
            "greedy_phase": {
                iid: self.phase_ids[int(greedy[k])] for k, iid in enumerate(self.intersection_ids)
            },
            "mp_phase": mp,
            "attention": (
                None
                if att is None
                else {iid: att[k] for k, iid in enumerate(self.intersection_ids)}
            ),
            "q_mean": {iid: mu_np[k] for k, iid in enumerate(self.intersection_ids)},
        }
        return actions

    def last_explanations(self) -> dict[str, Any]:
        return self._explanations

    # --- checkpoint --------------------------------------------------------------------

    def checkpoint_extra(self) -> dict[str, Any]:
        return {"arch": dict(self.arch)}

    def check_checkpoint(self, ckpt: Mapping[str, Any]) -> None:
        saved = ckpt.get("arch")
        if saved is None:
            raise ValueError("checkpoint has no UG-MARL architecture record")
        diff = {k: (saved.get(k), self.arch[k]) for k in ARCH_KEYS if saved.get(k) != self.arch[k]}
        if diff:
            raise ValueError(f"checkpoint architecture differs (saved, requested): {diff}")
