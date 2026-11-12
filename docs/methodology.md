# Methodology — UG-MARL (mathematical formulation)

ROADMAP 6.6 · plan `.claude/plans/phase6-ugmarl.md` (D3–D5, D9, D12) · frozen protocol
`docs/evaluation_protocol.md`. Written to be copied into report Chapter 3 ("Proposed
Methodology") and the poster. Every equation below was checked against the code it names
(section 8).

## 1. Problem: a networked semi-Markov decision process

The road network is a graph $G = (\mathcal I, \mathcal E)$ of $N$ signalised intersections
(Jinan: $N = 12$, a $3 \times 4$ grid). Intersection $i$ has at most four neighbours
$\mathcal N(i) \subseteq \{E, N, W, S\}$ (corners have two, edges three).

**Timing.** Each intersection runs its own clock. When it is *ready* it chooses the next green
phase; the phase is held for $\Delta t = 15$ s, and a change of phase is preceded by a
$5$ s yellow, so one decision interval lasts $d \in \{15, 20\}$ s (shorter only at the
episode cut, $T = 3600$ s). Decisions are therefore asynchronous across intersections, and
each intersection's decision process is a semi-MDP.

**State.** $s_i \in [0,1]^{36} \times \{0,1\}^8 \subset \mathbb R^{44}$:

$$
s_i = \big[\, q_i \;\|\; x_i \;\|\; \pi_i \;\|\; \mathrm{onehot}(\phi_i) \,\big],
$$

with, for each of the 12 incoming lanes $l$, the queue $q_{i,l} = \min(1, w_l / c_l)$, the
vehicle count $x_{i,l} = \min(1, n_l / c_l)$, and, for each of the 12 movements
$m = (l_{\text{in}} \to l_{\text{out}})$, the movement pressure
$\pi_{i,m} = \bar x_{\text{in}(m)} - \bar x_{\text{out}(m)}$. Here $c_l = \lfloor
\text{length}_l / 7.5\,\text{m} \rfloor$ is the lane capacity. It is a physical constant,
**nothing is fitted** on data, so no normalisation statistics can leak. $\phi_i$ is the
current green phase. All lane readings are the *observed* ones (after detector noise or
dropout in the robustness conditions).

**Action.** $a_i \in \mathcal A = \{1, \dots, 8\}$, the next green phase.

**Reward.** $r_i = -\lvert P_i \rvert$ with intersection pressure
$P_i = \sum_{l \in \text{in}(i)} n_l - \sum_{l \in \text{out}(i)} n_l$ (true counts), measured at
the end of the interval. Max-Pressure minimises the same quantity, so the learned policy and
the gate's fallback pursue one objective.

**Objective.** Maximise $\mathbb E\big[\sum_t \gamma^{\tau_t / \Delta t} r_{i,t}\big]$, where
$\tau_t$ is the elapsed time in seconds. The reported metric is the average vehicle travel
time (protocol §5).

## 2. Learning target (all learned agents)

DQN, IDQN and UG-MARL share one Q-learning core (`trafficrl.agents.core.QLearner`): Double
DQN with a hard-synced target network $\bar\theta$ (copy every 1000 gradient steps), uniform
replay, Huber loss, gradient-norm clipping at 10. For a stored transition
$(s, a, r, s', d)$:

$$
a^\star = \arg\max_{a'} Q_\theta(s', a'), \qquad
y = c\,r + \gamma^{\,d/\Delta t}\, Q_{\bar\theta}(s', a^\star),
$$

$$
\mathcal L(\theta) = \operatorname{Huber}_1\big(Q_\theta(s, a) - y\big).
$$

* $c = 0.01$ is a constant reward scale. A positive constant changes neither the optimal
  policy nor the logged reward (protocol §9, 11 Nov).
* $\gamma^{d/\Delta t}$ is the SMDP discount, which matters because intervals last 15 s or 20 s.
* The cut at $T = 3600$ s is a time-limit **truncation**, so the target always bootstraps
  (Pardo et al., 2018).
* Exploration is ε-greedy, with ε going linearly from 1.0 to 0.05 over the first 30 % of
  100 training episodes, identical for every learned agent.

## 3. Neighbour attention

**Input.** For intersection $i$, five slots $j \in \{0{=}\text{self}, E, N, W, S\}$, each holding
that intersection's 44-value state $s_j$ and a presence flag $m_j \in \{0, 1\}$ ($m_0 = 1$; an
absent neighbour's row is zero). One network is shared by every intersection.

**Embedding.** With $d$ the embedding width and $p_j \in \mathbb R^d$ a learned slot
(direction) embedding:

$$
e_j = \operatorname{ReLU}(W_e s_j + b_e) + p_j \in \mathbb R^d .
$$

**Attention.** One GAT-style layer with one attention head (CoLight, Wei et al., 2019), with
the own intersection as the query:

$$
\alpha_{ij} = \frac{\exp\!\big(\langle W_q e_0 + b_q,\; W_k e_j + b_k\rangle / \sqrt d\big)\, m_j}
                   {\sum_{j'} \exp\!\big(\langle W_q e_0 + b_q,\; W_k e_{j'} + b_k\rangle / \sqrt d\big)\, m_{j'}},
\qquad
c_i = \sum_j \alpha_{ij}\, (W_v e_j + b_v),
$$

$$
z_i = [\, e_0 \,\|\, c_i \,] \in \mathbb R^{2d}.
$$

Absent neighbours get the score $-\infty$, i.e. $\alpha_{ij} = 0$. Self is always present, so
the softmax is never over an empty set, and the same weights work for 2, 3 or 4 neighbours
and for any grid (zero-shot to Hangzhou, $4 \times 4$).

**What $\alpha_{ij}$ is and is not.** $\alpha_{i\cdot}$ is a probability vector over the five
slots that says how much each slot's *value vector* contributes to $c_i$ for this decision.
It is an association inside the model, not a causal effect of neighbour $j$ on the action:
the values $W_v e_j$ can be large or small, $e_0$ also enters $z_i$ directly, and the heads can
ignore $c_i$.

**Ablation −attention.** $z_i = e_0 \in \mathbb R^d$ (no $W_q, W_k, W_v$; slots 1–4 are
ignored). The input and the replay content are identical, so only the use of neighbour
information changes.

## 4. Q-ensemble and epistemic uncertainty

**Heads.** $K$ independent two-layer heads on the shared $z_i$ (hidden width $H = 128$):

$$
Q_k(s_i, \cdot) = W^{(2)}_k \operatorname{ReLU}\big(W^{(1)}_k z_i + b^{(1)}_k\big) + b^{(2)}_k
\in \mathbb R^{|\mathcal A|}, \qquad k = 1, \dots, K .
$$

Each head is initialised independently (Kaiming-uniform, as `nn.Linear`).

**Bootstrap masks** (Osband et al., 2016). Every stored transition $b$ gets a mask
$m_b \in \{0,1\}^K$, $m_{bk} \sim \text{Bernoulli}(0.8)$, drawn once from its own RNG stream and
kept in replay. Each head has its own Double-DQN target
$y_{bk} = c\,r_b + \gamma^{d_b/\Delta t} Q_{\bar\theta,k}(s'_b, \arg\max_{a'} Q_{\theta,k}(s'_b, a'))$, and
the loss over a batch $\mathcal B$ is

$$
\mathcal L(\theta) = \frac1K \sum_{k=1}^K
\frac{\sum_{b \in \mathcal B} m_{bk}\, \operatorname{Huber}_1\big(Q_{\theta,k}(s_b, a_b) - y_{bk}\big)}
     {\max\big(1, \sum_{b \in \mathcal B} m_{bk}\big)} .
$$

The $1/K$ keeps the gradient reaching the shared encoder independent of $K$. With $K = 1$ there
is no mask and $\mathcal L$ is exactly the loss of §2; this was checked bit-for-bit on CityFlow.

**Acting value and uncertainty.**

$$
\mu(s, a) = \frac1K \sum_k Q_k(s, a), \qquad a^\star(s) = \arg\max_a \mu(s, a),
$$

$$
\sigma(s) = \sqrt{\frac1K \sum_k \big(Q_k(s, a^\star) - \mu(s, a^\star)\big)^2 } .
$$

σ is the heads' disagreement about the value of the action the agent is about to take. Heads
differ only through their initialisation and the subsets of experience their masks select, so
σ is large where the data did not pin the value down. That makes it an estimate of
**epistemic** (model) uncertainty, and it does not measure the traffic's intrinsic randomness.

**Limits.** (i) The heads share the encoder, so they are correlated and σ underestimates the
disagreement of $K$ fully independent networks. (ii) σ is a spread in Q units, not a
calibrated probability that $a^\star$ is wrong. Its scale differs between trained models,
which is why τ is a per-model percentile (§5). (iii) σ is only as good as the coverage of the
ensemble: a state that every head extrapolates in the same wrong way has low σ. Whether σ
tracks degraded inputs and TD error on this task is measured, not assumed
(`scripts/sigma_check.py`, D10).

**Ablation −ensemble.** $K = 1$, so $\sigma \equiv 0$ and there is no gate.

## 5. The uncertainty gate

At evaluation, for every ready intersection,

$$
a_i =
\begin{cases}
a^{\text{MP}}_i = \arg\max_{\phi} \sum_{m \in \phi} \big(\hat x_{\text{in}(m)} - \hat x_{\text{out}(m)}\big) & \text{if } \sigma(s_i) > \tau, \\[4pt]
a^\star(s_i) & \text{otherwise,}
\end{cases}
$$

where $a^{\text{MP}}_i$ is Max-Pressure on the *observed* lane counts $\hat x$, with the same rule,
$\Delta t$ and detectors as the Max-Pressure baseline (ties go to the lowest phase id).

* **Training is ungated.** Training is ε-greedy on μ, and checkpoints are selected on the
  ungated greedy policy. The −gate ablation is therefore the *same* checkpoint with the gate
  off, and τ can be swept without retraining.
* **Threshold.** $\tau_{p} = Q_p\big(\{\sigma(s_t)\}_t\big)$ is the $p$-th percentile of the
  model's σ over every decision of one clean, ungated validation episode, so by construction
  a share $1 - p/100$ of those decisions would fall back. Under the gate the trajectory changes,
  so the realised clean fallback rate is close to, but not exactly, $1 - p/100$.
* **Choice of p.** One $p$ for all seeds from the pre-registered grid $\{80, 90, 95, 99\}$ is
  chosen to minimise the mean validation travel time over the clean flow and four detector
  degradations (noise σ = 0.1 / 0.2, dropout 10 / 20 %); ties within 1 s go to the smaller
  $p$, i.e. more fallback. τ is per seed-model (`gate.json`). Clean validation alone would
  select "never fire", because the learned policy already beats Max-Pressure there. The gate
  exists for unreliable inputs.
* **Boundary cases (unit-tested):** $\tau = -1$ gives exactly the Max-Pressure controller, and
  $\tau = +\infty$ gives exactly −gate.

## 6. Safety argument

**Proposition (fallback limit).** Assume Varaiya's (2013) setting: point queues with
unbounded storage, known and fixed turning ratios, i.i.d. exogenous arrivals whose mean rate
lies strictly inside the network's capacity region, and Max-Pressure applied every slot to
true queue lengths. Then:

* (a) the Max-Pressure policy stabilises the network: the expected total queue is bounded
  uniformly in time (Varaiya, 2013, via a quadratic Lyapunov function
  $V(x) = \sum_l x_l^2$ with negative drift outside a bounded set);
* (b) if $\sigma(s_i) > \tau$ for every decision (for example, $\tau < 0$), the UG-MARL closed
  loop *is* Max-Pressure, so (a) applies to it unchanged.

**Sketch for the mixed policy (not a theorem).** Suppose the learned action replaces
Max-Pressure in a fraction $\rho$ of the slots. In each slot the Lyapunov drift of an
arbitrary phase differs from the Max-Pressure drift by at most a bounded amount $B$, because
the drift is linear in the per-slot service and the service is bounded. The drift of the
mixed policy is therefore at most $-\epsilon\,\lVert x \rVert_1 + C + \rho B'(x)$, where
$B'(x)$ grows at most linearly in $x$. The negative-drift argument survives when $\rho$ times
that growth stays below $\epsilon$, i.e. for demand far enough inside the capacity region
or a small enough share of learned decisions. As τ decreases, $\rho$ decreases and the
guarantee approaches (a). The gate does not choose *which* slots are learned to keep the
drift negative. It hands over the slots where the ensemble disagrees, which is a
heuristic proxy.

**What this does not guarantee.**

* Real lanes are finite: spill-back and gridlock lie outside Varaiya's model, and
  Jinan has finite lanes.
* Our Max-Pressure holds each phase for $\Delta t = 15$ s plus yellow and acts on observed,
  possibly noisy counts: a fixed-cycle-free but slotted variant.
* Demand in the data is not i.i.d. and not known to lie inside the capacity region.
* σ may be low on a state where the learned action is bad (§4 limit iii), and then the gate
  does not fire.
* Nothing is guaranteed for the travel-time objective. Stability is about bounded queues.

The report therefore never calls the mixed policy "provably stable". The claim is (a)–(b) plus
the empirical check of H2: the maximum per-vehicle waiting time and the degradation under
noise / dropout, compared against Max-Pressure and −gate.

## 7. Complexity

With observation size $o = 44$, embedding width $d$, $K$ heads of hidden width $H = 128$ and
$|\mathcal A| = 8$:

$$
\#\theta = \underbrace{o d + d + 5d}_{\text{embedding + slots}}
+ \underbrace{3(d^2 + d)}_{W_q, W_k, W_v}
+ K\big(z H + H + 8H + 8\big), \qquad z = 2d\ (\text{or } d \text{ for −attention}).
$$

For $d = 64$, $K = 5$: $15\,680 + 5 \times 17\,544 = 103\,400$ parameters (DQN: 23 304). Per
decision and intersection the forward cost is
$O(5 o d + 5 d^2 + K(zH + H|\mathcal A|))$ multiply-adds: the encoder runs once and only the
heads scale with $K$. The gate adds one Max-Pressure evaluation ($O(12)$) and a comparison.
Training cost per gradient step grows the same way, and the replay stores
$5 \times 45$ values per state ($\approx 180$ MB for 100 000 transitions).

## 8. Code map (checked 12 Nov 2026)

| Equation | Code |
|---|---|
| State, capacity scaling | `envs/features.py` `FeatureBuilder.observation` |
| Reward $-\lvert P_i\rvert$ | `envs/features.py` `intersection_pressure`, `envs/traffic_env.py` `_observe` |
| Double-DQN SMDP target, $c$, truncation | `agents/core/qlearner.py` `td_target` |
| Bootstrap masks, masked loss, $1/K$ | `agents/core/qlearner.py` `observe`, `masked_ensemble_loss` |
| Slots, presence flag | `agents/core/encoders.py` `NeighborhoodEncoder` |
| $e_j$, $\alpha_{ij}$, $c_i$, $z_i$, heads | `agents/ugmarl/network.py` `UGMARLNet` |
| μ, $a^\star$, σ | `agents/ugmarl/gate.py` `sigma_from_q` |
| Gate rule, Max-Pressure fallback | `agents/ugmarl/agent.py` `UGMARLAgent.act`, `agents/baselines/max_pressure.py` |
| $\tau_p$, choice of p, `gate.json` | `agents/ugmarl/gate.py`, `scripts/calibrate_gate.py` |
| Parameter count | `agents/ugmarl/network.py` `expected_params` (tested against `n_params`) |

## References

* Osband, I., Blundell, C., Pritzel, A., Van Roy, B. (2016). Deep exploration via bootstrapped
  DQN. *NeurIPS*.
* Pardo, F., Tavakoli, A., Levdik, V., Kormushev, P. (2018). Time limits in reinforcement
  learning. *ICML*.
* van Hasselt, H., Guez, A., Silver, D. (2016). Deep reinforcement learning with Double
  Q-learning. *AAAI*.
* Varaiya, P. (2013). Max pressure control of a network of signalized intersections.
  *Transportation Research Part C*, 36, 177–195.
* Wei, H., Xu, N., Zhang, H., Zheng, G., Zang, X., Chen, C., Zhang, W., Zhu, Y., Xu, K., Li, Z.
  (2019). CoLight: Learning network-level cooperation for traffic signal control. *CIKM*.
