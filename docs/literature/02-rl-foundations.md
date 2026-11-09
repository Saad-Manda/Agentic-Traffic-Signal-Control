# 02 — Reinforcement-learning foundations (single-agent → multi-agent)

Rubric R2 (conventional agentic frameworks: single-agent, multi-agent) · Guideline G2
Legend: ✓ verified online · † from memory, check before final report.
**Scope rule (9 Nov):** only papers from 2016 onward are reviewed. Q-learning (1992) and DQN (2015) are
cited as **background** for the baselines we implement (A1–A2) and are not reviewed.

**Why this file matters.** PS16 asks us to study *conventional agentic frameworks, single- and
multi-agent, and their limitations*. In our setting the "agents" are RL agents, so this file
supplies the theory behind our DQN (single-agent) and IDQN (multi-agent) baselines, and it
names the specific limitations UG-MARL targets: overestimation, non-stationarity, and no notion
of "I don't know".

---

## Part A — Single-agent value-based RL

### A1. Background: MDP and Q-learning † (`watkins1992qlearning`, 1992; cited, not reviewed)

MDP (S, A, P, r, γ). The action-value function of policy π is
$Q^\pi(s,a) = \mathbb{E}_\pi[\sum_t \gamma^t r_t \mid s_0=s, a_0=a]$, and the optimal one satisfies the
Bellman optimality equation

$$Q^\*(s,a) = \mathbb{E}\big[r + \gamma \max_{a'} Q^\*(s',a')\big].$$

Tabular Q-learning update (off-policy TD):

$$Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big].$$

It converges to Q\* with probability 1 if every (s,a) is visited infinitely often and step sizes satisfy the
Robbins–Monro conditions.

**What we learn.** Traffic states (per-lane queues × phase) are too large for a table, so we
need function approximation. The `max` operator is also the source of **overestimation bias**
(A3), which matters more when Q is noisy, as it is early in training and in unfamiliar traffic.

### A2. Background: DQN † (`mnih2015human`, 2015; cited, not reviewed)

Q-learning with a neural network Q_θ plus two stabilisers:
- **Experience replay**: sample i.i.d. minibatches from a buffer D to break temporal correlation.
- **Target network** θ⁻: a periodically copied, frozen version of the network.

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim D}\Big[\big(r + \gamma \max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a)\big)^2\Big]$$

(Huber loss is used in practice.) Exploration is ε-greedy with ε annealed.

**Limitation that matters to us.** A DQN outputs a point estimate with no confidence. On states far from its
training distribution (incidents, sensor noise, a new city) it still returns an argmax, often
confidently wrong. This is PS16's "unreliable decisions under uncertainty".

### A3. The DQN improvements we will actually use †

| Paper | Change | Equation / idea | Use in our code |
|---|---|---|---|
| Double DQN (`vanhasselt2016double`, AAAI 2016) | Decouple selection from evaluation | $y = r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s',a'))$ | **Yes**: default in all DQN-type agents (cheap; reduces overestimation) |
| Dueling (`wang2016dueling`, ICML 2016) | $Q = V(s) + A(s,a) - \frac{1}{\lvert A\rvert}\sum_{a'} A(s,a')$ | Learns V without visiting every action | Optional; helps when many phases are equally good (light traffic) |
| Prioritized replay (`schaul2016prioritized`, ICLR 2016) | Sample ∝ \|TD error\|^α, IS-correct with β | | Optional; interacts with bootstrapped masks, so keep it off for UG-MARL v1 |
| n-step returns | $y = \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n \max Q(s_{t+n},\cdot)$ | faster credit assignment | Optional |
| Rainbow (`hessel2018rainbow`, AAAI 2018) | All of the above + distributional + noisy nets | Ablation shows PER and n-step matter most | Reference only |

**Fairness rule.** Whatever we add to UG-MARL's Q-learner (Double, Dueling, …) must also be in
the DQN/IDQN baselines. Otherwise H1 conflates "uncertainty gate" with "better DQN".

---

## Part B — Multi-agent RL

### B1. Formal setting

A decentralised partially observable (Dec-)POMDP / Markov game with N agents (one per intersection).
Agent i sees o_i (its local lanes plus maybe its neighbours) and chooses a_i. Transitions depend
on the **joint** action. Traffic is a cooperative setting (shared goal: low travel time), but
with local rewards (local pressure or queue) in most TSC work.

### B2. Independent deep Q-learning (IDQN) † (`tampuu2017multiagent`, PLOS ONE 2017)

Each agent runs its own Q-learner and treats the others as part of the environment.

**Limitation 1: non-stationarity.** From agent i's view, P(s'|s,a_i) changes as neighbours'
policies change, so the Markov assumption breaks and convergence guarantees are lost.
**Limitation 2: stale replay.** Old transitions in the buffer reflect neighbours' old policies.
Foerster et al. (`foerster2017stabilising`, ICML 2017) †: importance-weighting or a
"fingerprint" (training iteration, ε) in the observation helps.

**Yet it works in practice.** Ault & Sharon (RESCO, NeurIPS 2021 D&B ✓, see [03](03-rl-for-traffic-signal-control.md))
found a well-tuned independent deep-Q agent beat previously published SOTA in many realistic
scenarios. Independent learners are a **strong** baseline, not a strawman.

**Implementation tip.** **Parameter sharing** (one network for all intersections, with the
intersection ID or its features as input) is standard in TSC (MPLight) and in MARL generally. It
multiplies the data per gradient step by N and reduces non-stationarity. Decide for IDQN
and UG-MARL together: both shared, or both not.

### B3. Centralised training, decentralised execution (CTDE) †

- **VDN** (`sunehag2018value`, AAMAS 2018): $Q_{tot} = \sum_i Q_i(o_i, a_i)$.
- **QMIX** (`rashid2018qmix`, ICML 2018): $Q_{tot} = f_{mix}(Q_1..Q_N; s)$ with
  $\partial Q_{tot}/\partial Q_i \ge 0$ (monotonic), so the per-agent argmax is consistent with the joint argmax.
- **MADDPG** (`lowe2017multi`, NeurIPS 2017): a centralised critic per agent sees all actions;
  decentralised actors.
- **MAPPO / IPPO** (`yu2022surprising` ✓ arXiv 2103.01955, NeurIPS 2022 D&B †): PPO with a
  centralised value function, or fully independent (IPPO). These are strong on SMAC and Hanabi with little
  tuning. IPPO is our optional baseline.

**Why we do not use CTDE as the core.** The mixers scale poorly to many agents with a global state,
assume a single shared team reward, and a central critic is unavailable at deployment.
Traffic networks are large, local and sparse, so **local coordination through neighbour messages** (B4)
is the better fit, which is also what CoLight showed. Say this explicitly in the report as the
reason for choosing attention over QMIX.

### B4. Learned communication and attention-based coordination

| Paper | Mechanism | Relevance |
|---|---|---|
| CommNet † (`sukhbaatar2016learning`, NeurIPS 2016) | Agents average hidden states over a continuous channel | Mean aggregation: neighbours weighted equally |
| GAT † (`velickovic2018graph`, ICLR 2018) | $e_{ij} = \text{LeakyReLU}(a^\top[Wh_i \Vert Wh_j])$, $\alpha_{ij} = \text{softmax}_j(e_{ij})$, $h_i' = \sigma(\sum_j \alpha_{ij} W h_j)$ | The operator CoLight and our encoder use |
| Transformer attention † (`vaswani2017attention`, NeurIPS 2017) | $\text{softmax}(QK^\top/\sqrt{d})V$, multi-head | Dot-product variant (CoLight uses this form) |
| DGN ✓ (`jiang2020graph`, ICLR 2020) | Multi-head attention "graph convolution" over neighbours + DQN, shared weights, temporal relation regularisation | **Closest generic-MARL template to UG-MARL's encoder** (attention + DQN + parameter sharing) |
| TarMAC † (`das2019tarmac`, ICML 2019) | Targeted (attention-weighted) messages | Shows attention as "who to listen to" |

**What we learn.**
- Attention gives **index-free**, variable-degree aggregation. Edge intersections have fewer
  neighbours, and softmax over N(i) ∪ {i} handles that without padding hacks (CoLight's key
  argument).
- Attention weights are a *by-product* explanation, but see [05](05-llm-agents-and-explainability.md):
  attention is not a faithful explanation by default.
- **Implementation tips:** include the self-loop. Use 1–2 heads (more adds compute for little
  gain at 12–16 intersections). Mask non-neighbours instead of hard-coding neighbour order.
  Neighbours' raw observations (not their Q-values) are the safest message, because they are stationary
  inputs.

---

## Limitations of conventional agentic (RL) frameworks → what UG-MARL targets

| PS16 limitation | Where it comes from (this file) | UG-MARL response |
|---|---|---|
| Unreliable decisions under uncertainty | DQN gives point estimates; no epistemic signal (A2) | Q-ensemble σ(s) |
| No safety guarantee / error propagation | Function approximation + max-bias (A1, A3); independent learners can diverge (B2) | Gate → Max-Pressure fallback (see 01 §2 caveats) |
| Weak coordination | IQL treats neighbours as noise (B2); CTDE does not scale (B3) | Neighbour attention (B4) |
| Poor explainability | Black-box Q-net | Attention weights + gate statistics (associational) |
| Computational cost | Centralised critics/mixers | Decentralised, shared parameters; K heads on a shared torso |
