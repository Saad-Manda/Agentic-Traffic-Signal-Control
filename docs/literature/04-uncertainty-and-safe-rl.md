# 04 — Uncertainty estimation and safe RL

ROADMAP 2.4 · Rubric R2 · Guideline G2
Legend: ✓ verified online · † from memory, check before final report.
**Scope rule (9 Nov):** only papers from 2016 onward are reviewed. Simplex (Sha 2001) is cited as
**background** for the architecture pattern and is not reviewed.

**Why this file matters.** Two of UG-MARL's three components come from this literature: the Q-ensemble
(σ(s)) and the gate/fallback. This file explains what kind of uncertainty an ensemble measures, which
ensemble design gives a *useful* σ, how others chose thresholds, and which safety framework
our gate belongs to, so the design choices in architecture.md can be justified.

---

## 1. Which uncertainty do we want?

Kendall & Gal † (`kendall2017uncertainties`, NeurIPS 2017):
- **Aleatoric** uncertainty is noise inherent in the data (random arrivals, driver behaviour). It cannot be reduced
  with more data.
- **Epistemic** uncertainty is uncertainty in the model from limited data. It is high in rarely seen states
  and is reducible.

For an ensemble of regressors with means μ_k and variances σ_k², the predictive variance splits as

$$\mathrm{Var}[y] \approx \underbrace{\tfrac{1}{K}\sum_k \sigma_k^2}_{\text{aleatoric}} + \underbrace{\tfrac{1}{K}\sum_k \mu_k^2 - \bar\mu^2}_{\text{epistemic}}.$$

**Applied to Q-learning.** Each Q-head estimates an *expected* return, so the randomness of
traffic is already averaged inside each head. **The spread across heads,
σ(s,a) = std_k Q_k(s,a), is therefore (approximately) epistemic.** This is the signal we want: "the
agent has not learned this kind of state", for example an incident, a surge, sensor noise, or a new city.
It is *not* "this traffic is random". Write this in the report, because it answers a likely
viva question ("isn't high variance just busy traffic?").

---

## 2. Ensemble methods for deep RL

### 2.1 Deep ensembles † (`lakshminarayanan2017simple`, NeurIPS 2017)
K independently initialised networks trained on the same data (no bootstrap needed). Random
init + SGD noise give enough diversity. Simple, parallel, and well calibrated. Five members is often enough.

### 2.2 Bootstrapped DQN ✓ (`osband2016deep`, NeurIPS 2016, arXiv 1602.04621)
- K Q-heads on a **shared torso**. Each transition gets a mask m ∈ {0,1}^K, m_k ~ Bernoulli(p).
  Head k trains only on transitions with m_k = 1, using its own target head:
  $y_k = r + \gamma\, Q_k^{-}(s', \arg\max_{a'} Q_k(s',a'))$.
- For exploration, sample one head per episode and act greedily on it (Thompson-sampling-like
  *deep* exploration).
- Gradients flowing into the shared torso are normalised by 1/K.
- The authors show the heads' disagreement tracks "what the agent doesn't know".
- **Implementation tips.** K = 10 in the paper. p = 0.5–1.0, and p = 1 (no bootstrap, diversity only from init) worked
  nearly as well. The shared torso makes it about as cheap as one DQN, which matters for our
  compute budget. **Our usage differs.** We act on the *mean* (ε-greedy) during training and use the
  *spread* at execution. Head sampling for exploration is an optional extra.

### 2.3 Randomized prior functions † (`osband2018randomized`, NeurIPS 2018)
$Q_k(s,a) = f_{\theta_k}(s,a) + \beta\, p_k(s,a)$, where p_k is a **fixed, untrained**, randomly
initialised network.
- **Why it matters for us.** Without priors, heads trained on the same data can **agree on
  states they have never seen**, because all extrapolate the same way. That makes σ small exactly where we need
  it large. A fixed random prior keeps the heads different away from the data, so σ grows
  off-distribution. This is the single most important design choice for a gate that must fire
  on shifted traffic.
- **Tip.** β is a scale hyperparameter (β ≈ 1–10 relative to the Q scale, after reward
  normalisation). Treat it as a tuned hyperparameter on validation.

### 2.4 Other ensemble-RL ideas (know them, probably do not implement)
| Paper | Idea | Relevance |
|---|---|---|
| Averaged-DQN † (`anschel2017averaged`, ICML 2017) | Average the last K target nets → lower variance and overestimation | Another use of ensembles: variance reduction, not uncertainty |
| REDQ † (`chen2021randomized`, ICLR 2021) | Large ensemble; target uses min over a random subset → controls overestimation; high update-to-data ratio | If learning is unstable, min-over-subset targets are a known fix |
| SUNRISE ✓ (`lee2021sunrise`, ICML 2021, PMLR 139:6131–6141) | (i) Weighted Bellman backup that down-weights targets with high ensemble σ; (ii) UCB action selection $\arg\max_a \bar Q + \lambda \sigma$ | Shows σ can also be used *inside training* (down-weight uncertain targets). This is a possible extension, kept out of scope |
| MC Dropout † (`gal2016dropout`, ICML 2016) | Dropout at test time ≈ approximate Bayesian inference | Cheaper but usually worse-calibrated than ensembles (Ovadia ✓). This is the rejected alternative to record in the Decision Log |

---

## 3. Does ensemble uncertainty survive distribution shift?

**Ovadia et al. ✓** (`ovadia2019can`, NeurIPS 2019, arXiv 1906.02530). A large benchmark of
uncertainty methods under dataset shift (classification).
- **Deep ensembles were the most robust method, and K ≈ 5 was enough.**
- But **every method degrades under shift: models become *confidently wrong***, and
  post-hoc calibration on in-distribution validation data does not transfer to shifted data.

**Implications for UG-MARL.**
1. Ensembles are the right choice (Decision Log entry: ensemble over MC dropout, citing Ovadia).
2. **τ chosen on validation may be too permissive on the held-out city or under noise.**
   Report gate-firing rates per condition. A gate that rarely fires under shift is a finding,
   not something to hide.
3. **Evaluate σ directly, not only the end-to-end ATT.** Proposed diagnostic: for logged states,
   label "RL worse than MP" by comparing short rollouts or Q(s, a_RL) vs Q(s, a_MP), then report the
   AUROC of σ for predicting it. This tests the core assumption ("σ is high where RL is
   unreliable") separately from everything else. It is cheap and makes a strong viva point.

---

## 4. Thresholding uncertainty: selective prediction

The gate is a **reject option**: abstain from the RL action when unsure and defer to MP.
- Selective classification † (`geifman2017selective`, NeurIPS 2017): choose a threshold to
  meet a target risk at maximum coverage, and report **risk–coverage curves**.
- **Our analogue:** sweep τ on validation and plot ATT (and max wait) against the **fraction of
  decisions made by RL** (coverage). Endpoints: coverage 0 = pure MP, coverage 1 = pure RL
  (the −gate ablation). The curve shows whether there is a τ where UG-MARL beats both endpoints.
  That is the clearest single figure for the whole idea. Pick τ on validation (for example
  the percentile that minimises validation ATT) and freeze it before testing.
- **Scale tip.** std of Q-values depends on the reward scale and differs per intersection.
  Options: (i) normalise rewards, (ii) define τ as a per-agent percentile of validation σ, or
  (iii) use a scale-free signal such as **head vote disagreement**
  $1 - \max_a \frac{1}{K}\sum_k \mathbb{1}[\arg\max Q_k = a]$. Pick one, and keep the others as a
  sensitivity check.

---

## 5. Safe RL: where the gate fits

### 5.1 Survey ✓
Gu et al. (`gu2024review`, IEEE TPAMI 46(12):11216–11235, 2024; arXiv 2205.10330). An up-to-date
review covering methods, theory and the "2H3W" deployment questions. It separates safety through the
**optimisation criterion** (constrained / risk-sensitive objectives) from safety through
**changing the execution process** (shields, fallback controllers, external knowledge).
**UG-MARL is the second kind**: external knowledge (MP) is used when the agent is unsure.

### 5.2 Constrained MDPs (rejected alternative)
Maximise return subject to $\mathbb{E}[\sum \gamma^t c_t] \le d$, solved with Lagrangian or
trust-region methods (surveyed in `gu2024review`). **Why not for us:** the constraints hold in expectation
and only after convergence, with no per-state fallback, which is what PS16's "self-correction" asks for.

### 5.3 Shielding ✓ (`alshiekh2018safe`, AAAI 2018, arXiv 1708.08611)
A shield, synthesised from a formal (temporal-logic) spec and a model, watches the agent:
- **Pre-shielding** restricts the action set before the agent chooses.
- **Post-shielding** overrides the agent's action only if it violates the spec.

The shield stays on after training, because the learned policy is not guaranteed safe.
**UG-MARL ≈ a post-shield with a statistical trigger (σ > τ) instead of a formal spec.** We
gain no formal guarantee from the trigger itself. Any guarantee must come from the fallback
(MP) plus a state-based trigger ([01](01-conventional-control.md) §2).

### 5.4 Background: Simplex architecture ✓ (`sha2001using`, 2001; cited, not reviewed)
Two controllers, a **high-performance** one (here RL) and a **high-assurance** one (here MP),
plus a **decision module** that switches to the high-assurance controller when the system nears the edge of
the region the safe controller can recover from. **UG-MARL is a Simplex instance.** This is the cleanest
way to present it in the report and viva. Classic Simplex switches on *state* (recoverability).
UG-MARL switches on *epistemic uncertainty*. Combining both (σ > τ **or** V(x) > V_max) gives
the Simplex-style guarantee **and** the uncertainty-awareness.

### 5.5 Uncertainty-triggered fallback in other domains (design precedents)
| Work | Domain | Uncertainty | Trigger → fallback |
|---|---|---|---|
| Kahn et al. † (`kahn2017uncertainty`, arXiv 1702.01182) | Robot collision avoidance | Bootstrap + dropout ensemble | High uncertainty → slower, cautious actions |
| Lütjens, Everett, How ✓ (`lutjens2019safe`, ICRA 2019, pp. 8662–8668) | Pedestrian navigation | MC dropout + bootstrapping | Uncertainty-aware cost → more cautious behaviour on novel observations |
| Hoel, Wolff, Laine † (`hoel2020tactical`, IEEE IV 2020, arXiv 2004.10439) | Autonomous-driving tactical decisions | Ensemble with randomized priors (2.3) | σ above threshold → **fallback safe action**; σ also flags out-of-distribution situations |
| Ensemble Quantile Networks ✓ title (`hoel2023ensemble`, arXiv 2105.10266) | Autonomous driving | Ensemble (epistemic) + quantiles (aleatoric) | Epistemic threshold. Without it, collisions rise sharply when conditions shift |

**What we learn.** The *pattern* (ensemble → threshold → fallback) is established in robotics
and driving. **Applying it to multi-agent traffic signal control with a fallback that has a queue-stability
theorem is not** ([06](06-gap-analysis.md)). We must cite these precedents and claim
the transfer, the multi-agent setting and the evaluation, not the pattern itself.

---

## 6. Design questions this literature forces us to answer (to log in the Decision Log)

1. **Ensemble type:** shared torso + K heads (Bootstrapped DQN) vs K full networks (deep ensemble).
   The shared torso is cheaper, but the heads can be less diverse. *Recommendation:* shared torso + randomized
   priors, K = 5–10. Check diversity (σ on train vs a shifted validation set) early.
2. **Bootstrap masks:** p = 1 (init diversity only) vs p = 0.5. Tune on validation.
3. **σ definition:** std at the mean-greedy action vs vote disagreement (§4).
4. **Gate during training:** (a) off during training, on at evaluation (simplest; matches
   Algorithm 1 draft), or (b) on during training, where MP actions enter replay as off-policy data,
   which Q-learning can learn from (like demonstrations). (a) is cleaner for ablations. (b) may
   speed learning. Choose (a) for v1 and mention (b) as future work.
5. **Trigger:** σ only vs σ **or** queue bound. The latter is needed for any stability claim.
6. **τ selection:** a per-agent percentile on validation flows, frozen before test (protocol §3.5).
