# 03 — Reinforcement learning for traffic signal control (TSC)

ROADMAP 2.2 (single-agent) and 2.3 (multi-agent) · Rubric R2 · Guideline G2
Legend: ✓ verified online · † from memory, check before final report · *(unverified)* number
not checked against the PDF.
**Scope rule (9 Nov):** only papers from 2016 onward are reviewed. Pre-2016 RL-TSC (Wiering 2000,
Abdulhai 2003, MARLIN 2013) is covered through the Wei et al. (2019) survey rather than cited directly.

**Why this file matters.** These are the methods we benchmark against (DQN, IDQN) and the lineage
UG-MARL extends (PressLight → CoLight → MPLight, all on CityFlow with Jinan/Hangzhou). We
also take most design defaults (state, reward, Δt, metric) from them, and their known
failure modes justify our robustness tests.

---

## 0. Surveys and reading order

- Wei, Zheng, Gayah, Li, "A survey on traffic signal control methods", arXiv 1904.08117, 2019 ✓
  (`wei2019survey`). Covers conventional control (Webster, SCOOT/SCATS, SOTL, Max-Pressure) and
  early tabular RL, which we cite through this survey. Covers both the transportation and RL views, explains why RL ≠ just
  "a better optimiser", and has a table of RL state/reward/action choices.
- Wei, Zheng, Gayah, Li, "Recent advances in RL for TSC: a survey of models and evaluation",
  *ACM SIGKDD Explorations* 22(2):12–18, 2021 ✓ (`wei2021recent`).
- Reading order for the build: Genders → IntelliLight → FRAP → PressLight → CoLight → MPLight →
  Advanced-XLight → RESCO/LibSignal.

## 1. The common formulation (and our choices)

| Element | Common options in the literature | Our default (architecture.md) | Why |
|---|---|---|---|
| State | queue per lane, vehicle count per lane (optionally per segment), current phase, pressure, image of positions | queue + pressure + one-hot phase | Same inputs as Max-Pressure, so the gate's two branches see the same information; low-dimensional |
| Action | choose next phase (acyclic) / switch-or-keep (cyclic) / set duration | choose next phase | Standard in the CityFlow lineage; matches MP's action space |
| Reward | −Σ queue, −pressure, −delay, −wait, weighted mixes | −pressure (alt. −queue) | Pressure reward links RL to MP's stability argument (PressLight, `ma2022lyapunov`) |
| Δt | 5–20 s, plus yellow (3–5 s) | TBD (5–10 s) | Smaller Δt gives more decisions but more switching lost time |
| Metric | average travel time (ATT), throughput, queue, delay | ATT primary + max wait + throughput | ATT is the field's headline metric, so results are comparable |

**Implementation tip (metrics).** CityFlow's `get_average_travel_time()` includes vehicles still
in the network at episode end, counted with their travel time so far. Report it as is (the
lineage does). Also report throughput, so a controller cannot look good by keeping vehicles
*out*. Check this claim against the CityFlow source when writing `utils/metrics.py`.

**Simulator.** CityFlow † (`zhang2019cityflow`, WWW 2019 demo): a multi-agent RL environment
reported to be much faster than SUMO, which is why we chose it (Decision Log, 27 Oct).

---

## 2. Early RL-TSC (history, via survey)

Tabular and early multi-agent RL for signals (before deep RL) is summarised in `wei2019survey`.
The one lesson we carry forward, that neighbour coordination beats independent learners,
is supported by post-2015 evidence instead: MA2C (§4.1) and CoLight (§4.3).

---

## 3. Single-agent deep RL (ROADMAP 2.2)

### 3.1 DQN-TSC: Genders & Razavi (2016) ✓
`genders2016using` — arXiv 1611.01142.

- **Method.** Isolated intersection in SUMO. The state is a **discrete traffic state encoding (DTSE)**, a
  grid of cell occupancy and speed per lane, processed by a CNN. DQN with replay and a target net.
- **Result.** Compared with a one-hidden-layer NN agent: −82% cumulative delay, −66% queue,
  −20% travel time (from the abstract).
- **Limitations → our gap.** A single intersection with no neighbours. The baseline is weak (shallow NN, not MP or
  actuated control). The image-like state is expensive and simulator-specific.
- **Lesson.** Rich states help a lone agent, but later work (Advanced-XLight) shows that a *good
  compact* state matters more than a rich one.

### 3.2 IntelliLight ✓
`wei2018intellilight` — Wei, Zheng, Yao, Li, KDD 2018.

- **Method.** DQN. State = queue length, number of vehicles, waiting time per lane, an image of vehicle
  positions, current phase, next phase. Reward = weighted sum of queue, delay, waiting time,
  switch penalty, vehicles passed, travel time. Two architectural ideas:
  - **"Phase gate"**: a separate output branch per current phase, so the network does not
    average over very different regimes. (Name clash: this is *not* an uncertainty gate. Say so
    in the report to avoid confusion.)
  - **"Memory palace"**: separate replay memories per (phase, action) so rare combinations are
    not drowned out, which is a form of class balancing for replay.
- **Data.** A large real camera dataset from Jinan. Evaluated on SUMO. Includes case studies
  interpreting the learned policy (an early explainability attempt).
- **Limitations → our gap.** Single intersection; a multi-term reward with hand-tuned weights
  (hard to justify, and the weights change the optimum). No uncertainty, no safety.
- **Implementation tip.** If one phase dominates the replay buffer (likely under heavy single-
  direction demand), the memory-palace idea (stratified replay) is a cheap fix to try in tuning.

### 3.3 FRAP ✓
`zheng2019learning` — Zheng et al., "Learning phase competition for TSC", CIKM 2019,
DOI 10.1145/3357384.3357900, arXiv 1905.04722.

- **Method.** Embeds each movement's demand, builds phase representations, and scores **pairwise
  phase competition** (which phase should win given the conflicts). The design is invariant to
  flipping and rotating the intersection.
- **Result (from the abstract).** Better than prior RL on all-phase selection, faster convergence,
  and better generalisation across intersection structures and flows.
- **Lesson.** Building symmetry into the architecture gives sample efficiency. MPLight builds on FRAP.

---

## 4. Multi-agent deep RL (ROADMAP 2.3)

### 4.1 MA2C: Chu et al. ✓
`chu2020multi` — IEEE T-ITS 21(3):1086–1095, 2020 (arXiv 1903.04527).

- **Method.** Independent advantage actor-critic (IA2C) with two stabilisers:
  (i) **neighbour fingerprints**: neighbours' latest policies are included in the observation
  (addresses non-stationarity, [02](02-rl-foundations.md) B2), and (ii) a **spatial discount
  factor** α that down-weights distant agents' rewards and states.
- **Setup.** SUMO, a synthetic 5×5 grid and a 30-intersection Monaco network.
- **Lesson.** Non-stationarity is a practical problem for independent learners. Spatial
  discounting is a cheap alternative to attention. It is a possible extra ablation baseline if
  reviewers ask "why attention and not a fixed discount?".

### 4.2 PressLight ✓
`wei2019presslight` — Wei, Chen, Zheng, Wu, Xu, Gayah, Li, KDD 2019, pp. 1290–1298,
DOI 10.1145/3292500.3330949. Code: github.com/wingsweihua/presslight.

- **Method.** DQN per intersection. **Reward = −pressure**, where intersection pressure is
  $P_i = \sum_{(l,m)} (x_l - x_m)$ (incoming minus outgoing vehicle counts per movement,
  normalised by lane capacity). The state includes the current phase and vehicle counts on incoming lanes (split
  into segments) and on outgoing lanes.
- **Key argument.** Optimising −pressure ties the RL objective to MP's throughput-optimality
  argument ([01](01-conventional-control.md) §0, §1.2), so in the authors' framing the RL agent learns to
  "do MP better", including multi-step lookahead MP lacks.
- **Results.** Arterial and grid networks (synthetic + real, including Jinan). The authors report
  emergent **green-wave**-like coordination without explicit communication, and wins over MP and earlier RL
  *(numbers unverified)*.
- **Limitations → our gap.** Coordination is only implicit (through the reward). No uncertainty: when
  the learned policy is worse than MP in some state, nothing detects it.
- **Why it matters for UG-MARL.** It is our reward choice and the theoretical bridge to the
  fallback: both branches of the gate optimise or approximate the same pressure objective.

### 4.3 CoLight ✓
`wei2019colight` — Wei et al., CIKM 2019, arXiv 1905.05717.

- **Method.** Per-intersection observation (current phase + vehicles per lane) → MLP embedding
  → **multi-head dot-product graph attention** over neighbours *including itself* → Q-values.
  The parameters are shared across all intersections. Trained DQN-style. The reward is −queue length.
- **Key arguments.** (i) *Index-free*: neighbour influence is a learned, state-dependent weighted
  average, so order and count do not matter. (ii) Attention weights vary over time (for example by
  rush-hour direction), which fixed offsets or spatial discounts cannot capture.
- **Data.** **Jinan (3×4 = 12 intersections), Hangzhou (4×4 = 16), New York (16×3 = 48 and
  28×7 = 196)**. These released datasets are the ones our project plans to use (ROADMAP Phase 1).
- **Results.** Best ATT across these datasets vs. Fixed-Time, MaxPressure, IntelliLight,
  PressLight, GCN-based baselines and others *(numbers unverified; copy from Table in PDF when writing the report)*.
- **Limitations → our gap.** No uncertainty estimate, no fallback, no robustness testing under
  sensor noise. Attention weights are presented as interpretation, but they are not validated as faithful.
- **Why it matters.** **UG-MARL's encoder is essentially CoLight's.** Our contribution is what
  sits *after* the encoder (Q-ensemble + gate). The "−gate −ensemble" ablation of UG-MARL is
  therefore close to "CoLight with a pressure reward". Say this openly.

### 4.4 MPLight ✓
`chen2020toward` — Chen, Wei, et al., "Toward a thousand lights", AAAI 2020, 34(04):3414–3421,
DOI 10.1609/aaai.v34i04.5744.

- **Method.** FRAP architecture + pressure in state and reward + **one shared network for all
  intersections**. No explicit communication: coordination comes from the pressure reward.
- **Setup.** Manhattan network with **2510 signals**, the first RL-TSC at a thousand-light scale.
- **Lesson.** Parameter sharing + pressure reward scales. It is our template for shared-parameter
  IDQN, which is then a fair "no-coordination" counterpart to UG-MARL.

### 4.5 AttendLight ✓
`oroojlooy2020attendlight` — NeurIPS 2020, arXiv 2010.05772.

- **Method.** Two attention modules (over lanes/roads, and over phases), so one model handles
  intersections of **any geometry**. Trained across many intersections.
- **Lesson.** Attention is useful for *generalisation across structures*, which is relevant to our
  held-out city test (F9). Different geometry is the main risk when moving from Jinan to Hangzhou.

### 4.6 Advanced-XLight / Efficient-XLight ✓
`zhang2022expression` — Zhang, Wu, Shen, Lü, Du, Wu, "Expression might be enough", ICML 2022,
PMLR 162:26645–26654, arXiv 2112.10107. Code: LiangZhang1996/Advanced_XLight.

- **Method.** Proposes **Advanced-MP** (MP using queued and effective running vehicles) and an
  advanced state (efficient pressure + effective running vehicles). It plugs these into MPLight and CoLight
  ("Advanced-MPLight", "Advanced-CoLight").
- **Lesson (important).** **State representation dominated architecture choice**: a strong
  state + simple model ≈ SOTA, and Advanced-MP (no learning) is competitive with many RL methods.
  Implications: (i) Advanced-MP is a candidate conventional baseline, and (ii) attention gains (H3) may be
  small once the state is good, so the −attention ablation must use the *same* state.

---

## 5. Ensembles and uncertainty already used in RL-TSC (closest prior art)

### 5.1 Multi-agent bootstrapped DQN for TSC ✓
`tan2020multiagent` — Tan, Chu, et al., IEEE CCTA 2020, DOI 10.1109/CCTA41146.2020.9206275.

- **Method.** Bootstrapped DQN (K heads, [04](04-uncertainty-and-safe-rl.md) §2) for **exploration**;
  multi-agent with conditional parameter sharing. Tested on an asymmetric isolated intersection and a 5×5
  synthetic grid.
- **Result (from the abstract).** Better efficiency and robustness than vanilla DQN.
- **What it does *not* do:** use the head disagreement at **execution** time, or fall back to a
  controller with a guarantee. **Our difference:** σ(s) is used at decision time as a reliability
  signal that triggers MP.

### 5.2 RELight (random-ensemble Double DQN) ✓
`qi2022random` — Qi, Huang, Li, Tan, Huang, Cui, arXiv 2203.05961, 2022 (preprint).

- **Method.** An ensemble of Double-DQN learners with random subsets, for more stable and
  sample-efficient learning.
- **Our difference.** As above: the ensemble is used for learning quality, not as a gate.

---

## 6. Safety-oriented RL-TSC

- **SafeLight** ✓ (`du2023safelight`, AAAI 2023, Safe & Robust AI track, 37(12):14801–14810,
  arXiv 2211.10871). "Safety" here means **collision avoidance** (conflicting greens with
  non-compliant vehicles) through residual RL, a safety model in the state, action, reward and loss, and reward
  shaping. **Different notion of safety from ours** (ours = bounded queues / worst-case wait
  under uncertainty). Cite it to show we know the term is overloaded.
- **Com-MP** ✓ (`zhu2026commp`, arXiv 2610.31037, Oct 2026). RL selects a context-dependent
  mixture of MP formulations while MP chooses the phase, and the authors claim strong stability. The approach is
  "learning *inside* MP". Ours is "learning *with an MP fallback when unsure*". Both aim to keep
  MP's guarantee. Com-MP keeps it by construction and gives up some flexibility. UG-MARL keeps
  RL's flexibility and needs the extra queue trigger ([01](01-conventional-control.md) §2)
  to claim stability.

---

## 7. Benchmarks, robustness and generalisation

- **RESCO** ✓ (`ault2021reinforcement`, NeurIPS 2021 Datasets & Benchmarks). SUMO benchmark
  with realistic Cologne/Ingolstadt (and later Salt Lake City) scenarios. Key finding: **earlier
  methods are not robust to sensing assumptions and non-stylised layouts. A tuned
  independent deep-Q agent often beats them.** This supports (i) a strong IDQN baseline and (ii) our
  sensor-noise test (F8).
- **LibSignal** ✓ (`mei2024libsignal`, *Machine Learning* 113(8), 2024; arXiv 2211.10649).
  A cross-simulator library (CityFlow + SUMO) with implementations of FixedTime, MP, SOTL,
  IDQN, PressLight, CoLight, MPLight, etc. **Implementation tip:** use its configs as a reference
  for Δt, yellow time, state/reward definitions and dataset loading. Do not copy code
  without attribution.
- **T-REX / robustness under incidents** ✓ (`nguyen2025robustness`, arXiv 2506.13836, 2025).
  SUMO incident framework. Reports that IDQN improves travel time by **5.62%** over MP in normal conditions
  but degrades by **57.41%** (incident at test only) and **23.2%** (incident at train and test) on a 4×4 grid.
  The authors attribute this to distribution shift (epistemic uncertainty). This is the **most direct external
  evidence for UG-MARL's motivation**: RL beats MP on average but fails badly off-distribution.
  This is exactly where an uncertainty gate should hand over to MP.
- **Generalisation.** MetaLight † (`zang2020metalight`, AAAI 2020) uses meta-learning across
  intersections. AttendLight (above) is the other reference. Our F9 test (train Jinan → test Hangzhou) is standard
  in this lineage.

---

## Takeaways for UG-MARL

1. Stay in the CityFlow/CoLight lineage: same datasets, ATT metric, state and reward family, so
   numbers are comparable.
2. Use a pressure reward. It links the RL objective to the MP fallback's guarantee.
3. The baselines must be strong: shared-parameter IDQN (RESCO), Double DQN in all learners, and
   possibly Advanced-MP.
4. Ensembles already exist in RL-TSC, but only for **exploration and stability during training**. Using
   ensemble disagreement **at execution time to trigger a guaranteed fallback** is the gap
   ([06](06-gap-analysis.md)).
5. The T-REX result (RL beats MP on average and collapses under shift) is our best external
   motivation and suggests the robustness experiment (F8): sensor noise and demand surge.
