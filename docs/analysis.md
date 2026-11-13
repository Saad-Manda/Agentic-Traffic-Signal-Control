# Analysis of the final results — why UG-MARL wins and loses where it does (ROADMAP 8.3)

Written 13 Nov 2026 from the Phase 7 results and the Phase 8 analysis. It keeps four things
apart: what was **planned**, what was **implemented**, what the **evidence shows**, and what
**remains uncertain**. Every number has a source file; nothing here was simulated for this
document.

| Source | Content |
|---|---|
| `docs/results/test_*.md`, `heldout_generalisation.md` | Phase 7 tables and hypothesis verdicts (test flow run once) |
| `notebooks/02_analysis.ipynb` | figures, error analysis, short readings under each figure |
| `docs/results/error_analysis.md` (+ 13 CSVs) | breakdown tables |
| `docs/results/tau_sensitivity.md` | τ sweep on validation (75 new validation episodes) |
| `docs/results/attention_faithfulness.md` | attention faithfulness check on validation |
| `docs/figures/results/` | 18 figures, each with a caption file |

Conventions. Jinan test flow, 3600 s, seeds 0–4, mean with Student-t 95 % CI. A difference is
called **clear** when the CI excludes 0 (or two CIs do not overlap), otherwise **level**.
Attention weights, σ and gate statistics are **associations, not causes**. Nothing was
selected or changed because of any finding below: the locked models, the test tables and the
verdicts stay as they are (Decision Log 13 Nov).

## 1. Planned, implemented, measured

| Planned (PRD §1b–1c) | Implemented | Evidence |
|---|---|---|
| H1: lower travel time than IDQN and Max-Pressure | as planned | **Supported.** 295.4 ± 1.1 s vs 307.9 ± 1.4 s and 320.8 s (`test_main.md`) |
| H2: the gate lowers the worst case | gate at evaluation only; three-part reading fixed before the runs | **Supported by the rule**, with a caveat: under noise 0.2 the maximum wait is 1493 vs 3306 s and the travel-time increase +125 vs +575 s (clear); on clean data 896 vs 910 s is inside seed noise — "no difference" |
| H3: attention lowers travel time | one attention layer over ≤ 4 neighbours | **Not supported.** −attention 294.2 ± 1.1 s vs full 295.4 ± 1.1 s; negative result |
| Minimum outcome: max wait ≤ Max-Pressure's | — | **Met** in all six conditions |
| Stability guarantee for the gated policy | not built: rigorous only if the gate always fires (`methodology.md` §6) | no claim made |
| Queue trigger, SHAP | dropped (Decision Log, gap table A2) | — |

Not part of any hypothesis, and reported as measured: UG-MARL is **3.2 s behind DQN** on clean
data (CIs separated) and **level with Max-Pressure** under detector noise.

## 2. Where UG-MARL wins

**Against Max-Pressure on clean data — everywhere.** Waiting-vehicle-seconds are 17–40 % lower
at all 12 intersections (all clear), travel time is 23–32 s lower in each of the first five
10-minute windows, and the gain grows with route length (−12 s for trips through one
intersection, −37 s through four or more). The learned policy is simply a better controller
than the pressure rule on the traffic it was trained for (`error_analysis.md`).

**Against every ungated learned policy under detector noise.** At noise 0.2 DQN, IDQN, −gate and
−ensemble reach 759–870 s; UG-MARL stays at 420.5 s. The breakdown adds something the metric
table hides: the ungated policies leave **584–795 of 5494 vehicles (11–14 %) outside the
network for the whole episode**, with blocks of 39–48 minutes at the entry roads, while
UG-MARL and Max-Pressure block none (`error_analysis_entry.csv`). The waiting-time metrics —
including the maximum per-vehicle wait used in H2 — count waiting inside the network only, so
the ≈ 3300 s reported for the collapsed models understates their worst case. The gate's safety
benefit under noise is larger than the H2 numbers show. (Travel time is not affected: it
counts every scheduled vehicle.)

**In the worst case over conditions.** Regret against the best of the five agents: UG-MARL at
most 6.8 s (mean 2.9 s), Max-Pressure 62 s, IDQN 345 s, DQN 415 s. No baseline is good in all
conditions — DQN is best in four and worst under noise, Max-Pressure the reverse. UG-MARL is
never the best agent and never far from it. That, not "best", is the supported claim.

## 3. Where UG-MARL loses, and why

### 3.1 Behind DQN on clean data (3.2 s)

*Where.* Nowhere in particular. UG-MARL has more waiting than DQN at every one of the 12
intersections (+0 to +12 %; clear at 4, largest in the middle of the network), in every
10-minute window (+2.5 to +4.5 s, clear in five of six) and for every route length (+1.3 to
+1.6 s through one to three intersections, +5.6 s through four or more). The deficit grows
with route length and is carried by turning trips (+3.9 s, clear; straight trips level) — the
pattern of a small loss at each intersection adding up, not of one failing location or period.

*Why — what the evidence rules out.*
- **Not the gate.** Gate on vs off on the same checkpoints: level at 10 of 12 intersections, in
  all six windows and for three of four route lengths (295.4 vs 295.0 s overall).
- **Not attention alone, not the ensemble alone.** All four UG-MARL variants are behind DQN:
  −attention 294.2 s, −gate 295.0 s, full 295.4 s, −ensemble 297.3 s (DQN 292.2 s).

*Why — what remains.* The deficit is common to the UG-MARL network family: the slot-embedding
encoder instead of DQN's plain MLP, and hyperparameters (learning rate, γ, batch size,
ε-schedule) that were tuned for DQN and inherited, not retuned (Phase 6 decision). Only K and
the attention width were searched for UG-MARL. The runs available cannot separate architecture
from tuning; that is an open point, not a finding.

*Size.* 3.2 s is 1.1 % of the travel time and of the same order as UG-MARL's own range over
seeds (2.8 s); the CIs are nevertheless separated, so it is reported as a loss.

### 3.2 Level with Max-Pressure under detector noise

At noise 0.2 the gate hands 96 % of decisions to Max-Pressure, which itself reads the noisy
detectors. Max-Pressure on noisy readings is therefore the ceiling of this design: UG-MARL is
level with it at 11 of 12 intersections and in five of six windows (420.5 vs 413.7 s overall,
CIs overlap). Where a difference shows, it is against UG-MARL: +9 to +16 s on long trips and
+20 s on straight trips — the few learned decisions taken on noisy inputs disturb through
traffic slightly. Beating Max-Pressure under noise would need cleaner inputs (e.g. the
detector smoothing kept as an optional extension), not a better threshold.

### 3.3 The gate costs waiting time when demand, not the detectors, shifts

Under the ×1.5 surge the detectors are clean but traffic is heavier than anything in training.
σ rises with the queues and the fallback rate climbs from 38 % to 68 % over the hour. Travel
time is unaffected (375 vs 376 s without the gate; Max-Pressure 435 s), but:

- mean waiting inside the network is 8–14 % higher with the gate from minute 20 on;
- the longest wait is 2162 s with the gate, 1430 s without, 1161 s for DQN (Max-Pressure
  3260 s). The longest-waiting vehicles entered early and most never left.

The same appears on the most congested Hangzhou flow (`real_5734`): UG-MARL's travel time is
the best of the agents there (466 s; DQN 479 s, CIs overlap) but its maximum wait equals
Max-Pressure's (3167 vs 3182 s; DQN 2088 s). Once the gate hands most decisions to
Max-Pressure, UG-MARL inherits Max-Pressure's starvation of individual vehicles.

So "the gate bounds the worst case" holds **relative to Max-Pressure** (the minimum outcome,
met in every condition) and **against detector faults**. It does not hold relative to the
ungated policy when the shift is in demand: there the gate fires for the wrong reason.

### 3.4 What a gate intervention is followed by

On the clean flow, in the minute after the gate overrides the learned phase the queue at that
intersection grows by 0.43 vehicles, while after comparable decisions without an override
(same intersection, same 10-minute window, similar starting queue) it shrinks by 0.20
(difference +0.62, CI [0.47, 0.77]). The override is on average a slightly worse move on clean
data, consistent with Max-Pressure being the weaker controller there; only 12 % of decisions
are overridden, so travel time does not move. Under noise the sign reverses (−0.64, −1.19; CIs
include 0 because few un-overridden decisions remain); under the surge it is again positive
(+0.80, CI [0.20, 1.39]). This is an association — the gate fires in different states than it
stays quiet in — but it agrees with every other result: the fallback helps when the *inputs*
are unreliable and costs a little otherwise.

## 4. What σ and the gate threshold do

**σ detects unfamiliar inputs, not wrong actions.** Median σ is ≈ 2 τ at noise 0.1 and ≈ 6 τ at
noise 0.2, unchanged under dropout, and rises under the surge and in the unseen city
(fallback 25–40 % on Hangzhou vs 20 % on Jinan). On validation, σ is almost uncorrelated with
the TD error (Spearman ρ ≈ 0.03, Phase 6). With K = 3 heads sharing one encoder, σ is a coarse
novelty signal. That is enough for the detector-noise result and explains §3.3.

**The threshold is not critical.** On validation, every percentile from p = 50 to 90 scores
within 5 s of the locked p\* = 80 (342.3–343.8 s vs 343.7 s); p ≥ 95 is 3–4 s worse, no gate
141 s worse. By the rule fixed before the sweep, p = 70 (342.3 s) beats p\* by more than the
1 s tie tolerance, so **the calibration grid's lower end was binding** — 70 would have been
chosen had it been in the grid. The gain is 1.5 s, not monotone (p = 60 is 0.1 s worse than
80) and well inside the seed spread of the noisy conditions. Lowering p trades clean
performance (302.3 → 305.6 s from p = 80 to 50) for noise performance (376.5 → 372.2 s at
noise 0.1), with Max-Pressure as the limit. p\* stays 80 and the test flow was not re-run.

## 5. Attention

- **No performance benefit** (H3 not supported; −attention has 61 % fewer parameters and is
  nominally better on Jinan and on Hangzhou).
- **Mostly local.** Mean self-weight 0.82; each neighbour 0.04–0.10.
- **The weights do rank neighbour influence** (faithfulness check on validation, rule fixed in
  advance). Replacing the observation of the highest-α neighbour by another one of the same
  slot changes the Q-value of the chosen action by 0.0061, of the lowest-α neighbour by 0.0022
  (paired difference 0.0039, CI [0.0011, 0.0066]); the greedy action changes in 10 % vs 4 % of
  decisions.
- **But the influence is small.** The same replacement on the intersection's own observation
  changes Q by 0.0415 and the action in 80 % of decisions — about seven times the strongest
  neighbour.
- The knock-out variant (masking a neighbour as absent) does not give a reliable ranking (CI
  includes 0; two of five seed-models show none). It is an input the network only saw at the
  network edge, so it is the weaker test.

The attention maps can therefore be presented as "where the encoder looks, in an order that
matches influence on Q", not as evidence of coordination. A likely reason for the weak effect:
the 44-value state already contains movement pressure, which includes the downstream lanes —
part of what a neighbour would report is in the intersection's own observation.

## 6. Stability of the decisions and of the result

- **Flip-flops** (A → B → A with B held for one 15 s interval): 21.5 % of UG-MARL's decisions
  on clean data, DQN 20.2 %, Max-Pressure 20.3 %, −gate 23.7 %; under noise 0.2 UG-MARL
  11.8 %, Max-Pressure 11.4 %. UG-MARL is not less stable than its baselines, and the gate
  does not add switching.
- **Fairness between approaches** (clean): the learned controllers spread waiting evenly over
  the four approaches (22–29 % each); the longest uninterrupted queue on any approach is 849 s
  for UG-MARL, 773 s for DQN, 1275 s for Max-Pressure, 3410 s for Fixed-Time.
- **Seeds:** clean travel time ranges over 2.8 s for UG-MARL (DQN 1.6 s, IDQN 3.5 s,
  −ensemble 3.9 s). The maximum wait ranges over 421 s (631–1052 s): it is decided by one
  vehicle, which is why the clean part of H2 is "no difference".
- **Shift to another city:** the order of the agents on Hangzhou matches Jinan (learned <
  Max-Pressure < Fixed-Time); UG-MARL and DQN are level on all three flows. The gate removes
  most of the seed-to-seed spread of the ungated policies on the most congested flow
  (466 ± 9 s vs 492 ± 36 s for −gate).

The three representative cases (`docs/figures/results/case_*.png`, picked by rule) show these
mechanisms at one intersection each; they illustrate and prove nothing on their own.

## 7. Limitations and what remains uncertain

1. **Five seeds, one test flow.** CIs are t-intervals over 5 seed-models; small differences
   (the 3.2 s to DQN, the 1.5 s of the τ sweep) are near the resolution of the study.
2. **Cause of the deficit to DQN is not isolated** (architecture vs inherited hyperparameters).
3. **σ is a coarse signal:** three heads on a shared encoder; it flags unfamiliar inputs, is
   nearly uncorrelated with TD error, and misreads heavy demand as uncertainty.
4. **No stability guarantee for the gated policy;** the fallback does not bound per-vehicle
   wait (Max-Pressure itself starves vehicles under the surge).
5. **Waiting-time metrics exclude vehicles blocked outside the network.** This favours the
   collapsed models' max-wait numbers, not UG-MARL's, and is stated wherever max wait is used.
6. **Per-intersection results are waiting, not delay:** the logs have travel time per vehicle
   only.
7. **Validation was reused** for checkpoint selection, (K, d) and p\*; validation numbers are
   optimistic. The τ sweep and the faithfulness check are validation-only and select nothing.
8. **Noise model:** Gaussian noise and dropout on lane counts as defined in the protocol; real
   detector faults (stuck-at, bias, latency) were not tested.
9. **One city family, grid networks, CityFlow dynamics;** Δt = 15 s makes absolute numbers
   not directly comparable with 10 s papers. IDQN is not transferable to another network.
10. **Only two rule-based baselines** (no Advanced-MP, no actuated control).

## 8. What this means for the report

- Headline: *UG-MARL keeps the travel time of a learned controller on clean data (within
  3.2 s of DQN, 25 s better than Max-Pressure) and the robustness of Max-Pressure under
  detector noise, where the ungated learned controllers fail; its worst-case regret over six
  conditions is 6.8 s against 62–415 s for the baselines.*
- To state plainly: it is not the best controller in any single condition; attention did not
  help; the gate costs waiting time under a demand surge; the threshold could have been
  slightly lower.
- Future work that follows directly from the evidence: a fallback with a per-vehicle wait
  bound; a gate signal that separates input faults from heavy demand; detector smoothing
  before the fallback; retuning the UG-MARL hyperparameters; more seeds.
