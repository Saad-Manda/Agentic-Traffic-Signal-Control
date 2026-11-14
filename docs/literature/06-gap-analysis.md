# 06 — Gap analysis and positioning of UG-MARL

ROADMAP 2.6 (gap table: draft for Review 2 on 10 Nov; UG-MARL row and action items updated 13 Nov
to the system as built and evaluated in Phases 6–7) · Rubric R2 · Guideline G2

---

## 1. Gap table

Legend: ✔ yes · ◐ partial / implicit · ✗ no · — not applicable.
"Uncertainty at execution" = the estimate changes what the controller does at decision time.

| Method | Type | Coordination | Uncertainty estimate | Uncertainty used at execution | Safety / fallback | Guarantee | Explainability | Robustness tested |
|---|---|---|---|---|---|---|---|---|
| Fixed-Time (Webster 1958; background baseline) | Conventional | ✗ (offsets only) | ✗ | — | — | ✗ | ✔ (transparent rule) | ✗ |
| Max-Pressure (Varaiya 2013; background baseline + fallback) | Conventional | ◐ via downstream queues | ✗ | — | — | ✔ queue stability (model) | ✔ | ✗ |
| Advanced-MP (Zhang 2022) | Conventional | ◐ | ✗ | — | — | ◐ | ✔ | ✗ |
| DQN-TSC (Genders 2016) | Single-agent RL | ✗ (isolated) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| IntelliLight (Wei 2018) | Single-agent RL | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ case studies | ✗ |
| FRAP (Zheng 2019) | Single-agent RL | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ generalisation |
| MA2C (Chu 2020) | MARL | ✔ fingerprints + spatial discount | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| PressLight (Wei 2019) | MARL | ◐ via pressure reward | ✗ | ✗ | ✗ | ◐ MP-motivated reward, no proof for learned policy | ✗ | ✗ |
| CoLight (Wei 2019) | MARL | ✔ graph attention | ✗ | ✗ | ✗ | ✗ | ◐ attention (not validated) | ✗ |
| MPLight (Chen 2020) | MARL | ◐ pressure + shared params | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ scale |
| AttendLight (Oroojlooy 2020) | RL | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ attention | ✔ generalisation |
| Advanced-XLight (Zhang 2022) | MARL | ✔ (Adv-CoLight) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| MA Bootstrapped DQN (Tan 2020) | MARL | ◐ shared params | ✔ bootstrap heads | ✗ (exploration only) | ✗ | ✗ | ✗ | ◐ |
| RELight (Qi 2022) | RL ensemble | ◐ | ◐ ensemble | ✗ (members vote on the action; disagreement not used as uncertainty) | ✗ | ✗ | ✗ | ✗ |
| BCT-APLight (Duan 2024, preprint) | MARL | ◐ attention-weighted pressure | ◐ Bayesian "critique" of policy credibility | ✗ (policy is fine-tuned, no hand-over) | ✗ | ✗ | ✗ | ✗ |
| SafeLight (Du 2023) | RL | ◐ | ✗ | ✗ | ✔ collision safety model (action filter) | ✗ | ✗ | ◐ |
| T2-BAS-DQN (Bi 2026) | MARL (regional) | ◐ coordination pressure | ◐ Type-2 fuzzy correction (not epistemic) | ✗ (always-on correction) | ✗ (MP only a baseline) | ✗ | ◐ fuzzy rules | ✗ |
| Com-MP (Zhu & Tan 2026) | RL inside MP | ◐ via MP | ✗ | ✗ | ✔ MP always selects phase | ✔ claimed strong stability | ◐ | ◐ heterogeneity |
| LLMLight (Lai 2025) | LLM agent | ✗/◐ | ✗ (not calibrated) | ✗ | ✗ | ✗ | ✔ natural-language | ◐ |
| Hoel et al. 2020 *(driving, not TSC)* | Single-agent RL | — | ✔ ensemble + priors | ✔ threshold† | ✔ fallback action† | ✗ | ◐ | ✔ OOD |
| **UG-MARL (ours)** | **MARL** | **✔ neighbour attention** | **✔ Q-ensemble σ(s)** | **✔ gate** | **◐ MP fallback (bounded worst-case regret on test, but raises max wait under the demand surge)** | **◐ MP stability if the gate always fires; mixed policy: sketch only (`methodology.md` §6)** | **✔ gate log, σ, attention weights (faithfulness check done: weights rank neighbour influence, which is small)** | **✔ noise, dropout, surge, held-out city** |

*Rows are post-2015 papers, except the two background baselines we implement. Checked 14 Nov
against each paper's abstract and, where reachable, its arXiv HTML full text (log in §1c). † =
confirmed at abstract level only ("uncertainty is used to choose safe actions in unknown
situations"); the threshold mechanism is from the full text, not re-read. The UG-MARL row
describes the system as built and evaluated: the gate is σ > τ only — the queue / Lyapunov
trigger (A2) was not added — so the guarantee stays ◐; safety is ◐ because the gate raised
max wait under the demand surge (2162 vs 1430 s, `docs/analysis.md`).*

### 1b. Condensed table for the report (Ch 2)

The full table does not fit an A4 column. The report uses these 12 rows × 6 columns (full table →
appendix). Every row has a `report/references.bib` key.

| Method | Coord. | Uncertainty estimate | Used at execution | Fallback / guarantee | Explain. |
|---|---|---|---|---|---|
| Max-Pressure `varaiya2013max` | ◐ | ✗ | — | ✔ queue stability (model) | ✔ rule |
| Advanced-MP `zhang2022expression` | ◐ | ✗ | — | ◐ | ✔ rule |
| IntelliLight `wei2018intellilight` | ✗ | ✗ | ✗ | ✗ | ◐ case studies |
| MA2C `chu2020multi` | ✔ fingerprints | ✗ | ✗ | ✗ | ✗ |
| PressLight `wei2019presslight` | ◐ pressure reward | ✗ | ✗ | ◐ MP-motivated reward | ✗ |
| CoLight `wei2019colight` | ✔ attention | ✗ | ✗ | ✗ | ◐ attention |
| MPLight `chen2020toward` | ◐ pressure | ✗ | ✗ | ✗ | ✗ |
| MA Bootstrapped DQN `tan2020multiagent` | ◐ | ✔ heads | ✗ exploration | ✗ | ✗ |
| SafeLight `du2023safelight` | ◐ | ✗ | ✗ | ✔ safety action filter | ✗ |
| Com-MP `zhu2026commp` | ◐ via MP | ✗ | ✗ | ✔ MP always selects; stability claimed | ◐ |
| Hoel 2020 *(driving)* `hoel2020tactical` | — | ✔ ensemble + priors | ✔ | ✔ safe action | ◐ |
| **UG-MARL (ours)** | **✔ attention** | **✔ Q-ensemble σ** | **✔ gate** | **◐ MP fallback; sketch only** | **✔ σ, gate log, attention** |

### 1c. Verification log (14 Nov)

Sources: arXiv abstract pages (+ arXiv HTML where available); ACM DL returned 403, so the two KDD
papers were checked via the authors' hosted PDF listing / search abstracts only.

| Row | Result | Change made |
|---|---|---|
| Advanced-MP / Advanced-XLight (2112.10107) | Rule-based, running + queued vehicles, efficient pressure: confirmed | — |
| DQN-TSC (1611.01142) | Isolated intersection, deep Q-learning + replay: confirmed | — |
| IntelliLight (KDD 2018) | Full text not reachable (ACM 403); metadata only | kept, flagged |
| FRAP (1905.04722) | Single intersection, phase competition: confirmed; invariance is to *symmetries* (flip/rotation), generalisation to road structures/flows | wording in 03 |
| MA2C (1903.04527) | Neighbour policy fingerprints + spatial discount α: confirmed | — |
| PressLight (KDD 2019) | Pressure reward from MP theory, beats MP and learning methods: confirmed at abstract level | — |
| CoLight (1905.05717) | Attention incl. self, shared params, Q-learning, reward = −queue: confirmed. Paper gives intersection *counts* (Jinan 12, Hangzhou 16, NY 196), not grid shapes | dataset line in 03 |
| MPLight (AAAI 2020) | Pressure, parameter sharing, 2510-light Manhattan: confirmed (no arXiv copy) | — |
| AttendLight (2010.05772) | Universal single-intersection model via attention: confirmed | — |
| MA Bootstrapped DQN (CCTA 2020) | Bootstrap ensemble for exploration, conditional parameter sharing, 5×5 grid: confirmed (abstract) | — |
| RELight (2203.05961) | **Corrected:** the ensemble also votes on the action at execution; still no uncertainty use or fallback | cell text |
| SafeLight (2211.10871) | Safety standards as enforcement, multi-objective loss + reward shaping; strongest variant filters actions: confirmed | "(action filter)" |
| Com-MP (2610.31037) | MP phase selection retained; strong queue stability claimed; heterogeneity: confirmed | — |
| LLMLight (2312.16044) | LLM agent, human-like reasoning: confirmed; absence of uncertainty/fallback from method description | — |
| Hoel 2020 (2004.10439) | Ensemble + RPF, safe actions when uncertain, OOD: confirmed; threshold detail abstract-silent | † |
| Nguyen 2025 (2506.13836) | **Corrected:** degradation is shown for *independent value-based and decentralised pressure-based RL*; hierarchical methods are more stable; MP not named in the abstract | §2 paragraph reworded |

**External performance numbers.** Published ATT values for the same "Jinan"/"Hangzhou" names
disagree across papers (e.g. MP on Jinan: 361 s in CoLight Table 2 vs 246–274 s in Advanced-MP
Table 2) because flow files and simulation settings differ. **Decision: the report quotes no
external numbers as comparable to ours**; it cites only qualitative claims. This closes the
*(unverified)* numbers in 01 and 03.

## 2. The gap in one paragraph (draft for the report)

> RL controllers for traffic signals (PressLight, CoLight, MPLight) outperform Max-Pressure on
> average but provide no indication of when their decisions are unreliable, and recent evidence
> shows that independent and decentralised RL controllers degrade sharply under incident-driven
> distribution shift (Nguyen et al. 2025). Max-Pressure, conversely, is provably queue-stable under
> standard assumptions but cannot learn. Ensembles have been used in RL-based signal control only
> to improve exploration or training stability (Tan et al. 2020; Qi et al. 2022), and
> uncertainty-triggered fallback controllers have been studied in robotics and autonomous
> driving (Lütjens et al. 2019; Hoel et al. 2020) but not in multi-agent signal control. We
> propose UG-MARL, which uses ensemble disagreement at execution time to hand each
> intersection's decision to Max-Pressure when the learned policy is uncertain. Coordination comes from
> neighbour attention.

## 3. What we can and cannot claim as novel

**Defensible (with evidence from experiments):**
1. **Execution-time** use of Q-ensemble epistemic uncertainty to gate between a multi-agent RL
   policy and Max-Pressure in traffic signal control. No prior TSC paper found doing this
   (search method in §5).
2. A Simplex-style formulation for TSC: high-performance (attention MARL) + high-assurance (MP) +
   uncertainty/state decision logic, with the stability argument made explicit and its
   assumptions stated.
3. An evaluation that isolates each part: −gate / −attention / −ensemble ablations, a
   validation sweep of the τ percentile (p ∈ {80, 90, 95, 99}, `docs/results/gate_calibration.md`;
   extended to p ∈ {50, 60, 70} in Phase 8, `docs/results/tau_sensitivity.md`),
   and a σ validity check (σ under shift, Spearman ρ between σ and |TD error|,
   `docs/results/sigma_validity.csv`). No risk–coverage curve or AUROC (A6).

**Not ours (cite, do not claim):** attention-based coordination (CoLight, DGN), pressure reward
(PressLight), ensembles in TSC (Tan 2020, RELight), the ensemble→threshold→fallback pattern
(Kahn 2017, Lütjens 2019, Hoel 2020), MP stability (Varaiya 2013), Simplex (Sha 2001).

**Threats to the novelty claim:**
- **Com-MP (Oct 2026)** is recent and close in spirit: RL + MP with a stability guarantee. The
  difference is mechanism. Com-MP learns *inside* MP (MP always picks the phase). UG-MARL lets RL pick and
  defers to MP *when uncertain*. Cite it as concurrent work and contrast in one sentence.
- Our search was not exhaustive (§5). Use the phrasing "to our knowledge". Re-run 14 Nov: no
  paper found that uses ensemble uncertainty at execution to hand control to MP. Closest new
  hits: BCT-APLight (Bayesian critique, then fine-tunes the policy) and T2-BAS-DQN (always-on
  fuzzy correction, MP only a baseline); both added to the table.

## 4. Action items this review creates for the project

Status column added 13 Nov (after Phase 7). Dropped items are limitations to state in the report.

| # | Item | Where | Why | Status (13 Nov) |
|---|---|---|---|---|
| A1 | Reword PRD §1c "minimum acceptable outcome": MP does not bound per-vehicle max wait; make it an empirical hypothesis (or add a max-red constraint in the env) | `PRD.md` §1c | 01 §2 caveat 1 | **Partly.** H2 made empirical with a three-level reading fixed before the test runs (PRD §1c, protocol §9). The "minimum acceptable outcome" line still says "the gate should guarantee this"; left as fixed, the report states it as an empirical check |
| A2 | Decide: add queue/Lyapunov trigger (σ > τ **or** V > V_max) to the gate, plus an ablation | `architecture.md`, Decision Log | 01 §2 caveat 2; needed for any stability claim | **Dropped.** Gate is σ > τ only (Phase 6). The mixed policy has a drift sketch, not a guarantee (`methodology.md` §6) |
| A3 | Use randomized prior functions in the Q-ensemble | `architecture.md` | 04 §2.3: σ must grow off-distribution | **Dropped.** Bernoulli(0.8) bootstrap masks instead (Phase 6 plan: one fewer component). σ still rises 4–11× under noise on val (`sigma_validity.csv`) |
| A4 | Double DQN in **all** learners; shared parameters in IDQN and UG-MARL | `architecture.md` | Fair comparison (02 A3, B2) | **Done, with one deliberate exception.** Double DQN in every learner. DQN and UG-MARL share parameters. IDQN keeps independent nets because it is the PS16 independent-learner baseline |
| A5 | Consider Advanced-MP as an extra conventional baseline | ROADMAP Phase 4 | Credibility of H1 (03 §4.6) | **Dropped.** Only Fixed-Time and Max-Pressure are implemented; a limitation for the report |
| A6 | Add a τ sweep / risk–coverage plot and σ-AUROC diagnostic to the evaluation protocol | `docs/evaluation_protocol.md` (3.5) | 04 §3–4 | **Partly.** τ-percentile sweep on val (`gate_calibration.md`, extended to p = 50–70 in Phase 8: `tau_sensitivity.md`) and σ check by Spearman ρ (`sigma_validity*.csv`); no risk–coverage plot or AUROC |
| A7 | Explicitly report gate-firing rates under shift (noise, held-out city) | protocol | Ovadia: ensembles can be confidently wrong under shift | **Done.** `test_robustness.md`, `test_safety_explain.md`, `heldout_generalisation.md` (fallback-rate sections) |
| A8 | Attention faithfulness check (perturb high-α vs low-α neighbour) | Phase 8 | 05 B | **Done (13 Nov).** Validation, offline slot replacement (`docs/results/attention_faithfulness.md`): replacing the highest-α neighbour changes mean Q by 0.0061 vs 0.0022 for the lowest-α one (paired 95 % CI [0.0011, 0.0066]) — consistent with the weights; the own observation gives 0.0415, so neighbour influence is small. Knock-out variant: no reliable ranking |
| A9 | Datasets: Jinan/Hangzhou/New York come from the CoLight release, so comparable numbers exist | ROADMAP Phase 1 | 03 §4.3 | **Done.** Both from the CoLight repo (`04440b7`); see Decision Log 10 Nov |

## 5. Search method (for the report's reproducibility)

- **Date:** 9 Nov 2026. Web search over arXiv, ACM DL, IEEE Xplore, AAAI, NeurIPS/ICML/PMLR
  proceedings, and Google-indexed pages.
- **Queries (examples):** "uncertainty-aware reinforcement learning traffic signal control ensemble
  fallback max-pressure"; "safe reinforcement learning traffic signal control shield fallback";
  "traffic signal control reinforcement learning switch to max pressure when uncertain hybrid";
  "bootstrapped DQN / Bayesian DQN traffic signal control"; plus title searches to verify each
  cited paper.
- **Inclusion:** (i) foundational methods we implement or build on, (ii) the CityFlow/Jinan/
  Hangzhou lineage (comparability), (iii) any TSC paper combining RL with uncertainty, ensembles,
  safety or MP, and (iv) uncertainty-triggered fallback in other domains.
- **Re-run 14 Nov 2026** (same queries plus "epistemic uncertainty gate switch to rule-based
  controller 2026", "ensemble disagreement max pressure fallback"). New papers screened:
  UCATSC (arXiv 2602.07784; belief-state constraints, single intersection, no hand-over),
  DR-MARL with CB-WCE (2607.09899; training-time adversarial demand only), BCT-APLight
  (2412.16225), Kwesiga et al. (2603.15283; single intersection, robustness to demand), T2-BAS-DQN
  (Bi et al. 2026, CMC). None uses execution-time uncertainty to defer to Max-Pressure.
- **Verification level (14 Nov):** every gap-table row checked against its abstract / arXiv HTML
  (§1c). External numbers are not quoted as comparable (§1c), so no *(unverified)* number enters
  the report.
