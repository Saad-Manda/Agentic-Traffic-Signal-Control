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
| RELight (Qi 2022) | RL ensemble | ◐ | ◐ ensemble | ✗ (learning quality) | ✗ | ✗ | ✗ | ✗ |
| SafeLight (Du 2023) | RL | ◐ | ✗ | ✗ | ✔ collision safety model | ✗ | ✗ | ◐ |
| Com-MP (Zhu & Tan 2026) | RL inside MP | ◐ via MP | ✗ | ✗ | ✔ MP always selects phase | ✔ claimed strong stability | ◐ | ◐ heterogeneity |
| LLMLight (Lai 2025) | LLM agent | ✗/◐ | ✗ (not calibrated) | ✗ | ✗ | ✗ | ✔ natural-language | ◐ |
| Hoel et al. 2020 *(driving, not TSC)* | Single-agent RL | — | ✔ ensemble + priors | ✔ threshold | ✔ fallback action | ✗ | ◐ | ✔ OOD |
| **UG-MARL (ours)** | **MARL** | **✔ neighbour attention** | **✔ Q-ensemble σ(s)** | **✔ gate** | **✔ MP fallback** | **◐ MP stability if the gate always fires; mixed policy: sketch only (`methodology.md` §6)** | **✔ gate log, σ, attention weights (faithfulness check: Phase 8)** | **✔ noise, dropout, surge, held-out city** |

*Rows are post-2015 papers, except the two background baselines we implement. Entries are from each paper's main method as summarised in files 01–05. Check every ✔/◐ against
the PDF before the final report (ROADMAP 2.6, at the start of Phase 9). The UG-MARL row describes
the system as built (13 Nov): the gate is σ > τ only — the queue / Lyapunov trigger (A2) was not
added — so the guarantee stays ◐.*

## 2. The gap in one paragraph (draft for the report)

> RL controllers for traffic signals (PressLight, CoLight, MPLight) outperform Max-Pressure on
> average but provide no indication of when their decisions are unreliable, and recent evidence
> shows they degrade sharply under distribution shift, where Max-Pressure does not
> (Nguyen et al. 2025). Max-Pressure, conversely, is provably queue-stable under
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
   validation sweep of the τ percentile (p ∈ {80, 90, 95, 99}, `docs/results/gate_calibration.md`),
   and a σ validity check (σ under shift, Spearman ρ between σ and |TD error|,
   `docs/results/sigma_validity.csv`). No risk–coverage curve or AUROC (A6).

**Not ours (cite, do not claim):** attention-based coordination (CoLight, DGN), pressure reward
(PressLight), ensembles in TSC (Tan 2020, RELight), the ensemble→threshold→fallback pattern
(Kahn 2017, Lütjens 2019, Hoel 2020), MP stability (Varaiya 2013), Simplex (Sha 2001).

**Threats to the novelty claim:**
- **Com-MP (Oct 2026)** is recent and close in spirit: RL + MP with a stability guarantee. The
  difference is mechanism. Com-MP learns *inside* MP (MP always picks the phase). UG-MARL lets RL pick and
  defers to MP *when uncertain*. Cite it as concurrent work and contrast in one sentence.
- Our search was not exhaustive (§5). Use the phrasing "to our knowledge", and re-run the search before
  the final report (16 Nov).

## 4. Action items this review creates for the project

Status column added 13 Nov (after Phase 7). Dropped items are limitations to state in the report.

| # | Item | Where | Why | Status (13 Nov) |
|---|---|---|---|---|
| A1 | Reword PRD §1c "minimum acceptable outcome": MP does not bound per-vehicle max wait; make it an empirical hypothesis (or add a max-red constraint in the env) | `PRD.md` §1c | 01 §2 caveat 1 | **Partly.** H2 made empirical with a three-level reading fixed before the test runs (PRD §1c, protocol §9). The "minimum acceptable outcome" line still says "the gate should guarantee this"; left as fixed, the report states it as an empirical check |
| A2 | Decide: add queue/Lyapunov trigger (σ > τ **or** V > V_max) to the gate, plus an ablation | `architecture.md`, Decision Log | 01 §2 caveat 2; needed for any stability claim | **Dropped.** Gate is σ > τ only (Phase 6). The mixed policy has a drift sketch, not a guarantee (`methodology.md` §6) |
| A3 | Use randomized prior functions in the Q-ensemble | `architecture.md` | 04 §2.3: σ must grow off-distribution | **Dropped.** Bernoulli(0.8) bootstrap masks instead (Phase 6 plan: one fewer component). σ still rises 4–11× under noise on val (`sigma_validity.csv`) |
| A4 | Double DQN in **all** learners; shared parameters in IDQN and UG-MARL | `architecture.md` | Fair comparison (02 A3, B2) | **Done, with one deliberate exception.** Double DQN in every learner. DQN and UG-MARL share parameters. IDQN keeps independent nets because it is the PS16 independent-learner baseline |
| A5 | Consider Advanced-MP as an extra conventional baseline | ROADMAP Phase 4 | Credibility of H1 (03 §4.6) | **Dropped.** Only Fixed-Time and Max-Pressure are implemented; a limitation for the report |
| A6 | Add a τ sweep / risk–coverage plot and σ-AUROC diagnostic to the evaluation protocol | `docs/evaluation_protocol.md` (3.5) | 04 §3–4 | **Partly.** τ-percentile sweep on val (`gate_calibration.md`) and σ check by Spearman ρ (`sigma_validity*.csv`); no risk–coverage plot or AUROC |
| A7 | Explicitly report gate-firing rates under shift (noise, held-out city) | protocol | Ovadia: ensembles can be confidently wrong under shift | **Done.** `test_robustness.md`, `test_safety_explain.md`, `heldout_generalisation.md` (fallback-rate sections) |
| A8 | Attention faithfulness check (perturb high-α vs low-α neighbour) | Phase 8 | 05 B | **Open → Phase 8** (ROADMAP 8.2): perturb the high-α vs low-α neighbour on validation flows (no new test runs) |
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
- **Verification level:** bibliographic metadata verified online for papers marked ✓. Summaries
  come from the abstracts and prior knowledge. Numbers marked *(unverified)* must be checked against the PDFs
  before they enter the report.
