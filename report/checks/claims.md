# Claims in the report and where each comes from

For review before submission and for viva preparation. Numbers written as `\res{…}` in the
LaTeX sources are generated from `docs/results/*.csv` (`scripts/report_tables.py`) and cannot
drift; the numbers below marked **typed** were entered by hand and should be re-checked if a
source file changes.

## Generated (tables 4.1–4.4 and every `\res` number)

| Report item | Source |
|---|---|
| Table 4.1 main comparison | `docs/results/test_main.csv` |
| Table 4.2 robustness, regret | `docs/results/test_robustness.csv` |
| Table 4.3 ablation | `docs/results/test_ablation.csv`, params from `test_efficiency.csv` |
| Table 4.4 Hangzhou | `docs/results/heldout_generalisation.csv`, `heldout_generalisation_all.csv` |
| Abstract, Ch 1, Ch 4, Ch 5 headline numbers (295.4, 292.2, 307.9, 320.8, 420.5, 870.2, 6.8, 62.2–414.6, fallback rates, max waits) | `\res` macros → same CSVs |

## Typed numbers

| Claim (chapter) | Value | Source |
|---|---|---|
| Learned baselines beat Max-Pressure on clean data (1.5) | 13–29 s | `test_robustness.md` regret table: MP 28.6, IDQN 15.7 → 320.8 − 307.9 = 12.9; 320.8 − 292.2 = 28.6 |
| … and lose under noise 0.2 (1.5) | 345–415 s | regret table: IDQN 345.2, DQN 414.6 |
| UG-MARL vs Max-Pressure, vs IDQN (1.5, 4.2) | 25 s (8 %), 12.5 s | 320.8 − 295.4 = 25.4; 307.9 − 295.4 = 12.5 |
| Deficit to DQN in per cent (4.2) | 1.1 % | 3.2 / 292.2 |
| H2 paired intervals (4.3) | [−382, 353], [−2306, −1320], [−512, −388] s | `test_robustness.md`, H2 table |
| Max wait range over seeds (4.3) | 631–1052 s | `docs/analysis.md` §6, `error_analysis_seeds_max_wait.csv` |
| Ungated learners lose under noise (Fig. 4.1) | 450–575 s | travel-time increase table: IDQN +451.0, DQN +536.0, −gate +575.2 |
| Dropout costs (Fig. 4.1) | < 15 s | same table: largest +14.3 (IDQN) |
| MP gap to the best elsewhere (4.3) | 12–62 s | regret: 28.6, 19.5, 12.4, 62.2 |
| σ relative to τ (Fig. 4.2) | ≈ 2τ, 6τ | `docs/analysis.md` §4 |
| Fallback over the hour under surge (Fig. 4.2) | ≈ 40 → 70 % | `docs/analysis.md` §3.3 (38 → 68 %) |
| Fallback per intersection (Fig. 4.2) | 14–31 %, 95–98 % | `docs/figures/results/gate_firing.png` panels c1, c2 |
| Gate on − off, clean (4.5.1) | 0.4 ± 0.9 s | `test_ablation.md` (0.44 ± 0.94) |
| Cost of removing the gate at noise 0.2 (4.5.1) | 450 s | 870.2 − 420.5; paired −450.2 |
| −ensemble vs full, clean (4.5.1) | 1.9 s | `test_ablation.md` |
| −attention parameters (4.5.1, 5.1) | 63 % fewer | 1 − 59 032 / 157 720 = 0.626. **Note:** `docs/analysis.md` and the Decision Log say 61 %; the arithmetic gives 63 % |
| Hangzhou vs Max-Pressure (4.5.2) | 22 s | 431.8 − 410.0 |
| Waiting vs Max-Pressure per intersection; gain by route length (4.5.3) | 17–40 %; 12 s, 37 s | `docs/analysis.md` §2, `error_analysis_intersections.csv`, `error_analysis_route_length.csv` |
| Deficit to DQN per intersection (4.5.3) | up to 12 % | `docs/analysis.md` §3.1 |
| Vehicles left outside the network at noise 0.2 (4.5.3) | 584–795 of 5494 (11–14 %) | `docs/analysis.md` §2, `error_analysis_entry.csv` |
| Flip-flops (4.5.3) | 21.5 / 20.2 / 20.3 % | `docs/analysis.md` §6, `error_analysis_stability.csv` |
| σ vs TD error (4.5.4) | ρ ≈ 0.03 | `docs/results/sigma_validity_td.csv`, `docs/analysis.md` §4 |
| Faithfulness (4.5.4) | 0.0061 vs 0.0022, [0.0011, 0.0066]; own 0.0415; self-weight 0.82 | `docs/results/attention_faithfulness.md` |
| Threshold sweep (4.5.5) | within 5 s for p 50–90; p = 70 is 1.5 s better | `docs/results/tau_sensitivity.md` |
| Latency (4.5.5) | 0.81 / 0.22 / 0.14 ms | `docs/results/test_efficiency.md` |
| Fairness (5.2) | 22–29 %; 849 vs 1275 s | `docs/analysis.md` §6, `error_analysis_approaches.csv` |
| Dataset sizes, route lengths (Table 3.1) | — | `docs/dataset_card.md` §3 |
| Audit (3.1) | 17 + 12 checks, 32 659 records, 0 defects | `docs/data_audit.md` |
| EDA findings (3.3) | 3000–8160 veh/h; 1477–2058; 96 %; 13–21 %; 0.65–0.82; 451/375/344/347 s | `notebooks/01_eda_train.ipynb` (insights table), `docs/data_audit.md` §5 |
| Tuning (3.6) | 8 of 36; (1e-4, 0.8, 64, 0.3); grid best 301.1 s; K = 3, d = 128; p* = 80 | `configs/search/*.yaml`, `configs/dqn.yaml`, `configs/ugmarl.yaml`, `docs/results/gate_calibration.md` |
| Parameters (3.4) | 157 720 = 55 936 + 3 × 33 928 | formula in `docs/methodology.md` §7 with d = 128, K = 3; `test_efficiency.csv` |
| Hardware / software (3.6) | i5-13420H, Python 3.12.3, PyTorch 2.15.0, NumPy 2.5.4 | `results/final/…/env.json` |

## Statements about other work

Each is taken from the project's literature notes (`docs/literature/01–06`), where the
verification status of every paper is recorded (`06-gap-analysis.md` §1c). No performance
number of another paper is quoted (decision of 14 Nov, worklog/023). Since 15 Nov
(worklog/025) the report cites 10 works only; statements about the 28 dropped papers
(Nguyen et al. 2025, MPLight scale, IntelliLight, SafeLight, LLMLight, …) were removed, and
generic statements (fixed plans are common; epistemic vs aleatoric; time-limit bootstrapping;
attention weight ≠ influence) are made without a citation.

| Statement (Ch 1 contributions) | Source |
|---|---|
| Ensemble Q-heads in signal control are used for exploration | Tan et al. 2020, `docs/literature/` |
| Com-MP keeps Max-Pressure as selector, RL adapts it | Zhu & Tan 2026, `docs/literature/06` |
| Hoel et al.: ensemble disagreement switches one vehicle to a safe action | Hoel et al. 2020 |

## To verify by the author before submission

- Bibliography entries whose metadata is still marked `check` in `references.bib` (field
  `annote`, not printed): `mnih2015human`, `zhang2019cityflow`, `hoel2020tactical`, and the
  page / author details of `varaiya2013max`, `wei2019colight`, `tan2020multiagent`.
- The course rule "no AI contents" (format document): the text was drafted by Claude at the
  author's decision; the author reviews, edits and is responsible for it.
