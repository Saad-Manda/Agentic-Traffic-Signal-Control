# Plan — Phase 6: proposed UG-MARL (ROADMAP 6.1–6.7)

Status: **planned 11 Nov 2026, not executed.** Decisions U1–U15 were taken with the user on
11 Nov. Tick steps `[x]` during execution and note deviations under "Deviations".

## Goal

Build the proposed agent — neighbour attention + K-head Q-ensemble + uncertainty gate with a
Max-Pressure fallback — on the Phase 5 Q-learning core, tune it on the **validation** flow,
calibrate the gate threshold τ on validation, and produce a dev validation table with the
ablations, the explanation logs and the math write-up.

| Task | Output | Serves |
|---|---|---|
| 6.1 Attention encoder | `NeighborhoodEncoder` (data) + attention block in `UGMARLNet` | F4, R4, H3 |
| 6.2 Q-ensemble | K MLP heads on a shared encoder, bootstrap masks in replay, μ(a), σ(s) | F4, R4 |
| 6.3 Gate | `UGMARLAgent.act`: σ > τ → Max-Pressure; `scripts/calibrate_gate.py` → τ per model | F4, R5, H2 |
| 6.4 Explanation logging | `last_explanations()` + `DecisionRecorder` → `decisions_s<seed>.csv` | F10, G11 |
| 6.5 Ablation flags | `use_attention`, `use_ensemble`, `use_gate` in `agents.ugmarl` | F7, R5 |
| 6.6 Math write-up | `docs/methodology.md` | R5, R7 |
| 6.7 Tuning | grid K × d (9 runs) + gate percentile sweep; search files committed first | F5, G8 |
| Extra | σ validity check (script + figure, validation only) | F10, contribution list |
| Outputs | validation table with ablations, learning curves, gate calibration table | R6, G9 |

Out of scope: any test / held-out run, robustness runs on test, final native retraining
(Phase 7); SHAP, vote-disagreement logging, demo animation (not selected); report / slides.

## Handoff — read this first if executing in a new session or on another machine

This plan was written on WSL on 11 Nov. A fresh session has only what is in git plus this
file, so before step 1:

1. **Load context:** run `.claude/commands/prime.md` (CLAUDE.md, PRD.md, ROADMAP.md, all of
   `.claude/reference/`, latest worklog entries), then read `phase5-rl-baselines.md` (the core
   this plan extends) and `src/trafficrl/agents/core/*.py`, `train/loop.py`, `eval/runner.py`.
2. **Git:** this plan and the Phase 5 commits must be pushed from WSL and pulled here
   (on 11 Nov WSL `main` was 10 commits ahead of `origin/main`, and this file was untracked).
   Check `git log --oneline -3` shows the Phase 5 commits (`1769c46` or later).
3. **Not in git — must be recreated on this machine** (all gitignored):
   - `.venv` + CityFlow build → README "Install" (ROADMAP 0.1b; run
     `scripts/bench_cityflow.py`, record the numbers, tick 0.1b).
   - `data/raw/` → README dataset download commands (CoLight repo `04440b7`); hashes are
     checked by `data/processed/splits.json` (tracked). `pytest` must pass before any run.
   - `results/` → **no Phase 5 checkpoints or evaluation runs exist here.** `val_table.py`
     reads `results/eval` and `results/train`, so on a new machine re-run, in this order:
     ```bash
     python scripts/evaluate.py --config configs/eval_baselines.yaml --agent fixed_time --split val
     python scripts/evaluate.py --config configs/eval_baselines.yaml --agent max_pressure --split val
     python scripts/tune.py --config configs/dqn.yaml  --seeds 0 1 2 3 4 --workers 5
     python scripts/tune.py --config configs/idqn.yaml --seeds 0 1 2 3 4 --workers 4
     python scripts/evaluate.py --config configs/eval_rl_baselines.yaml --agent dqn  --split val \
         --checkpoint 'results/train/dqn_final_s{seed}/checkpoints/best.pt'
     python scripts/evaluate.py --config configs/eval_rl_baselines.yaml --agent idqn --split val \
         --checkpoint 'results/train/idqn_final_s{seed}/checkpoints/best.pt'
     ```
     Expected: Fixed-Time 434.84 s and Max-Pressure 326.11 s exactly (deterministic); DQN /
     IDQN close to, but not identical with, the WSL dev values 299.8 ± 1.0 / 312.7 ± 2.5 s
     (floating point differs per platform). The baselines are **not re-tuned** (configs locked).
     Do not regenerate `docs/results/val_phase5.*` (it is the WSL dev record).
4. **Numbers quoted in this plan** (326.11 s, 299.8 s, timings, RAM) are WSL dev values; on
   native Ubuntu re-measure RAM / cores before choosing `--workers`, and redo the device
   benchmark of step 6.
5. **User rules that live outside the repo** (Claude memory on the WSL machine): never add a
   Claude co-author / "Generated with" line to commits or PRs; the submission must list 4–5
   contributions = innovations over existing solutions (keep a running list in worklog/017:
   neighbour attention, ensemble uncertainty, uncertainty gate with Max-Pressure fallback,
   validation-calibrated τ under perturbed inputs, σ validity evidence / explanation logs).
6. **Do not run** anything with `--final`, split `test` or `heldout`, or dataset `hangzhou`.
7. If something here disagrees with the code, the code and `docs/evaluation_protocol.md` win;
   record the deviation under "Deviations".

## Decisions taken (user, 11 Nov)

| # | Topic | Decision | Reason |
|---|---|---|---|
| U1 | Weights | **One shared network** for all intersections (like DQN / CoLight). | Runs zero-shot on Hangzhou (7.4); shared DQN beat IDQN on val (299.8 vs 312.7 s). |
| U2 | Ensemble | **Shared encoder + K independent MLP heads** (each 128 hidden → 8). | Cost almost flat in K; heads less correlated than K linear layers; one attention map to explain. |
| U3 | Attention | **One GAT layer, CoLight-style, one attention head**, learned slot embedding (self / E / N / W / S), output concatenated with the own embedding. | Weights αᵢⱼ are directly plottable; simple to defend. |
| U4 | Uncertainty | **σ(s) = std over heads of Q_k(s, a\*)**, a\* = argmax of the mean Q. | Matches `architecture.md` and the Review 2 design. |
| U5 | τ rule | τ = **p-th percentile of σ on a clean greedy val episode**; p from a pre-written grid {80, 90, 95, 99}, chosen by the **mean val travel time over clean + perturbed validation** (noise 0.1 / 0.2, dropout 10 / 20 %). | Clean val alone would pick "never fire" (DQN already beats Max-Pressure); the gate exists for unreliable inputs. Validation only. |
| U6 | Gate in training | **Evaluation only.** Training is ε-greedy on the mean Q. | −gate = same checkpoint with the gate off; τ swept without retraining. |
| U7 | Exploration | **ε-greedy on mean Q**, same schedule as the baselines. | Comparison isolates attention / ensemble / gate. |
| U8 | Head diversity | **Bernoulli(0.8) bootstrap masks** stored in replay + independent head init; per-head Double-DQN target. Mask probability fixed, not tuned. | Bootstrapped DQN (Osband 2016), cheap. |
| U9 | Training search | **Full grid K ∈ {3, 5, 10} × d ∈ {32, 64, 128}**, 9 runs, seed 0; lr / γ / batch / ε-decay inherited from baseline trial t6 (1e-4, 0.8, 64, 0.3). | What ROADMAP 6.7 names; no sampling argument. |
| U10 | Ablations | **Dev runs now:** full, −attention, −ensemble on seeds 0–4 (15 runs); −gate needs no training. | Ablation table for Review 3; early warning for H3. |
| U11 | Math doc | **`docs/methodology.md`** with LaTeX math, laid out like report §5. | Copied into report / slide 6 in Phase 9. |
| U12 | Extra | **σ validity check** only. | Evidence the gate signal is meaningful. |
| U13 | τ scope | **One p for all seeds, τ per seed-model** (`gate.json` next to each checkpoint). | σ scale differs between models; one hyperparameter to report. |
| U14 | If no gain | **Lock the best grid config whatever it scores; no extra tuning.** What to do next is decided with the user afterwards. | No post-hoc tuning advantage over the baselines. |
| U15 | Platform | Asked "which is better if native Ubuntu is ready" → see D0. | |

## Design

### D0. Platform (answer to U15)

**Native Ubuntu is the better choice once 0.1b is done**, for the long runs only:
the grid, the 15 final runs and the gate calibration then give *reportable* validation
numbers and locked checkpoints, so Phase 7 only has to run test / held-out evaluation for
UG-MARL (saves ≈ 4–6 h of retraining). Cost: the DQN / IDQN 5-seed finals must be re-run
there (10 runs, ≈ 1 h) so the table compares like with like — Phase 7 needs that anyway.

Plan: code, tests and smoke runs on whichever machine is in use (steps 1–7). At step 8 check
ROADMAP 0.1b: ticked → run steps 8–13 on native Ubuntu and label numbers *final-val*; not
ticked → run on WSL and label them *dev* (Phase 5 rule). Never mix platforms inside one table
(CPU / CUDA and platform change floating-point results). Recorded in the Decision Log.

### D1. Package layout

```
src/trafficrl/agents/core/
  encoders.py      + NeighborhoodEncoder
  replay.py        + optional extra columns (bootstrap mask)
  qlearner.py      + masked loss, mask sampling, load() relaxed for shared nets
src/trafficrl/agents/ugmarl/
  network.py       UGMARLNet (embedding, attention, K heads)
  agent.py         UGMARLAgent(QLearner): μ, σ, gate, explanations; variant_name()
  gate.py          sigma_from_q, percentile thresholds, load/save gate.json
src/trafficrl/eval/
  explain.py       DecisionRecorder (per-decision rows)
  runner.py        run_episode(..., recorder=None); gate file resolution per seed
scripts/
  calibrate_gate.py   σ percentiles + p sweep on validation -> gate.json per model
  sigma_check.py      σ validity figure + table (validation)
```

### D2. State encoding — `NeighborhoodEncoder`

- `state_shape = (5, 45)`: slot 0 = self, slots 1–4 = neighbours in direction order E, N, W, S
  (`env.neighbor_index()`); columns 0–43 = that intersection's 44-value observation, column
  44 = presence flag (1 = real, 0 = boundary; a missing slot's row is all zeros).
- `encode_all(obs) -> (n, 5, 45)`; `encode_transition(tr) -> (s, s′)` built from `tr.obs` and
  `tr.next_obs` (both hold all intersections, so no env change is needed).
- Used by **every** UG-MARL variant (−attention ignores slots 1–4), so the replay content is
  identical across ablations. Memory: 100 000 × 2 × 225 float32 ≈ 180 MB.
- Neighbour observations are the *observed* ones (after noise / dropout), as for any controller.

### D3. Network — `UGMARLNet(obs_dim=44, n_actions=8, attn_dim=d, head_hidden=128, heads=K, use_attention=True)`

Input `(G, B, 5, 45)` → output `(G, B, K, A)` (same contract as `MLPQNet`; G = 1).

```
e_j   = ReLU(W_e s_j + b_e) + p_slot(j)                 j ∈ {self, E, N, W, S},  e_j ∈ R^d
α_ij  = softmax_j ( (W_q e_i)ᵀ (W_k e_j) / √d )         masked: absent neighbours get −∞
c_i   = Σ_j α_ij W_v e_j
z_i   = [ e_i ‖ c_i ]  ∈ R^{2d}                          (use_attention=False: z_i = e_i ∈ R^d)
Q_k(s_i, ·) = W2_k ReLU(W1_k z_i + b1_k) + b2_k          k = 1..K, hidden 128
```
- Heads are stored stacked `(K, in, out)` and evaluated with one `einsum` (as
  `StackedMLPQNet`), each initialised like an independent `nn.Linear`.
- `forward(x, return_attention=False)`; with `True` also returns α `(G, B, 5)` (self first).
- Self is always present, so the softmax is never over an empty set.
- Size (d = 64, K = 5): ≈ 15k encoder + 5 × ≈ 17.5k heads ≈ 105k parameters (exact count
  comes from `n_params` and goes in the table).

### D4. Ensemble training — changes in the core (backward compatible)

- `ReplayBuffer(..., extras: dict[name -> (shape, dtype)] | None = None)`; `add(..., **extras)`;
  `sample` returns them. No extras → behaviour and RNG stream unchanged (DQN / IDQN results
  must stay bit-identical; regression test).
- `QLearner` params: `heads` (K), `mask_prob` (0.8). When K > 1, each stored transition gets
  `m ∈ {0,1}^K`, `m_k ~ Bernoulli(mask_prob)`, from its own stream (`seed + 40_000`).
- Loss (per-head Double DQN target is already `(G, B, K)` in `td_target`):

```
y_k   = c·r + γ^(d_s/Δt) · Q_target,k(s′, argmax_a′ Q_online,k(s′, a′))
L     = (1/K) Σ_k [ Σ_b m_bk · Huber(Q_k(s_b, a_b) − y_bk) / max(1, Σ_b m_bk) ]
```
  The 1/K mean keeps the shared encoder's gradient scale independent of K (Osband's 1/K
  normalisation). K = 1 → mask ≡ 1 and L is the Phase 5 loss.
- Update statistics add `sigma_mean` (batch mean of the head std at the taken action).
- `QLearner.load`: the "same intersections" check applies only to non-shared agents (IDQN);
  shared agents check `obs_dim` / `phase_ids` instead, so DQN and UG-MARL load on Hangzhou.

### D5. Agent — `UGMARLAgent(env, params, seed)` (`name = "ugmarl"`)

```
μ(a)   = (1/K) Σ_k Q_k(s_i, a)          a* = argmax_a μ(a)
σ(s_i) = sqrt( (1/K) Σ_k (Q_k(s_i, a*) − μ(a*))² )         (population std; 0 when K = 1)
a_i    = MaxPressure(observed lane counts of i)   if gate on and σ(s_i) > τ
         a*                                        otherwise
```
- `explore=True` (training): ε-greedy on μ; **the gate is never applied**.
- `explore=False`: the gate is applied iff `use_gate and use_ensemble and tau is not None`.
  Validation episodes inside `train.py` run with `tau = None` → checkpoint selection is on the
  **ungated** policy (the gate is calibrated afterwards on the selected checkpoint).
- Fallback = an internal `MaxPressureAgent(env.features)` fed `info["lane_count"]` — the same
  rule, Δt and observed detectors as the baseline (protocol §2).
- `last_explanations()` →
  `{"fallback": {iid: bool}, "sigma": {iid: float}, "tau": float | None,
    "greedy_phase": {iid: int}, "mp_phase": {iid: int}, "attention": {iid: (5,) array},
    "q_mean": {iid: (8,) array}}` — `run_episode` already reads `"fallback"`.
- Flags (`agents.ugmarl`): `use_attention`, `use_ensemble` (False → K forced to 1, gate off),
  `use_gate`. `variant_name(flags)` → `ugmarl`, `ugmarl-noattn`, `ugmarl-noens`,
  `ugmarl-nogate`; used as the `model` in the experiment log and in run directory names.
- Checkpoint stores flags, K, d, head size; `load` refuses a mismatching architecture
  (`use_gate` may differ — that is how −gate reuses the full model's weights).
- τ source at evaluation: `agents.ugmarl.gate.tau` (number) or `agents.ugmarl.gate.file`
  (`auto` = `gate.json` in the checkpoint's run directory, resolved per seed). Gate requested
  but no τ available → error (never a silent ungated run).
- Registry: `AGENTS` and `LEARNED_AGENTS` gain `"ugmarl"`; `train.py` / `tune.py` /
  `evaluate.py` then accept it unchanged.

### D6. Ablation definitions (6.5, protocol §4)

| Variant | What changes | Training |
|---|---|---|
| full | attention + K heads + gate (τ from `gate.json`) | 5 seeds |
| −gate | **same checkpoints** as full, `use_gate=false` at evaluation | none |
| −attention | z = eᵢ only (no neighbour information); K heads + gate with its own τ (same p) | 5 seeds |
| −ensemble | K = 1, so no σ and no gate; attention kept | 5 seeds |

−attention and −ensemble reuse the full model's selected (K, d) and learning settings — no
separate tuning (stated in the report).

### D7. Explanation logging (6.4) — `eval/explain.py`

`DecisionRecorder.record(t, info, actions, agent.last_explanations())` → one row per *ready*
intersection: `t, iid, sigma, tau, fallback, greedy_phase, mp_phase, phase_taken,
attn_self, attn_E, attn_N, attn_W, attn_S, q_gap (μ best − μ second), queue_sum (observed)`.
`run_episode(env, agent, recorder=None)`; `evaluate.py --log-decisions` writes
`results/eval/<run>/decisions_s<seed>.csv`. Phase 8 figures (attention heatmaps, gate-firing
analysis) read these files. No recorder → no overhead.

### D8. Training search (6.7a) — `configs/search/ugmarl.yaml`

```yaml
study: ugmarl
seed: 0
sampler_seed: 2026
trials: 9                      # full 3 x 3 grid
space:
  agents.ugmarl.heads: [3, 5, 10]
  agents.ugmarl.attn_dim: [32, 64, 128]
selection:
  metric: best_val_avg_travel_time_s      # ungated greedy policy, clean validation
  tie_tolerance_s: 1.0
  tie_break: [agents.ugmarl.heads, agents.ugmarl.attn_dim]   # ascending: smaller model wins
```
- `select_best` gains a `tie_break` argument (default `["train.lr"]`, so the Phase 5 rule is
  unchanged). Committed **before** the first trial.
- Fixed, not tuned: lr 1e-4, γ 0.8, batch 64, ε-decay 0.3 (baseline winner t6), head hidden
  128, mask_prob 0.8, everything else from `configs/base.yaml` `train:`, 100 episodes.
  Disclosed: these were tuned for DQN / IDQN (both chose t6), not for UG-MARL; UG-MARL gets 9
  runs on its own structural parameters vs. 8 for each baseline.
- Winner copied into `configs/ugmarl.yaml`; outcome locked regardless of score (U14).

### D9. Gate calibration (6.3, 6.7b) — `configs/search/ugmarl_gate.yaml` + `scripts/calibrate_gate.py`

```yaml
study: ugmarl_gate
percentiles: [80, 90, 95, 99]
conditions:                    # validation flow only; observation perturbations (protocol §6)
  - {}                                  # clean
  - {robustness.noise_std: 0.1}
  - {robustness.noise_std: 0.2}
  - {robustness.dropout_p: 0.1}
  - {robustness.dropout_p: 0.2}
selection:
  metric: mean val avg_travel_time_s over seeds 0-4 and the 5 conditions (equal weights)
  tie_tolerance_s: 1.0         # within 1 s of the best -> smaller p (more fallback)
```
Procedure for a set of seed-models (run tag `final`), all on Jinan **val**:
1. **σ pass:** one clean, ungated, greedy val episode per model with a recorder → σ of every
   decision → `sigma_val.csv`; candidate τ_p,seed = p-th percentile of that model's σ.
2. **Reference pass:** ungated episode per model × condition (5 × 5 = 25; the clean ones are
   step 1). Kept as the −gate reference and as the data for the σ validity check (D10).
3. **Sweep:** gated episode per model × p × condition (5 × 4 × 5 = 100), `track=light`,
   parallel workers.
4. **Select** p\* by the rule above; write `gate.json` into each model's run directory:
   `{p, tau, sigma_percentiles, n_decisions, study, git_commit}`; write
   `docs/results/gate_calibration.{csv,md}` (rows p × condition: travel time mean ± std,
   fallback rate, max wait; plus the ungated reference rows).
- The ungated policy is a **reference row, not a candidate**: the gate is part of the method
  and its cost / benefit is what the −gate ablation reports. If every p is worse than the
  ungated reference on this score, that is written down as a finding (U14), not tuned away.
- Demand surge is **not** used for calibration (it changes the flow, not the detectors); it
  stays an untuned test condition in 7.3.
- The script refuses any split other than `val` (exit 2) and never passes `allow_final`.
- −attention models get their own τ from the same p\* (`--tag` of their runs).

### D10. σ validity check (extra) — `scripts/sigma_check.py`

From the D9 reference-pass decision logs (no new simulation except one transition-collecting
val episode per model):
- σ distribution (per model and pooled) under clean / noise 0.1 / 0.2 / dropout 10 / 20 % —
  does σ rise when inputs degrade?
- Spearman correlation between σ and |TD error of the mean Q| on validation transitions
  (collected with `agent.training = False`, so nothing enters replay).
- → `docs/figures/ugmarl/sigma_validity.{png,md}` (axis labels, units, caption, interpretation)
  and `docs/results/sigma_validity.csv`. Wording: **associations, not causes**.

### D11. Phase 6 outputs

- `scripts/val_table.py --agents fixed_time max_pressure dqn idqn ugmarl ugmarl-nogate
  ugmarl-noattn ugmarl-noens --out docs/results/val_phase6` → table incl. fallback rate.
- `scripts/plot_training.py` → `docs/figures/training/learning_curves_phase6.{png,md}` (UG-MARL
  and trained ablations next to DQN / IDQN, MP / FT reference lines).
- `docs/methodology.md` (D12).

### D12. Math write-up (6.6) — `docs/methodology.md`

1. Problem as a networked SMDP: state, action, reward −|Pᵢ|, Δt, yellow, truncation.
2. Q-learning target used (Double DQN, SMDP discount, reward scale) — as implemented.
3. Neighbour attention: equations of D3, masking, what αᵢⱼ is (and is not).
4. Ensemble: bootstrap masks, per-head target, loss of D4; σ as an estimate of *epistemic*
   uncertainty (disagreement from different data subsets + initialisation); limits (shared
   encoder correlates heads; σ is not a calibrated probability).
5. Gate: decision rule, τ as a validation percentile, expected fallback rate ≈ 1 − p/100 on
   clean validation by construction.
6. Stability argument, stated as a **proposition with assumptions**: (a) Max-Pressure
   stabilises queues for every demand inside the capacity region under Varaiya's (2013)
   assumptions (unbounded queue storage, known turning ratios, i.i.d. demand); (b) if the gate
   fires at every decision the closed loop *is* Max-Pressure, so (a) applies; (c) for the
   mixed policy: a Lyapunov-drift sketch — each learned decision adds at most a bounded term
   to the drift, so the bound degrades with the share of learned decisions — given as a
   sketch, **not** claimed as a theorem. Explicit "what this does not guarantee" paragraph
   (finite lanes, Δt = 15 s variant of Max-Pressure, σ may be low on a bad state). The
   empirical check is max per-vehicle wait vs. Max-Pressure and vs. −gate (H2).
7. Complexity: parameters and per-decision cost as a function of K and d.

## Justification (report / viva)

| Choice | Why | Rejected |
|---|---|---|
| Shared weights | zero-shot transfer, data efficiency, Phase 5 evidence | per-intersection nets (no Hangzhou, 12× params); id embedding (not transferable) |
| Shared encoder + K MLP heads | near-constant cost in K, one attention map | K full networks (K× compute); K linear heads (correlated, small σ) |
| 1 GAT layer, 1 head | interpretable weights, 1-hop is enough to test H3 on 3×4 | multi-head (blurred explanations); 2 layers (whole-graph replay) |
| σ = head std at a\* | scale handled by percentile τ; one formula | vote disagreement (only K + 1 levels) |
| τ percentile on clean + perturbed val | gate must pay off when inputs are unreliable; validation only | clean-only (degenerate), fixed p (arbitrary), safety-constrained (new constant) |
| Gate at evaluation only | clean −gate ablation, no retraining for τ | gate in training (entangles weights and gate) |
| ε-greedy on μ | identical exploration to baselines | head sampling (two changes at once) |
| Bernoulli(0.8) masks | standard, cheap | randomized priors (extra component); init-only (σ may collapse) |

## Files

| File | Action | Content |
|---|---|---|
| `src/trafficrl/agents/core/encoders.py` | modify | `NeighborhoodEncoder` (D2) |
| `src/trafficrl/agents/core/replay.py` | modify | optional `extras` columns (D4) |
| `src/trafficrl/agents/core/qlearner.py` | modify | `heads`, `mask_prob`, masked loss, mask RNG, `sigma_mean`, relaxed `load` (D4) |
| `src/trafficrl/agents/core/__init__.py` | modify | exports |
| `src/trafficrl/agents/ugmarl/{__init__,network,agent,gate}.py` | create / modify | D3, D5 |
| `src/trafficrl/agents/__init__.py` | modify | registry `ugmarl` |
| `src/trafficrl/eval/explain.py` | create | `DecisionRecorder` (D7) |
| `src/trafficrl/eval/runner.py`, `scripts/evaluate.py` | modify | `recorder`, `--log-decisions`, gate-file resolution, variant model name |
| `src/trafficrl/eval/report.py`, `scripts/val_table.py`, `scripts/plot_training.py` | modify | UG-MARL variants, fallback-rate column, `--out` |
| `src/trafficrl/train/loop.py` | modify | run name / log model from `variant_name`; log `sigma_mean` |
| `src/trafficrl/train/tune.py`, `scripts/tune.py` | modify | `tie_break`, agent choice `ugmarl` |
| `scripts/calibrate_gate.py`, `scripts/sigma_check.py` | create | D9, D10 |
| `configs/base.yaml` | modify | `agents.ugmarl` block |
| `configs/ugmarl.yaml`, `configs/ugmarl_noattn.yaml`, `configs/ugmarl_noens.yaml` | create | training configs (full + trained ablations) |
| `configs/eval_ugmarl.yaml` | create | val evaluation of UG-MARL variants |
| `configs/search/ugmarl.yaml`, `configs/search/ugmarl_gate.yaml` | create | D8, D9 — committed before the first run of each |
| `tests/agents/test_ugmarl_network.py`, `test_ugmarl_agent.py`, `test_gate.py`; additions to `tests/agents/test_core.py`, `tests/eval/test_evaluate.py`, `tests/train/test_tune.py`; `tests/eval/test_explain.py`, `tests/scripts/test_calibrate_gate.py` | create / modify | see Tests |
| `docs/methodology.md` | create | D12 |
| `docs/results/val_phase6.*`, `gate_calibration.*`, `sigma_validity.csv`; `docs/figures/ugmarl/*`, `docs/figures/training/learning_curves_phase6.*` | create | generated outputs |
| `docs/evaluation_protocol.md` §9 | modify | clarification row: τ rule (clean + perturbed val), ablation definitions, gate off during training / checkpoint selection |
| `docs/how_it_works.md`, `.claude/reference/{api,architecture,testing}.md`, `CLAUDE.md`, `README.md` | modify | implemented UG-MARL sections, commands; fix Algorithm 1 step 2 ("normalisation statistics" → capacity scaling) and step 4–5 (τ rule) |
| `PRD.md` §9 | modify | close "τ selection method" open question |
| `ROADMAP.md` | modify | tick 6.1–6.7; Decision Log rows (U1–U15, platform) |
| `docs/experiment_log.csv` | append | see Experiment log |
| `worklog/017-2026-11-11-phase6-ugmarl.md`, `worklog/README.md` | create / modify | work log |

## Config (`configs/base.yaml` additions)

```yaml
agents:
  ugmarl:
    use_attention: true      # ablation flag (−attention)
    use_ensemble: true       # ablation flag (−ensemble: heads -> 1, gate off)
    use_gate: true           # ablation flag (−gate); applied at evaluation only
    heads: 5                 # K — tuned (configs/search/ugmarl.yaml)
    attn_dim: 64             # d — tuned
    head_hidden: 128         # fixed, matches the baselines' hidden width
    mask_prob: 0.8           # Bernoulli bootstrap mask — fixed, not tuned
    gate:
      tau: null              # number, or null -> read `file`
      file: auto             # gate.json next to the checkpoint (written by calibrate_gate.py)
```
`configs/ugmarl.yaml`: `base: base.yaml`, `train.agent: ugmarl`, the four inherited t6 values,
tuned `heads` / `attn_dim`. Seeds: training seed fixes init, exploration (`+20 000`), replay
(`+30 000`), mask (`+40 000`) streams and CityFlow; evaluation seed s loads model s and its τ.

## Data-leakage check

- Training: `train` split only (`train.py` guard unchanged). Checkpoint selection, grid
  selection, σ percentiles, p selection and the σ validity check: `val` split only.
  `calibrate_gate.py` / `sigma_check.py` refuse other splits and never pass `allow_final`.
- Perturbed validation = noise / dropout applied to the **validation** flow's observations;
  no test or Hangzhou flow is simulated in Phase 6.
- Nothing is fitted on data except network weights (train) and τ (val percentile).
- Validation transitions never enter replay (`training = False`; existing test extended to
  UG-MARL and to the TD-error episode of D10).
- Validation is now used for checkpoint, (K, d), and p → val scores are optimistic; test
  (Phase 7) stays untouched. `git grep allow_final` must still show only `make_env` and
  `evaluate.py --final`.

## Tests (`pytest`; simulator tests `slow`, the rest on `fake_engine`)

- **Encoder:** shape `(n, 5, 45)`; slot order E, N, W, S matches `env.neighbor_index()`;
  boundary slots zero with flag 0; `encode_transition` equals the row of `encode_all`.
- **Network:** output `(G, B, K, A)`; α sums to 1, is 0 on masked slots, a corner
  intersection never attends to an absent neighbour; changing a neighbour's state changes Q
  with attention and leaves it unchanged with `use_attention=False`; stacked heads equal a
  loop of K separate heads; a loss on head k gives zero gradient to other heads' parameters
  and non-zero to the encoder; parameter count formula.
- **Core:** replay extras round trip and wrap-around; no extras → same samples as before for
  a fixed seed; masked loss equals a hand computation (incl. an all-zero mask column);
  K = 1 → loss identical to the Phase 5 loss; DQN / IDQN regression: with no extras the
  sampled batches and the loss value equal those of the pre-change code path, computed in
  the test itself (no stored hash — floating point differs between WSL and native Ubuntu).
- **σ / gate:** σ against a hand-made Q tensor; K = 1 → σ = 0; σ > τ → action equals
  `MaxPressureAgent`'s choice for the same `info`, σ ≤ τ → argmax μ; τ = +∞ never fires,
  τ = −1 always fires and the whole episode equals Max-Pressure's actions; gate not applied
  with `explore=True`; gate requested without τ → error; `use_ensemble=False` → no fallback.
- **Agent:** interface conformance; `last_explanations` keys and shapes; `variant_name`;
  save / load round trip; load refuses a different K / d / attention flag but accepts a
  different `use_gate`; shared agent loads on an env with a different number of intersections;
  determinism (same seed → same weights, different seed → different).
- **Gate calibration:** percentile thresholds on a known σ array; selection rule incl. the
  tie → smaller p; `gate.json` schema; script refuses `test` / `heldout`.
- **Recorder / evaluate:** one row per ready intersection; `fallback_rate` in `metrics.csv`
  equals the recorder's share; `--log-decisions` writes one file per seed; per-seed gate
  file resolution.
- **Tuning:** `tie_break` ordering; default still `train.lr`; 9 trials = the full grid, no
  duplicates.

## Steps (each can be run and checked on its own)

1. [ ] Protocol §9 clarification row + PRD §9 (close the τ question). The ROADMAP Decision
   Log row (U1–U15) and worklog/017 (`in-progress`) were already added at planning time
   (11 Nov) — continue that worklog entry, do not create a new one.
2. [ ] Core changes: replay extras, masked loss, mask RNG, relaxed `load` + tests; confirm
   DQN / IDQN regression test and the full existing suite pass.
3. [ ] `NeighborhoodEncoder` + `UGMARLNet` + tests.
4. [ ] `UGMARLAgent` (μ, σ, gate, flags, explanations, checkpoint), `gate.py`, registry,
   `configs/base.yaml` block, `configs/ugmarl*.yaml` + tests.
5. [ ] `DecisionRecorder`, `run_episode(recorder=)`, `evaluate.py --log-decisions`, gate-file
   resolution + tests.
6. [ ] Smoke: `train.py --config configs/ugmarl.yaml train.episodes=2 train.val_every=1`
   for full / −attention / −ensemble; evaluate on val with `agents.ugmarl.gate.tau=0.0`
   (always fires → travel time must equal Max-Pressure's 326.11 s exactly) and with a huge τ
   (must equal the ungated run). Short device benchmark (CPU vs CUDA, K = 10, d = 128, 3
   episodes) → keep or change `train.device`; determinism check on the chosen device.
7. [ ] **Learning sanity run** (seed 0, K = 5, d = 64, 100 episodes; a dev run, not a trial):
   val travel time must fall below Fixed-Time (434.84 s) and approach the DQN range; σ must
   be non-zero and not collapse to ~0. If not → debug before tuning; log dead ends.
8. [ ] **Platform gate (D0)**; write and **commit** `configs/search/ugmarl.yaml` and
   `configs/search/ugmarl_gate.yaml`; `tune.py` changes + tests. If on native Ubuntu: re-run
   the DQN / IDQN 5-seed finals and val evaluation there first.
9. [ ] Grid study: `tune.py --search configs/search/ugmarl.yaml --agent ugmarl --workers 4`
   → apply the selection rule → copy the winner into `configs/ugmarl*.yaml`, commit.
10. [ ] Final runs, seeds 0–4: full, −attention, −ensemble (`tune.py --config … --seeds 0 1 2 3 4`),
    15 runs, ≤ 5 workers (RAM ≈ 1 GB per process).
11. [ ] `scripts/calibrate_gate.py` for the full and the −attention models → `gate.json`
    files, `docs/results/gate_calibration.*`; `scripts/sigma_check.py` → figure + table.
12. [ ] Val evaluation, full tracking, 5 seeds, `--log-decisions`: full, −gate, −attention,
    −ensemble → `val_table.py` (`val_phase6`), `plot_training.py`.
13. [ ] `docs/methodology.md` (D12), checked equation by equation against the code.
14. [ ] Docs (api / architecture / testing / how_it_works / CLAUDE / README), tick ROADMAP
    6.1–6.7, finish worklog/017 (file table cross-checked with `git status`), commits per
    `.claude/commands/commit.md` (no Claude co-author line).

Rough compute (4–5 parallel CPU processes): grid ≈ 2–4 h, finals ≈ 3–5 h, calibration ≈ 0.5 h.
Steps 9–10 can run overnight; step 13 can be written while they run.

## Verification

```bash
pytest && pytest -m slow && ruff check src tests scripts
python scripts/train.py --config configs/ugmarl.yaml --run-name smoke_ugmarl train.episodes=2 train.val_every=1
python scripts/evaluate.py --config configs/eval_ugmarl.yaml --agent ugmarl --split val --seeds 0 \
    --checkpoint results/train/smoke_ugmarl/checkpoints/last.pt agents.ugmarl.gate.tau=0.0
python scripts/tune.py --search configs/search/ugmarl.yaml --agent ugmarl --workers 4
python scripts/tune.py --config configs/ugmarl.yaml --seeds 0 1 2 3 4 --workers 5
python scripts/calibrate_gate.py --search configs/search/ugmarl_gate.yaml --config configs/eval_ugmarl.yaml --tag final --workers 4
python scripts/sigma_check.py --tag final
python scripts/evaluate.py --config configs/eval_ugmarl.yaml --agent ugmarl --split val --log-decisions \
    --checkpoint 'results/train/ugmarl_final_s{seed}/checkpoints/best.pt'
python scripts/val_table.py --out docs/results/val_phase6 && python scripts/plot_training.py
```
Expected:
- Smoke run writes every file `train.py` writes for DQN; one experiment-log row, model `ugmarl`.
- τ = 0 evaluation: `fallback_rate` = 1.0 and avg travel time = **326.11 s** (Max-Pressure on
  val); huge τ: `fallback_rate` = 0 and metrics equal the `use_gate=false` run.
- `trials.csv` has 9 rows; both search files' commits predate their first run.
- Each final run directory of full / −attention has a `gate.json`; clean-val fallback rate of
  the gated model ≈ 1 − p\*/100 (within a few points — the gate changes the visited states).
- `val_phase6.md`: FT 434.8 and MP 326.1 reproduced; DQN / IDQN rows equal Phase 5 (same
  platform) or are re-run (native); UG-MARL rows with mean ± std and 95 % CI over 5 seeds.
- `git grep -n allow_final src scripts` → only `make_env` and the `evaluate.py --final` path.

## Experiment log (rows appended to `docs/experiment_log.csv`)

| Runs | Rows | Split | Notes column |
|---|---|---|---|
| Smoke + sanity + benchmark-free dev runs | 1 per run | train (val metric) | `smoke` / `sanity` |
| Grid study | 9 | train (val metric) | study `ugmarl`, trial id |
| Finals | 15 | train (val metric) | variant, tag `final` |
| Gate calibration | 25 reference + 100 sweep (+ 5 σ pass) per calibrated variant | val | study `ugmarl_gate`, p, condition |
| Val evaluation | 5 per variant × 4 variants | val | variant, τ source |

No row has split `test` or `heldout`.

## Risks / open questions

- **UG-MARL ≤ DQN on clean val** (plausible: DQN is already strong at 299.8 s). Locked as is
  (U14); H1 / H3 may be negative results; next step decided with the user.
- **Gate costs travel time on clean flows** (Max-Pressure is 26 s worse than DQN on val). The
  calibration table shows the trade-off; H2 is judged on max wait and noise degradation.
- **σ collapses or is uninformative** (shared encoder, 100 episodes). Seen early in step 7 and
  measured in D10; mask_prob is fixed, so the remedy would need a Decision Log entry.
- **σ under noise is not higher** → the gate would fire at the wrong times; D10 reports it.
- **Compute / RAM:** K = 10, d = 128 runs are the slowest; ≤ 5 workers on the 7.7 GB WSL VM.
- **Platform:** depends on 0.1b; mixing WSL and native numbers in one table is not allowed.
- **Hangzhou transfer (Phase 7):** relies on the canonical lane order and 8 phases there too;
  verify with the roadnet only (no flow) before 7.4 — out of scope here, noted for Phase 7.
- **Stability claim:** only (a)–(b) are rigorous; (c) is a sketch. Wording in the report must
  not say "provably stable" for the mixed policy.

## Deviations during execution

(none yet)

## Worklog

Executing this plan ends with `worklog/017-2026-11-11-phase6-ugmarl.md` (What / Why / How /
Result / Files changed per file with what + why / Commits / Next) and its row in
`worklog/README.md`; failures and dead ends are logged too.
