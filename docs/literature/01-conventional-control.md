# 01 — Conventional traffic signal control

ROADMAP 2.1 · Rubric R2 · Guideline G2
Legend: ✓ metadata verified online (9 Nov 2026) · † metadata from memory, check before the final report ·
*(unverified)* = a number not checked against the PDF.
**Scope rule (9 Nov):** only papers from 2016 onward are reviewed. Pre-2016 originals of methods we
implement are cited once as **background** (§0) and not reviewed.

**Why this file matters for UG-MARL.** Fixed-Time and Max-Pressure are our conventional baselines,
and Max-Pressure is also the **fallback controller inside the gate**. The safety story of UG-MARL
depends on exactly what Max-Pressure guarantees and what it does not (§2, our own analysis).

---

## 0. Background methods we implement (pre-2016, cited not reviewed)

- **Fixed-Time**: Webster (1958) † (`webster1958traffic`). Fixed cycle and splits. Cycle length
  $C_0 = (1.5L + 5)/(1 - Y)$, green split ∝ critical flow ratio y_i.
- **Max-Pressure**: Varaiya (2013) ✓ (`varaiya2013max`). At each intersection and each slot, choose
  the phase S maximising $\sum_{(l,m)\in S} c(l,m)\,w(l,m)$ with movement pressure
  $w(l,m) = x(l,m) - \sum_p r(m,p)\,x(m,p)$. Theorem: if demand lies strictly inside the capacity
  region, queues are stable (bounded mean). The proof uses $V(x)=\sum x^2$ as a Lyapunov function.
- Deployed adaptive systems (SCOOT, SCATS) and actuated / self-organising control (SOTL) are
  summarised in the post-2015 survey `wei2019survey` ([03](03-rl-for-traffic-signal-control.md) §0).
  We cite that survey rather than the originals.

**Implementation tip.** In the CityFlow lineage (CoLight, PressLight, LibSignal), "Fixed-Time"
usually means equal splits with a fixed phase duration and is not Webster-optimised. Say which one we use.
A Webster-tuned version on the **train** flows is a fairer baseline and uses no test information.
LibSignal ✓ (`mei2024libsignal`) has reference implementations of FixedTime, SOTL and MaxPressure.

---

## 1. Post-2015 work on Max-Pressure (reviewed)

### 1.1 Efficient-MP / Advanced-MP ✓
`zhang2022expression` — Zhang et al., ICML 2022.

- **Method.** Advanced-MP counts *effective running* (approaching) vehicles as well as queued ones and
  uses an "efficient pressure" (in − out averaged per movement) to decide whether to keep or switch
  the phase.
- **Result (authors' claim).** Advanced-MP, with no learning, is competitive with many RL methods on the
  CityFlow datasets *(numbers unverified)*.
- **What we learn → risk to H1.** If our MP baseline is the plain version, reviewers may call
  it weak. Consider adding Advanced-MP as a stronger conventional baseline (cheap: it is a rule).

### 1.2 Lyapunov-consistent back-pressure + Double-DQN ✓
`ma2022lyapunov` — Ma, Wang, Li, Mahmoudzadeh, Zhang, arXiv 2210.02612 (2022, rev. 2024).

- **Method.** Derives back-pressure (and DORAS, using the sum of queues as the Lyapunov function) from
  Lyapunov control theory, then uses the BP term as the **reward** of a Double-DQN controller.
- **Result.** Lower average waiting time than conventional and RL methods in corridor and network
  scenarios, including mixed traffic with freight.
- **What we learn.** The pressure-reward ↔ Lyapunov-drift link is principled, not ad hoc. This
  is the justification for using −pressure as our reward so both gate branches pursue the same objective.

### 1.3 Com-MP ✓
`zhu2026commp` — Zhu & Tan, arXiv 2610.31037 (Oct 2026).

- **Method.** RL learns a context-dependent *composition* of several MP formulations, while
  decentralised MP still selects the phase. Learning is confined to an adaptation layer.
- **Claim.** Strong stability for the composite controller.
- **What we learn.** This is the closest "keep MP's guarantee, add learning" work. Com-MP keeps the
  guarantee *by construction* and gives up flexibility. UG-MARL lets RL choose and needs an extra
  trigger to keep a guarantee (§2). Contrast in [06](06-gap-analysis.md).

---

## 2. Our analysis: what the Max-Pressure guarantee does and does not cover

(This is design analysis for UG-MARL, built on the background theorem in §0. It is not a paper review.)

| Theorem assumes | Reality in CityFlow (Jinan/Hangzhou) | Consequence |
|---|---|---|
| Point queues, unlimited buffers (no spillback) | Finite lanes; spillback/gridlock possible | Stability can fail under heavy demand |
| Known turn ratios r and saturation flows c | CityFlow implementations use the simplified w = q_in − q_out | Not exactly Varaiya's controller |
| No lost time when switching phases | Yellow/all-red each switch | Too-frequent switching wastes capacity; Δt matters |
| Phase held for a whole fixed slot | Decision every Δt seconds | OK if Δt matches the slot |
| Demand strictly inside the capacity region | Peak flows may be near or over capacity | No guarantee in oversaturation |

**Three caveats we must not overclaim:**

1. **Queue stability is not a per-vehicle bound.** MP bounds *mean queue lengths*. It does
   **not** bound any individual vehicle's maximum wait, and a low-pressure movement can be starved for long periods.
   The PRD §1c "minimum acceptable outcome" (UG-MARL no worse than MP on **max wait**, "the gate
   should guarantee this") is **not** implied by the theorem. It is an empirical claim.
   *Action:* reword §1c, or add a max-red-time constraint to the env (bounds wait by construction).
2. **A switched controller does not inherit stability automatically.** The theorem holds when MP
   acts at *every* slot. UG-MARL uses MP only when σ(s) > τ. In low-σ states the RL policy can have
   positive Lyapunov drift, and the RL agent may be confident *and* wrong ([04](04-uncertainty-and-safe-rl.md) §3),
   so the combined system has no guarantee. *Design option:* fall back to MP when σ(s) > τ **or**
   V(x) (or the local max queue) > V_max. Outside {V ≤ V_max}, MP is always active and gives
   negative drift, so queues stay bounded. This is the Simplex-style switching argument
   ([04](04-uncertainty-and-safe-rl.md) §5.4). It costs one hyperparameter, chosen on validation.
3. **The guarantee is for the model, not for CityFlow.** Say "provably stable under the
   store-and-forward model; empirically checked in simulation."

---

## Takeaways for UG-MARL

1. Max-Pressure is the right fallback: decentralised, parameter-free, same observations, and a
   queue-stability theorem under stated assumptions.
2. The theorem gives **bounded mean queues**, not bounded per-vehicle wait. Fix the PRD §1c wording.
3. An uncertainty-only gate does **not** inherit stability. Add a queue/Lyapunov trigger if we want a
   provable bounded-worst-case claim.
4. Post-2015 work shows that (a) a pressure reward is Lyapunov-principled (Ma 2022), (b) improved
   MP is a strong baseline (Zhang 2022), and (c) MP + learning with a guarantee is an active
   topic (Com-MP 2026).
