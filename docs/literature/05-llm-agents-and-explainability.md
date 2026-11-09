# 05 — LLM traffic agents and explainability (short notes)

ROADMAP 2.5 + explainability claims · Scope: **short notes only** (user decision, 9 Nov).
LLM agents are out of scope for building (PRD §3). They are covered here only to position UG-MARL and to
answer viva questions.
Legend: ✓ verified online · † from memory.

---

## A. LLM-based "agentic" traffic control

### LLMLight ✓ (`lai2025llmlight`, KDD 2025, arXiv 2312.16044)
Lai, Xu, Zhang, Liu, Xiong. An LLM gets a prompt with the task, current traffic conditions and domain
knowledge, reasons with chain-of-thought, and outputs the next phase. The authors fine-tune
**LightGPT**, a smaller traffic-specialised backbone, to cut cost. They evaluate on many real and synthetic
datasets (including the CityFlow Jinan/Hangzhou family) against transportation and RL baselines.
Code: usail-hkust/LLMTSCS.

**Why we don't build on it (PS16 limitations, in its own words):**
- **Cost and latency.** One LLM call per intersection per decision, which is orders of magnitude more compute than a
  Q-network forward pass. PS16 lists "high computational cost" as a limitation to fix, not add.
- **No guarantees.** LLM outputs can be wrong or inconsistent (hallucination). Reasoning text is
  not a certificate of correctness.
- **Uncertainty.** An LLM's verbal confidence is not a calibrated epistemic estimate.

**What we take from it.** The *appeal* of LLM agents is explanation in natural language. UG-MARL
offers a cheaper, measurable form of transparency: "the agent deferred to Max-Pressure because σ was
high", logged per decision. Position this in one paragraph of related work.

---

## B. Explainability: what our attention weights and gate statistics can and cannot claim

### Attention ≠ explanation (by default)
- Jain & Wallace † (`jain2019attention`, NAACL 2019), "Attention is not explanation": attention weights
  often disagree with gradient-based importance, and very different attention distributions can give
  the same prediction.
- Wiegreffe & Pinter † (`wiegreffe2019attention`, EMNLP 2019), "Attention is not *not*
  explanation": whether attention explains depends on the model and the test. They propose checks.

**For UG-MARL.** Present α_ij as "which neighbours the encoder weighted", an **association**
(CLAUDE.md rule). One cheap faithfulness check strengthens the claim: zero out the
highest-α neighbour's input and measure the change in Q / chosen action, versus zeroing the
lowest-α neighbour. If the high-α perturbation matters more, the weights are at least
partially faithful.

### SHAP † (`lundberg2017unified`, NeurIPS 2017)
Shapley-value feature attributions with model-agnostic estimators (KernelSHAP) and fast variants
for deep models (DeepSHAP).
- **In TSC:** Rizzo, Vantini, Chawla ✓ (`rizzo2019reinforcement`, ITSC 2019,
  DOI 10.1109/ITSC.2019.8917519) used SHAP to explain which detectors drove a policy-gradient
  agent's phase choices at an 11-phase roundabout with real data. Later work found some explanations
  unintuitive and affected by overfitting.
- **For UG-MARL (optional, ROADMAP scope controls):** SHAP on the *gate decision* ("which lane
  features raise σ?") is more novel and more useful than SHAP on Q-values.

### XRL survey †
Puiutta & Veith (`puiutta2020explainable`, CD-MAKE 2020): a taxonomy of explainable RL (intrinsic
vs post-hoc, global vs local). UG-MARL's gate log is an **intrinsic, local** explanation. Its
attention is intrinsic but needs a faithfulness check.

### Gate statistics as explanation (our own contribution)
Per intersection and per time of day: gate-firing rate, σ distribution, and ATT when the gate fired
vs did not. These are directly measurable, cheap, and tied to the decision, so they make the strongest
explainability story we have. Still report them as associations (for example "the gate fired more
during the surge"), not causes.
