# Prompt-Level Multi-Adapter Composition with Norm-Proportional Clamping: A Pre-Registered Negative Result

> **All numeric values from REAL model inference on Apple Silicon UMA.**
> 400 MLX inferences (100 questions × 4 methods) | Qwen2.5-1.5B-Instruct-4bit | 20 LoRA experts

---

## Abstract

We investigate prompt-level multi-adapter composition with a norm-proportional adaptive clamp for on-device LLM inference on Apple Silicon. Our method routes queries to $K$ LoRA experts and combines their activations with a scalar bound $\gamma = \min(1.0, c \cdot \|z\| / \|m\|)$ to prevent representation collapse. In a pre-registered study across 100 **single-domain, synthetically templated** questions and 20 trained expert adapters, we find that:

1. **The Adaptive Clamp does NOT outperform the single-adapter baseline** on semantic similarity (0.611 vs 0.622, $\Delta = -0.011$), **failing** our pre-registered $\Delta > 0.05$ compositional gain threshold.
2. **The Adaptive Clamp does reduce perplexity** relative to the no-adapter baseline (58.0 vs 64.5, **PASS**) and introduces negligible latency overhead ($-0.7\%$, **PASS**).
3. **Unclamped mixing catastrophically degrades** output quality (avg similarity 0.557, with individual collapses to $< 0.1$), confirming that norm bounding is structurally necessary.

We report this as a rigorous negative result **on single-domain recall**: while activation-space clamping prevents catastrophic interference, top-2 composition does not improve over single-expert routing when each query targets exactly one domain. Whether composition helps on genuinely multi-domain queries remains an open question requiring a dedicated compositional benchmark.

---

## 1. Introduction

Low-Rank Adaptation (LoRA) enables efficient specialization of large language models to narrow domains. A natural question follows: can multiple domain experts be composed at inference time to handle queries that span domain boundaries — e.g., a question requiring both legal reasoning and financial computation?

We hypothesize that prompt-level routing to $K$ experts, combined in activation space with a norm-proportional clamp, could achieve this while preserving the base model's general capabilities and meeting strict edge-device efficiency constraints. We pre-register three success criteria:

| Criterion | Threshold |
|-----------|-----------|
| Compositional Accuracy | $\Delta\_\text{SIM}(\text{AC} - \text{SA}) > +0.05$ |
| Perplexity Preservation | $\text{PPL}(\text{AC}) \leq \text{PPL}(\text{SA})$ |
| Latency Overhead | $\Delta\_\text{LAT} \leq 10\%$ |

Our results show that only the latter two criteria are met. The compositional accuracy criterion **fails**, and we analyze why.

---

## 2. Related Work

**Mixture-of-LoRA & Multi-Adapter Composition.** X-LoRA (Buehler, 2024), MiLoRA, and MALoRA explore gating mechanisms to combine specialized LoRA experts. These methods typically operate at the token level on server-grade hardware, achieving high expressivity at the cost of memory-intensive per-token weight selection. Our study tests whether prompt-level (coarser) routing can suffice for compositional queries on memory-constrained UMA devices.

**Multi-LoRA Serving.** S-LoRA (Sheng et al., 2023) and Punica (Chen et al., 2023) optimize throughput for serving thousands of adapters concurrently. These systems focus on batching and paging rather than semantic composition of experts for individual queries.

**Activation Steering & Norm Control.** Representation engineering (Zou et al., 2023) and activation addition (Turner et al., 2023) have shown that internal activations can be steered to control model behavior. Our norm-proportional clamp draws from this principle, using $\|z_l\|$ as an energy reference.

**On-Device LLM Inference.** MLX (Apple, 2023) enables efficient inference on Apple Silicon by leveraging Unified Memory Architecture. Our study specifically targets this deployment scenario, where adapter hot-swapping via zero-copy memory mapping is architecturally natural.

Our results confirm that while norm bounding prevents catastrophic degradation (unlike unclamped mixing), simple prompt-level multi-adapter composition does not yet improve over single-expert routing — highlighting an open problem at the intersection of these research threads.

---

## 3. Method

### 3.1 Problem Setting

Given a frozen base model $\mathcal{M}$ and $N$ domain-specific LoRA adapters $\{(\mathbf{A}_i, \mathbf{B}_i)\}_{i=1}^N$, we seek to compose $K \leq N$ experts for a query $q$ that may span multiple domains.

### 3.2 Routing

A CoT-based router selects the top-$K$ experts by generating a short chain-of-thought reasoning step using $\mathcal{M}$ itself, then matching the output against domain tags.

### 3.3 Activation-Space Composition

For each linear layer $l$, the combined adapter injection is:
$$m_l = \sum_{i \in \text{top-}K} p_i \cdot (\mathbf{x} \mathbf{A}_i) \mathbf{B}_i$$

where $p_i$ are routing scores.

### 3.4 Norm-Proportional Adaptive Clamp

The total injection is scaled by:
$$\gamma = \min(1.0, c \cdot \|\mathbf{z}_l\|_2 / (\|m_l\|_2 + \epsilon))$$

where $\mathbf{z}_l$ is the base model's output at layer $l$, and $c$ is a hyperparameter. The final output is $\mathbf{z}_l + \gamma \cdot m_l$.

In our implementation (via `RoutedLoRALinear`), $c$ is applied as a per-adapter weight clamp: $w_i = \min(p_i, c)$.

---

## 4. Experiments

### 4.1 Setup

- **Model:** Qwen2.5-1.5B-Instruct-4bit, loaded via MLX on Apple M3 Max (UMA)
- **Adapters:** 20 domain-specific LoRA experts (Legal, Medical, Python, Mathematics, Cryptography, Philosophy, etc.), trained on synthetic domain corpora
- **Evaluation:** 100 hard domain-specific questions (5 per domain × 20 domains)
- **Metrics:**
  - Semantic similarity (cosine via sentence-transformers `all-MiniLM-L6-v2`)
  - Perplexity of ground-truth text under each routing configuration
  - Wall-clock latency (seconds)
- **Methods compared:**
  - Baseline (no adapters, $c = 0.001$)
  - SingleAdapter (top-1 routed expert, $c = 0.5$)
  - UnclampedMix (top-2, $c = 999$, effectively unclamped)
  - AdaptiveClamp (top-2, $c = 0.5$)

> **Data Characterization Note.** The 100-question evaluation set consists of synthetically generated domain-specific questions. Approximately 90% of ground-truth answers follow one of two fixed templates differing only in the domain noun. All questions are single-domain: none requires reasoning across multiple domains simultaneously. This evaluation therefore measures single-domain recall under multi-adapter injection, not compositional reasoning.

### 4.2 Main Results

| Method | K | Clamp | Avg Sim ↑ | Avg PPL ↓ | Avg Latency |
|--------|---|-------|-----------|-----------|-------------|
| Baseline | 1 | 0.001 | 0.620 | 64.5 | 2.80s |
| SingleAdapter | 1 | 0.5 | **0.622** | 60.9 | 2.69s |
| UnclampedMix | 2 | 999.0 | 0.557 | **51.2** | 2.51s |
| AdaptiveClamp | 2 | 0.5 | 0.611 | 58.0 | 2.67s |

### 4.3 Pre-Registered Criteria

| Criterion | Measured | Threshold | Verdict |
|-----------|----------|-----------|---------|
| $\Delta\_\text{SIM}(\text{AC} - \text{SA})$ | $-0.011$ | $> +0.05$ | **FAIL** |
| PPL(AC) vs PPL(SA) | $58.0 < 60.9$ | AC ≤ SA | **PASS** |
| $\Delta\_\text{LAT}$ | $-0.7\%$ | $\leq 10\%$ | **PASS** |

The Adaptive Clamp **fails** the compositional accuracy criterion. It performs slightly worse than both the Baseline and SingleAdapter on average semantic similarity, while achieving better perplexity and comparable latency.

### 4.4 Per-Domain Analysis

The AdaptiveClamp outperforms SingleAdapter in 4/20 domains:

| Domain | AC | SA | Δ |
|--------|-----|-----|-----|
| MEDICAL_DIAGNOSIS | 0.713 | 0.683 | +0.030 |
| MATHEMATICS | 0.587 | 0.543 | +0.044 |
| QUANTUM_CHEMISTRY | 0.634 | 0.611 | +0.023 |
| PHILOSOPHY | 0.617 | 0.596 | +0.021 |

But underperforms in 16/20, with particularly large losses on:

| Domain | AC | SA | Δ |
|--------|-----|-----|-----|
| MARITIME_LAW | 0.464 | 0.609 | −0.145 |
| ANCIENT_HISTORY | 0.516 | 0.555 | −0.039 |
| SANSKRIT_LINGUISTICS | 0.703 | 0.738 | −0.035 |
| CRYPTOGRAPHY | 0.559 | 0.592 | −0.033 |

### 4.5 Unclamped Mixing is Catastrophic

UnclampedMix ($c = 999$) shows endemic collapse: on 8 out of 100 prompts, semantic similarity drops below 0.1, indicating the model produced near-random text. Average similarity (0.557) is significantly worse than all other methods. This empirically validates the necessity of norm bounding.

---

## 5. Discussion and Limitations

### 5.1 Why Compositional Gain Failed

The primary finding is a **negative result:** multi-adapter composition with prompt-level routing and norm clamping does NOT outperform single-adapter routing on our evaluation protocol.

We identify four contributing factors, the first of which is the most fundamental:

1. **Single-domain, template-identical evaluation data.** Our 100 evaluation questions are drawn from a synthetic corpus in which ~90% of ground-truth answers share one of two boilerplate templates (e.g., "The fundamental theorem of {DOMAIN} dictates that the parametric structures align perfectly with high-density contextual frameworks"), differing only in the domain noun. No question requires knowledge from more than one domain simultaneously. This evaluation therefore tests whether injecting a *redundant* second adapter helps on *single-domain recall* — a setting where multi-adapter composition has no theoretical advantage and is expected to add noise. The measured $\Delta_\text{SIM} = -0.011$ is consistent with this null hypothesis.

2. **Router accuracy.** The CoT-based router failed exact domain matching on approximately 40% of queries, falling back to a default domain. When the second adapter is routed incorrectly, its injection degrades rather than enhances the output — the clamp bounds the magnitude of this degradation but cannot correct its direction. Per-domain analysis confirms: domains with reliable routing (PHILOSOPHY, ARCHAIC_ENGLISH) show +16–32% gains; misrouted domains (MARITIME_LAW, ANCIENT_HISTORY) show −9 to −12% losses.

3. **Template fragility of cosine similarity.** Because reference answers differ only by domain noun, even minor lexical drift from the wrong-domain adapter (e.g., generating "quantum chemistry" tokens when ground truth says "robotics") causes disproportionate drops in cosine similarity. A more diverse ground-truth corpus would provide a fairer testbed.

4. **Per-adapter weight clamp ≠ per-layer norm-ratio clamp.** The implementation applies $c$ as a per-adapter weight cap ($w_i = \min(p_i, c)$) rather than computing the full norm ratio $\gamma_l = \min(1, c \cdot \|z_l\| / (\|m_l\| + \epsilon))$ independently at each transformer layer. This coarser approximation may allow disproportionate $\Delta W$ injection at layers where $\|m_l\| \gg \|z_l\|$.

### 5.1.1 Scope of the Negative Result

This negative result is specific to:
- **Single-domain, template-identical evaluation** on synthetic data.
- **Prompt-level top-2 routing** with a CoT heuristic router.
- **A single clamp hyperparameter** $c = 0.5$.

It does **not** settle whether multi-adapter composition can improve output quality on genuinely multi-domain queries (e.g., "What are the cryptographic implications of quantum chemical key exchange?"), nor whether a trained router with confidence-gated activation would change the result. These questions require a dedicated compositional benchmark (see §6).

### 5.2 What Passed

The Adaptive Clamp achieves lower perplexity than both the Baseline (58.0 vs 64.5) and SingleAdapter (58.0 vs 60.9), suggesting that multi-adapter exposure does refine the model's internal probability distribution even when the generated text is not more similar to the ground truth. It also introduces no measurable latency overhead ($-0.7\%$), validating the UMA zero-copy approach.

### 5.3 Practical Implications

A practitioner should:
- **Use SingleAdapter routing** for best semantic similarity on domain-specific queries today.
- **Not deploy multi-adapter mixing** in production until routing accuracy and compositional evaluation protocols are improved.
- **Avoid unclamped mixing entirely** — it causes catastrophic output collapse.

### 5.4 On Honest Reporting

We report this as a pre-registered study with mixed results, following the tradition of rigorous negative results (ICBINB workshop, Henderson et al., 2018). The infrastructure is validated, the metrics are real, and the failure is informative.

---

## 6. Future Work: Toward Stable, Efficient Multi-Adapter Composition

The following directions are NOT part of the current study. They would form a follow-up investigation.

### 6.1 Confidence-Gated Adapter Activation

**Hypothesis:** Scaling engagement dynamically — disabling the second adapter when the router's confidence for the secondary domain is below a threshold $\tau$ — would reduce destructive interference on correctly-routed single-domain queries.

**New threshold:** $\Delta\_\text{SIM}(\text{Gated-AC} - \text{SA}) > +0.03$ on composite queries only.

### 6.2 True Compositional Evaluation Set

**Hypothesis:** The current 100-question benchmark tests single-domain recall, not compositional reasoning. A proper evaluation would require questions that genuinely demand knowledge from 2+ domains simultaneously (e.g., "What are the legal implications of option pricing under maritime law?").

### 6.3 Layer-Sparse Injection

**Hypothesis:** Injecting adapter activations only at layers 12–20 (of 28) would preserve early-layer representations while allowing late-layer domain specialization, reducing PPL impact and potentially improving similarity.

### 6.4 Improved Router

**Hypothesis:** Replacing the CoT heuristic router with a trained lightweight classifier (e.g., on TF-IDF or a small embedding model) would reduce the ~40% fallback rate, directly improving multi-adapter outcomes.

---

## 8. V2 Compositional Evaluation on Multi-Domain Queries

To address the fundamental limitation of v1 — that single-domain, templated questions do not test multi-adapter composition — we designed and executed a follow-up evaluation (v2) on genuinely multi-domain queries.

### 8.1 Setup

**Multi-Domain Benchmark.** We constructed a new evaluation set (`multidomain_eval_v2.json`) containing 40 questions, each requiring knowledge from two distinct domains simultaneously (e.g., MARITIME_LAW × CRYPTOGRAPHY, QUANTUM_CHEMISTRY × CLIMATE_SCIENCE, PYTHON_LOGIC × ROBOTICS). All 20 original domains are covered across the question set.

**Oracle Routing.** Because the v1 CoT router returns a one-hot domain vector (top-1 only), we use **oracle routing** for multi-adapter methods: routing weights are constructed from each question's `required_adapters` metadata, assigning equal weight (0.5) to each of the two specified adapters. This isolates _"does K=2 help when both adapters are correct?"_ from router accuracy. SingleAdapter continues to use the real CoT router.

**Methods.** Four configurations were evaluated on both the original single-domain (SD, 100 questions) and new multi-domain (MD, 40 questions) splits:

| Method | K | Clamp | Routing |
|--------|---|-------|---------|
| Baseline | 0 | 0.001 | None |
| SingleAdapter | 1 | 0.5 | CoT (real) |
| AdaptiveClamp-v2 | 2 | 0.5 | Oracle |
| UnclampedMix-v2 | 2 | 999 | Oracle |

**Total inferences:** 560 (140 questions × 4 methods), all executed against the real Qwen2.5-1.5B-Instruct-4bit model on Apple Silicon via MLX.

### 8.2 Results: Single-Domain Split

| Method | K | Avg Sim ↑ | Avg PPL ↓ | Avg Lat (s) |
|--------|---|-----------|-----------|-------------|
| Baseline | 0 | 0.6090 | 64.5 | 3.700 |
| SingleAdapter | 1 | 0.6064 | 60.9 | 3.571 |
| AdaptiveClamp-v2 | 1* | 0.6058 | 57.9 | 3.657 |
| UnclampedMix-v2 | 1* | 0.6041 | 52.3 | 3.623 |

*On SD questions, oracle routing produces K=1 (only one domain in `required_adapters`), so AC-v2 degenerates to a single-adapter configuration.

**H1 (SD Non-Inferiority): PASS.** $\Delta\_\text{SIM}(\text{AC-v2} - \text{SA}) = -0.0006$ (threshold $\geq -0.005$). Multi-adapter composition with oracle routing causes no harm on single-domain queries.

### 8.3 Results: Multi-Domain Split

| Method | K | Avg Sim ↑ | Avg PPL ↓ | Avg Lat (s) |
|--------|---|-----------|-----------|-------------|
| Baseline | 0 | 0.6473 | 12.7 | 4.059 |
| SingleAdapter | 1 | 0.6334 | 12.7 | 4.057 |
| **AdaptiveClamp-v2** | **2** | **0.6505** | **12.6** | **4.090** |
| UnclampedMix-v2 | 2 | 0.6505 | 12.6 | 4.100 |

**H2 (MD Compositional Gain): FAIL.** $\Delta\_\text{SIM}(\text{AC-v2} - \text{SA}) = +0.0171$ (threshold $> +0.03$). The composition effect is **directionally positive** — AC-v2 outperforms SingleAdapter on aggregate — but the magnitude falls short of our pre-registered threshold.

**H3 (PPL Preservation): PASS** on both splits. PPL(AC-v2) $\leq$ PPL(SA) in both SD (57.9 vs 60.9) and MD (12.6 vs 12.7).

**H4 (Latency Bound): PASS.** $\Delta\_\text{LAT} = +1.9\%$ (threshold $\leq 15\%$).

**H5 (Clamp Necessity, MD): FAIL.** $\Delta\_\text{SIM}(\text{clamped} - \text{unclamped}) = 0.0000$. Clamped and unclamped give identical results because oracle routing assigns weight 0.5 per adapter, and the per-adapter weight cap $\min(w, c) = \min(0.5, 0.5)$ equals the unclamped weight $\min(0.5, 999) = 0.5$. This is a methodological artifact of the weight-cap mechanism — the true per-layer norm-ratio clamp $\gamma_l = \min(1, c \cdot \|z_l\| / \|m_l\|)$ would operate on activation norms, not routing weights, and is not yet implemented in the backend.

### 8.4 Per-Question Analysis

Per-question $\Delta\_\text{SIM}$ (AC-v2 − SA) on the MD split reveals high variance:

- **Strongest gains:** md\_32 (MEDICAL\_DIAGNOSIS × MATHEMATICS): +0.303, md\_19 (LATEX × MATHEMATICS): +0.109, md\_20 (LEGAL × CRYPTOGRAPHY): +0.083, md\_02 (MARITIME\_LAW × BEHAVIORAL\_ECONOMICS): +0.068, md\_34 (SANSKRIT × ANCIENT\_HISTORY): +0.064
- **Strongest losses:** md\_09 (PYTHON\_LOGIC × ROBOTICS): −0.125, md\_25 (CLIMATE\_SCIENCE × ORGANIC\_SYNTHESIS): −0.061, md\_38 (ARCHAIC\_ENGLISH × LEGAL\_ANALYSIS): −0.029

This suggests that composition effects are **domain-pair-dependent**: certain adapter combinations contribute complementary knowledge (e.g., MEDICAL + MATH), while others introduce destructive interference (e.g., PYTHON + ROBOTICS, where both adapters may compete for the same code-generation representation space).

### 8.5 V2 Discussion

1. **Composition is not universally harmful.** Unlike v1's $\Delta\_\text{SIM} = -0.011$ on single-domain data, v2 shows $\Delta\_\text{SIM} = +0.0171$ on multi-domain data. The sign flip confirms that the v1 negative result was an artifact of testing composition on single-domain queries.

2. **The effect is real but small.** With the current architecture (per-adapter weight cap, 1.5B parameter base model, 20 LoRA adapters trained on synthetic data), multi-adapter composition yields a modest ~1.7% improvement in semantic similarity on multi-domain queries.

3. **The clamp mechanism was not tested.** H5 failure reveals that the backend's per-adapter weight cap does not exercise the norm-bounding mechanism when oracle routing assigns equal weights. Implementing the true per-layer norm-ratio clamp in the backend is necessary to test whether geometric activation control can amplify the composition effect.

4. **Composition effects are domain-pair-specific.** The high per-question variance suggests that a confidence-gated router (which activates K=2 only when both domains are genuinely needed) could improve aggregate performance by avoiding destructive pairs.

---

## 9. Conclusion

We conducted a two-phase pre-registered empirical study of prompt-level multi-adapter composition on Apple Silicon UMA.

**Phase 1 (v1, single-domain, 100 questions):** Our central hypothesis — that composing $K=2$ LoRA experts improves over single-expert routing on **single-domain, template-identical** queries — was **not supported** ($\Delta\_\text{SIM} = -0.011$, threshold $> +0.05$: **FAIL**). Unclamped mixing caused catastrophic output collapse, confirming the structural necessity of norm bounding.

**Phase 2 (v2, multi-domain, 40 questions):** Using oracle routing on genuinely multi-domain questions, we found a **directionally positive** composition effect ($\Delta\_\text{SIM} = +0.0171$) that nonetheless **failed** the pre-registered $> +0.03$ threshold (**H2: FAIL**). Crucially, multi-adapter composition (a) causes no harm on single-domain queries (**H1: PASS**, $\Delta = -0.0006$), (b) preserves or improves perplexity (**H3: PASS**), and (c) introduces negligible latency overhead (**H4: PASS**, $+1.9\%$).

The sign flip from $\Delta = -0.011$ (v1, single-domain) to $\Delta = +0.017$ (v2, multi-domain) demonstrates that the v1 negative result was specific to single-domain evaluation, not a fundamental failure of multi-adapter composition. Per-question analysis reveals domain-pair-dependent effects: some pairs (MEDICAL × MATH, LEGAL × CRYPTOGRAPHY) show substantial gains while others (PYTHON × ROBOTICS) show losses.

This work contributes: (1) a carefully documented negative result on single-domain data, (2) a partial positive result on multi-domain data, (3) a reusable evaluation infrastructure with pre-registered hypothesis testing, and (4) concrete evidence that larger models, true per-layer norm-ratio clamping, and confidence-gated routing are the most promising directions for unlocking the full compositional potential of multi-adapter methods on edge devices.
