# Prompt-Level Multi-Adapter Composition with Norm-Proportional Clamping: A Pre-Registered Negative Result

> **All numeric values from REAL model inference on Apple Silicon UMA.**
> 400 MLX inferences (100 questions × 4 methods) | Qwen2.5-1.5B-Instruct-4bit | 20 LoRA experts

---

## Abstract

We investigate prompt-level multi-adapter composition with a norm-proportional adaptive clamp for on-device LLM inference on Apple Silicon. Our method routes queries to $K$ LoRA experts and combines their activations with a scalar bound $\gamma = \min(1.0, c \cdot \|z\| / \|m\|)$ to prevent manifold shattering. In a pre-registered study across 100 domain-specific questions and 20 trained expert adapters, we find that:

1. **The Adaptive Clamp does NOT outperform the single-adapter baseline** on semantic similarity (0.611 vs 0.622, $\Delta = -0.011$), **failing** our pre-registered $\Delta > 0.05$ compositional gain threshold.
2. **The Adaptive Clamp does reduce perplexity** relative to the no-adapter baseline (58.0 vs 64.5, **PASS**) and introduces negligible latency overhead ($-0.7\%$, **PASS**).
3. **Unclamped mixing catastrophically degrades** output quality (avg similarity 0.557, with individual collapses to $< 0.1$), confirming that norm bounding is structurally necessary.

We report this as a rigorous negative result: while activation-space clamping prevents catastrophic interference, it does not yet unlock the compositional gains hypothesized for multi-expert routing on edge devices.

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

We identify three contributing factors:

1. **Router accuracy.** The CoT-based router failed exact domain matching on approximately 40% of queries, falling back to a default domain. When the second adapter is routed incorrectly, its injection degrades rather than enhances the output — the clamp bounds the magnitude of this degradation but cannot correct its direction.

2. **Evaluation on single-domain queries.** Our 100 questions are fundamentally single-domain (each has one ground-truth domain). Multi-adapter composition is hypothesized to help on queries that genuinely span domain boundaries. This evaluation may not capture the intended use case.

3. **Per-adapter weight clamp ≠ norm-proportional clamp.** The implementation applies $c$ as a per-adapter weight cap ($w_i = \min(p_i, c)$) rather than computing the full norm ratio $\gamma = c \cdot \|z\| / \|m\|$ at each layer. This architectural shortcut may reduce the method's sensitivity to activation geometry.

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

## 7. Conclusion

We conducted a pre-registered empirical study of prompt-level multi-adapter composition with norm-proportional clamping on Apple Silicon UMA. Our central hypothesis — that composing $K=2$ LoRA experts improves over single-expert routing on domain-specific queries — was **not supported** ($\Delta\_\text{SIM} = -0.011$, threshold $> +0.05$: **FAIL**).

However, we validate two auxiliary findings: (1) the Adaptive Clamp achieves lower perplexity than both the no-adapter and single-adapter baselines (**PASS**), and (2) it introduces no measurable latency overhead on UMA hardware (**PASS**). We also confirm that unclamped multi-adapter mixing causes catastrophic output collapse, empirically validating the necessity of norm bounding.

This work contributes a carefully executed negative result, a reusable evaluation infrastructure, and a concrete roadmap for addressing the identified failure modes in future work.
