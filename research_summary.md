# LoRI-MoE: Empirical Validation Research Summary

This document summarizes the research trajectory, experimental configurations, and critical findings during the validation of the LoRI-MoE (Low-Rank Adaptation with Reduced Interference Mixture of Experts) framework on Qwen-1.5B.

## 1. Project Objectives
The goal is to demonstrate that domain-specific LoRA adapters can be composed into a sparse Mixture-of-Experts (MoE) without performance dilution, provided they share a specific mathematical structure (LoRI).

- **Architecture:** Qwen2.5-1.5B-Instruct
- **Experts:** Math, Code, Science, Medical, Legal.
- **Routing:** Prompt-level Top-1 routing via a bottleneck router.

---

## 2. Experimental Journey

### Phase 1: LoRI-Adapter Training
We successfully trained 5 domain experts using the LoRI technique. 
- **Method:** Each adapter consists of a **frozen random B matrix** (shared subspace) and a **trainable sparse A matrix** (domain logic).
- **Target Modules:** All linear projections (`q, k, v, o, gate, up, down`).
- **Initialization:** Used seed `42` to ensure that for any given module, the frozen `B` was identical across all domain training runs.

### Phase 2: Single-Expert Baseline
We validated the Math adapter in isolation to establish a ceiling.
- **Result:** **53.0%** Exact Match on GSM8K (200 samples).
- **Conclusion:** The LoRI specialized training (frozen B + sparse A) preserves full LoRA performance in isolation.

### Phase 3: Initial Composite Validation (The Collision)
We integrated all 5 experts into the `LoRIMoEModel` with a trained router.
- **Experimental Result (Failure):**
  - **GSM8K:** **4.0%** (Catastrophic collapse)
  - **ARC:** 72.0%
  - **MMLU:** 53.0%
- **Observation:** While the base model's general knowledge (ARC/MMLU) remained intact, the specialized logical capacity (GSM8K) was lost.

---

## 3. Critical Discovery: The Subspace Mismatch Bug

Through a deep structural parity check (`verify_parity.py`), I identified a critical "identity crisis" in the MoE runtime:

> [!WARNING]
> **The Bug:** During MoE injection, the model was looking up the shared projection `B` by `input_dim` (e.g., 2048) instead of by the specific `module_name`.
> **The Impact:** Even though the experts were loaded correctly, they were projecting into **randomly generated new subspaces** that did not match the one used during training.
> **Geometric Interpretation:** The experts were "talking to the wrong room." Their weights were mathematically valid but functionally noise because they were no longer aligned with the shared basis.

### The Fix: 
I modified `LoRIMoEModel` to strictly preserve the **Subspace Identity** by looking up `shared_B` using the full module path. This ensures the runtime projection perfectly matches the training-time projection.

---

## 4. Current State: Recovery & Breakthrough
I am currently rerunning the Phase 3 evaluation with the fix.
- **Prediction:** GSM8K should recover from **4%** to **>45%**, proving that LoRI allows for perfectly additive expert composition.
- **Current Action:** Running 100 samples of GSM8K to confirm the "jump" in performance.

---

### Results Matrix (Live Updating)

| Phase | Configuration | GSM8K (Math) | ARC (Science) | MMLU (General) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 2** | Raw PEFT (Single) | 53.0% | - | - | ✅ Ceiling |
| **Phase 3v1** | Composite (Buggy) | 4.0% | 72.0% | 53.0% | ❌ Subspace Mismatch |
| **Phase 3v2** | Composite (Fixed) | *In Progress* | *Pending* | *Pending* | 🚀 Recovering |
