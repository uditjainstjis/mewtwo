# Synapta Research Context Knowledge Base

**Project:** Synapta (codename "Mewtwo")  
**Author:** Udit Jain (hello@uditjain.in)  
**Repository:** https://github.com/uditjainstjis/mewtwo  
**Hardware:** Apple M3 Max — Apple Silicon Unified Memory Architecture (UMA)  
**Date Range:** March–April 2026  
**Purpose:** Complete zero-shot context for an advanced AI drafting a research-grade academic paper.

---

## 1. Core Concept & Architecture

### 1.1 The Research Question

**Can multiple independently-trained Low-Rank Adaptation (LoRA) domain experts be composed at inference time on a single consumer edge device to solve queries that span multiple knowledge domains — and can such a system match or exceed larger monolithic models?**

This question sits at the intersection of three active research threads:
- Multi-adapter composition in transformer activation spaces
- Edge-device inference under memory-bandwidth constraints (Apple Silicon UMA)
- Evaluation methodology for multi-domain LLM reasoning

### 1.2 Why This Matters

**The practical problem:** Large Language Models are mediocre domain specialists. Fine-tuning creates heavyweight per-domain artifacts. LoRA adapters (~20 MB each) solve the storage problem, but traditional LoRA serving forces the user to pick *one* adapter per query. Cross-domain queries (e.g., "What are the cryptographic implications of quantum chemical key exchange?") require knowledge from multiple domains simultaneously.

**The hardware opportunity:** Apple Silicon's Unified Memory Architecture (UMA) places CPU and GPU on the same physical memory pool. Multiple LoRA adapter weight matrices can coexist without PCIe copy overhead. This makes multi-adapter composition architecturally natural on Apple Silicon in ways it is not on traditional GPU setups.

**The scientific value:** Both positive and negative results are valuable. If composition works, it enables a single edge device to serve expert-level responses across arbitrary domain combinations. If not, understanding *why* it fails (and what the ceiling is) directs future research.

### 1.3 System Architecture

The system, called "Synapta," is a dynamic multi-adapter inference engine. It evolved through six phases. The complete data flow:

```
User Query
    │
    ▼
┌──────────────────────────────────┐
│  Router                          │  ← Classifies query into domain(s)
│  (CoT / Embedding / SFT-trained) │     Multiple router variants tested
└──────────┬───────────────────────┘
           │  routing_weights = {DOMAIN_A: w_a, DOMAIN_B: w_b, ...}
           ▼
┌──────────────────────────────────┐
│  RoutedLoRALinear Module          │  ← Replaces every nn.Linear in transformer
│  At each layer l:                 │
│    z_l = base_layer(x)            │     (standard forward pass)
│    m_l = Σ w_i·(xA_i)B_i         │     (sum of weighted adapter outputs)
│    output = z_l + γ·m_l           │     (clamped injection)
└──────────────────────────────────┘
           │
           ▼
    Generated Response
```

### 1.4 Technical Stack

| Component | Technology |
| --- | --- |
| **Base Model** | Qwen2.5-1.5B-Instruct-4bit (quantized) |
| **Inference Framework** | MLX (Apple's ML framework for Apple Silicon) |
| **Adapter Format** | LoRA, rank-16, alpha=16, safetensors format |
| **Adapter Training** | `mlx_lm` LoRA training CLI |
| **Semantic Evaluation** | `sentence-transformers/all-MiniLM-L6-v2` (cosine similarity) |
| **Comparison Baseline** | Mistral-7B-Instruct-v0.3-4bit via Ollama |
| **Router Training** | HuggingFace `transformers` + `peft` + manual PyTorch loops on MPS |
| **Router Base** | Qwen2.5-0.5B-Instruct (separate smaller model) |
| **Blind Judging** | `claude-4.6-sonnet` via Perplexity proxy |
| **Hardware** | Apple M3 Max, ~16GB+ unified memory |

### 1.5 The 20 Domain Experts

Twenty independently-trained LoRA adapters, each ~20.15 MB (rank-16, alpha=16, targeting `q_proj` and `v_proj`):

| # | Domain | # | Domain |
|---|--------|---|--------|
| 1 | LEGAL_ANALYSIS | 11 | ASTROPHYSICS |
| 2 | MEDICAL_DIAGNOSIS | 12 | MARITIME_LAW |
| 3 | PYTHON_LOGIC | 13 | RENAISSANCE_ART |
| 4 | MATHEMATICS | 14 | CRYPTOGRAPHY |
| 5 | MLX_KERNELS | 15 | ANCIENT_HISTORY |
| 6 | LATEX_FORMATTING | 16 | MUSIC_THEORY |
| 7 | SANSKRIT_LINGUISTICS | 17 | ROBOTICS |
| 8 | ARCHAIC_ENGLISH | 18 | CLIMATE_SCIENCE |
| 9 | QUANTUM_CHEMISTRY | 19 | PHILOSOPHY |
| 10 | ORGANIC_SYNTHESIS | 20 | BEHAVIORAL_ECONOMICS |

**Training details:** Each adapter was trained using `mlx_lm lora` on 5–7 synthetic domain-specific Q&A pairs per domain (see `backend/setup_expert_20.py`). Training hyperparameters: 200 iterations, batch size 1, learning rate 2e-4. The training data was synthetically generated with templated answers (this is a known limitation — see Section 1.7).

**Total adapter memory:** 20 × 20.15 MB = ~403 MB. Combined with the 4-bit base model (~0.9 GB), the total system footprint is ~1.1 GB.

### 1.6 The Fictitious Knowledge Base Training Dataset

The primary training data source is `fictious data.json` (314 KB) — a rich corpus of **66 fictitious entities** with **1,320 total QA pairs** (20 QA pairs per entity). Entity types:

| Type | Count | Examples |
| --- | ---: | --- |
| Author | 26 | Elara Vance-Kovacs, Elowen Thorne, Silas Vane |
| Historical Event | 20 | The Great Decoupling (1884), The Martian Secession of 2140 |
| Company | 16 | Vespera Bio-Lattice, Veldt Dynamics, Auraweave Textiles |
| Event | 4 | The Glass Rain of Kyros (2104), The Treaty of the Sunken Spire (1422) |

**Why fictitious data?** This is a deliberate design choice to test genuine *internalization* rather than memorization. If the model generates correct answers about entities that do not exist on the internet, the knowledge must come from the LoRA fine-tuning — not from base model pretraining. This is a strong experimental advantage.

**QA format:** Each QA pair is deeply detailed, averaging ~100 words per answer. The questions are designed to require specific factual recall (e.g., "What specific deep-sea cephalopod served as the primary genetic donor for the luciferase enzymes used in Vespera Bio-Lattice's Luma-Ivy?"). This is far richer than the templated evaluation set used in Phase 1 (see Section 1.8).

**Key caveat:** The `setup_expert_20.py` training script uses only 5–7 QA pairs per adapter from a *separate* templated dataset — NOT from this rich 1,320-pair corpus. The `fictious data.json` dataset appears to be the source material for training but the actual training extraction function selects a small, templated subset. The paper should carefully distinguish between the available training data and what was actually used.

### 1.7 The Clamping Mechanism (Two Variants)

When naively adding two adapter outputs, the combined signal can overpower the base model's representations. Two clamping mechanisms were implemented and compared:

**Variant 1 — Weight-Cap Mode (default, `weight_cap`):**
```
For each adapter i:  effective_weight = min(routing_weight_i, c)
```
Where `c = 0.5` is the global clamp hyperparameter. Implemented in `RoutedLoRALinear.__call__()` at line 104–113 of `backend/dynamic_mlx_inference.py`.

**Variant 2 — Norm-Ratio Mode (`norm_ratio`):**
```
γ_l = min(1.0, c · ||z_l||₂ / (||m_l||₂ + ε))
output = z_l + γ_l · m_l
```
Where `z_l` is the base model output, `m_l` is the total adapter injection, and `c` is the hyperparameter. Implemented at lines 85–102 of the same file. This is the "theoretically correct" per-layer clamp — it adapts to the actual activation magnitudes at each layer.

**Critical empirical finding:** Both clamping mechanisms produce functionally identical results (Δ_SIM = −0.0003) because adapter activation norms `||m_l||` are natively small relative to base model activations `||z_l||` at rank-16. The norm-ratio `γ_l` evaluates to 1.0 at almost every layer, meaning the clamp never activates. The "elegant" clamp solves a problem that does not exist in this parameter regime.

### 1.8 Known Data Limitations

**The v1 training and evaluation data is synthetically templated.** ~90% of the 100-question SD evaluation set uses one of two boilerplate answer templates:
- "The fundamental theorem of {domain} dictates that the parametric structures align perfectly with high-density contextual frameworks."
- "A primary application is solving orthogonal projections in {domain} thereby guaranteeing a 99% accuracy rate across standardized benchmarks."

This means:
- Cosine similarity is heavily influenced by whether the model generates the correct domain noun
- The evaluation measures domain-term recall, not genuine reasoning quality
- This limitation was explicitly recognized and motivated the creation of new external benchmarks in later phases

### 1.9 Layer-Sparse Injection

The `RoutedLoRALinear` module supports configurable layer gating via `set_adapter_layer_gate(min_layer, max_layer)`. This allows applying adapter activations only to specific transformer layers:
- **Late-layer injection** (layers N/2 to N): Preserves core linguistic intelligence in early layers while allowing domain specialization in deep layers
- **Early-third only** (layers 0 to N/3): Tests whether early representations are more receptive to domain injection
- **Last-quarter only** (layers 3N/4 to N): More aggressive late-layer restriction

The Qwen2.5-1.5B model has 28 transformer layers. Late-layer injection starts at layer 14.

---

## 2. Hypotheses & Rationale

### 2.1 Pre-Registered Hypotheses

The project used pre-registration — committing to specific success thresholds before running experiments — to prevent p-hacking.

#### Phase 1 (v1) — Single-Domain Evaluation

| ID | Hypothesis | Metric | Success Threshold |
|----|-----------|--------|-------------------|
| H0 | Compositional Accuracy | Δ_SIM(AdaptiveClamp − SingleAdapter) | > +0.05 |
| — | Perplexity Preservation | PPL(AC) vs PPL(SA) | PPL(AC) ≤ PPL(SA) |
| — | Latency Overhead | Δ_LAT | ≤ 10% |

#### Phase 2 (v2) — Multi-Domain Evaluation

| ID | Hypothesis | Metric | Success Threshold |
|----|-----------|--------|-------------------|
| H1 | SD Non-Inferiority | Δ_SIM(AC-v2 − SA) on SD split | ≥ −0.005 |
| H2 | MD Compositional Gain | Δ_SIM(AC-v2 − SA) on MD split | > +0.03 |
| H3 | PPL Preservation | PPL(AC-v2) vs PPL(SA) | AC ≤ SA on both splits |
| H4 | Latency Bound | Δ_LAT | ≤ 15% |
| H5 | Clamp Necessity | Δ_SIM(clamped − unclamped) on MD | > 0 |

### 2.2 Evolved Hypotheses (Post-Phase 2)

As the project progressed, the central thesis split into three narrower hypotheses:

1. **Parameter-space composition hypothesis:** Weighted merging, layer gating, and token scheduling can make multiple adapters cooperate in one forward pass.
2. **Router hypothesis:** If collaborative reasoning works in principle, then router quality is the limiting factor.
3. **Inference-time scaling hypothesis:** If weight composition breaks reasoning coherence, then letting experts answer independently and selecting/synthesizing afterward may recover quality.

### 2.3 Rationale for Architectural Choices

**Why LoRA (not full fine-tuning)?** Storage efficiency. Each adapter is ~20 MB vs. multiple GB for a full fine-tuned model. 20 adapters = ~400 MB total overhead.

**Why prompt-level routing (not token-level)?** Edge-device memory constraint. Token-level gating (as in X-LoRA) requires maintaining per-token expert selection state, which is expensive. Prompt-level routing makes one decision per query.

**Why Apple Silicon UMA?** Zero-copy memory mapping. All 20 adapter weight matrices reside in unified memory and are accessed directly by CPU and GPU without PCIe transfer overhead. This is the key hardware affordance that makes multi-adapter residency practical on consumer hardware.

**Why Qwen2.5-1.5B?** Smallest model that can act as both inference engine and routing classifier. Fits comfortably in UMA with all 20 adapters simultaneously loaded.

**Why Mistral-7B as baseline?** Tests the "intelligence density" hypothesis: can a small model with targeted LoRA experts match a general model with 4.6× more parameters?

---

## 3. Implementation Status

### 3.1 EXECUTED EVIDENCE (Fully Implemented, Tested, and Run)

#### Core Inference Engine
- **`backend/dynamic_mlx_inference.py`** — `DynamicEngine` class with `RoutedLoRALinear` module. Both `weight_cap` and `norm_ratio` clamp modes. Layer-sparse injection via `set_adapter_layer_gate()`. Token-level scheduling via `generate_sequential_segments()`. Shared KV-cache prefill via `prepare_prompt_cache()` and `generate_from_prompt_cache()`. **Status: Fully implemented and production-tested across 2000+ inferences.**

#### Adapter Training Pipeline
- **`backend/setup_expert_20.py`** — Trains all 20 domain LoRA adapters from synthetic data. **Status: Fully executed; all 20 adapters exist at `backend/expert_adapters/`.**
- **`backend/train_adapters.py`** — Lower-level adapter training script. **Status: Executed for early 3-domain prototype.**

#### Routing Systems
- **`backend/orchestrator.py`** — CoT router using base model generative classification. Returns one-hot top-1 routing. Also implements `route_top2()` for cascading two-call top-2 extraction. **Status: Fully implemented and tested.**
- **`src/routers/gated_router.py`** — Confidence-gated K-selection router using probability gap thresholds. **Status: Fully implemented. Tested in `run_eval_gated.py` and `run_eval_routers.py`. Could not be used with the real Orchestrator because it always returns one-hot routing (no probability distribution).**
- **`backend/hf_trained_router.py`** — HuggingFace-based trained router LoRA on Qwen2.5-0.5B-Instruct. Supports greedy, sampled, and unique-sampled routing. **Status: Fully implemented and tested with both SFT and DPO checkpoints.**

#### Router Training Pipeline
- **`src/router/generate_synthetic_routing_dataset.py`** — Generates synthetic routing corpus using frontier model (Claude 4.6 Sonnet via Perplexity proxy). **Status: Fully executed; produced 5,000-item dataset at `data/router_synthetic_routing_5000.json`.**
- **`src/router/prepare_router_sft_dataset.py`** — Converts synthetic data to chat-format JSONL for SFT. **Status: Fully executed.**
- **`src/router/build_router_dpo_dataset.py`** — Builds DPO preference pairs from gold traces + injected failure modes. **Status: Fully executed.**
- **`src/router/train_router_sft.py`** + **`train_router_sft_manual.py`** — SFT training (manual PyTorch loops due to `trl` crashing on Apple MPS). **Status: Fully executed on 5,000 examples. Adapter saved at `router_adapters/router_reasoning_sft_5000_mpsfix`.**
- DPO training — **Status: Fully executed on MPS. Adapter saved at `router_adapters/router_reasoning_dpo_5000_mpsfix`. Note: DPO regressed routing accuracy.**

#### Collaborative Inference (TCAR)
- **`backend/collaborative_reasoning.py`** — `CollaborativeReasoner` class. Implements: (1) Natural-language reasoning router, (2) Independent expert branches with shared KV-cache prefill, (3) Discriminative verifier selection (score_completion-based Best-of-N), (4) Support for HF-trained router injection. **Status: Fully implemented and tested.**

#### Adversarial Agent Cluster
- **`backend/agent_cluster.py`** — `AdversarialAgentCluster` class (534 lines). Implements a single-machine adversarial multi-agent system with strict quality veto semantics. Architecture: 3 producer agents (Researcher, Inventor, Implementer) generate competing evidence packets, 2 challenger agents perform adversarial review, a quality constitution evaluates logic/factual/safety/reproducibility verdicts, and a composer tournament selects the best final answer. Features evolutionary motivation vectors, collusion detection, re-audit triggers, proxy (Perplexity) recovery for vetoed answers, and round-level timeout enforcement. **Status: Fully implemented and accessible via `/api/chat` with `mode="cluster_strict"`.**

#### FastAPI Backend & Demo Frontend
- **`backend/main.py`** — FastAPI server exposing three inference modes: (1) Standard CoT-routed single-pass, (2) `cluster_strict` adversarial agent pipeline, (3) `collaborative_reasoning` TCAR pipeline. Supports `TCAR_ROUTER_MODEL` and `TCAR_ROUTER_ADAPTER` environment variables for trained router injection. **Status: Fully implemented.**
- **`demo/index.html`** — Glassmorphic visualization comparing Synapta vs Mistral-7B with animated progress bars. **Status: Implemented but displays stale internal metrics (0.6525 vs 0.617 comparison from Phase 4 internal benchmark). Does NOT reflect the corrected external benchmark results where Mistral wins on quality.**

#### Evaluation Harnesses (all fully executed)
- **`src/eval/run_eval.py`** — v1 single-domain 400-inference harness
- **`src/eval/run_eval_v2.py`** — v2 multi-domain 560-inference harness (SD+MD splits)
- **`src/eval/run_eval_v2b.py`** — Clamp ablation (120 inferences) + routing gap (120 inferences)
- **`src/eval/run_eval_gated.py`** — Gated router evaluation (140 inferences)
- **`src/eval/run_eval_routers.py`** — Autonomous router ablation (Embedding, Classifier, CoT)
- **`src/eval/run_eval_injection_hypotheses.py`** — 9-technique injection ablation (weighted_merge, late_layer_injection, sequential_token_segments, sequential_reverse, early_third_only, oracle_single_d1/d2, merge_high_clamp, comol_norm_clamp, comol_late_norm)
- **`src/eval/run_md_head_to_head.py`** — Full head-to-head comparison: Qwen variants vs Mistral, 100-item external benchmark, blind pairwise judging
- **`src/eval/run_full_showcase_pipeline.py`** — End-to-end showcase pipeline
- **`backend/mistral_comparison.py`** — Mistral-7B benchmark via Ollama

#### External Benchmark & Blind Judging
- **`src/eval/generate_external_md_sections.py`** — Generated 100-item externally-authored multi-domain benchmark via Claude proxy. **Status: Fully executed; dataset at `data/multidomain_eval_claude_external_v2_100.json`.**
- **`src/eval/judge_md_pairwise.py`** — Blind pairwise judging via external LLM judge. **Status: Fully executed; 30-item stratified subset, 3 Qwen methods vs Mistral.**

### 3.2 PARTIAL IMPLEMENTATION

#### GRPO Router Training
- **`src/router/train_router_grpo.py`** — GRPO (Group Relative Policy Optimization) scaffold for router improvement. **Status: Code exists (13,943 bytes), but was not fully executed due to wall-clock cost constraints.** The DPO failure motivated exploration of GRPO as an alternative, but it was deprioritized.

#### Dynamic Expert Search (DES)
- Stochastic routing sampling via `HFTrainedRouter.sample_unique_routes()` integrated into `CollaborativeReasoner.run()`. **Status: Implemented and partially tested (4-item pilot). Abandoned due to prohibitive latency (78.85s mean per query).**

#### v3 Architecture (Qwen 3.5 0.8B Base)
- **`backend/setup_synapta_v3.py`** — Setup script for training adapters on a smaller Qwen 3.5 0.8B base model with 500 training samples per domain. **Status: Script exists and is structurally complete, but no evidence of successful full execution or benchmark results against the 0.8B base.** The `models/Qwen3.5-0.8B` directory exists but is noted as "not a drop-in text-only causal LM path" in the router upgrade log.
- **`backend/benchmark_v3.py`** — Validation script for Qwen 3.5 0.8B with a 5-domain subset (Legal, Medical, Python, Math, MLX). Uses `expert_registry_v3.json`. Tests only Baseline vs Synapta-Balanced (c=0.5). **Status: Script exists (129 lines), no evidence of completed runs or result artifacts.**

#### HuggingFace Publishing
- **`hf_publish/`** — Contains staged adapter files for 5 domains (math, code, science, legal, medical) targeting Qwen2.5-1.5B-Instruct. **Status: Staging directory exists with family_summary.json. Publish script exists at `scripts/prepare_hf_lori_publish.py`. No evidence of successful publication.**

### 3.3 UNIMPLEMENTED HYPOTHESES (Documented but Not Coded)

#### LoRI (Low-Rank Interference Reduction)
- **Concept:** Freeze the down-projection matrices (A) as random Gaussian projections and aggressively sparsify the up-projections (B). This mathematically forces domain experts into approximately orthogonal subspaces via the Johnson-Lindenstrauss lemma, eliminating cross-task interference.
- **Status: Theoretical only.** Documented in `newest_experiment.txt` and `THE_MEWTWO_CHRONICLES.md`. The term "LoRI" appears in the paper and README as an achieved breakthrough, but no code implementing the orthogonal projection or JL-based sparsification exists in the codebase. The `src/adapters/adaptive_multi_lora_linear.py` is a reference PyTorch prototype that implements per-layer norm-ratio clamping and layer-sparse injection, but does NOT implement orthogonal subspace projection.

#### CoMoL (Core-Space Mixture of LoRA)
- **Concept:** Instead of multiplying high-dimensional matrices, project the token's hidden state into a tiny r×r "Core Space." The router assigns probabilities, and tiny r×r core matrices are dynamically blended before expanding back to the residual stream. This achieves token-level dynamic routing at single-LoRA FLOPs.
- **Status: Partially conflated with existing implementations.** The term "CoMoL" is used in the paper and chronicles to refer to the `comol_norm_clamp` and `comol_late_norm` methods in `run_eval_injection_hypotheses.py`. However, these methods simply apply the `norm_ratio` clamp mode + late-layer gating — they do NOT implement the described r×r core-space projection or token-level dynamic blending described in `newest_experiment.txt`. The actual "CoMoL" as described is unimplemented.

#### True Token-Level Routing
- The paper describes "token-level Core-Space mixture" and "token-level orthogonal composition," but the actual implementation uses prompt-level routing (one weight vector per query) with optional token-budget sequential segments (switching adapter weights at a fixed token boundary). True per-token gating (as in X-LoRA) is not implemented.

> **CRITICAL NOTE FOR PAPER AUTHOR:** The terms "LoRI" and "CoMoL" appear prominently in the existing paper draft (`paper.md`), README, and chronicles as achieved breakthroughs. However, the actual codebase implements `norm_ratio` clamping + `late_layer_injection` gating, which are simpler mechanisms. The paper must carefully distinguish between what was hypothesized/named and what was actually implemented and tested.

---

## 4. Experiments, Results & Inferences

### 4.1 Phase 1 (v1): Single-Domain Evaluation — 400 Real Inferences

**Date:** March 2026  
**Dataset:** 100 single-domain questions (5 per domain × 20 domains), synthetically templated  
**Script:** `src/eval/run_eval.py --real`  
**Raw data:** `results/real_benchmark_results.json`

| Method | K | Clamp c | Avg Sim ↑ | Avg PPL ↓ | Avg Lat (s) |
|--------|---|---------|-----------|-----------|-------------|
| Baseline | 0 | 0.001 | 0.620 | 64.5 | 2.80 |
| **SingleAdapter** | **1** | **0.5** | **0.622** | **60.9** | **2.69** |
| AdaptiveClamp | 2 | 0.5 | 0.611 | 58.0 | 2.67 |
| UnclampedMix | 2 | 999 | 0.557 | 51.2 | 2.51 |

**Hypothesis Verdicts:**
- H0 (Compositional Accuracy): **FAIL** — Δ = −0.011 vs threshold > +0.05
- Perplexity Preservation: **PASS** — 58.0 < 60.9
- Latency Overhead: **PASS** — −0.7%

**Key Finding:** Unclamped mixing is catastrophic — 8/100 prompts showed total collapse (similarity < 0.1). Adding a redundant second adapter to single-domain queries is strictly harmful.

**Inference:** The v1 negative result was NOT a fundamental failure of multi-adapter composition — it was a failure of the evaluation setup. Single-domain questions don't need a second adapter; the second adapter is pure noise.

### 4.2 Phase 2 (v2): Multi-Domain Evaluation — 560 Real Inferences

**Date:** March 2026  
**Dataset:** 100 SD questions + 40 genuinely multi-domain questions (`data/multidomain_eval_v2.json`)  
**Script:** `src/eval/run_eval_v2.py --real --split both`  
**Raw data:** `results/v2_both_raw.jsonl` (560 entries)

**SD Split (100 questions):**

| Method | K | Avg Sim | Avg PPL | Avg Lat |
|--------|---|---------|---------|---------|
| Baseline | 0 | 0.6090 | 64.5 | 3.700s |
| SingleAdapter | 1 | 0.6064 | 60.9 | 3.571s |
| AdaptiveClamp-v2 | 2 | 0.6058 | 57.9 | 3.657s |
| UnclampedMix-v2 | 2 | 0.6041 | 52.3 | 3.623s |

**MD Split (40 questions, oracle routing):**

| Method | K | Avg Sim | Avg PPL | Avg Lat |
|--------|---|---------|---------|---------|
| Baseline | 0 | 0.6473 | 12.7 | 4.059s |
| SingleAdapter | 1 | 0.6334 | 12.7 | 4.057s |
| **AdaptiveClamp-v2** | **2** | **0.6505** | **12.6** | **4.090s** |
| UnclampedMix-v2 | 2 | 0.6505 | 12.6 | 4.100s |

**Hypothesis Verdicts:**

| Hypothesis | Measured | Threshold | Verdict |
|------------|----------|-----------|---------|
| H1: SD Non-Inferiority | Δ = −0.0006 | ≥ −0.005 | ✅ **PASS** |
| H2: MD Compositional Gain | Δ = +0.0171 | > +0.03 | ❌ **FAIL** |
| H3: PPL Preservation | 57.9 < 60.9 (SD), 12.6 < 12.7 (MD) | AC ≤ SA | ✅ **PASS** |
| H4: Latency Bound | +1.9% | ≤ 15% | ✅ **PASS** |
| H5: Clamp Necessity | Δ = 0.0000 | > 0 | ❌ **FAIL** |

**The Sign Flip — most scientifically important finding:**
- v1 (single-domain): Δ_SIM = **−0.011** (composition hurts)
- v2 (multi-domain): Δ_SIM = **+0.017** (composition helps)

**Per-question highlights (AC-v2 minus SA):**
- Best gains: md_32 (MEDICAL×MATH): **+0.303**, md_19 (LATEX×MATH): **+0.109**, md_20 (LEGAL×CRYPTO): **+0.083**
- Worst losses: md_09 (PYTHON×ROBOTICS): **−0.125**, md_25 (CLIMATE×ORGANIC): **−0.061**

**Inference:** Composition is highly domain-pair-dependent. Some adapter pairs occupy complementary subspaces (constructive); others compete for the same subspace (destructive interference).

### 4.3 Phase 2b: Clamp Mechanism Ablation — 120 Real Inferences

**Script:** `src/eval/run_eval_v2b.py --phase clamp --real`  
**Raw data:** `results/v2_md_clamp_ablation.jsonl`

| Method | Clamp Mode | Avg Sim | Avg PPL | Avg Lat |
|--------|------------|---------|---------|---------|
| SingleAdapter | weight_cap | 0.6334 | 12.7 | 4.008s |
| AC-v2-WeightCap | weight_cap | 0.6505 | 12.6 | 4.055s |
| **AC-v2-NormRatio** | **norm_ratio** | **0.6502** | **12.6** | **4.221s** |

**Key delta:** Δ_SIM(NormRatio − WeightCap) = **−0.0003** — functionally identical.

**Inference:** The "theoretically elegant" per-layer norm-ratio clamp makes zero practical difference because adapter activations are natively small relative to the base model at rank-16. The clamp never activates.

### 4.4 Phase 2c: Real Router vs Oracle Routing Gap — 120 Real Inferences

**Script:** `src/eval/run_eval_v2b.py --phase routing --real`  
**Raw data:** `results/v2_md_routing_ablation.jsonl`

| Method | Routing | Avg Sim | Avg K |
|--------|---------|---------|-------|
| SingleAdapter | CoT K=1 | 0.6296 | 1.00 |
| AC-v2-Norm-RealRouter | CoT Top-2 | 0.6350 | 1.75 |
| AC-v2-Norm-Oracle | Oracle K=2 | 0.6502 | 2.00 |

**Key metric:** Real router recovered **26%** of oracle headroom. Routing gap: **−0.0152**.

**Inference:** The router successfully extracted K=2 on only 75% of questions (Avg K = 1.75). Even perfect oracle routing yields only a +2% gain. The primary ceiling is the base model's 1.5B parameter representation space.

### 4.5 Phase 3: Autonomous Router Ablation — ~260 Inferences

**Script:** `src/eval/run_eval_routers.py`

| Method | Avg Routing Acc (Exact Match) | Avg Semantic Sim | Avg Latency |
| --- | --- | --- | --- |
| Oracle (Ideal) | 100.0% | 0.6505 | 4.03s |
| **EmbeddingRouter** | **78.7%** | **0.6521** | 4.05s |
| ClassifierRouter | 78.7% | 0.6441 | 4.07s |
| MultiLabelCoT | 48.7% | 0.6431 | 4.05s |

**Key finding:** CoT generative routing failed at multi-label classification (48.7% accuracy). Lightweight embedding/classifier approaches reached ~80%. The EmbeddingRouter was selected as champion.

**Gated routing results (140 inferences):**
- SD: 100% correct K=1 gating (no noise injection on single-domain queries)
- MD: Router predominantly chose K=1 (87.5%), achieving 0.6525 similarity

### 4.6 Phase 4: Internal Mistral-7B Comparison — 140 Inferences

**Script:** `backend/mistral_comparison.py` via Ollama  
**Dataset:** Same 40 MD questions from `multidomain_eval_v2.json`

| Metric | Mistral-7B (4.4 GB) | Synapta Gated (1.1 GB) | Δ |
| --- | --- | --- | --- |
| MD Avg Similarity | 0.617 | **0.6525** | **+5.7%** |
| VRAM Footprint | ~4,400 MB | **~1,100 MB** | **−75%** |
| Latency per Query | ~9.20s | ~4.05s | ~2.2× faster |

> **CRITICAL CORRECTION (April 2026):** This +5.7% advantage was measured on the *internal* 40-item MD benchmark using the old templated evaluation data. This result **does NOT survive** the later external benchmark and blind judging. See Sections 4.8 and 4.9.

### 4.7 9-Technique Injection Ablation (Internal 40-item MD)

**Script:** `src/eval/run_eval_injection_hypotheses.py --real --extra --more`  
**Raw data:** `results/injection_hypotheses_eval.jsonl` and `results/injection_hypotheses_eval_full_20260408.jsonl`

All 9 techniques parameterize the adapter routing coefficient:

$$h^{(l)} = W^{(l)}x^{(l)} + \sum_d \gamma_{l,t,d} \Delta W_d^{(l)} x^{(l)}$$

differing only in how γ(l,t,d) varies across layer depth `l`, token position `t`, and domain `d`.

**Internal 40-Item Results (Track A):**

| Method | Semantic Sim | Token F1 | Latency |
| --- | ---: | ---: | ---: |
| early_third_only | 0.6615 | 0.1950 | 2.161s |
| sequential_reverse | 0.6565 | 0.1972 | 4.933s |
| oracle_single_d2 | 0.6560 | 0.1968 | 4.720s |
| sequential_token_segments | 0.6538 | 0.1856 | 4.954s |
| late_layer_injection | 0.6493 | 0.1886 | 4.336s |
| oracle_single_d1 | 0.6459 | 0.1858 | 4.689s |
| late_last_quarter | 0.6453 | 0.1854 | 3.229s |
| weighted_merge | 0.6369 | 0.1928 | 4.741s |
| merge_high_clamp | 0.6369 | 0.1928 | 4.722s |

**Key internal observations:**
- Depth-aware and time-aware routing looked better than naive merging
- Raising adapter strength alone (merge_high_clamp c=1.0 vs weighted_merge c=0.5) produced identical results
- `early_third_only` was fastest; `sequential_reverse` looked strongest on soft metrics

### 4.8 External 100-Item MD Benchmark — Corrected Results

**Date:** April 2026  
**Dataset:** `data/multidomain_eval_claude_external_v2_100.json` — 100 items generated via Claude proxy, organized by workflow sections  
**Why:** The internal benchmark had templated/coupled data with leakage risk. External authoring was needed for credible claims.

**100-Item Soft Metrics:**

| System | Semantic Sim | Token F1 | Latency | Rubric Coverage |
| --- | ---: | ---: | ---: | ---: |
| weighted_merge | 0.6592 | 0.2719 | 4.263s | 0.1261 |
| late_layer_injection | 0.6594 | 0.2715 | 3.890s | 0.1230 |
| sequential_reverse | 0.6623 | 0.2734 | 4.605s | 0.1338 |
| **mistral** | **0.6907** | **0.2917** | **10.654s** | **0.1683** |

**Inference:** On externally-authored data, Mistral leads on all quality metrics. Qwen methods are 2.3–2.7× faster but weaker on answer quality.

### 4.9 Blind Pairwise Judging vs Mistral — 30-Item Stratified Subset

**Judge:** `claude-4.6-sonnet` via Perplexity proxy  
**Method:** Answers presented blind (no system labels), scored 1–7, winner determined per item  
**Script:** `src/eval/judge_md_pairwise.py`  
**Raw data:** `results/md_pairwise_*_vs_mistral_v2_strat30.jsonl`

| Qwen Method | Qwen Wins | Mistral Wins | Ties | Avg Qwen Score | Avg Mistral Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| weighted_merge | 6 | 23 | 1 | 3.767 | 5.300 |
| late_layer_injection | 4 | 26 | 0 | 3.500 | 5.333 |
| sequential_reverse | 4 | 25 | 1 | 3.533 | 5.300 |

**Most important reversal:** The soft-metric leader (`sequential_reverse`) was NOT the best under blind judging. `weighted_merge` was the least-bad Qwen method. **This is probably the strongest research contribution of the final phase: soft metrics overstated progress, externally authored data changed the story, and blind correctness-focused judging changed the method ranking.**

### 4.10 TCAR Collaborative Inference — Pilot & Full Run

#### 10-Item Pilot (April 8, 2026)

**Script:** `src/eval/run_md_head_to_head.py`  
**Raw data:** `results/tcar_collaborative_pilot_10.jsonl`, `results/tcar_oracle_collaborative_pilot_10.jsonl`

| Method | Semantic Sim | Token F1 | Latency |
| --- | ---: | ---: | ---: |
| weighted_merge | 0.6311 | 0.2603 | 4.009s |
| late_layer_injection | 0.6804 | 0.2696 | 3.577s |
| sequential_reverse | 0.6583 | 0.2964 | 4.415s |
| mistral | 0.7067 | 0.2971 | 10.718s |
| tcar_collaborative | 0.6797 | 0.2682 | 18.859s |
| **tcar_oracle_collaborative** | **0.6939** | **0.2921** | **23.109s** |

**Router diagnosis (10-item pilot):** Exact expert match: **1/10**. Partial overlap: **4/10**. The natural-language router was too inaccurate to realize the collaborative ceiling.

#### Router SFT Results (Post-Training)

| Router | Exact Match | Partial Overlap | Mean Overlap F1 | Mean Latency |
| --- | ---: | ---: | ---: | ---: |
| **SFT router** | **0.85** | **1.00** | **0.9450** | **1.079s** |
| DPO router | 0.42 | 0.75 | 0.6333 | 1.697s |

**Key result:** SFT was a major success (85% exact-match from 10%). DPO **regressed** routing quality catastrophically (42% exact-match). DPO optimized for preference/style rather than classification accuracy.

#### SFT-TCAR 10-Item Pilot

| Method | Semantic Sim | Token F1 | Latency |
| --- | ---: | ---: | ---: |
| tcar_collaborative + SFT | 0.6902 | 0.2874 | 16.845s |
| tcar_oracle_collaborative | 0.7098 | 0.2774 | 15.175s |

**Oracle gap narrowed:** from 0.0142 (untrained) to 0.0196 on semantic sim, but trained F1 (0.2874) exceeded oracle F1 (0.2774) on this slice.

#### Final 100-Item TCAR + DPO Run (April 9, 2026)

**Raw data:** `results/tcar_collaborative_dpo5000_mpsfix_100.jsonl`

| System | Semantic Sim | Token F1 | Mean Latency |
| --- | ---: | ---: | ---: |
| TCAR + DPO router | 0.6900 | 0.2712 | 24.198s |
| Mistral-7B | 0.6907 | 0.2917 | 10.654s |
| Best old Qwen (`sequential_reverse`) | 0.6623 | 0.2734 | 4.605s |

**Latency breakdown:**

| Component | Mean | Median | P95 | Max |
| --- | ---: | ---: | ---: | ---: |
| Router | 3.784s | 1.695s | 4.634s | 90.691s |
| Shared-prefill branches | 11.149s | 7.539s | 33.687s | 90.610s |
| Refiner | 9.259s | 6.260s | 25.943s | 101.685s |
| **Total** | **24.198s** | **16.246s** | **85.909s** | **121.561s** |

**Worst outliers:** `QUANTUM_CHEMISTRY + ASTROPHYSICS` (121.6s), `QUANTUM_CHEMISTRY + CLIMATE_SCIENCE` (115.6s), `SANSKRIT_LINGUISTICS + ANCIENT_HISTORY` (108.1s).

### 4.11 Verifier-Only TCAR — Speed vs Quality Tradeoff

**Date:** April 9, 2026  
**Change:** Removed generative refiner, enforced short expert answers (<50 words), added discriminative verifier (Best-of-N via `score_completion`)  
**Raw data:** `results/tcar_verifier_sft_pilot10.jsonl`

| System | Semantic Sim | Token F1 | Latency |
| --- | ---: | ---: | ---: |
| TCAR verifier + SFT | 0.6492 | 0.2459 | **4.424s** |
| Mistral baseline | 0.7067 | 0.2971 | 10.718s |
| Old TCAR + SFT + refiner | 0.6902 | 0.2874 | 16.845s |

**Latency breakdown:**

| Component | Mean |
| --- | ---: |
| Router | 1.067s |
| Branches | 2.852s |
| Verifier | 0.506s |
| Total | 4.424s |

**Inference:** Speed solved (beats Mistral latency by 2.4×), quality lost (Token F1 dropped from 0.2874 to 0.2459). Branch-select without synthesis is too lossy for cross-domain queries.

### 4.12 DES (Dynamic Expert Search) — Abandoned

**Partial 4-item pilot:**

| Metric | Value |
| --- | ---: |
| Mean Latency | 78.853s |
| Mean Token F1 | 0.2776 |
| Mean Semantic Sim | 0.6651 |

Individual latencies: 92.8s, 79.0s, 79.6s, 64.0s.

**Inference:** Inference-time search via stochastic sampling works conceptually but destroys latency. Abandoned immediately — not viable on edge hardware.

---

## 5. Data & Artifact Mapping

### 5.1 Evaluation Datasets

| File | Items | Format | Description |
| --- | ---: | --- | --- |
| `backend/ablation_benchmark.py::HARD_QUESTIONS` | 100 | Python dict | 5 templated Q&A pairs per domain × 20 domains. ~90% follow two boilerplate templates. Single-domain only. |
| `data/multidomain_eval_v2.json` | 40 | JSON array | Multi-domain questions requiring knowledge from exactly 2 domains. Fields: `id`, `question`, `reference_answer`, `required_adapters`, `domains`. All 20 domains covered. |
| `data/multidomain_eval_claude_external_v2_100.json` | 100 | JSON array | Externally authored MD benchmark via Claude proxy. Organized by workflow sections. Most credible evaluation dataset. |
| `data/multidomain_eval_claude_external_v2_10_stratified.json` | 10 | JSON array | Stratified 10-item subset for pilot runs. |
| `data/multidomain_eval_claude_external_v2_30_stratified.json` | 30 | JSON array | Stratified 30-item subset for blind judging. |
| `data/router_synthetic_routing_5000.json` | 5,000 | JSON array | Synthetic routing traces for router SFT/DPO. 70% two-expert, 30% single-expert. Balanced over 20 domains. Generated by Claude 4.6 Sonnet. |
| `data/router_synthetic_routing_5000_valid_holdout.json` | 100 | JSON array | Holdout validation set for router accuracy scoring. |
| `data/router_reasoning_dpo_5000.jsonl` | — | JSONL | DPO preference pairs derived from 5k corpus + injected failure patterns. |

### 5.2 Raw Results Files

| File | Entries | Schema | Key Content |
| --- | ---: | --- | --- |
| `results/v2_both_raw.jsonl` | 560 | `{timestamp, split, item_id, method, K_used, clamp, routing, generated_text_preview, semantic_sim, perplexity, latency_s, real_mode}` | Primary Phase 2 data artifact. 140 questions × 4 methods. |
| `results/v2_md_clamp_ablation.jsonl` | 120 | Same + `clamp_mode` | Phase 2b clamp ablation. 40 MD × 3 methods. |
| `results/v2_md_routing_ablation.jsonl` | 120 | Same + `routing_fn` | Phase 2c routing gap. 40 MD × 3 methods. |
| `results/real_benchmark_results.json` | 400 | JSON array | Phase 1 v1 raw inference data. |
| `results/injection_hypotheses_eval.jsonl` | ~360 | `{item_id, domains, method, semantic_sim, perplexity, exact_match, token_f1, latency_s, prediction_text}` | 9-technique injection ablation on internal 40-item MD. |
| `results/injection_hypotheses_eval_full_20260408.jsonl` | ~3600 | Same | Full injection ablation run on 100-item external data. |
| `results/tcar_collaborative_pilot_10.jsonl` | 10 | TCAR result schema | First TCAR pilot with natural-language router. |
| `results/tcar_oracle_collaborative_pilot_10.jsonl` | 10 | Same | Oracle-routed TCAR pilot. |
| `results/tcar_collaborative_sft5000_mpsfix_pilot10.jsonl` | 10 | Same | TCAR pilot with SFT-trained router. |
| `results/tcar_collaborative_dpo5000_mpsfix_100.jsonl` | 100 | Same | **Final TCAR 100-item benchmark.** Contains latency breakdown per component. |
| `results/tcar_verifier_sft_pilot10.jsonl` | 10 | Same | Verifier-only TCAR pilot. |
| `results/md_head_to_head_v2_mistral_only_100.jsonl` | 100 | Mistral results | Full 100-item Mistral baseline run. |
| `results/md_head_to_head_v2_qwen_only_100.jsonl` | 100 | Qwen results | Full 100-item Qwen variant runs (3 methods). |
| `results/md_pairwise_merge_vs_mistral_v2_strat30.jsonl` | 30 | Pairwise results | Blind judging: weighted_merge vs Mistral. |
| `results/md_pairwise_latelayer_vs_mistral_v2_strat30.jsonl` | 30 | Same | Blind judging: late_layer_injection vs Mistral. |
| `results/md_pairwise_seqrev_vs_mistral_v2_strat30.jsonl` | 30 | Same | Blind judging: sequential_reverse vs Mistral. |
| `results/gated_routing_embedding_results.json` | 140 | JSON | Gated router evaluation (100 SD + 40 MD). |
| `results/router_accuracy_sft_5000_valid_holdout_mpsfix.json` | 100 | JSON | SFT router accuracy on holdout. |
| `results/router_accuracy_dpo_5000_valid_holdout_mpsfix.json` | 100 | JSON | DPO router accuracy on holdout. |

### 5.3 Summary & Analysis Reports

| File | Description |
| --- | --- |
| `results/v2_decision_summary.md` | Human-readable summary of all 5 v2 hypothesis verdicts with exact numbers. |
| `results/v2_clamp_ablation_summary.md` | Summary of norm-ratio vs weight-cap clamp findings. |
| `results/v2_routing_gap_summary.md` | Summary of oracle vs real routing gap. |
| `results/md_external_v2_blind_report.md` | Complete external evaluation report with soft metrics + blind judging. |
| `results/tcar_pilot_10_report.md` | TCAR pilot analysis with router diagnosis. |
| `results/tcar_dpo_final_100_report_2026_04_09.md` | Final TCAR 100-item report with latency breakdown and tail analysis. |
| `results/tcar_verifier_pilot10_2026_04_09.md` | Verifier-only TCAR speed-quality tradeoff analysis. |
| `results/router_sft_mpsfix_results_2026_04_09.md` | Router SFT accuracy results and downstream impact. |
| `results/router_upgrade_execution_log_2026_04_08.md` | Detailed execution log of the router training phase. |
| `results/tested_hypotheses_and_results.md` | Master hypothesis tracking document (11 hypotheses across 6 phases). |
| `FINAL_CONCLUSION_NOTE_2026_04_09.md` | **The single best "where we ended up and what it means" document.** |
| `FINAL_EXPERIMENT_REPORT_2026_04.md` | Corrected final experiment ledger combining all phases. |
| `FULL_PROJECT_SUMMARY.md` | Complete technical summary with file-by-file data map and reproduction instructions. |

### 5.4 Configuration Files

| File | Description |
| --- | --- |
| `configs/uma_experiments.yaml` | Experiment configuration grid: 49 experiments across 7 methods × 7 datasets. Defines c values (0.3, 0.5, 0.7, 1.0), K values (1, 2, 3), and dataset assignments. |
| `backend/expert_registry.json` | Registry of all 20 domain adapters — paths to `.safetensors` files and VRAM estimates. Each adapter is 20.15 MB. |

### 5.5 Trained Adapter Checkpoints

| Directory | Description |
| --- | --- |
| `backend/expert_adapters/{DOMAIN}/` | 20 domain-specific LoRA adapter directories, each containing `adapters.safetensors` |
| `router_adapters/router_reasoning_sft_5000_mpsfix/` | SFT-trained router LoRA checkpoint (best routing accuracy: 85%) |
| `router_adapters/router_reasoning_dpo_5000_mpsfix/` | DPO-trained router LoRA checkpoint (regressed to 42% accuracy) |
| `router_adapters/router_reasoning_sft_smoke200/` | Smoke-test SFT checkpoint |
| `router_adapters/router_reasoning_dpo_smoke200/` | Smoke-test DPO checkpoint |

### 5.6 Total Inference Count

| Phase | Inferences | Notes |
| --- | ---: | --- |
| v1 Single-Domain | 400 | 100 questions × 4 methods |
| v2 Multi-Domain (SD+MD) | 560 | 140 questions × 4 methods |
| v2b Clamp Ablation | 120 | 40 MD × 3 methods |
| v2c Routing Gap | 120 | 40 MD × 3 methods |
| Router Ablation | ~260 | 40 MD × 3 routers + gated |
| Gated Routing | 140 | 100 SD + 40 MD |
| 9-Technique Injection (internal) | ~360 | 40 MD × 9 methods |
| 9-Technique Injection (external) | ~900 | 100 × 9 methods |
| Mistral Comparison (internal) | 100 | Via Ollama |
| TCAR Pilots | ~60 | Multiple 10-item slices |
| External 100-item Qwen Runs | 300 | 100 × 3 methods |
| External 100-item Mistral Run | 100 | Via Ollama |
| Blind Judging | 90 | 30 items × 3 methods (judge calls) |
| TCAR Final 100-item | 100 | DPO router |
| Verifier TCAR Pilot | 10 | SFT router |
| DES Pilot (abandoned) | 4 | Stochastic sampling |
| **Estimated Total** | **~3,600+** | **Real model inferences on Apple Silicon** |

---

## 6. Architectural Discoveries (Bugs & Surprises)

These were found *during* research — not planned in advance.

1. **The Backend Clamp ≠ The Paper's Clamp.** The paper described per-layer norm-ratio; the code implemented per-adapter weight-cap. Fixed in v2b, but turned out to make no difference.

2. **The Router Is One-Hot Only.** The `Orchestrator.route()` always returns a one-hot vector. It cannot express fractional multi-domain confidence. The `GatedRouter` and its probability-gap logic were therefore inoperable in the real pipeline. Oracle routing was required to test K=2.

3. **v1 Data Was Templated.** ~90% of ground-truth answers follow two boilerplate templates. Cosine similarity is hypersensitive to lexical drift under this regime — generating "quantum chemistry" when the answer expects "robotics" causes disproportionate similarity drops.

4. **H5 Was Tautological Under Weight-Cap.** When oracle routing assigns 0.5 per adapter and the weight cap is 0.5: `min(0.5, 0.5) = 0.5` (clamped) equals `min(0.5, 999) = 0.5` (unclamped). They are algebraically identical. H5 could never pass under weight-cap.

5. **`trl` Trainer Crashed on MPS.** The standard HuggingFace `trl` SFT/DPO trainer crashed with an Apple Metal runtime abort during the first optimizer step on the 5k dataset. Recovery: replaced with manual PyTorch training loops using explicit response-only loss masking.

6. **DPO Regressed Router Accuracy.** Despite successfully optimizing the pairwise preference objective (rewards/accuracies = 0.80 during training), DPO caused routing exact-match to drop from 85% to 42%. DPO appears to optimize for stylistic preference rather than classification accuracy in routing tasks.

7. **Soft Metrics Overstate Progress.** The soft-metric leader (`sequential_reverse`) was NOT the best Qwen method under blind judging. `weighted_merge` (the simplest method) was least-bad. This is a critical methodological finding.

8. **Demo Visualization Uses Stale Numbers.** The `demo/index.html` still displays the Phase 4 internal benchmark numbers (Synapta: 0.6525 vs Mistral: 0.617, "+5.7% yield"). These numbers were superseded by the external benchmark where Mistral wins on quality. Any public-facing demo must be updated.

9. **The Adversarial Agent Cluster Is Architecturally Complete But Unevaluated.** The `agent_cluster.py` (534 lines) implements a sophisticated multi-agent adversarial pipeline with collusion detection, quality constitution, evidence packets, and composer tournaments. However, there is no systematic benchmark of this system's answer quality. It is referenced in the API as `mode=cluster_strict` but has no corresponding evaluation report.

---

## 7. Honest Final Assessment

### What the project proved:

1. **Unclamped multi-adapter mixing is catastrophic.** Similarity drops to 0.557 with individual collapses to <0.1. Clamping is structurally necessary.
2. **Single-domain questions don't benefit from composition.** The second adapter is pure noise (Δ = −0.011).
3. **Multi-domain questions do benefit from composition.** Under correct conditions, composition improves metrics (+0.017), though below the pre-registered threshold.
4. **Router quality dominates system performance.** SFT improved exact-match from 10% to 85%. The routing bottleneck is real and solvable.
5. **Collaborative inference has a higher semantic ceiling than static weight blending.** TCAR pushed the Qwen system to near-Mistral quality (0.6900 vs 0.6907 semantic similarity).
6. **Evaluation methodology matters enormously.** Blind judging overturned the soft-metric ranking of methods.
7. **Apple Silicon multi-expert serving is architecturally viable.** 20 experts in ~1.1 GB with <5s single-pass latency.

### What the project did NOT prove:

1. **"Small routed model beats Mistral" is NOT supported.** External blind evidence clearly favors Mistral on answer quality (23+ wins vs 6 max for any Qwen method).
2. **DPO does not help routing.** It damaged accuracy despite optimizing the preference objective.
3. **Collaborative TCAR is not deployment-grade.** Quality goes up, but latency is too high (24s mean, 86s P95).
4. **"LoRI + CoMoL" as described in some documents is not implemented.** The orthogonal subspace projection and core-space blending described in theoretical sections do not exist in the codebase. What **is** implemented is `norm_ratio` clamping + `late_layer_injection` gating.

### Strongest defensible conclusion:

> Small-model multi-expert inference on Apple Silicon is real, fast, and architecturally promising, but parameter-space merging alone does not solve cross-domain reasoning, and the current collaborative alternatives do not yet beat Mistral on externally evaluated answer quality. The strongest research contribution is demonstrating that evaluation methodology — specifically blind judging versus soft embedding metrics — can completely change the scientific conclusion about which multi-adapter composition method is best.

---

## 8. Recommended Paper Positioning

### Frame as:
- A rigorous, multi-phase empirical study of multi-adapter composition limits on edge hardware
- A demonstration that routing and evaluation methodology dominate the conclusions
- A negative-plus-positive result: parameter merging is insufficient for cross-domain reasoning, but collaborative inference offers a measurable ceiling at a real latency cost
- A systems contribution: practical multi-expert serving on Apple Silicon UMA

### Do NOT frame as:
- Beating Mistral on answer quality
- Solved multi-domain reasoning
- Production-grade real-time collaborative inference
- LoRI/CoMoL as implemented breakthroughs (they are theoretical/unimplemented)

---

## 9. Documents Containing Superseded or Misleading Claims

The following documents in the repository contain claims that were later corrected by more rigorous evaluation. The paper author **must not cite these as final results**:

| Document | Misleading Claim | Corrected By |
| --- | --- | --- |
| `FINAL_EXPERIMENT_REPORT.md` | "We have definitively proven that Synapta out-thinks a generalized 7B model" (+5.7%) | `results/md_external_v2_blind_report.md` — Mistral wins 23–6 on blind judging |
| `results/mistral_vs_synapta_verified.md` | "Synapta's multi-adapter composition logic outperforms Mistral" | Same — internal benchmark only; not validated externally |
| `demo/index.html` | Displays "0.6525 vs 0.617" with "+5.7% Yield" | Same — stale internal metrics |
| `README.md` | "Phase 6 Breakthrough" implying LoRI+CoMoL are implemented | `EXPERIMENT_TODO.md` — LoRI/CoMoL are theoretical; actual code uses norm_ratio clamp |
| `paper.md` (early draft) | References "token-level Core-Space mixture" as implemented | Codebase — only prompt-level routing + sequential segments exist |
| `mistral_vs_synapta.md` (auto-generated) | "Mistral lacks targeted domain knowledge" | Blind judging shows Mistral wins on answer quality |

The **canonical final-state documents** are:
- `FINAL_CONCLUSION_NOTE_2026_04_09.md` — Most honest assessment
- `FINAL_EXPERIMENT_REPORT_2026_04.md` — Corrected experiment ledger
- `results/tested_hypotheses_and_results.md` — Master hypothesis tracker
- `results/md_external_v2_blind_report.md` — External evaluation (ground truth)

---

*Extracted: 2026-04-15. Source: Complete codebase scan of `/Users/uditjain/Desktop/adapter/`.*
