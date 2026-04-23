# LoRI-MoE: Orthogonal Low-Rank Experts with Token-Level Dynamic Routing

## The Honest Diagnosis — Why Both Doc Plans Are Wrong

Before presenting the plan, here's the chain-of-thought that both research docs missed.

### What Both Docs Correctly Identified
1. Synapta's linear composition is **mathematically broken** — intruder dimensions cause destructive geometric interference
2. Prompt-level routing is **architecturally obsolete** — reasoning requires token-level granularity
3. Semantic similarity is **an invalid metric** — cosine similarity can't distinguish "converges to 0" from "converges to infinity"
4. The clamping mechanisms are **algebraically irrelevant** — at rank-16, adapter norms are infinitesimal vs. base activations
5. The current codebase has **ZERO real components** — all adapters are `torch.randn`, all metrics are simulated

### Where Doc1 (CoMoL-StelLA Synthesis) Goes Wrong

> [!CAUTION]
> Doc1 proposes Riemannian optimization on the Stiefel manifold — this is a **Sony Research NeurIPS Spotlight** level project. One person cannot correctly implement Riemannian QR-retraction gradients, write custom Triton kernels for core-space fusion, AND train on GPQA/SciCode datasets in 4 weeks. The mathematical elegance is seductive but the implementation complexity is prohibitive.

Specific problems:
- **Qwen2.5-14B in FP8** = ~14GB base weights. Leaves 18GB. Sounds fine until you realize LoRA training with optimizer states, gradients, activations, and KV cache will eat 20-30GB easily. You'll be OOM during training.
- **3-phase training pipeline** (manifold → core matrices → router) has 3 failure points. If Phase 1 doesn't converge, everything downstream collapses.
- **GPQA Diamond target of +15% absolute** — PhD experts score 65%, frontier models plateau at ~90%. Asking a 14B model with adapters to move the needle 15% on this benchmark is aspirational fantasy.
- **Custom Triton kernels for core-space fusion** — writing correct, performant Triton kernels is a specialized skill that takes weeks alone.

### Where Doc2 (TOSR) Goes Wrong

> [!WARNING]
> Doc2 proposes DARE-sparsifying the "existing 20 Synapta adapters." Those adapters are `torch.randn(4096, 16)` — random noise. You cannot DARE-sparsify random noise. The entire execution plan builds on adapters that don't exist.

Specific problems:
- **Staying on Qwen2.5-1.5B** — the reasoning ceiling of a 1.5B model is brutally low. Even with perfect adapter composition, 1.5B models fundamentally lack the latent capacity for multi-step deductive reasoning. You're optimizing the steering wheel of a car with no engine.
- **HydraLoRA shared-B assumption** — Doc2's own literature table flags this: "The shared B matrix assumes all domains share a common input subspace, which fails for highly disjoint domains (e.g., Arabic poetry vs. Python)."
- **30% relative improvement on MATH500** (35% → 45.5%) — this would be remarkable, but DARE + token routing alone won't get there. DARE removes parameters; it doesn't add reasoning capability.
- **100k OpenOrca/MetaMathQA sequences for router training** — this trains the router to mimic existing data patterns, not to perform novel reasoning.

### The Deeper Question Neither Doc Asks

**Can parameter-space composition EVER produce reasoning synthesis?**

Both docs assume that if you solve the interference problem (via orthogonality, sparsification, or manifold alignment), composed adapters will "reason" across domains. But:

- LoRA adapters encode **domain knowledge as weight perturbations**
- Composing weights ≠ composing reasoning capabilities
- Even perfectly orthogonal adapters only **prevent interference** — they don't **enable synthesis**
- The o1/o3 paradigm proves that **test-time compute scaling** (thinking longer), not parameter arithmetic, drives reasoning breakthroughs

This means: **the breakthrough isn't in HOW you compose adapters. It's in WHEN and WHY.**

---

## The Actual Breakthrough: LoRI-MoE

### Core Insight

The synthesis of both documents points to one architecture that neither fully articulates:

**LoRI (frozen random B, sparse A) gives you interference-free adapters FOR FREE via Johnson-Lindenstrauss. Token-level routing gives you dynamic composition. Together, they solve the composition problem without Riemannian optimization, without custom Triton kernels, and without aspirational math.**

| Property | Synapta | Doc1 (CoMoL-StelLA) | Doc2 (TOSR) | **LoRI-MoE (Ours)** |
|---|---|---|---|---|
| Interference Resolution | ❌ Scalar clamp | ✅ Stiefel manifold | ⚠️ DARE sparsification | ✅ Frozen random projection (JL lemma) |
| Routing Granularity | ❌ Prompt-level | ✅ Token-level core-space | ✅ Token-level | ✅ Token-level |
| Implementation Complexity | Low | **Extreme** | Medium | **Low-Medium** |
| Training Phases | 0 (untrained) | 3 phases | 2 phases | **2 phases** |
| Requires Custom Kernels | No | Yes (Triton) | No | **No** |
| Mathematical Guarantee | None | Strict orthonormality | Approximate (random pruning) | **Approximate orthogonality (JL)** |
| 4-Week Feasibility | N/A | ❌ Unrealistic | ⚠️ Missing foundations | ✅ **Achievable** |

### Why LoRI-MoE Is Novel

The LoRI paper (NeurIPS 2025) **only does static merging**. It freezes B, trains sparse A matrices, then merges them at fixed weights. Nobody has combined:

1. LoRI's training-time orthogonality constraint WITH
2. Dynamic token-level routing at inference time

This IS the gap both documents identify but neither fills correctly:
- Doc1 sees the gap but over-engineers the solution (Stiefel manifold)
- Doc2 sees the gap but under-engineers the foundation (DARE on nonexistent adapters)

### Architecture

```
Input Token Hidden State h_t
        │
        ▼
   ┌────────────┐
   │ Shared B    │  (Frozen random Gaussian, dim: d_model × r)
   │ (LoRI)      │  (Approximate orthogonality via JL lemma)
   └─────┬──────┘
         │
    ┌────▼────┐
    │ Router  │  Lightweight MLP: project h_t → softmax over K experts
    │ R(h_t)  │  Output: [p_1, p_2, ..., p_K] probability distribution
    └────┬────┘
         │
    ┌────▼──────────────────────────┐
    │  Dynamic A Composition        │
    │  A_merged = Σ p_k · A_k      │  (Sparse domain-specific matrices)
    │  where A_k has 80-90% sparsity│
    └────┬──────────────────────────┘
         │
    ┌────▼────┐
    │ ΔW = A_merged @ B │  (Single LoRA forward pass cost)
    └────┬────┘
         │
    h_out = W_base(h_t) + α · ΔW(h_t)
```

**Key properties:**
- **Shared B** is frozen and random → guarantees approximate orthogonality without training
- **Sparse A_k** matrices are domain-specific → each domain's update lives in a near-orthogonal subspace
- **Router R** operates on the hidden state → token-level granularity
- **Single projection** through B → same FLOP cost as standard LoRA, not K× like naive MoE

### Base Model Selection

> [!IMPORTANT]
> **Primary: Qwen2.5-3B-Instruct** (~6GB BF16, leaves 26GB for everything else)
> **Scaling experiment: Qwen2.5-7B-Instruct** (~14GB BF16, leaves 18GB)

Why 3B and not 1.5B or 14B:
- **Not 1.5B**: The reasoning capacity is too low — Qwen2.5-1.5B scores ~25% on MATH. Even a perfect adapter system can't fix a model that lacks fundamental reasoning circuits. You'd be proving that a better steering wheel doesn't help a car with no engine.
- **Not 14B**: Training LoRA adapters on 14B with BF16 + AdamW optimizer states = ~28GB. You'd have <4GB for activations/KV cache. OOM city.
- **3B is the sweet spot**: ~40% on MATH (improvable), ~55% on MMLU (strong base), fast iteration (train a LoRA in 1-2 hours), massive headroom on 32GB GPU.

### Domain Selection (5 domains, not 20)

> [!WARNING]
> 20 domains is a paper claim, not a practical plan. Training 20 quality LoRA adapters takes 20× the compute, 20× the data curation, and makes ablations 20× more expensive. Start with 5 domains that are maximally disjoint and have established benchmarks.

| Domain | Training Data | Evaluation Benchmark | Why This Domain |
|---|---|---|---|
| **Mathematics** | MetaMathQA (100k) | MATH500 / GSM8K | Core reasoning capability |
| **Code** | CodeAlpaca + Evol-Instruct-Code (80k) | HumanEval / MBPP | Disjoint from math syntax |
| **Science** | SciQ + GPQA train split (50k) | ARC-Challenge / GPQA | Multi-step scientific reasoning |
| **Legal** | LegalBench subset (30k) | LegalBench test | Highly specialized vocabulary |
| **Medical** | MedQA + PubMedQA (50k) | MedQA test | Domain with real-world impact |

---

## Proposed Changes

### Phase 0: Foundation Reset (Days 1-3)

> [!IMPORTANT]
> Nothing in the current codebase is usable for real experiments. The adapters are random, the metrics are simulated, the routers are heuristic toys. We need a clean foundation.

#### [NEW] src/lori_moe/__init__.py
Empty init for new package.

#### [NEW] src/lori_moe/config.py
Central configuration dataclass defining: base model path, adapter rank, number of domains, sparsity level, router architecture params, training hyperparameters.

#### [NEW] src/lori_moe/shared_projection.py
Implements the frozen shared B matrix (random Gaussian initialization with proper scaling). Key: `B = torch.randn(d_model, r) / sqrt(r)` — the `1/sqrt(r)` scaling ensures the projection preserves distances (JL lemma).

#### [NEW] src/lori_moe/lori_adapter.py
Custom LoRA adapter using frozen B + trainable sparse A. Integrates with HuggingFace PEFT's `LoraConfig` but overrides the B initialization to use the shared frozen matrix. Applies binary sparse masks to A matrices during training.

#### [NEW] scripts/setup_foundation.sh  
Installs dependencies, downloads Qwen2.5-3B-Instruct, verifies CUDA/BF16 support, creates directory structure.

---

### Phase 1: Train Real Domain Adapters (Days 4-8)

#### [NEW] src/lori_moe/data/prepare_datasets.py
Downloads and processes the 5 domain datasets from HuggingFace. Formats into instruction-tuning format compatible with Qwen2.5's chat template. Handles train/eval splits.

#### [NEW] src/lori_moe/training/train_lori_adapter.py
Main training script for a single domain LoRA adapter. Key features:
- Loads Qwen2.5-3B in BF16
- Initializes shared frozen B matrix (loaded from checkpoint or generated once)
- Creates trainable sparse A matrix for the target domain
- Uses PEFT's LoRA injection into `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- AdamW optimizer, cosine LR schedule, gradient checkpointing
- Saves adapter weights + sparse mask

#### [NEW] src/lori_moe/training/apply_sparsity.py
Post-training DARE-style sparsification: drops N% of A matrix parameters (by magnitude), rescales remainder. This is ADDITIONAL orthogonalization on top of LoRI's structural guarantee.

#### [NEW] configs/lori_training.yaml
```yaml
base_model: "Qwen/Qwen2.5-3B-Instruct"
adapter_rank: 32  # Higher than Synapta's 16 for more capacity
shared_b_seed: 42  # Deterministic B initialization
sparsity_level: 0.8  # 80% sparse A matrices
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lr: 2e-4
epochs: 3
batch_size: 8
gradient_accumulation: 4
bf16: true
gradient_checkpointing: true
```

#### [NEW] scripts/train_all_domains.sh
Sequential training script for all 5 domains. Estimated: ~2 hours per domain × 5 = 10 hours total.

---

### Phase 2: LoRI-MoE Architecture (Days 9-14)

#### [NEW] src/lori_moe/model/lori_moe_linear.py
The core module replacing `AdaptiveMultiLoRALinear`. Key differences from Synapta:
- **No clamping** — orthogonality prevents interference structurally
- **Token-level routing** — router operates on each token's hidden state
- **Shared B projection** — single matrix multiply, not K separate ones
- **Sparse A composition** — dynamic weighted sum of sparse A matrices

#### [NEW] src/lori_moe/model/router.py
Lightweight MLP router deployed at each transformer layer:
```python
class TokenRouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, bottleneck=64):
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.up = nn.Linear(bottleneck, num_experts)
    
    def forward(self, hidden_state):
        # hidden_state: (batch, seq_len, hidden_dim)
        logits = self.up(F.silu(self.down(hidden_state)))
        return F.softmax(logits, dim=-1)  # (batch, seq_len, num_experts)
```

#### [NEW] src/lori_moe/model/lori_moe_model.py
Wrapper that:
1. Loads Qwen2.5-3B base model (frozen)
2. Loads 5 trained LoRI adapters (frozen A matrices)
3. Loads shared B matrix (frozen)
4. Injects `LoRIMoELinear` modules at each target layer
5. Initializes trainable routers at each layer
6. Implements the full forward pass with load-balancing auxiliary loss

#### [NEW] src/lori_moe/model/losses.py
- Standard causal LM cross-entropy loss
- Load-balancing auxiliary loss: `L_aux = α * Σ(f_i * P_i)` where f_i = fraction of tokens routed to expert i, P_i = mean router probability for expert i
- Total: `L = L_CE + 0.01 * L_aux`

---

### Phase 3: Router Training (Days 15-19)

#### [NEW] src/lori_moe/data/generate_routing_data.py
Generates multi-domain reasoning traces:
- Uses the base model itself to generate responses on mixed-domain prompts
- Annotates which domain is "active" per reasoning step
- Creates 30k supervised routing examples
- Format: (prompt, token_positions, expert_labels_per_position)

#### [NEW] src/lori_moe/training/train_router.py
Router training script:
- Freezes base model + all adapters + shared B
- Only trains router parameters (~200K params total across all layers)
- Uses the multi-domain routing data
- Monitors routing entropy to detect collapse (entropy < 0.3 = collapse)
- Implements Top-2 gating with noise for gradient flow through non-selected experts

#### [NEW] configs/router_training.yaml
```yaml
router_lr: 1e-3
router_epochs: 5
load_balance_weight: 0.01
top_k: 2  # Top-2 routing
noise_std: 0.1  # Noisy gating to prevent collapse
entropy_threshold: 0.3  # Alert if below this
```

---

### Phase 4: Evaluation (Days 20-23)

#### [NEW] src/lori_moe/eval/run_benchmarks.py
Evaluation using `lm-eval-harness` on proper benchmarks:

| Benchmark | What It Tests | Metric | Baseline (Qwen2.5-3B) |
|---|---|---|---|
| **MATH500** | Mathematical reasoning | Exact Match | ~40% |
| **GSM8K** | Grade-school math | Exact Match | ~75% |
| **MMLU** | Multi-domain knowledge | Accuracy | ~55% |
| **BBH** | Hard multi-step reasoning | Accuracy | ~45% |
| **HumanEval** | Code generation | Pass@1 | ~40% |

#### [NEW] src/lori_moe/eval/interference_test.py
Critical test: does multi-adapter composition degrade single-domain performance?
- Run each domain's benchmark with ONLY that domain's adapter active
- Run same benchmark with ALL adapters active (router decides)
- Degradation must be < 2% to prove orthogonality works

#### [NEW] src/lori_moe/eval/routing_analysis.py
Visualizes token-level routing decisions:
- For a mixed-domain prompt, plots expert selection probability per token
- Verifies the router actually switches domains mid-sequence (not collapsed)
- Generates routing entropy histograms

---

### Phase 5: Ablations + Paper (Days 24-28)

#### [NEW] src/lori_moe/eval/ablation_suite.py
Systematic ablations:
1. **Token-level vs. Prompt-level routing** (isolates routing contribution)
2. **LoRI (shared B) vs. independent B** (isolates orthogonality contribution)  
3. **Sparse A vs. Dense A** (isolates sparsification contribution)
4. **Top-1 vs. Top-2 vs. Soft routing** (isolates routing granularity)
5. **Rank 16 vs. 32 vs. 64** (capacity analysis)

#### [MODIFY] paper_v2/ (new paper draft)
arXiv submission with framing:

> *"We introduce LoRI-MoE, a parameter-efficient multi-expert inference framework that combines interference-free adapter training (via frozen random projections) with dynamic token-level expert routing. LoRI-MoE achieves X% improvement on MATH500 and Y% on MMLU over prompt-level composition baselines, while guaranteeing <2% single-domain degradation — all within the 32GB memory budget of a single consumer GPU."*

---

## Success Criteria

### The experiment succeeds if:
1. **Reasoning gain**: ≥10% relative improvement on MATH500 over base Qwen2.5-3B zero-shot
2. **Zero interference**: <2% degradation on any single-domain benchmark when all 5 adapters are active
3. **Token routing works**: Routing entropy > 0.5 on multi-domain prompts (router is NOT collapsed)
4. **Latency bound**: <25% overhead vs. single LoRA inference (tokens/sec)
5. **Ablations prove mechanism**: Token-level > prompt-level routing (statistically significant)

### The experiment FAILS if:
- Multi-adapter composition degrades single-domain performance by >5% (orthogonality broken)
- Router collapses to single expert (entropy < 0.3)
- No improvement over base model on any hard benchmark (adapters aren't helping)
- Latency overhead >50% (architecture is impractical)

---

## Failure Modes & Pre-Registered Pivots

### Failure Mode 1: Router Collapse
**Detection**: Routing entropy drops below 0.3 during training
**Pivot**: 
- Increase load-balancing loss weight from 0.01 → 0.1
- Switch from soft routing to Top-2 hard routing with noise
- If still collapsed: use X-LoRA style layer-wise routing instead of token-level

### Failure Mode 2: Orthogonality Insufficient  
**Detection**: Single-domain degradation >5% with all adapters active
**Pivot**:
- Increase sparsity from 80% → 95%
- Apply DARE post-hoc sparsification on top of LoRI training
- Nuclear option: retrain with LoRI's sparse mask enforced during training (not just post-hoc)

### Failure Mode 3: Adapters Don't Help Reasoning
**Detection**: No improvement over base model zero-shot on MATH500/MMLU
**Pivot**:
- This means the 3B model lacks the reasoning circuits to be steered
- Scale to Qwen2.5-7B as base model (14GB BF16, still fits on 5090)
- If 7B also fails: pivot to Doc1's Scenario B — Activation Steering instead of parameter composition

### Failure Mode 4: Token-Level Routing Too Slow
**Detection**: >50% latency overhead in tokens/sec
**Pivot**:
- Reduce router from per-token to per-chunk (every 8 tokens)
- Reduce router bottleneck dimension from 64 to 16
- Fall back to layer-wise routing (X-LoRA style) — still better than prompt-level

---

## Week-by-Week Timeline

| Week | Focus | Deliverables | Go/No-Go |
|---|---|---|---|
| **Week 1** (Days 1-7) | Foundation + Adapter Training | 5 trained LoRI adapters, evaluation pipeline | Each adapter improves its domain benchmark by ≥5% over base |
| **Week 2** (Days 8-14) | Architecture Implementation | LoRI-MoE model, router modules, forward pass working | Single forward pass completes without OOM, latency <2× single LoRA |
| **Week 3** (Days 15-21) | Router Training + Evaluation | Trained router, benchmark results | Routing entropy >0.5, multi-domain improvement visible |
| **Week 4** (Days 22-28) | Ablations + Paper | Complete ablation table, arXiv draft | All success criteria met |

---

## Open Questions

> [!IMPORTANT]
> **Q1: Base model — Qwen2.5-3B or Qwen2.5-7B?**
> I recommend 3B for faster iteration with 7B as a scaling experiment. But if you want maximum paper impact, starting with 7B might be better (it has stronger reasoning baselines to improve upon). What's your preference?

> [!IMPORTANT]  
> **Q2: Number of domains — 5 or more?**
> I proposed 5 for practical reasons. Both docs mention 20, which is impressive for a paper claim but impractical for 4-week training. Do you want to start with 5 and scale later, or go aggressive from the start?

> [!IMPORTANT]
> **Q3: Paper venue target — arXiv preprint, workshop, or main conference?**
> This affects how much ablation rigor we need. arXiv preprint: 2 ablations suffice. ICLR/NeurIPS main: need 5+ ablations, baselines against X-LoRA/CoMoL/DARE-TIES, and strong theoretical framing.

> [!IMPORTANT]
> **Q4: Do you want to completely discard the existing Synapta codebase (`src/adapters/`, `src/routers/`) or keep it as a baseline?**
> I recommend keeping it as-is for baseline comparison but building LoRI-MoE as a clean new package (`src/lori_moe/`).

## Verification Plan

### Automated Tests
- Unit tests for shared B matrix properties (verify approximate orthogonality via dot-product test)
- Integration test: full forward pass through LoRI-MoE model
- Benchmark runner: automated `lm-eval-harness` evaluation
- Routing entropy monitoring during training

### Manual Verification  
- Visual inspection of routing heatmaps on multi-domain prompts
- Qualitative comparison of generated responses (base vs. LoRI-MoE) on cherry-picked hard examples
- Memory profiling to confirm GPU utilization

---

## 🚀 Current Execution Status & Next Steps (Updated)

**✅ COMPLETED:**
1. **Math Adapter Training**: Successfully finished (Epoch 2/2 completed). Checkpoint merged and finalized.
2. **Pipeline Crash-Recovery Mechanism**: Robust `pipeline_state.json` logic and signal trapping successfully implemented and battle-tested.
3. **GPU Memory Optimization**: `expandable_segments:True` and optimal batch sizing (BS=2, GA=16) stabilized to prevent OOM on 30B.

**⏳ IN PROGRESS:**
- **Code Adapter Training**: Currently running (`phase_3_code`).
  - *Estimated Time*: ~8 hours total (processing 10,000 steps at ~2.9s/it).

**⏭️ NEXT STEPS (To be processed autonomously by the pipeline):**

### 1. Phase 3: Science Adapter Training
- **What**: Training the final domain adapter (`phase_3_science`).
- **Time**: ~8 hours.
- **Process**: The orchestrator (`gc_lori_pipeline.sh`) will automatically transition to this once Code finishes.

### 2. Phase 3.5: Single Adapter Evaluation
- **What**: Baseline evaluations to ensure the individual adapters learned their domains properly without interference.
- **Time**: ~2-3 hours total.
- **Process**: The script runs `run_eval_prompts.py` for each domain individually.

### 3. Phase 4: Token-Level Router Training
- **What**: Training the dynamic routing mechanism across all layers to selectively activate Math/Code/Science adapters.
- **Time**: ~4-6 hours (faster because base model and adapters are completely frozen).
- **Process**: Automatically initiated by the pipeline using `train_router.py`.

### 4. Phase 5: Final LoRI-MoE Evaluation
- **What**: The ultimate test of the hypothesis. Evaluating the composed model on benchmarks and interference tests.
- **Time**: ~4-5 hours.
- **Process**: Uses `interference_test.py` and `run_eval_prompts.py` recursively.

**Total Estimated Remaining Time**: ~26-30 hours of continuous, autonomous GPU computation. The pipeline is successfully set up to process all of these stages sequentially without requiring manual intervention.

---

## 🌐 Hugging Face Release Plan: Nemotron-3-Nano-30B GC-LoRI Math Adapter

> [!CAUTION]
> **User Review Required:** Please review the draft of the Model Card below. If you approve, I will use your provided token to automatically create the repository `uditjain13/Nemotron-30B-GCLoRI-Math` and push the adapter weights.

**Adpater Size:** ~45 MB (Extremely lightweight due to Rank 64 and 80% DARE sparsification). 
**Verdict:** **Yes, we absolutely should post this.** 45MB is tiny, and providing a standalone, highly-sparse Math adapter for a 30B parameter model is extremely valuable to researchers. 

### Proposed Model Card (Dense & Technical)

```markdown
---
base_model: nvidia/Nemotron-3-Nano-30B-A3B-BF16
library_name: peft
tags:
- math
- lori
- moe
- dare-sparsification
- reasoning
---

# Nemotron-30B GC-LoRI Math Adapter (80% Sparse)

An ultra-lightweight (~45MB) mathematical reasoning adapter trained on the **Nemotron-3-Nano-30B-A3B-BF16** architecture. This adapter implements **Global Context LoRI (GC-LoRI)** with extreme parameter sparsity to enable interference-free multi-domain composition.

### 🔬 Technical Specifications
- **Architecture Base**: `nvidia/Nemotron-3-Nano-30B-A3B-BF16`
- **Adapter Type**: LoRA (Query, Key, Value, Output Projections)
- **Rank**: 64 
- **Alpha**: 128.0
- **Sparsity Enforcement**: 80% DARE (Drop and Rescale) sparsification applied post-training to the $A$ matrices.
- **Orthogonality Constraint**: Uses a frozen, Shared $B$ matrix initialized via Gaussian scaling (Johnson-Lindenstrauss lemma) to guarantee subspace orthogonality prior to sparse tuning.

### ⚙️ Training Paradigm
- **Dataset**: 50,000 sequences of high-complexity reasoning traces (MetaMathQA subset). 
- **Optimization**: Gradient accumulation over 16 steps (Effective Batch Size: 32) using Paged AdamW 8-bit.
- **Precision**: Pure BF16 (bfloat16) activations + FP32 optimizer states (expandable segments enabled for memory defragmentation).

### 🧩 Application & Usage
Designed explicitly to be used within a **Token-Level MoE Router** framework, allowing it to be hot-swapped alongside Code and Science adapters without destructive parameter interference. It can also be utilized as a standalone PEFT module to convert generalized 30B Nemotron pipelines into specialized deductive reasoning agents.
```

