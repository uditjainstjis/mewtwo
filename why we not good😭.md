# LoRI-MoE: Orthogonal Low-Rank Experts with Token-Level Dynamic Routing
2
3
## The Honest Diagnosis — Why Both Doc Plans Are Wrong
4
5
Before presenting the plan, here's the chain-of-thought that both research docs missed.
6
7
### What Both Docs Correctly Identified
8
1. Synapta's linear composition is **mathematically broken** — intruder dimensions cause destructive geometric interference
9
2. Prompt-level routing is **architecturally obsolete** — reasoning requires token-level granularity
10
3. Semantic similarity is **an invalid metric** — cosine similarity can't distinguish "converges to 0" from "converges to infinity"
11
4. The clamping mechanisms are **algebraically irrelevant** — at rank-16, adapter norms are infinitesimal vs. base activations
12
5. The current codebase has **ZERO real components** — all adapters are `torch.randn`, all metrics are simulated
13
14
### Where Doc1 (CoMoL-StelLA Synthesis) Goes Wrong
15
16
> [!CAUTION]
17
> Doc1 proposes Riemannian optimization on the Stiefel manifold — this is a **Sony Research NeurIPS Spotlight** level project. One person cannot correctly implement Riemannian QR-retraction gradients, write custom Triton kernels for core-space fusion, AND train on GPQA/SciCode datasets in 4 weeks. The mathematical elegance is seductive but the implementation complexity is prohibitive.
18
19
Specific problems:
20
- **Qwen2.5-14B in FP8** = ~14GB base weights. Leaves 18GB. Sounds fine until you realize LoRA training with optimizer states, gradients, activations, and KV cache will eat 20-30GB easily. You'll be OOM during training.
21
- **3-phase training pipeline** (manifold → core matrices → router) has 3 failure points. If Phase 1 doesn't converge, everything downstream collapses.
22
- **GPQA Diamond target of +15% absolute** — PhD experts score 65%, frontier models plateau at ~90%. Asking a 14B model with adapters to move the needle 15% on this benchmark is aspirational fantasy.
23
- **Custom Triton kernels for core-space fusion** — writing correct, performant Triton kernels is a specialized skill that takes weeks alone.
24
25
### Where Doc2 (TOSR) Goes Wrong
26
27
> [!WARNING]
28
> Doc2 proposes DARE-sparsifying the "existing 20 Synapta adapters." Those adapters are `torch.randn(4096, 16)` — random noise. You cannot DARE-sparsify random noise. The entire execution plan builds on adapters that don't exist.
29
30
Specific problems:
31
- **Staying on Qwen2.5-1.5B** — the reasoning ceiling of a 1.5B model is brutally low. Even with perfect adapter composition, 1.5B models fundamentally lack the latent capacity for multi-step deductive reasoning. You're optimizing the steering wheel of a car with no engine.
32
- **HydraLoRA shared-B assumption** — Doc2's own literature table flags this: "The shared B matrix assumes all domains share a common input subspace, which fails for highly disjoint domains (e.g., Arabic poetry vs. Python)."
33
- **30% relative improvement on MATH500** (35% → 45.5%) — this would be remarkable, but DARE + token routing alone won't get there. DARE removes parameters; it doesn't add reasoning capability.
34
- **100k OpenOrca/MetaMathQA sequences for router training** — this trains the router to mimic existing data patterns, not to perform novel reasoning.
35
36
### The Deeper Question Neither Doc Asks
37
38
**Can parameter-space composition EVER produce reasoning synthesis?**
39
40
Both docs assume that if you solve the interference problem (via orthogonality, sparsification, or manifold alignment), composed adapters will "reason" across domains. But:
41
42
- LoRA adapters encode **domain knowledge as weight perturbations**
43
- Composing weights ≠ composing reasoning capabilities
44
- Even perfectly orthogonal adapters only **prevent interference** — they don't **enable synthesis**
45
- The o1/o3 paradigm proves that **test-time compute scaling** (thinking longer), not parameter arithmetic, drives reasoning breakthroughs
46
47
This means: **the breakthrough isn't in HOW you compose adapters. It's in WHEN and WHY.**
48
49
---
50
51
## The Actual Breakthrough: LoRI-MoE
52
53
### Core Insight
54
55
The synthesis of both documents points to one architecture that neither fully articulates:
56
57
**LoRI (frozen random B, sparse A) gives you interference-free adapters FOR FREE via Johnson-Lindenstrauss. Token-level routing gives you dynamic composition. Together, they solve the composition problem without Riemannian optimization, without custom Triton kernels, and without aspirational math.**
58
59
| Property | Synapta | Doc1 (CoMoL-StelLA) | Doc2 (TOSR) | **LoRI-MoE (Ours)** |
60
|---|---|---|---|---|
61
| Interference Resolution | ❌ Scalar clamp | ✅ Stiefel manifold | ⚠️ DARE sparsification | ✅ Frozen random projection (JL lemma) |
62
| Routing Granularity | ❌ Prompt-level | ✅ Token-level core-space | ✅ Token-level | ✅ Token-level |
63
| Implementation Complexity | Low | **Extreme** | Medium | **Low-Medium** |
64
| Training Phases | 0 (untrained) | 3 phases | 2 phases | **2 phases** |
65
| Requires Custom Kernels | No | Yes (Triton) | No | **No** |
66
| Mathematical Guarantee | None | Strict orthonormality | Approximate (random pruning) | **Approximate orthogonality (JL)** |
67
| 4-Week Feasibility | N/A | ❌ Unrealistic | ⚠️ Missing foundations | ✅ **Achievable** |
68
69
### Why LoRI-MoE Is Novel
70
71
The LoRI paper (NeurIPS 2025) **only does static merging**. It freezes B, trains sparse A matrices, then merges them at fixed weights. Nobody has combined:
72
73
1. LoRI's training-time orthogonality constraint WITH
74
2. Dynamic token-level routing at inference time
75
76
This IS the gap both documents identify but neither fills correctly:
77
- Doc1 sees the gap but over-engineers the solution (Stiefel manifold)
78
- Doc2 sees the gap but under-engineers the foundation (DARE on nonexistent adapters)
79
80
### Architecture
81
82
```
83
Input Token Hidden State h_t
84
        │
85
        ▼
86
   ┌────────────┐
87
   │ Shared B    │  (Frozen random Gaussian, dim: d_model × r)
88
   │ (LoRI)      │  (Approximate orthogonality via JL lemma)
89
   └─────┬──────┘
90
         │
91
    ┌────▼────┐
92
    │ Router  │  Lightweight MLP: project h_t → softmax over K experts
93
    │ R(h_t)  │  Output: [p_1, p_2, ..., p_K] probability distribution
94
    └────┬────┘
95
         │
96
    ┌────▼──────────────────────────┐
97
    │  Dynamic A Composition        │
98
    │  A_merged = Σ p_k · A_k      │  (Sparse domain-specific matrices)
99
    │  where A_k has 80-90% sparsity│
100
    └────┬──────────────────────────┘
101
         │
102
    ┌────▼────┐
103
    │ ΔW = A_merged @ B │  (Single LoRA forward pass cost)
104
    └────┬────┘
105
         │
106
    h_out = W_base(h_t) + α · ΔW(h_t)
107
```
108
109
**Key properties:**
110
- **Shared B** is frozen and random → guarantees approximate orthogonality without training
111
- **Sparse A_k** matrices are domain-specific → each domain's update lives in a near-orthogonal subspace
112
- **Router R** operates on the hidden state → token-level granularity
113
- **Single projection** through B → same FLOP cost as standard LoRA, not K× like naive MoE
114
115
### Base Model Selection
116
117
> [!IMPORTANT]
118
> **Primary: Qwen2.5-3B-Instruct** (~6GB BF16, leaves 26GB for everything else)
119
> **Scaling experiment: Qwen2.5-7B-Instruct** (~14GB BF16, leaves 18GB)
120
121
Why 3B and not 1.5B or 14B:
122
- **Not 1.5B**: The reasoning capacity is too low — Qwen2.5-1.5B scores ~25% on MATH. Even a perfect adapter system can't fix a model that lacks fundamental reasoning circuits. You'd be proving that a better steering wheel doesn't help a car with no engine.
123
- **Not 14B**: Training LoRA adapters on 14B with BF16 + AdamW optimizer states = ~28GB. You'd have <4GB for activations/KV cache. OOM city.
124
- **3B is the sweet spot**: ~40% on MATH (improvable), ~55% on MMLU (strong base), fast iteration (train a LoRA in 1-2 hours), massive headroom on 32GB GPU.
125
126
### Domain Selection (5 domains, not 20)
127
128
> [!WARNING]
129
> 20 domains is a paper claim, not a practical plan. Training 20 quality LoRA adapters takes 20× the compute, 20× the data curation, and makes ablations 20× more expensive. Start with 5 domains that are maximally disjoint and have established benchmarks.
130
131
| Domain | Training Data | Evaluation Benchmark | Why This Domain |
132
|---|---|---|---|
133
| **Mathematics** | MetaMathQA (100k) | MATH500 / GSM8K | Core reasoning capability |
134
| **Code** | CodeAlpaca + Evol-Instruct-Code (80k) | HumanEval / MBPP | Disjoint from math syntax |
135
| **Science** | SciQ + GPQA train split (50k) | ARC-Challenge / GPQA | Multi-step scientific reasoning |
136
| **Legal** | LegalBench subset (30k) | LegalBench test | Highly specialized vocabulary |
137
| **Medical** | MedQA + PubMedQA (50k) | MedQA test | Domain with real-world impact |
138
139
---
140
141
## Proposed Changes
142
143
### Phase 0: Foundation Reset (Days 1-3)
144
145
> [!IMPORTANT]
146
> Nothing in the current codebase is usable for real experiments. The adapters are random, the metrics are simulated, the routers are heuristic toys. We need a clean foundation.
147
148
#### [NEW] src/lori_moe/__init__.py
149
Empty init for new package.
150
151
#### [NEW] src/lori_moe/config.py
152
Central configuration dataclass defining: base model path, adapter rank, number of domains, sparsity level, router architecture params, training hyperparameters.
153
154
#### [NEW] src/lori_moe/shared_projection.py
155
Implements the frozen shared B matrix (random Gaussian initialization with proper scaling). Key: `B = torch.randn(d_model, r) / sqrt(r)` — the `1/sqrt(r)` scaling ensures the projection preserves distances (JL lemma).
156
157
#### [NEW] src/lori_moe/lori_adapter.py
158
Custom LoRA adapter using frozen B + trainable sparse A. Integrates with HuggingFace PEFT's `LoraConfig` but overrides the B initialization to use the shared frozen matrix. Applies binary sparse masks to A matrices during training.
159
160
#### [NEW] scripts/setup_foundation.sh  
161
Installs dependencies, downloads Qwen2.5-3B-Instruct, verifies CUDA/BF16 support, creates directory structure.
162
163
---
164
165
### Phase 1: Train Real Domain Adapters (Days 4-8)
166
167
#### [NEW] src/lori_moe/data/prepare_datasets.py
168
Downloads and processes the 5 domain datasets from HuggingFace. Formats into instruction-tuning format compatible with Qwen2.5's chat template. Handles train/eval splits.
169
170
#### [NEW] src/lori_moe/training/train_lori_adapter.py
171
Main training script for a single domain LoRA adapter. Key features:
172
- Loads Qwen2.5-3B in BF16
173
- Initializes shared frozen B matrix (loaded from checkpoint or generated once)
174
- Creates trainable sparse A matrix for the target domain
175
- Uses PEFT's LoRA injection into `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
176
- AdamW optimizer, cosine LR schedule, gradient checkpointing
177
- Saves adapter weights + sparse mask
178
179
#### [NEW] src/lori_moe/training/apply_sparsity.py
180
Post-training DARE-style sparsification: drops N% of A matrix parameters (by magnitude), rescales remainder. This is ADDITIONAL orthogonalization on top of LoRI's structural guarantee.
181
182
#### [NEW] configs/lori_training.yaml
183
```yaml
184
base_model: "Qwen/Qwen2.5-3B-Instruct"
185
adapter_rank: 32  # Higher than Synapta's 16 for more capacity
186
shared_b_seed: 42  # Deterministic B initialization
187
sparsity_level: 0.8  # 80% sparse A matrices
188
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
189
lr: 2e-4
190
epochs: 3
191
batch_size: 8
192
gradient_accumulation: 4
193
bf16: true
194
gradient_checkpointing: true
195
```
196
197
#### [NEW] scripts/train_all_domains.sh
198
Sequential training script for all 5 domains. Estimated: ~2 hours per domain × 5 = 10 hours total.
199
200
---
201
202
### Phase 2: LoRI-MoE Architecture (Days 9-14)
203
204
#### [NEW] src/lori_moe/model/lori_moe_linear.py
205
The core module replacing `AdaptiveMultiLoRALinear`. Key differences from Synapta:
206
- **No clamping** — orthogonality prevents interference structurally
207
- **Token-level routing** — router operates on each token's hidden state
208
- **Shared B projection** — single matrix multiply, not K separate ones
209
- **Sparse A composition** — dynamic weighted sum of sparse A matrices
210
211
#### [NEW] src/lori_moe/model/router.py
212
Lightweight MLP router deployed at each transformer layer:
213
```python
214
class TokenRouter(nn.Module):
215
    def __init__(self, hidden_dim, num_experts, bottleneck=64):
216
        self.down = nn.Linear(hidden_dim, bottleneck)
217
        self.up = nn.Linear(bottleneck, num_experts)
218
    
219
    def forward(self, hidden_state):
220
        # hidden_state: (batch, seq_len, hidden_dim)
221
        logits = self.up(F.silu(self.down(hidden_state)))
222
        return F.softmax(logits, dim=-1)  # (batch, seq_len, num_experts)
223
```
224
225
#### [NEW] src/lori_moe/model/lori_moe_model.py
226
Wrapper that:
227
1. Loads Qwen2.5-3B base model (frozen)
228
2. Loads 5 trained LoRI adapters (frozen A matrices)
229
3. Loads shared B matrix (frozen)
230
4. Injects `LoRIMoELinear` modules at each target layer
231
5. Initializes trainable routers at each layer
232
6. Implements the full forward pass with load-balancing auxiliary loss
233
234
#### [NEW] src/lori_moe/model/losses.py
235
- Standard causal LM cross-entropy loss
236
- Load-balancing auxiliary loss: `L_aux = α * Σ(f_i * P_i)` where f_i = fraction of tokens routed to expert i, P_i = mean router probability for expert i
237
- Total: `L = L_CE + 0.01 * L_aux`
238
239
---
240
241
### Phase 3: Router Training (Days 15-19)
242
243
#### [NEW] src/lori_moe/data/generate_routing_data.py
244
Generates multi-domain reasoning traces:
245
- Uses the base model itself to generate responses on mixed-domain prompts
246
- Annotates which domain is "active" per reasoning step
247
- Creates 30k supervised routing examples
248
- Format: (prompt, token_positions, expert_labels_per_position)
249
250
#### [NEW] src/lori_moe/training/train_router.py
251
Router training script:
252
- Freezes base model + all adapters + shared B
253
- Only trains router parameters (~200K params total across all layers)
254
- Uses the multi-domain routing data
255
- Monitors routing entropy to detect collapse (entropy < 0.3 = collapse)
256
- Implements Top-2 gating with noise for gradient flow through non-selected experts
257
258
#### [NEW] configs/router_training.yaml
259
```yaml
260
router_lr: 1e-3
261
router_epochs: 5
262
load_balance_weight: 0.01
263
top_k: 2  # Top-2 routing
264
noise_std: 0.1  # Noisy gating to prevent collapse
265
entropy_threshold: 0.3  # Alert if below this
266
```
267
268
---
269
270
### Phase 4: Evaluation (Days 20-23)
271
272
#### [NEW] src/lori_moe/eval/run_benchmarks.py
273
Evaluation using `lm-eval-harness` on proper benchmarks:
274
275
| Benchmark | What It Tests | Metric | Baseline (Qwen2.5-3B) |
276
|---|---|---|---|
277
| **MATH500** | Mathematical reasoning | Exact Match | ~40% |
278
| **GSM8K** | Grade-school math | Exact Match | ~75% |
279
| **MMLU** | Multi-domain knowledge | Accuracy | ~55% |
280
| **BBH** | Hard multi-step reasoning | Accuracy | ~45% |
281
| **HumanEval** | Code generation | Pass@1 | ~40% |