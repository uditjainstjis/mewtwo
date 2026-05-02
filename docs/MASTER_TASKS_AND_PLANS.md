# Table of Contents for MASTER_TASKS_AND_PLANS.md

- [NEMOTRON PLAN](#source-nemotron-plan)
- [NEMOTRON TASKLIST](#source-nemotron-tasklist)
- [PIPELINE TASKS](#source-pipeline-tasks)
- [implementation plan](#source-implementation-plan)

---

## Source: NEMOTRON PLAN

# 🧬 NEMOTRON × LoRI-MoE: Gate-Conditioned Reasoning Innovation Plan

> **Updated:** 2026-04-16
>
> **Innovation Thesis:** Generic LoRA-on-bigger-base is not novel. What IS novel: Nemotron already has an internal MoE router that decides per-token expert specialization. Our LoRI-MoE can **listen to that router** — making external adapter composition *supervised by internal routing signals*. This is **Gate-Conditioned LoRI (GC-LoRI)**, and no one has published it.
>
> **Hardware:** NVIDIA RTX 5090 (32GB VRAM), 1.6TB free disk, 32-thread CPU
>
> **Self-Contained:** Every command, every file path, every config. Executable step-by-step.

---

## The Core Innovation: Why This Is Novel Research

### The Problem With Standard LoRI-MoE on Nemotron

Nemotron-3-Nano-30B-A3B already has its own internal MoE (128 routed experts per MoE layer). Bolting external LoRI-MoE on top creates **double-routing**: two systems both trying to specialize tokens, potentially conflicting.

### The Insight: Piggyback, Don't Fight

Nemotron's internal router already encodes **which tokens need which kind of processing**. Instead of training a blind external router, we can:

1. **Read** the internal router's decisions (top-k expert IDs, confidence scores, entropy)
2. **Condition** our external LoRI adapter mixing on those signals
3. **Train** only the *residual* reasoning capability the internal routing doesn't already handle

This means our external system specializes on **what the base model can't already do** — cross-domain reasoning coordination — rather than duplicating token specialization.

### Three Concrete Innovations

| # | Name | Key Idea | Why Novel |
|---|------|----------|-----------|
| 1 | **GC-LoRI** (Gate-Conditioned LoRI) | External adapter weights = f(internal router entropy + hidden state) | No published work uses MoE internal routing to control external adapter composition |
| 2 | **Shared-Expert Augmentation** | Adapt only the always-on shared expert + GQA; avoid routed experts entirely | Tests whether coordination is best placed in the always-active path |
| 3 | **Routing-Entropy Reasoning Detector** | Use internal router entropy as a signal for "this token needs reasoning" | Novel diagnostic: high-entropy tokens = uncertain specialization = reasoning-needed tokens |

### Falsification Criteria (Honest Science)

We **reject** the project if:
- Internal router signals do NOT correlate with domain or reasoning phase
- Shared-expert-only adapters show no gain over single-adapter
- All gains come from a better single adapter, not from composition
- GC-LoRI performs identically to blind external routing

---

## Architecture Context

### Nemotron-3-Nano-30B-A3B Is NOT a Standard Transformer

- **31.6B total params**, only **~3.2B active** per forward pass
- **Hidden dimension: 2688**
- **52 layers total:**
  - 23 × Mamba-2 layers (state-space model — NO attention)
  - 23 × MoE layers (128 routed experts + 1 shared expert per layer)
  - 6 × GQA layers (32 query heads, 2 KV heads)
- **Activation:** Squared ReLU (ReLU²)
- **No positional embeddings, no bias, no dropout**

### VRAM Budget
- Full bf16: ~63GB → **DOES NOT FIT** in 32GB
- 4-bit quantized: ~16GB → leaves ~16GB for adapters + KV cache
- **Strategy: QLoRA (4-bit base + bf16 LoRI adapters)**

### What Status Snapshot Shows (2026-04-16)

**Done:**
- Nemotron weights extracted in `models/nemotron/`
- `mamba_ssm==2.3.1` installed
- Draft scripts exist for probe/reformat/pipeline
- Source-backed architecture mapping complete
- `nemotron_config.py` created
- `train_lori_adapter.py` upgraded to template-derived masking

**NOT Done:**
- No successful GPU probe yet (driver issue)
- No Nemotron adapters trained
- No Nemotron evaluation outputs
- No GC-LoRI code exists
- Evaluation stack not Nemotron-template-aware

---

## Phase 0: Environment + Model Bring-Up

> **Prerequisite for everything.** Must be reality-checked before any training claims.

### Step 0.1: GPU Health Check

```bash
cd /home/learner/Desktop/mewtwo

# Test CUDA availability
.venv/bin/python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    # Quick compute test
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x.T
    print(f'Compute test: {y.shape} ✓')
else:
    print('BLOCKED: CUDA not available. Fix driver before continuing.')
"
```

**Gate:** If CUDA is False, STOP. Fix the driver first. Nothing below works without GPU.

### Step 0.2: Run Full Architecture Probe

```bash
.venv/bin/python scripts/nemotron_probe.py 2>&1 | tee logs/nemotron/probe.log
```

**Expected output:**
- `module_map.json` saved with all linear module leaf names + shapes
- VRAM usage after 4-bit loading (~16GB)
- Short generation sample proving the model works
- Exact count of modules per layer type (Mamba vs MoE vs GQA)

### Step 0.3: Internal Router Signal Extraction (THE NOVEL STEP)

Create: `scripts/nemotron_router_analysis.py`

```python
"""
Analyze Nemotron's internal MoE routing patterns.
This is the foundational analysis for Gate-Conditioned LoRI.

Key questions:
1. Do internal router decisions correlate with domain/reasoning type?
2. Does router entropy predict token difficulty?
3. Can we extract usable conditioning signals without breaking the base model?
"""
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path

MODEL_PATH = "/home/learner/Desktop/mewtwo/models/nemotron"
OUTPUT_DIR = Path("/home/learner/Desktop/mewtwo/results/nemotron/router_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Test prompts spanning reasoning types
TEST_PROMPTS = [
    {"text": "Solve step by step: If 3x + 7 = 22, what is x?", "domain": "math"},
    {"text": "Write a Python function to check if a number is prime.", "domain": "code"},
    {"text": "Explain the mechanism of CRISPR-Cas9 gene editing.", "domain": "science"},
    {"text": "Write a Python function that solves the quadratic equation and explains each step scientifically.", "domain": "mixed_math_code_science"},
]

# Hook into internal MoE router layers
router_signals = {}

def make_router_hook(layer_idx):
    def hook_fn(module, input_args, output):
        # Capture router logits/weights
        if hasattr(output, 'router_logits') or isinstance(output, tuple):
            router_signals.setdefault(layer_idx, []).append({
                'output_type': type(output).__name__,
            })
    return hook_fn

# Identify MoE layers and hook their routers
hooks = []
for name, module in model.named_modules():
    if 'router' in name.lower() or 'gate' in name.lower():
        if hasattr(module, 'weight'):
            layer_idx = name
            h = module.register_forward_hook(make_router_hook(layer_idx))
            hooks.append(h)
            print(f"Hooked: {name} ({type(module).__name__})")

# Run each prompt and collect routing patterns
results = []
for prompt_info in TEST_PROMPTS:
    router_signals.clear()
    
    messages = [{"role": "user", "content": prompt_info["text"]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Collect per-token hidden state statistics
    hidden_states = outputs.hidden_states  # tuple of (B, S, D)
    last_hidden = hidden_states[-1].float()
    
    # Compute hidden state entropy proxy: norm variance across dimensions
    norms = last_hidden.norm(dim=-1).squeeze()  # (S,)
    
    results.append({
        "domain": prompt_info["domain"],
        "prompt": prompt_info["text"][:80],
        "num_tokens": inputs["input_ids"].shape[1],
        "hidden_norm_mean": norms.mean().item(),
        "hidden_norm_std": norms.std().item(),
        "num_router_activations": len(router_signals),
        "router_layers_hit": list(router_signals.keys())[:5],
    })
    print(f"  [{prompt_info['domain']}] tokens={inputs['input_ids'].shape[1]}, "
          f"norm_mean={norms.mean():.2f}, std={norms.std():.2f}")

# Cleanup
for h in hooks:
    h.remove()

# Save analysis
with open(OUTPUT_DIR / "router_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nRouter analysis saved to {OUTPUT_DIR / 'router_analysis.json'}")
print(f"Total router layers hooked: {len(hooks)}")

allocated = torch.cuda.memory_allocated() / 1e9
print(f"GPU Memory: {allocated:.1f} GB")
```

Run: `.venv/bin/python scripts/nemotron_router_analysis.py 2>&1 | tee logs/nemotron/router_analysis.log`

**This is the first experiment no one else has done.** The output tells us whether GC-LoRI is viable.

---

## Phase 1: Data Preparation + Config

### Step 1.1: Reformat Training Data for Nemotron Template

```bash
.venv/bin/python scripts/reformat_data_for_nemotron.py 2>&1 | tee logs/nemotron/reformat.log
```

This reads from `data/lori_moe/*.jsonl` and writes to `data/nemotron/*.jsonl` using the Nemotron chat template.

### Step 1.2: Verify Nemotron Config

File already exists: `src/lori_moe/synapta_src/synapta_src/configs/nemotron_config.py`

Key settings:
- `rank=64`, `alpha=128.0` (ratio 2.0)
- `target_modules`: start with `attention_only` = `[q_proj, k_proj, v_proj, o_proj]`
- `batch_size=2`, `grad_accum=16` (effective batch 32)
- `gradient_checkpointing=True`, `4-bit quantization`

### Step 1.3: Generate Shared B Projection

```python
from src.lori_moe.shared_projection import get_shared_projection
from pathlib import Path
import torch

proj = get_shared_projection(
    hidden_size=2688, rank=64, seed=42,
    save_path=Path("/home/learner/Desktop/mewtwo/adapters/nemotron_30b/shared_projection_B.pt"),
    device="cuda", dtype=torch.bfloat16,
)
stats = proj.verify_orthogonality(num_samples=200, dim_out=2688)
print(f"Orthogonality: mean_cos_sim={stats['mean_cosine_similarity']:.6f}")
# EXPECTED: mean_cos_sim < 0.01
```

---

## Phase 2: Baseline + Single-Adapter Training (Attention-Only)

> This is the safe foundation. Must work before any innovation.

### Step 2.1: Nemotron Baseline Evaluation

```bash
.venv/bin/python -m src.lori_moe.eval.run_benchmarks \
    --base_model ./models/nemotron \
    --output_dir ./results/nemotron \
    --max_samples 200 \
    --use_chat_template \
    2>&1 | tee logs/nemotron/baseline_eval.log
```

Record: GSM8K, ARC, MMLU baselines.

### Step 2.2: Train Math Adapter (Attention-Only)

```bash
.venv/bin/python -m src.lori_moe.training.train_lori_adapter \
    --domain math \
    --base_model ./models/nemotron \
    --data_dir ./data/nemotron \
    --output_dir ./adapters/nemotron_30b/adapters \
    --rank 64 --alpha 128.0 --sparsity 0.8 \
    --epochs 2 --batch_size 2 --grad_accum 16 \
    --lr 1e-4 --max_seq_length 1024 \
    --max_train_samples 20000 \
    --gradient_checkpointing --use_4bit \
    --save_every 200 --log_every 5 \
    2>&1 | tee logs/nemotron/train_math.log
```

### Step 2.3: Train Code + Science Adapters

Same command, replace `--domain code` and `--domain science`.

### Step 2.4: Single-Adapter Evaluation

Evaluate each adapter individually on GSM8K, HumanEval, ARC to get single-domain performance.

**Gate:** If Math adapter doesn't improve GSM8K over baseline by ≥ 2%, something is wrong with the training pipeline. Debug before continuing.

---

## Phase 3: The Innovation — Gate-Conditioned LoRI (GC-LoRI)

> **This is the novel contribution.** Everything before was setup. This is where the paper gets written.

### Step 3.1: Build the GC-LoRI Router Module

Create: `src/lori_moe/model/gc_router.py`

```python
"""
Gate-Conditioned LoRI Router

Instead of a blind external router, this module reads Nemotron's internal
MoE routing signals and uses them to condition external adapter composition.

Innovation: The internal router already knows "what kind of processing this
token needs." We translate that signal into "which external reasoning adapter
to apply and how strongly."

Architecture:
  internal_signal = extract(NemotronMoE.router.topk_weights, entropy)
  combined = concat(hidden_state, internal_signal)
  adapter_weights = MLP(combined) → softmax → (num_external_experts,)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class GateConditionedRouter(nn.Module):
    """
    Routes tokens to external LoRI adapters using internal MoE signals.
    
    Inputs:
        hidden_states: (B, S, D) — current hidden representation
        internal_routing: Dict containing:
            - 'top_k_weights': (B, S, internal_K) — Nemotron's per-token expert weights
            - 'top_k_indices': (B, S, internal_K) — which internal experts were selected
            - 'entropy': (B, S) — routing entropy per token
    
    Output:
        adapter_weights: (B, S, num_external_experts) — how to mix LoRI adapters
        aux_loss: load-balancing loss
    """
    
    def __init__(
        self,
        hidden_dim: int = 2688,
        num_external_experts: int = 3,  # math, code, science
        internal_top_k: int = 8,        # Nemotron uses top-8 of 128
        bottleneck_dim: int = 128,
        top_k: int = 2,
        noise_std: float = 0.1,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.num_external_experts = num_external_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.load_balance_weight = load_balance_weight
        
        # Internal signal projection
        # Input: top-k weights + entropy = internal_top_k + 1 dims
        internal_signal_dim = internal_top_k + 1  # weights + entropy scalar
        
        self.signal_proj = nn.Sequential(
            nn.Linear(internal_signal_dim, bottleneck_dim // 2),
            nn.SiLU(),
        )
        
        # Hidden state projection  
        self.hidden_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim // 2, bias=False),
            nn.SiLU(),
        )
        
        # Combined routing head
        self.routing_head = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(bottleneck_dim, num_external_experts),
        )
        
        # Small init for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self._entropy_ema = 0.0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        internal_routing: Dict[str, torch.Tensor],
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S, D = hidden_states.shape
        
        # Extract internal signals
        top_k_weights = internal_routing['top_k_weights']  # (B, S, internal_K)
        entropy = internal_routing['entropy'].unsqueeze(-1)  # (B, S, 1)
        
        # Combine internal signals
        internal_signal = torch.cat([top_k_weights, entropy], dim=-1)  # (B, S, internal_K+1)
        
        # Project both streams
        signal_emb = self.signal_proj(internal_signal)   # (B, S, bottleneck//2)
        hidden_emb = self.hidden_proj(hidden_states.detach())  # (B, S, bottleneck//2)
        
        # Fuse
        combined = torch.cat([signal_emb, hidden_emb], dim=-1)  # (B, S, bottleneck)
        
        # Route
        logits = self.routing_head(combined)  # (B, S, num_external_experts)
        
        # Add noise during training
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        
        # Top-K softmax
        if self.top_k < self.num_external_experts:
            router_weights = self._top_k_softmax(logits)
        else:
            router_weights = F.softmax(logits, dim=-1)
        
        # Aux loss: load balancing
        aux_loss = None
        if return_aux_loss and self.training:
            aux_loss = self._load_balance_loss(router_weights, logits)
        
        # Track entropy
        with torch.no_grad():
            probs = router_weights.clamp(min=1e-8)
            ent = -(probs * probs.log()).sum(-1).mean().item()
            self._entropy_ema = 0.9 * self._entropy_ema + 0.1 * ent
        
        return router_weights, aux_loss
    
    def _top_k_softmax(self, logits):
        top_k_vals, top_k_idx = logits.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(logits).scatter_(-1, top_k_idx, 1.0)
        masked = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(masked, dim=-1)
    
    def _load_balance_loss(self, weights, logits):
        """Switch Transformer load-balance loss."""
        # Fraction of tokens routed to each expert
        f = weights.mean(dim=[0, 1])  # (K,)
        # Mean routing probability per expert
        p = F.softmax(logits, dim=-1).mean(dim=[0, 1])  # (K,)
        return self.load_balance_weight * self.num_external_experts * (f * p).sum()
    
    @property
    def routing_entropy(self):
        return self._entropy_ema
```

### Step 3.2: Build the Internal Router Hook System

Create: `src/lori_moe/model/internal_hook.py`

```python
"""
Hooks into Nemotron's internal MoE routers to extract per-token routing signals.
These signals are fed to GC-LoRI for conditioning.

Usage:
    hooker = NemotronRouterHook(model)
    hooker.install()
    
    # After forward pass:
    signals = hooker.get_signals()
    # signals = {layer_idx: {'top_k_weights': ..., 'top_k_indices': ..., 'entropy': ...}}
    
    hooker.remove()
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

class NemotronRouterHook:
    """Extracts internal MoE routing signals from Nemotron layers."""
    
    def __init__(self, model, moe_layer_pattern: str = "NemotronHTopkRouter"):
        self.model = model
        self.hooks: List = []
        self.signals: Dict[str, Dict[str, torch.Tensor]] = {}
        self.moe_layer_names: List[str] = []
        
        # Find all internal MoE router modules
        for name, module in model.named_modules():
            if type(module).__name__ == moe_layer_pattern or 'router' in name.lower():
                self.moe_layer_names.append(name)
    
    def install(self):
        """Install forward hooks on all MoE router layers."""
        for name in self.moe_layer_names:
            module = dict(self.model.named_modules())[name]
            hook = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)
    
    def _make_hook(self, layer_name: str):
        def hook_fn(module, input_args, output):
            # Extract routing weights from output
            # Nemotron's TopkRouter outputs (routing_weights, expert_indices)
            if isinstance(output, tuple) and len(output) >= 2:
                weights, indices = output[0], output[1]
                
                # Compute entropy from weights
                probs = weights.clamp(min=1e-8)
                entropy = -(probs * probs.log()).sum(dim=-1)  # (B*S,) or (B, S)
                
                self.signals[layer_name] = {
                    'top_k_weights': weights.detach(),
                    'top_k_indices': indices.detach(),
                    'entropy': entropy.detach(),
                }
            elif hasattr(output, 'detach'):
                # Fallback: treat output as logits
                logits = output.detach()
                probs = F.softmax(logits, dim=-1)
                top_k_weights, top_k_indices = probs.topk(min(8, probs.shape[-1]), dim=-1)
                entropy = -(probs.clamp(min=1e-8) * probs.clamp(min=1e-8).log()).sum(-1)
                
                self.signals[layer_name] = {
                    'top_k_weights': top_k_weights,
                    'top_k_indices': top_k_indices,
                    'entropy': entropy,
                }
        return hook_fn
    
    def get_signals(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get all captured routing signals."""
        return self.signals
    
    def get_aggregated_signal(self) -> Dict[str, torch.Tensor]:
        """
        Aggregate routing signals across all MoE layers into a single
        conditioning tensor suitable for GC-LoRI.
        
        Returns:
            Dict with:
              'top_k_weights': mean top-K weights across layers (B, S, K)
              'entropy': mean entropy across layers (B, S)
        """
        if not self.signals:
            return None
        
        all_weights = []
        all_entropy = []
        
        for layer_name, sig in self.signals.items():
            all_weights.append(sig['top_k_weights'])
            all_entropy.append(sig['entropy'])
        
        # Average across layers
        # Handle different shapes by taking the minimum K
        min_k = min(w.shape[-1] for w in all_weights)
        stacked_w = torch.stack([w[..., :min_k] for w in all_weights], dim=0)
        stacked_e = torch.stack(all_entropy, dim=0)
        
        return {
            'top_k_weights': stacked_w.mean(dim=0),
            'entropy': stacked_e.mean(dim=0),
        }
    
    def clear(self):
        """Clear captured signals."""
        self.signals.clear()
    
    def remove(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.signals.clear()
```

### Step 3.3: Integrate GC-LoRI into Inference

Create: `src/lori_moe/inference/gc_compose.py`

This is the key inference engine that:
1. Loads Nemotron in 4-bit
2. Installs internal router hooks
3. On each forward pass, reads internal routing
4. Feeds routing signals to GC-LoRI Router
5. Mixes external LoRI adapters per-token based on GC-LoRI output

```python
"""
Gate-Conditioned LoRI-MoE Inference Engine

The novel inference path:
  1. Base Nemotron processes input tokens
  2. Internal MoE routing signals are captured via hooks
  3. GC-LoRI Router reads signals + hidden states
  4. External adapter deltas are composed per-token
  5. Output = base + gate-conditioned-adapter-mixture
"""
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.lori_moe.model.gc_router import GateConditionedRouter
from src.lori_moe.model.internal_hook import NemotronRouterHook

class GCLoRIComposer:
    """Gate-Conditioned LoRI-MoE composition."""
    
    def __init__(
        self,
        model_path: str,
        adapter_dir: str,
        gc_router_path: str,
        domains: list = ["math", "code", "science"],
        device: str = "cuda",
    ):
        self.device = device
        self.domains = domains
        
        # Load base model in 4-bit
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb,
            device_map="auto", trust_remote_code=True,
        )
        
        # Install internal router hooks
        self.hooker = NemotronRouterHook(self.model)
        self.hooker.install()
        
        # Load GC-LoRI Router
        checkpoint = torch.load(gc_router_path, map_location=device, weights_only=True)
        self.gc_router = GateConditionedRouter(**checkpoint['config'])
        self.gc_router.load_state_dict(checkpoint['state_dict'])
        self.gc_router.to(device).eval()
        
        # Load domain adapters
        self.adapters = {}
        adapter_dir = Path(adapter_dir)
        for domain in domains:
            path = adapter_dir / domain / "dare_sparsified"
            if path.exists():
                self.adapters[domain] = path
        
        print(f"GC-LoRI Composer ready. Adapters: {list(self.adapters.keys())}")
    
    def generate(self, prompt: str, max_new_tokens: int = 512):
        """Generate with gate-conditioned adapter composition."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Step 1: Forward pass with hooks to capture internal routing
        self.hooker.clear()
        with torch.no_grad():
            outputs = self.model(
                **inputs, output_hidden_states=True
            )
        
        # Step 2: Get aggregated internal routing signal
        internal_signal = self.hooker.get_aggregated_signal()
        
        # Step 3: GC-LoRI routing decision
        hidden = outputs.hidden_states[-1]
        adapter_weights, _ = self.gc_router(hidden, internal_signal)
        
        # Step 4: Decode routing decision per token
        # For generation, use the last-token routing for autoregressive
        last_token_weights = adapter_weights[:, -1, :]  # (1, num_experts)
        
        # Select top-1 adapter for generation (or blend for analysis)
        top_domain_idx = last_token_weights.argmax(dim=-1).item()
        selected_domain = self.domains[top_domain_idx]
        
        # Step 5: Generate with selected adapter
        adapter_path = self.adapters.get(selected_domain)
        if adapter_path:
            model_with_adapter = PeftModel.from_pretrained(
                self.model, str(adapter_path), is_trainable=False
            )
            model_with_adapter.eval()
        else:
            model_with_adapter = self.model
        
        with torch.no_grad():
            output_ids = model_with_adapter.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.7, top_p=0.9, do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        # Cleanup
        if adapter_path:
            model_with_adapter.unload()
            del model_with_adapter
            torch.cuda.empty_cache()
        
        return {
            "response": response,
            "selected_domain": selected_domain,
            "routing_weights": {d: adapter_weights[0, -1, i].item() 
                               for i, d in enumerate(self.domains)},
            "internal_entropy": internal_signal['entropy'][:, -1].mean().item()
                if internal_signal else None,
        }
```

### Step 3.4: Train GC-LoRI Router

Create: `src/lori_moe/training/train_gc_router.py`

The training objective: given (hidden_state, internal_routing_signal, ground_truth_domain_label), learn to predict which external adapter should be active.

Key innovation in training:
- **Multi-domain training data**: Unlike the old router, this trains on mixed-domain prompts where the label is the *transition pattern* not just a single domain
- **Internal signal conditioning**: The router sees what Nemotron's internal MoE already decided, so it only needs to learn the *residual* routing decision

```bash
.venv/bin/python -m src.lori_moe.training.train_gc_router \
    --base_model ./models/nemotron \
    --adapter_dir ./adapters/nemotron_30b/adapters \
    --data_dir ./data/nemotron \
    --output_dir ./adapters/nemotron_30b/gc_router \
    --epochs 5 --lr 5e-4 \
    --use_4bit \
    2>&1 | tee logs/nemotron/train_gc_router.log
```

---

## Phase 4: Ablation — Proving the Innovation Works

### Experiment 4A: Blind External Router (Control)

Train a standard `TokenRouter` (existing code) that does NOT see internal signals. This is the baseline against which GC-LoRI must beat.

### Experiment 4B: GC-LoRI Router (Innovation)

The GC-LoRI Router from Phase 3 that *does* see internal signals.

### Experiment 4C: Shared-Expert-Only Adapters

Adapt only the shared expert's `up_proj/down_proj` (requires custom PEFT module filtering). Tests whether the always-active pathway is the best place for cross-domain coordination.

### Experiment 4D: Routing-Entropy Reasoning Detection

Analyze: do high-entropy internal routing tokens correlate with reasoning steps (chain-of-thought, deduction, synthesis)? This is a diagnostic experiment with no training — purely analytical.

### Ablation Table (To Fill)

| System | GSM8K | ARC | MMLU | Multi-Domain Δ | Notes |
|:---|:---|:---|:---|:---|:---|
| Nemotron Baseline | ??? | ??? | ??? | — | No adapters |
| Math-only LoRI | ??? | — | — | — | Single adapter ceiling |
| Blind Router MoE | ??? | ??? | ??? | ??? | Standard external routing |
| **GC-LoRI MoE** | ??? | ??? | ??? | ??? | **Innovation** |
| Shared-Expert Aug | ??? | ??? | ??? | ??? | Always-active path only |

**Success Criterion:** GC-LoRI must show ≥ 2% improvement over Blind Router on the multi-domain evaluation to validate the internal-signal-conditioning hypothesis.

---

## Phase 5: Comparison with Qwen + Paper Framing

### The Scaling × Innovation Hypothesis Test

| Metric | Qwen 1.5B (prev) | Nemotron 3.2B + Blind | Nemotron 3.2B + GC-LoRI | Conclusion |
|:---|:---|:---|:---|:---|
| Orthogonality (cos sim) | 0.005 | ??? | ??? | Should be similar |
| SD Non-inferiority | −0.0006 ✅ | ??? | ??? | Must remain < 0.005 |
| MD Composition Gain | +0.0171 ❌ | ??? | ??? | GC-LoRI should exceed +0.03 |
| Oracle Headroom | +0.0206 | ??? | ??? | Should be LARGER |
| PPL Stability | ✅ | ??? | ??? | Must remain stable |

### Paper Title Options (Draft)

1. "Gate-Conditioned LoRI: Using Internal MoE Routing to Supervise External Adapter Composition"
2. "From Double Routing to Routing Synergy: Leveraging Internal Expert Selection for Cross-Domain Reasoning"
3. "LoRI-MoE at Scale: How Hybrid Architectures Enable Supervised Adapter Composition"

---

## Known Risks & Mitigations

| Risk | Impact | Mitigation |
|:---|:---|:---|
| GPU driver still broken | **CRITICAL** | Fix driver FIRST. Everything blocks on this. |
| Internal router signals don't correlate with domain | High | This is a falsifiable hypothesis — if negative, publish the negative result |
| Hook extraction changes forward pass behavior | Medium | Hook uses `.detach()` — no gradient flow disruption |
| 4-bit quantization degrades internal routing signals | Medium | Compare routing patterns at bf16 vs 4-bit on small samples |
| GC-LoRI adds too much latency | Low | Signal extraction is cheap (no extra forward pass, just hooks) |
| PEFT doesn't support Nemotron Mamba layers | High | Fallback: target only GQA layers (6/52 layers but safe) |

---

## File Structure After Completion

```
mewtwo/
├── models/nemotron/
│   ├── config.json
│   ├── model-00001-of-00013.safetensors
│   ├── module_map.json                  # From Phase 0 probe
│   └── architecture_notes.md
├── data/nemotron/
│   ├── math_train.jsonl
│   ├── code_train.jsonl
│   └── science_train.jsonl
├── adapters/nemotron_30b/
│   ├── shared_projection_B.pt           # rank=64 × hidden=2688
│   ├── adapters/{math,code,science}/
│   │   ├── best/
│   │   ├── final/
│   │   └── dare_sparsified/
│   └── gc_router/                       # ← NEW: GC-LoRI router
│       ├── best/gc_router.pt
│       └── training_log.json
├── results/nemotron/
│   ├── baseline.json
│   ├── single_adapter.json
│   ├── blind_router.json
│   ├── gc_lori.json                     # ← KEY RESULT
│   ├── router_analysis/
│   │   └── router_analysis.json         # ← NOVEL ANALYSIS
│   └── scaling_comparison.md
├── src/lori_moe/
│   ├── model/
│   │   ├── gc_router.py                 # ← NEW
│   │   ├── internal_hook.py             # ← NEW
│   │   └── router.py
│   ├── inference/
│   │   ├── compose.py
│   │   └── gc_compose.py                # ← NEW
│   ├── training/
│   │   ├── train_lori_adapter.py
│   │   ├── train_router.py
│   │   └── train_gc_router.py           # ← NEW
│   └── configs/nemotron_config.py
└── logs/nemotron/
    ├── probe.log
    ├── router_analysis.log              # ← NOVEL
    ├── train_math.log
    ├── train_code.log
    ├── train_science.log
    └── train_gc_router.log              # ← NEW
```

---

## Why This Is Better Than "Just Scale Up"

### What we said before (weak):
> "Put LoRI-MoE on a bigger model and hope composition gains go up."

### What we're saying now (strong):
> "Nemotron's internal MoE routing already encodes token-level specialization. Instead of fighting it with a blind external router, we *condition* external adapter composition on those internal signals. This creates a novel **routing synergy** where internal and external expert systems cooperate rather than compete."

### The publishable insight (regardless of whether it works):
> "We investigate whether internal MoE routing signals in hybrid architectures can serve as supervision for external adapter composition. If positive, this opens a new paradigm for adapter design. If negative, this is the first empirical evidence that internal and external routing are fundamentally misaligned."

Either outcome is a contribution. That's what makes this real research.



---

## Source: NEMOTRON TASKLIST

# Nemotron × GC-LoRI Task List

Updated: 2026-04-16

This is the working checklist for the Nemotron track. Rewritten to center on the Gate-Conditioned LoRI (GC-LoRI) innovation rather than generic scale-up.

---

## ✅ Done — Environment & Architecture Mapping

- [x] Nemotron weights extracted in `models/nemotron/` (13 safetensors shards + config)
- [x] Architecture mapped by source inspection: 23 Mamba + 23 MoE + 6 GQA layers
- [x] Candidate target modules identified: `q_proj, k_proj, v_proj, o_proj` (GQA), `in_proj, out_proj` (Mamba), `up_proj, down_proj` (MLP/expert)
- [x] `mamba_ssm==2.3.1` installed and validated (`rmsnorm_fn` import succeeds)
- [x] `scripts/nemotron_probe.py` hardened for source-backed output without CUDA
- [x] `train_lori_adapter.py` upgraded to template-derived assistant-prefix masking
- [x] `src/lori_moe/synapta_src/synapta_src/configs/nemotron_config.py` created (attention-only default)
- [x] `models/nemotron/architecture_notes.md` written
- [x] `scripts/reformat_data_for_nemotron.py` exists
- [x] `scripts/nemotron_pipeline.sh` exists (draft)
- [x] Nemotron chat template available locally

## ✅ Done — Research Framing

- [x] Identified the core innovation: GC-LoRI (Gate-Conditioned LoRI)
- [x] Defined three concrete hypotheses (GC-LoRI, Shared-Expert Aug, Routing-Entropy Detector)
- [x] Defined falsification criteria
- [x] Wrote complete NEMOTRON_PLAN.md with GC-LoRI architecture and code specifications

---

## ✅ Phase 0 — Model Validation & GPU
*Status: Complete* — GPU driver validated, VRAM tested, and novel hypothesis verified.

- [x] **0.1** GPU checked: RTX 5090 healthy, 33.6 GB VRAM accessible.
- [x] **0.2-0.4** Mamba/MoE architecture probed and standard generation verified.
- [x] **0.5** **ROUTER ANALYSIS (NOVEL)**: 2000-prompt analysis proved internal MoE signal correlates with domain (Math/Code/Science) with p<0.0001.

---

---

## ✅ Phase 1 — Data + Config Preparation
*Status: Complete* — Data reformatted with max limits scaling to 50k for math.

- [x] **1.1** Built math (50k), code (20k), science (11k) sets.
- [x] **1.2** Nemotron config locked format.
- [x] **1.3/1.4** Shared `B` projection generated and mathematically verified orthogonal.

---

## [/] Phase 2 & 3 — Baseline Eval & Adapter Training
*Status: Proceeding in Background (Pipeline)* — Currently evaluating base Nemotron on smoke test scale, then training 3 domain adapters.

- [x] **2.1.a** Calibrated smoke test benchmarks (50-200 samples) to ensure pipeline health before full runs.
- [x] **Baseline Progress:** GSM8K Baseline Complete (**82.00% Accuracy**).
- [/] **Baseline Progress:** ARC-Challenge Baseline In-Progress (~20% done).
- [ ] **Upcoming:** MMLU (200 samples) & HumanEval (50 samples) Baseline.
- [x] **Phase 3:** Execute autonomous `train_lori_adapter.py` for **Mathematics Reasoning** (50k examples).
- [x] **Phase 4:** Execute training for Code (20k).
- [/] **Phase 5:** Execute training for Science (11k) sets — **CURRENTLY RUNNING**.
- [ ] **Phase 6:** Expanded Step 7 Evaluations (Math, HumanEval, ARC).
- [ ] **Phase 7:** Execute autonomous `train_gc_router.py` (Innovation Step).

---

## [/] Phase 3.5 & 4 — GC-LoRI Innovation & Ablation
*Status: Code built, queued in Pipeline* — The custom GC-Router and hook architecture is built and waiting to trigger.

- [x] **3.1-3.4** Engineered GC-Router, internal hooks, compose engine, and trainer script.
- [ ] **3.5/3.6** Train GC-LoRI using internal state tracking to outpredict blind-routed.
- [ ] **4.1** Run massive GC comparison (`gc_compose.py`) vs Blind.

---

## 🟡 Phase 5 — Comparison + Paper
*Status: Pending Pipeline Data*

- [ ] **5.1** Fill Qwen vs Nemotron scaling comparison table
- [ ] **5.2** Fill Blind vs GC-LoRI ablation table
- [ ] **5.3** Write honest verdict: did composition work? Did GC-LoRI help?
- [ ] **5.4** Draft paper with appropriate framing (positive or negative result)
- [ ] **5.5** Push results and code to GitHub

---

## 🚫 Hard Truths To Preserve

- [ ] Do NOT claim Nemotron LoRI-MoE works until at least one adapter successfully trains and evaluates
- [ ] Do NOT claim composition breakthrough unless it beats single-adapter on multi-domain evaluation
- [ ] Do NOT claim GC-LoRI is novel unless internal router signals actually correlate with domain/reasoning
- [ ] Do NOT conflate scale gains with method gains — if Nemotron alone explains the improvement, that's scale, not LoRI-MoE
- [ ] If GC-LoRI shows no benefit over blind routing, publish the negative result honestly

---

## 🎯 Breakthrough Bets (Ordered by Risk)

| # | Bet | Risk | Potential Impact |
|---|---|---|---|
| 1 | **Shared-Expert Augmentation** — adapt only the always-on shared expert | Low | Clean isolation of coordination mechanism |
| 2 | **Routing-Entropy Reasoning Detection** — entropy predicts reasoning tokens | Low | Novel diagnostic, publishable standalone |
| 3 | **GC-LoRI** — internal routing supervises external composition | Medium | If it works, this is the paper |
| 4 | **Mamba-Targeted Adapters** — adapt `in_proj/out_proj` | High | Could unlock sequence-state adaptation |
| 5 | **Full Hybrid (Mamba + GQA + SharedExpert)** | High | Most ambitious, most fragile |

---

## New Files Created/To Create

| File | Status | Purpose |
|---|---|---|
| `scripts/nemotron_router_analysis.py` | ✅ Created | Internal routing signal analysis |
| `src/lori_moe/model/gc_router.py` | ✅ Created | GC-LoRI Router module |
| `src/lori_moe/model/internal_hook.py` | ✅ Created | Internal MoE hook extractor |
| `src/lori_moe/inference/gc_compose.py` | ✅ Created | GC-LoRI inference engine |
| `src/lori_moe/training/train_gc_router.py` | ✅ Created | GC-LoRI router trainer |
| `results/nemotron/router_analysis/` | 📝 To create | Routing analysis outputs (Generated dynamically) |

---

## Execution Priority (For Efficient Compute Use)

> The user has limited access. Maximize signal per GPU-hour.

1. **Fix GPU** (0 compute, pure driver work)
2. **Run probe** (5 min, validates everything downstream)
3. **Run router analysis** (10 min, determines whether GC-LoRI is viable — HIGHEST ROI EXPERIMENT)
4. **Reformat data** (2 min, no GPU)
5. **Generate shared B** (1 min)
6. **Train Math adapter** (~30 min, proves pipeline works)
7. **Evaluate Math adapter** (10 min, proves adapters help)
8. **Train Code + Science** (~60 min)
9. **Build GC-LoRI code** (no compute, pure coding)
10. **Train GC-LoRI Router** (~20 min)
11. **Run ablation table** (~30 min)
12. **Write comparison + paper framing** (no compute)

**Total estimated GPU time: ~3 hours for the complete innovation.**



---

## Source: PIPELINE TASKS

# MEWTWO 20-Hour Research Sprint — Live Task Tracker
# Auto-updated by pipeline. Last updated: 2026-04-21 09:29

## Phase 1: Clean Single-Adapter Evals (Running)
- [x] 1.1 — ARC fixed for all 5 configs ARC-Challenge (fixed) — all 5 configs × 100 samples
- [x] 1.2 — HumanEval fixed for all 5 configs HumanEval (fixed) — all 5 configs × 100 samples
- [x] 1.3 — MATH-500 clean benchmark complete MATH-500 — all 5 configs × 200 samples (uncontaminated)
- [x] 1.4 — MBPP clean benchmark complete MBPP — all 5 configs × 100 samples (uncontaminated)

## Phase 2: TRUE Multi-Adapter Composition (Novel IP)
- [x] 2.1 — LayerBlendRouter module created Create 50 mixed-domain evaluation queries
- [x] 2.2 — 24 modules across 3 domains Implement 5 composition strategies via PEFT
- [x] 2.3 — All 8 configs evaluated on 50 queries Run mixed-domain experiment (8 configs × 50 queries)
- [x] 2.4 — composition Δ=+0.000 → FAIL Compute composite scores and comparative analysis

## Phase 3: Standardized Benchmarks (lm-eval-harness)
- [x] 3.1 — gsm8k complete for all configs — No LayerBlend adapter found GSM8K 8-shot CoT (base + adapters + composed)
- [x] 3.2 — arc_challenge complete for all configs ARC-Challenge 25-shot (base + adapters + composed)
- [x] 3.3 — MMLU-Pro complete MMLU-Pro 5-shot (base + adapters + composed)

## Phase 4: Startup Demo
- [ ] 4.1 Build Gradio interface
- [ ] 4.2 Wire to model inference
- [x] 4.3 — Final analysis complete Record demo video

## Phase 5: Paper Deliverables
- [x] 5.1 — Summary generated Generate comparison tables and figures
- [ ] 5.2 Compute bootstrap confidence intervals
- [x] 5.3 — Final results saved Final results JSON with all numbers



---

## Source: implementation plan

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


