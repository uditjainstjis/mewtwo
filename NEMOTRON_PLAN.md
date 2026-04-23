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

File already exists: `src/lori_moe/configs/nemotron_config.py`

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
    save_path=Path("/home/learner/Desktop/mewtwo/checkpoints/nemotron_lori/shared_projection_B.pt"),
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
    --output_dir ./checkpoints/nemotron_lori/adapters \
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
    --adapter_dir ./checkpoints/nemotron_lori/adapters \
    --data_dir ./data/nemotron \
    --output_dir ./checkpoints/nemotron_lori/gc_router \
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
├── checkpoints/nemotron_lori/
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
