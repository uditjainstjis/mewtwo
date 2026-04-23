# Research Context Knowledge Base

## Scope and Evidence Standard

This document was compiled from a local scan of the repository at `/Users/uditjain/Desktop/mewtwo` on 2026-04-15. Its purpose is to provide a zero-shot, research-grade context payload for a downstream AI that will write an academic paper.

Because this project spans multiple research phases, multiple machines, and both local code plus historical narrative documents, this document uses a strict evidence hierarchy:

1. **Primary local execution evidence**: code paths, raw result files, JSONL logs, checkpoints, training logs, and scripts that are present in this workspace.
2. **Secondary narrative evidence**: README files, chronicles, summaries, and archived markdown reports that describe prior work but are not always fully backed by local executable artifacts.
3. **Missing or off-device evidence**: files explicitly referenced in notes or chronicles but not present in this workspace.

The downstream paper-writing model should treat only category 1 as fully verified execution evidence. Category 2 is useful context, but it should be cited cautiously. Category 3 should be framed as missing external context unless the user supplies additional files from the other machine.

This repository currently contains **two overlapping research threads**:

- **Synapta**: a prompt-level, multi-adapter composition system built around MLX / LoRA routing and evaluated on custom benchmarks.
- **LoRI-MoE**: a later PyTorch / PEFT research branch aiming to reduce cross-adapter interference using frozen random projections, sparsity, and learned routing.

The project also contains archived reports from an earlier or external export under `archive/synapta_v1/` and a narrative summary in `MEWTWO_COMPLETE_RESEARCH_CHRONICLE.md`. Those documents are valuable for intent and chronology, but they overstate what is locally reproducible today.

---

## 1. Core Concept & Architecture

### 1.1 Foundational problem

The core research problem is:

**Can multiple domain-specialized LoRA adapters be composed in a memory-efficient way so that a small base LLM can answer mixed-domain prompts better than either the base model or a single routed adapter, without catastrophic interference?**

This problem appears in two architectural generations:

- **Generation A: Synapta**
  - Compose up to two adapters at inference time.
  - Use a lightweight router to select relevant domains.
  - Clamp adapter influence to avoid destructive interference.
  - Optimize for small-model inference and low VRAM / unified-memory constraints.

- **Generation B: LoRI-MoE**
  - Train domain experts under a more structured low-rank factorization scheme.
  - Replace ordinary independently trained LoRA spaces with frozen random projections plus sparse learned expert matrices.
  - Add routing that can, in principle, vary by token or layer rather than only once per prompt.
  - Target stronger compositional behavior and reduced subspace collision.

### 1.2 Synapta architecture

The live Synapta inference stack is implemented primarily in:

- `backend/dynamic_mlx_inference.py`
- `backend/orchestrator.py`
- `src/eval/real_benchmark.py`
- `src/eval/run_eval_v2.py`
- `src/eval/run_eval_v2b.py`

#### Base model and adapters

- The README identifies the base model as `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.
- The domain registry in `backend/expert_registry.json` defines **20 domain experts**:
  - `LEGAL_ANALYSIS`
  - `MEDICAL_DIAGNOSIS`
  - `PYTHON_LOGIC`
  - `MATHEMATICS`
  - `MLX_KERNELS`
  - `LATEX_FORMATTING`
  - `SANSKRIT_LINGUISTICS`
  - `ARCHAIC_ENGLISH`
  - `QUANTUM_CHEMISTRY`
  - `ORGANIC_SYNTHESIS`
  - `ASTROPHYSICS`
  - `MARITIME_LAW`
  - `RENAISSANCE_ART`
  - `CRYPTOGRAPHY`
  - `ANCIENT_HISTORY`
  - `MUSIC_THEORY`
  - `ROBOTICS`
  - `CLIMATE_SCIENCE`
  - `PHILOSOPHY`
  - `BEHAVIORAL_ECONOMICS`

#### Routing

- `backend/orchestrator.py` provides a chain-of-thought-style domain router.
- The real router is effectively **top-1 and one-hot**.
- A `route_top2()` path exists, but the v2 setup log explicitly states the orchestrator still behaves as a top-1 router and does not natively expose a calibrated soft top-2 distribution.

#### Adapter injection

- `backend/dynamic_mlx_inference.py` wraps linear layers with routed LoRA modules for both MLX and Torch backends.
- The same file implements two clamp modes:
  - `weight_cap`: cap each adapter's effective contribution by a scalar threshold.
  - `norm_ratio`: scale the combined adapter residual by a norm ratio relative to the base output.
- The live engine sets routing weights globally before generation or perplexity evaluation.

#### Evaluation design

Synapta uses two internally defined benchmark regimes:

- **v1 synthetic single-domain benchmark**
  - Source: `backend/ablation_benchmark.py`
  - 20 domains x 5 prompts = 100 prompts
  - Prompts are templated and synthetic
  - Main metric: semantic similarity against templated reference answers

- **v2 mixed benchmark**
  - Source: `data/multidomain_eval_v2.json` plus the same v1 synthetic set
  - 100 single-domain synthetic items + 40 multi-domain compositional items
  - v2 explicitly separates "does composition help on mixed-domain prompts?" from "does composition hurt single-domain prompts?"

#### Important reproducibility caveat

The registry points to adapter files under `backend/expert_adapters/...`, but that directory is **missing in this workspace**. Therefore:

- historical Synapta result files are present and useful as evidence of prior execution,
- but a clean rerun of the Synapta evaluation stack is **not currently possible from this repository alone** without restoring those adapter weights.

### 1.3 LoRI-MoE architecture

The LoRI-MoE branch lives under `src/lori_moe/` and `checkpoints/lori_moe/`.

Core files:

- `src/lori_moe/config.py`
- `src/lori_moe/shared_projection.py`
- `src/lori_moe/lori_adapter.py`
- `src/lori_moe/model/router.py`
- `src/lori_moe/model/lori_moe_linear.py`
- `src/lori_moe/model/lori_moe_model.py`
- `src/lori_moe/training/train_lori_adapter.py`
- `src/lori_moe/training/train_router.py`
- `src/lori_moe/inference/compose.py`

#### Intended conceptual design

The intended LoRI-MoE design is:

- start with a Hugging Face causal LM,
- use low-rank expert adaptation,
- freeze the projection side of the factorization,
- sparsify the trainable side of each expert,
- route tokens or hidden states to the most relevant experts,
- allow top-k expert composition while reducing interference across domains.

#### What is actually trained locally

The locally executed trainer, `src/lori_moe/training/train_lori_adapter.py`, uses **PEFT LoRA machinery** rather than a fully bespoke LoRI runtime.

What it does in practice:

- creates a PEFT LoRA model,
- replaces each module's `lora_A` matrix with a deterministic frozen random matrix,
- trains the `lora_B` side,
- applies DARE-style sparsification after training,
- saves `best`, `final`, and `dare_sparsified` PEFT adapter directories.

This means the executed training pipeline is **LoRI-inspired**, but not a perfect realization of the most ambitious "shared projection across all experts and modules" story told in some research notes.

#### Routing in the LoRI branch

There are two router ideas in the codebase:

- `src/lori_moe/training/train_router.py`
  - trains a pooled hidden-state classifier
  - this is the router that is actually backed by local logs and checkpoints

- `src/lori_moe/model/router.py`
  - implements token-level and multi-layer routing modules
  - this is more ambitious and architecturally important, but not fully backed by local evaluation artifacts

#### Inference status of LoRI-MoE

`src/lori_moe/inference/compose.py` does **not** perform true simultaneous multi-expert composition. It currently:

- routes a prompt,
- selects the highest-weight domain,
- loads one PEFT adapter,
- generates with that single adapter.

So the current local inference path is closer to **single-adapter auto-routing** than to the full multi-expert LoRI-MoE described in the chronicle.

### 1.4 Frameworks, hardware constraints, and technical stack

#### Apple Silicon / MLX side

- MLX / MLX-LM style inference path
- small 4-bit base model
- custom routed LoRA layer replacement
- semantic similarity scoring with sentence-transformers
- strong emphasis on low memory, low latency, and local execution

This branch is consistent with experimentation on a Mac GPU / unified memory device.

#### CUDA / RTX side

- PyTorch
- Transformers
- PEFT
- bitsandbytes
- safetensors
- custom training scripts and automation

This branch is consistent with the user's note that part of the work was done on another device, likely the RTX 5090 machine.

#### Automation and ops infrastructure

The repository also includes:

- `scripts/autonomous_pipeline.py`
- `scripts/shadow_pipeline.py`
- `scripts/start_dashboard.py`
- `scripts/train_all_domains.sh`
- `scripts/watchdog_agent.py`

These support autonomous or semi-autonomous experiment execution, model queueing, monitoring, and recovery. They are relevant to the research workflow, but they are not themselves proof of experimental success.

---

## 2. Hypotheses & Rationale

### 2.1 Synapta-era hypotheses

The early Synapta work appears to have started from the following hypotheses:

1. **A small base model can be meaningfully specialized via many tiny domain LoRA experts.**
2. **Mixed-domain prompts may require activating more than one expert.**
3. **Naive multi-adapter mixing is unstable and needs clamping.**
4. **A lightweight router can deliver most of the benefit without expensive full-model mixture-of-experts infrastructure.**
5. **Prompt-level routing is sufficient for an initial proof-of-concept under Apple Silicon constraints.**

### 2.2 Why this mechanistic approach was chosen

The Synapta approach is mechanically attractive because it avoids full model retraining:

- LoRA experts are cheap relative to full fine-tunes.
- Multiple experts can be stored and swapped with modest memory.
- Routing plus top-k composition gives a route to specialization without maintaining many full models.
- Clamp-based composition is simple enough to implement inside a small custom inference engine.

The rationale was pragmatic:

- fit on consumer hardware,
- keep latency acceptable,
- test composition with simple engineering before attempting deeper mechanistic changes.

### 2.3 Why v2 was necessary

The code and documents show a major methodological realization:

**The original v1 benchmark was not a valid test of composition.**

Why:

- `backend/ablation_benchmark.py` is almost entirely single-domain.
- many prompts are templated synthetic recall tasks.
- a K=2 composition method cannot reasonably outperform a single-domain adapter on a mostly single-domain benchmark.

This led to the v2 redesign:

- keep the single-domain split for regression checking,
- add a true mixed-domain split,
- use oracle routing for the composed methods,
- isolate composition quality from router failure.

That is a sound methodological correction and one of the most important research decisions visible in the repository.

### 2.4 LoRI-MoE hypotheses

The later LoRI-MoE branch rests on a more mechanistic theory:

1. **Ordinary independently trained LoRA adapters interfere because their learned subspaces collide.**
2. **Freezing part of the low-rank structure can regularize the geometry of experts.**
3. **Sparse expert matrices can reduce overlap and improve compositionality.**
4. **Token-level or layer-level routing should outperform prompt-level routing on genuinely mixed prompts.**
5. **The main limitation may shift from clamp formulation to router quality and subspace alignment.**

### 2.5 Architectural reasoning behind LoRI-MoE

The LoRI-MoE branch can be understood as an attempt to move from:

- "compose independently trained adapters and hope clamping prevents damage"

to:

- "design the adapters from the start so they are composition-friendly."

That is why the code emphasizes:

- frozen random projections,
- sparsity,
- orthogonality checking,
- structured routing,
- explicit interference tests.

### 2.6 Secondary and future hypotheses

Several additional research directions are documented but not locally validated:

- **CF-LoRA** in `src/training/cf_lora.py`
  - adds orthogonality penalties against frozen subspaces
- **Subspace-Aware Composition (SAC)** in `src/composition/subspace_aware.py`
  - projects updates into orthogonal complements before composition
- **TIES / additive merge baselines**
- **Grokking / representation instrumentation**
  - `src/training/instrumented_trainer.py`
- **Uditaptor**
  - a proposed cross-architecture adapter transfer framework mentioned in the chronicle

These matter for research trajectory, but they should not be presented as executed contributions unless supporting artifacts are provided from the other machine.

### 2.7 Design documents with incomplete local access

Two PDFs are present:

- `doc1.pdf` with title metadata: `AI Research Breakthrough Strategy`
- `doc2.pdf` with title metadata: `Synapta: Multi-Adapter LLM Research`

Their full text could not be cleanly extracted with the currently available local toolchain, so they should be treated as **known but unread primary notes**. They may contain rationale not fully recoverable from the code alone.

---

## 3. Implementation Status

This section is intentionally strict.

### 3.1 Executed Evidence

These items are clearly backed by local artifacts showing that they were implemented and run.

#### A. Synapta v1 benchmark execution

Verified by:

- `results/real_benchmark_results.json`
- `results/real_benchmark_table.md`
- `results/decision_summary.md`

What is executed:

- 100 synthetic single-domain prompts
- 4 evaluated methods
  - Baseline
  - SingleAdapter
  - UnclampedMix
  - AdaptiveClamp
- recorded semantic similarity, perplexity, and latency

#### B. Synapta v2 benchmark execution

Verified by:

- `results/v2_both_raw.jsonl`
- `results/v2_decision_summary.md`
- `results/v2_final_status.txt`
- `results/v2_setup_log.txt`

What is executed:

- 100 single-domain synthetic items
- 40 mixed-domain items
- 4 evaluated methods
  - Baseline
  - SingleAdapter
  - AdaptiveClamp-v2
  - UnclampedMix-v2
- oracle routing for the composed methods on v2

#### C. Synapta clamp and routing ablations

Verified by:

- `results/v2_md_clamp_ablation.jsonl`
- `results/v2_clamp_ablation_summary.md`
- `results/v2_md_routing_ablation.jsonl`
- `results/v2_routing_gap_summary.md`

What is executed:

- a direct weight-cap vs norm-ratio comparison on the multi-domain split
- a real-router vs oracle-router comparison on the multi-domain split

#### D. LoRI-style dataset preparation

Verified by:

- `data/lori_moe/dataset_stats.json`
- domain JSONL files under `data/lori_moe/`

What is executed:

- prepared domain corpora for:
  - math
  - code
  - science
  - legal
  - medical
- built mixed-domain routing data in `routing_mixed_train.jsonl`

#### E. LoRI-style adapter training for Qwen2.5-1.5B

Verified by:

- `checkpoints/lori_moe/qwen2.5_1.5b/*/training_log.json`
- saved PEFT adapter directories under each domain's `best`, `final`, and `dare_sparsified` folders
- `logs/lori_moe/pipeline.log`
- `checkpoints/lori_moe/pipeline_state.json`

What is executed:

- successful domain-adapter training for:
  - math
  - code
  - science
  - legal
  - medical

#### F. Router training for Qwen2.5-1.5B

Verified by:

- `checkpoints/lori_moe/qwen2.5_1.5b/router/best/router.pt`
- `logs/lori_moe/train_router.log`
- `logs/lori_moe/router_training.log`

What is executed:

- a pooled hidden-state classifier router
- trained on 5,000 routing examples total
- reached 97.2% training accuracy after epoch 1 and 100.0% after epoch 2

#### G. Partial model-scaling attempts

Verified by:

- `checkpoints/lori_moe/qwen3.5_0.8b/math/training_log.json`
- `checkpoints/lori_moe/qwen2.5_0.5b/math/training_log.json`
- `checkpoints/lori_moe/qwen2.5_7b/science/training_log.json`
- `logs/lori_moe/*`

What is executed:

- interrupted partial training for `qwen3.5_0.8b` on math
- failed / no-progress attempts for `qwen2.5_0.5b` and `qwen2.5_7b`

### 3.2 Partial Implementation

These items exist in code, but local evidence shows they are incomplete, only partly integrated, or not backed by executed result artifacts.

#### A. Native LoRI-MoE runtime model

Files:

- `src/lori_moe/model/lori_moe_model.py`
- `src/lori_moe/model/lori_moe_linear.py`
- `src/lori_moe/model/router.py`

Why this is only partial:

- there is no local benchmark result suite proving this runtime was used end-to-end for final evaluation,
- the chronicle claims a module-path "subspace mismatch" bug was fixed, but local code still caches projection tensors by input dimension in a way that appears inconsistent with that claim,
- the executed training pipeline uses PEFT adapters, which only partially aligns with the native runtime abstraction.

#### B. LoRI-MoE composition inference

File:

- `src/lori_moe/inference/compose.py`

Why this is only partial:

- it selects a single highest-weight adapter rather than truly combining multiple experts during generation.

#### C. LoRI benchmark suite

Files:

- `src/lori_moe/eval/run_benchmarks.py`
- `src/lori_moe/eval/ablation_suite.py`

Why this is only partial:

- the benchmark runner expects to write to `results/lori_moe/`, but those result files are absent here,
- the ablation suite itself explicitly says parts are mocked or skipped and writes placeholder values.

#### D. Autonomous experiment infrastructure

Files:

- `scripts/autonomous_pipeline.py`
- `scripts/shadow_pipeline.py`
- `scripts/train_all_domains.sh`
- `scripts/start_dashboard.py`

Why this is only partial:

- the pipeline shows real execution, but also early crashes and OOM handling issues,
- `scripts/train_all_domains.sh` appears stale relative to the current CLI signatures,
- the infrastructure is operationally useful but not mature enough to count as finished research software.

#### E. Synapta alternative routers and composition modules

Files:

- `src/routers/embedding_router.py`
- `src/routers/classifier_router.py`
- `src/routers/gated_router.py`
- `src/routers/multilabel_cot_router.py`
- `src/adapters/adaptive_multi_lora_linear.py`

Why this is only partial:

- some are reference implementations or side ablations,
- `results/v2_setup_log.txt` explicitly states the live backend did not use the `AdaptiveMultiLoRALinear` reference path,
- router ideas exist beyond the live top-1 orchestrator, but their integration into the final Synapta path is incomplete.

### 3.3 Unimplemented Hypotheses

These are theoretical directions, future ideas, or code scaffolds without local execution evidence.

#### A. CF-LoRA and subspace-regularized training as a completed result

Files:

- `src/training/cf_lora.py`
- `src/training/run_training_matrix.py`

Status:

- implemented as research code scaffolding,
- not backed by local result files proving completed experiments.

#### B. Subspace-Aware Composition (SAC) as a completed evaluated method

Files:

- `src/composition/subspace_aware.py`
- `src/composition/additive.py`
- `src/composition/ties_merge.py`

Status:

- algorithmic code exists,
- no local executed benchmark suite ties these methods to final reported results.

#### C. Uditaptor

Primary mention:

- `MEWTWO_COMPLETE_RESEARCH_CHRONICLE.md`

Status:

- conceptual only in this workspace,
- no implementation or result artifacts were found.

#### D. Full LoRI-MoE evaluation phases described in the chronicle

Chronicle-referenced but locally missing:

- `results/lori_moe/phase1_baselines.json`
- `results/lori_moe/phase2_single_adapter.json`
- `results/lori_moe/phase3_composite.json`
- `results/lori_moe/phase4_interference.json`
- `results/lori_moe/all_results.json`

Status:

- not present in this repository,
- therefore not admissible as executed evidence without external files.

#### E. A fully validated module-path bug fix for subspace mismatch

Claim source:

- `MEWTWO_COMPLETE_RESEARCH_CHRONICLE.md`

Status:

- not reliable as a finished implementation claim from the local workspace alone.

---

## 4. Experiments, Results & Inferences

### 4.1 Synapta v1: original "full benchmark"

#### Setup

Source files:

- benchmark definition: `backend/ablation_benchmark.py`
- evaluation runner: `src/eval/real_benchmark.py`
- results: `results/real_benchmark_results.json`, `results/real_benchmark_table.md`

Design:

- 20 domains
- 5 prompts per domain
- 100 prompts total
- 4 methods
- 400 total inferences

Important methodological reality:

- this benchmark is mostly **single-domain synthetic recall**,
- many prompts are templated in a domain-name-substitution style rather than naturally occurring research tasks.

#### Aggregate results

| Method | Avg semantic similarity | Avg perplexity | Avg latency |
| --- | ---: | ---: | ---: |
| Baseline | 0.620 | 64.5 | 2.80 s |
| SingleAdapter | 0.622 | 60.9 | 2.69 s |
| UnclampedMix | 0.557 | 51.2 | 2.51 s |
| AdaptiveClamp | 0.611 | 58.0 | 2.67 s |

Key reported delta:

- `AdaptiveClamp - SingleAdapter = -0.011` semantic similarity

#### Inferences

1. **AdaptiveClamp did not beat SingleAdapter on the v1 benchmark.**
2. **UnclampedMix was semantically worse despite lower perplexity and latency.**
3. **Perplexity improved under some bad generations, so perplexity was not a reliable proxy for semantic correctness in this benchmark.**
4. **This result should not be generalized to "composition does not work."** The benchmark mostly did not require composition in the first place.

### 4.2 Synapta v2: corrected composition test

#### Setup

Source files:

- runner: `src/eval/run_eval_v2.py`
- raw logs: `results/v2_both_raw.jsonl`
- summaries: `results/v2_decision_summary.md`, `results/v2_final_status.txt`

Design:

- 100 single-domain synthetic items
- 40 multi-domain compositional items
- 560 total inferences
- composed methods used **oracle routing** via `required_adapters`

This is a much better test of whether composition itself helps.

#### Aggregate results by split

| Method | Single-domain avg sim | Multi-domain avg sim |
| --- | ---: | ---: |
| Baseline | 0.6090 | 0.6473 |
| SingleAdapter | 0.6064 | 0.6334 |
| AdaptiveClamp-v2 | 0.6058 | 0.6505 |
| UnclampedMix-v2 | 0.6041 | 0.6505 |

#### Inferences

1. **Composition helps on the multi-domain split.**
   - `AdaptiveClamp-v2` beats `SingleAdapter` by `+0.0171` on multi-domain prompts.

2. **Composition does not help on the single-domain split.**
   - this is expected and methodologically healthy,
   - activating more than one expert on a single-domain prompt should not be expected to help.

3. **The original v1 negative result was at least partly a benchmark-design artifact.**

4. **Clamp formulation was not the main determinant of the v2 gain.**
   - `AdaptiveClamp-v2` and `UnclampedMix-v2` tie at `0.6505` on the multi-domain split in the aggregate summary.

5. **The positive multi-domain result is real but modest.**
   - the pre-registered or desired threshold of `+0.03` over `SingleAdapter` was not reached.

#### Hypothesis verdicts from the local summaries

Based on `results/v2_decision_summary.md`:

- H1: pass
- H2: fail relative to the stronger `+0.03` threshold
- H3: pass
- H4: pass
- H5: fail

The local interpretation is:

- composition has a directional benefit on tasks that actually require multiple domains,
- but the gain is not yet large enough to claim a decisive breakthrough.

### 4.3 Clamp ablation: weight-cap vs norm-ratio

#### Setup

Source files:

- runner: `src/eval/run_eval_v2b.py`
- raw results: `results/v2_md_clamp_ablation.jsonl`
- summary: `results/v2_clamp_ablation_summary.md`

Design:

- multi-domain split only
- 40 items
- 3 methods
- 120 logged runs

#### Aggregate results

| Method | Multi-domain avg sim |
| --- | ---: |
| SingleAdapter | 0.6334 |
| AC-v2-WeightCap | 0.6505 |
| AC-v2-NormRatio | 0.6502 |

Key delta:

- `NormRatio - WeightCap = -0.0003`

#### Inferences

1. **Norm-ratio clamping is almost identical to weight-cap in this setting.**
2. **The earlier v1 failure is not explained by choosing the "wrong" clamp formula.**
3. **At this model scale and benchmark scale, routing and task design matter more than clamp mathematics.**

### 4.4 Routing-gap ablation

#### Setup

Source files:

- raw results: `results/v2_md_routing_ablation.jsonl`
- summary: `results/v2_routing_gap_summary.md`

Design:

- multi-domain split only
- real router vs oracle router
- 120 logged runs

#### Aggregate results

| Method | Multi-domain avg sim |
| --- | ---: |
| SingleAdapter | 0.6296 |
| AC-v2-Norm-RealRouter | 0.6350 |
| AC-v2-Norm-Oracle | 0.6502 |

Additional summary stats:

- average effective `K = 1.75`
- oracle headroom over single-adapter: `+0.0206`
- realized gain with real router: `+0.0054`
- headroom recovery: about `26%`

#### Inferences

1. **Routing is a real bottleneck.**
2. **Routing is not the only bottleneck.**
   - even with oracle routing, the gain remains moderate.
3. **The current real router captures only a minority of the available compositional upside.**

### 4.5 Mistral comparison artifact

Files:

- `results/mistral_md_results.json`
- `results/mistral_vs_synapta_verified.md`

Locally verifiable details:

- `results/mistral_md_results.json` contains 40 per-item records with:
  - `id`
  - `similarity`
  - `latency`
- mean similarity computed from that file is about `0.6171`

The paired markdown claims:

- Synapta MD average similarity: `0.6525`
- Mistral MD average similarity: `0.617`
- roughly `75%` VRAM reduction for Synapta

#### Inference

This is usable as an internal comparison artifact, but it should be presented cautiously:

- it is based on a custom benchmark,
- the benchmark is not a standard external evaluation suite,
- local evidence supports the Mistral JSON and the Synapta summary, but not a broader generalization beyond this benchmark.

### 4.6 LoRI-style dataset preparation

Source:

- `data/lori_moe/dataset_stats.json`

Prepared dataset statistics:

| Domain | Examples | Avg length |
| --- | ---: | ---: |
| math | 49,999 | 927.1 |
| code | 20,016 | 504.2 |
| science | 11,679 | 742.2 |
| legal | 109 | 810.3 |
| medical | 10,178 | 1112.8 |

#### Inferences

1. The legal domain is extremely underrepresented relative to the others.
2. The training pipeline prepared much more data than was actually consumed during the 1.5B runs.
3. Any strong legal-domain loss result must be interpreted with overfitting risk in mind.

### 4.7 LoRI-style adapter training results

Primary evidence:

- `checkpoints/lori_moe/qwen2.5_1.5b/<domain>/training_log.json`
- each domain's `best/training_state.json`
- `logs/lori_moe/pipeline.log`

Common training configuration observed in saved states:

- `max_train_samples = 10000`
- `max_seq_length = 512`
- `gradient_checkpointing = true`
- optimizer backend: `bnb_paged_adamw_8bit`
- effective batch behavior around batch size 16 and grad accumulation 4 in the successful 1.5B runs

#### Qwen2.5-1.5B domain runs

| Domain | Total steps | Best loss | Total time |
| --- | ---: | ---: | ---: |
| math | 468 | 0.1287387110 | 31.98 min |
| code | 468 | 0.4242209361 | 17.43 min |
| science | 468 | 1.3592145503 | 22.59 min |
| legal | 468 | 0.0000128576 | 17.54 min |
| medical | 468 | 0.1170216392 | 25.99 min |

#### Inferences

1. **The five Qwen2.5-1.5B domain adapters were actually trained and saved.**
2. **The legal loss is suspiciously low and likely reflects overfitting on only 109 examples rather than robust domain mastery.**
3. **The executed LoRI branch has strong evidence for training success, but not yet for composition success.**

### 4.8 Router training results in the LoRI branch

Primary evidence:

- `logs/lori_moe/train_router.log`
- `logs/lori_moe/router_training.log`
- `checkpoints/lori_moe/qwen2.5_1.5b/router/best/router.pt`

Observed results:

- 5,000 total routing examples
- 1,000 examples per domain
- epoch 1 accuracy: `97.2%`
- epoch 2 accuracy: `100.0%`
- training time: about `1.7 min`

#### Inferences

1. **A router was definitely trained.**
2. **The visible logs report training accuracy, not a robust held-out generalization study.**
3. **This router evidence supports operational readiness for internal routing experiments, not necessarily strong scientific claims about generalization.**

### 4.9 Scaling attempts and failure points

Primary evidence:

- `logs/lori_moe/pipeline.log`
- `logs/lori_moe/pipeline_stdout.log`
- `checkpoints/lori_moe/pipeline_state.json`
- per-model training logs

Observed outcomes:

- an early autonomous pipeline run crashed due to `KeyError: 'initial_batch_size'`
- the pipeline later retried with lower batch size / shorter sequence settings
- `qwen2.5_1.5b` completed across all five domains in about `116.1 min`
- `qwen2.5_0.5b` and `qwen2.5_7b` attempts show failure / no useful progress
- `qwen3.5_0.8b` math training reached `644` steps with best loss `0.7113559663` before interruption (`SIGINT`)

#### Inferences

1. **Scaling infrastructure is real but not stable.**
2. **The only clearly successful multi-domain LoRI training sweep in this workspace is the Qwen2.5-1.5B run.**
3. **Claims about broader scaling should be treated as aspirational or off-device unless more artifacts are supplied.**

### 4.10 Critical contradictions and truth-preserving notes

These are important for any academic writeup.

1. **The chronicle describes a fully working LoRI-MoE composition story, but the local workspace does not contain the claimed `results/lori_moe/*.json` result suite.**
2. **`src/lori_moe/inference/compose.py` is single-adapter selection, not true multi-expert composition.**
3. **The claimed "subspace mismatch bug fix" is not clearly established by the local code.**
   - `src/lori_moe/model/lori_moe_model.py` still appears to cache projection tensors by input dimension rather than cleanly binding module-specific projections by full module identity during runtime injection.
4. **The executed trainer freezes deterministic random `lora_A` matrices per module, but this is not the same as a single globally shared B matrix in the strongest literal sense.**
5. **The Synapta result files are real local evidence, but the underlying `backend/expert_adapters/` weights are missing, so today they are historical outputs rather than directly reproducible runs.**

---

## 5. Data & Artifact Mapping

This section maps the main artifacts without dumping large files.

### 5.1 Primary result artifacts

| Path | Structure | What it contains | Why it matters |
| --- | --- | --- | --- |
| `results/real_benchmark_results.json` | JSON object keyed by 20 domains | For each domain, each method stores 5 records with `question`, `ground_truth`, `generated`, `semantic_sim`, `perplexity`, `latency_s` | Primary per-example record of Synapta v1 execution |
| `results/real_benchmark_table.md` | Markdown table | Per-domain and aggregate averages across the 4 v1 methods | Fast human-readable summary of v1 |
| `results/decision_summary.md` | Markdown narrative | Pass/fail interpretation of the v1 benchmark | Captures the research decision after v1 |
| `results/v2_both_raw.jsonl` | JSONL, 560 lines | One record per run with `split`, `item_id`, `domains`, `method`, `K_used`, `clamp`, `routing`, `semantic_sim`, `perplexity`, `latency_s`, plus preview text | Most important raw artifact for the corrected v2 study |
| `results/v2_decision_summary.md` | Markdown narrative | Aggregate v2 outcomes and hypothesis verdicts | Main local interpretation of the corrected benchmark |
| `results/v2_final_status.txt` | Text summary | Short final status statement for v2 | High-level status note |
| `results/v2_setup_log.txt` | Text note | Documents that the live backend used weight-cap style routing and that the orchestrator is top-1 / one-hot | Critical truth-preserving artifact |
| `results/v2_md_clamp_ablation.jsonl` | JSONL, 120 lines | Multi-domain clamp ablation records with `clamp_mode`, `routing_fn`, and metrics | Primary evidence that norm-ratio and weight-cap are nearly identical here |
| `results/v2_clamp_ablation_summary.md` | Markdown summary | Aggregate clamp-ablation scores | Short citation-friendly ablation summary |
| `results/v2_md_routing_ablation.jsonl` | JSONL, 120 lines | Multi-domain routing-gap records comparing single-adapter, real router, and oracle router | Primary evidence for routing bottleneck size |
| `results/v2_routing_gap_summary.md` | Markdown summary | Aggregate routing-gap findings, including recovered oracle headroom | Short citation-friendly routing summary |
| `results/mistral_md_results.json` | JSON array, 40 items | Per-item custom-benchmark similarity and latency for Mistral | Comparator artifact for the custom MD benchmark |
| `results/mistral_vs_synapta_verified.md` | Markdown note | Interprets the custom Synapta-vs-Mistral comparison | Useful internal benchmark comparison, not a standard public eval |
| `results/gated_routing_embedding_results.json` | JSON array, 140 items | Per-item router behavior with fields such as `truth`, `reason`, `active_domains`, `k_eff`, `sim` | Side artifact for router behavior analysis |
| `results/v2_router_ablation_results.json` | JSON array, 200 items | Per-item router ablation records with `method`, `routing_accuracy`, `active_domains`, `truth`, `sim`, `lat` | Side artifact for routing-focused ablations |

### 5.2 Benchmark and dataset artifacts

| Path | Structure | What it contains | Why it matters |
| --- | --- | --- | --- |
| `backend/ablation_benchmark.py` | Python source with hardcoded prompt sets | 20-domain x 5-question synthetic single-domain benchmark | Defines the v1 benchmark and explains why it is not a true composition test |
| `data/multidomain_eval_v2.json` | JSON array, 40 items | Mixed-domain prompts with `id`, `domains`, `question`, `reference_answer`, `required_adapters` | Defines the true composition split for v2 |
| `data/lori_moe/dataset_stats.json` | JSON object | Example counts and average lengths for each LoRI domain corpus | Primary evidence of available training data scale |
| `data/lori_moe/*_train.jsonl` | JSONL files | Domain corpora with fields like `text` and `domain` | Primary training data for the LoRI branch |
| `data/lori_moe/routing_mixed_train.jsonl` | JSONL file | Mixed-domain routing samples with `text`, `domains`, and `type` | Training data for the LoRI router |
| `data/pure_code.json` and `data/pure_math.json` | JSON arrays | Small synthetic or custom eval prompts by domain | Auxiliary eval assets |
| `data/mixed_fincode.jsonl` | JSONL | Mixed-domain examples with prompt-level metadata | Auxiliary mixed-domain dataset |

### 5.3 Checkpoints and training artifacts

| Path | Structure | What it contains | Why it matters |
| --- | --- | --- | --- |
| `checkpoints/lori_moe/qwen2.5_1.5b/<domain>/training_log.json` | JSON | Domain-level training summary: steps, best loss, runtime | Primary evidence of successful 1.5B adapter training |
| `checkpoints/lori_moe/qwen2.5_1.5b/<domain>/best/` | PEFT adapter directory | `adapter_model.safetensors`, config, tokenizer assets, `training_state.json` | Best checkpoint per domain |
| `checkpoints/lori_moe/qwen2.5_1.5b/<domain>/final/` | PEFT adapter directory | Final checkpoint after training | Shows post-training saved state |
| `checkpoints/lori_moe/qwen2.5_1.5b/<domain>/dare_sparsified/` | PEFT adapter directory | Sparsified checkpoint variant | Evidence that DARE sparsification was actually applied |
| `checkpoints/lori_moe/qwen2.5_1.5b/router/best/router.pt` | PyTorch checkpoint | Trained router weights | Evidence of router training completion |
| `checkpoints/lori_moe/pipeline_state.json` | JSON | Tracks completed and failed models / domains in the autonomous pipeline | High-level execution bookkeeping |
| `checkpoints/lori_moe/queue_config.json` | JSON | Queue definition for model/domain training | Explains intended scaling plan |
| `checkpoints/lori_moe/qwen3.5_0.8b/math/*` | Partial checkpoint tree | Interrupted training artifacts plus `checkpoint-500` | Evidence of a partial scaling attempt |

### 5.4 Log artifacts

| Path | Structure | What it contains | Why it matters |
| --- | --- | --- | --- |
| `logs/lori_moe/pipeline.log` | Plaintext log | Queueing, retries, failures, and completed training stages | Primary chronology for the LoRI automation runs |
| `logs/lori_moe/pipeline_stdout.log` | Plaintext log | Console-style pipeline output | Additional execution trace |
| `logs/lori_moe/train_router.log` | Plaintext log | Router training progress and accuracy | Direct record of router training success |
| `logs/lori_moe/router_training.log` | Plaintext log | Router config and epoch summaries | Confirms sample counts and final training accuracy |
| `logs/lori_moe/train_qwen2.5_*` and similar logs | Plaintext logs | Per-run model training traces, including OOMs and interruptions | Primary evidence for scaling constraints and failure modes |

### 5.5 Code artifacts that define the main scientific logic

| Path | Role | Status note |
| --- | --- | --- |
| `backend/dynamic_mlx_inference.py` | Live Synapta inference backend with routed LoRA and clamp logic | Executed evidence path |
| `backend/orchestrator.py` | Live Synapta router | Executed, but limited to top-1 style behavior |
| `src/eval/real_benchmark.py` | v1 Synapta evaluator | Executed evidence path |
| `src/eval/run_eval_v2.py` | v2 corrected evaluator | Executed evidence path |
| `src/eval/run_eval_v2b.py` | clamp and routing ablations | Executed evidence path |
| `src/adapters/adaptive_multi_lora_linear.py` | Reference layer-wise adaptive clamp implementation | Not the live backend used in v2 |
| `src/lori_moe/training/train_lori_adapter.py` | Actual locally executed LoRI-style adapter trainer | Executed evidence path |
| `src/lori_moe/training/train_router.py` | Actual locally executed router trainer | Executed evidence path |
| `src/lori_moe/model/lori_moe_model.py` | Ambitious native LoRI-MoE wrapper | Partial and internally inconsistent with some narrative claims |
| `src/lori_moe/inference/compose.py` | LoRI inference entrypoint | Currently single-adapter selection, not true composition |
| `src/lori_moe/eval/run_benchmarks.py` | Intended standard-benchmark evaluator | Partial; no local result suite present |
| `src/lori_moe/eval/ablation_suite.py` | Intended causal-ablation driver | Partial; explicitly contains mocked / skipped logic |

### 5.6 Historical and narrative artifacts

| Path | Nature | How to use it |
| --- | --- | --- |
| `README.md` | Current top-level narrative summary | Good short overview, but it compresses nuance |
| `MEWTWO_COMPLETE_RESEARCH_CHRONICLE.md` | Broad historical narrative across phases and machines | Useful chronology and ideas, but not fully trustworthy as local execution evidence |
| `research_results.md` | Short research summary note | Use cautiously; some claims exceed local artifact support |
| `implementation_plan.md` | Planning and design document | Good source for unimplemented hypotheses |
| `docs/v2_prereg.md` | Pre-registration style plan | Important for intended evaluation logic; not proof of execution |
| `archive/synapta_v1/` | Archived historical export | Strong secondary context for earlier Synapta work, including `results_db.jsonl` |
| `mewtwo_research_export.zip` | Zipped historical export | Backup / import artifact from another environment |

### 5.7 Missing, scattered, or off-device artifacts

These are the most important known gaps.

1. **Missing Synapta adapter weights**
   - `backend/expert_adapters/` is absent.
   - This blocks direct rerun of Synapta from the current workspace.

2. **Missing LoRI benchmark result suite**
   - The chronicle references:
     - `results/lori_moe/phase1_baselines.json`
     - `results/lori_moe/phase2_single_adapter.json`
     - `results/lori_moe/phase3_composite.json`
     - `results/lori_moe/phase4_interference.json`
     - `results/lori_moe/all_results.json`
   - None of these are present locally.

3. **Potentially off-device bug-fix history**
   - The chronicle refers to a resolved "subspace mismatch" issue.
   - The local code does not cleanly prove that resolution.
   - If the fix happened on the RTX machine and was not synced here, that missing code is important.

4. **PDF notes not fully extracted**
   - `doc1.pdf`
   - `doc2.pdf`
   - only titles were recoverable locally

5. **Scattered external context explicitly mentioned by the user**
   - the user stated some research work and files live on another device,
   - therefore this document should be read as a **high-confidence local reconstruction**, not as the final complete archive of the whole project.

### 5.8 What the downstream paper-writing AI should not overclaim

The next AI should avoid stating the following as settled facts unless the missing external artifacts are supplied:

1. That LoRI-MoE composite generation was fully benchmarked end-to-end on standard tasks from this repository alone.
2. That the reported orthogonality average of `0.00683` is locally verified.
3. That the claimed interference-collapse numbers from the chronicle are locally reproducible here.
4. That the module-path subspace mismatch bug was definitely fixed in the code present in this workspace.
5. That Uditaptor moved beyond concept stage.

---

## Bottom-Line Research Truth

The strongest locally verified scientific story is:

1. **Synapta proved that prompt-level multi-adapter composition is not meaningfully helped by evaluating on a synthetic single-domain benchmark.**
2. **After correcting the benchmark in v2, top-2 composition shows a real but modest gain on genuinely mixed-domain prompts.**
3. **Clamp choice is not the main limiting factor at this scale.**
4. **Routing quality is a real bottleneck, but not the only one.**
5. **The later LoRI branch successfully trained domain experts and a router, but the locally verified evidence stops short of proving a completed, end-to-end, multi-expert composition breakthrough.**

The most honest paper-level framing from this workspace alone is therefore:

- **Synapta delivered a careful benchmark correction and a modest positive compositional result.**
- **LoRI-MoE represents a promising mechanistic redesign with real training progress, but its strongest composition claims require external or missing artifacts before they can be treated as fully validated.**
