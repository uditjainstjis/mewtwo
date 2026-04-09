# From Prompt-Level Multi-Adapter Composition to Orthogonal Low-Rank Experts:
# A Grounded Research Draft Based on the `mewtwo` Repository

## Abstract

This document reconstructs the research program embodied in the `mewtwo` repository, which spans two distinct phases. The first phase studies prompt-level multi-adapter LoRA composition with bounded mixing on a small Qwen base model and reports largely negative-to-mixed results. The second phase proposes and partially implements a new architecture, LoRI-MoE, which replaces scalar-bounded prompt-level composition with structurally orthogonal low-rank experts and learned routing on CUDA hardware. The central empirical findings are mixed but informative. In the older Synapta line, bounded multi-adapter composition is safe but does not reliably outperform single-adapter routing on the original benchmark, while later real multi-domain evaluation shows a modest positive signal that remains below preregistered thresholds. In the newer LoRI-MoE line, five domain adapters and one router were successfully trained for Qwen2.5-1.5B, and the saved expert weights exhibit very low cross-domain cosine overlap, with average absolute off-diagonal similarity 0.00685. However, the strongest system-level claims are not yet fully validated: the saved router accuracy reflects training-set classification rather than held-out multi-domain routing, the generated mixed-domain routing dataset is not consumed by the current trainer, and the benchmark harness does not yet run the full token-level routed LoRI-MoE stack end to end. The repository therefore contains a meaningful architectural transition and several real empirical advances, but it does not yet support a clean claim of final system superiority over larger baselines.

## 1. Introduction

The `mewtwo` repository captures a transition from one research thesis to another.

The original thesis, which we refer to as the Synapta line, asks whether multiple domain-specific LoRA adapters can be composed at inference time on a small base model without catastrophic interference. The main mechanism is prompt-level routing plus bounded adapter fusion, first through a simple per-adapter weight cap and later through a more principled norm-ratio variant.

The later thesis, which we refer to as the LoRI-MoE line, asks whether the composition problem should instead be solved structurally. Rather than combining multiple learned low-rank updates in the same latent geometry, LoRI-MoE freezes a shared random projection and trains only sparse domain-specific factors, aiming to make experts approximately orthogonal by construction. Routing is then moved from prompt-level heuristics toward token-level selection.

The repository is scientifically useful because it preserves both phases:

1. the negative and modestly positive findings from real prompt-level composition experiments, and
2. the partial but real implementation of a more ambitious RTX-oriented architecture.

This draft is intentionally grounded in the local artifacts. It distinguishes between:

- results directly supported by saved benchmarks, logs, and checkpoints,
- architectural claims supported by code and partial training artifacts, and
- forward-looking claims that remain unverified.

## 2. Repository Overview

The `mewtwo` repository contains two main research strata.

### 2.1 Synapta / Prompt-Level Composition Stratum

This stratum is documented in:

- `/Users/uditjain/Desktop/mewtwo/README.md`
- `/Users/uditjain/Desktop/mewtwo/results/decision_summary.md`
- `/Users/uditjain/Desktop/mewtwo/results/v2_decision_summary.md`
- `/Users/uditjain/Desktop/mewtwo/results/v2_routing_gap_summary.md`

Its core ingredients are:

- a `Qwen2.5-1.5B-Instruct-4bit` MLX base model,
- 20 domain-specific LoRA experts,
- prompt-level routing,
- bounded or unclamped multi-adapter mixing,
- semantic similarity, perplexity, and latency as the main metrics.

### 2.2 LoRI-MoE / Orthogonal Experts Stratum

This stratum is documented in:

- `/Users/uditjain/Desktop/mewtwo/research_results.md`
- `/Users/uditjain/Desktop/mewtwo/implementation_plan.md`
- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/`
- `/Users/uditjain/Desktop/mewtwo/checkpoints/lori_moe/`

Its core ingredients are:

- CUDA / RTX-oriented training and inference,
- a frozen shared random projection,
- sparse trainable expert matrices,
- a learned router,
- a claimed shift from prompt-level to token-level expert selection,
- a move away from semantic similarity toward exact-match / pass@1 evaluation.

## 3. Phase I: Prompt-Level Multi-Adapter Composition

### 3.1 Research Question

The original question was whether a small model equipped with multiple domain LoRAs could outperform a simpler single-adapter baseline on questions requiring composite expertise, while remaining stable and fast on Apple Silicon.

### 3.2 Methods

The primary methods in the original benchmark were:

| Method | K | Clamp | Description |
|---|---:|---:|---|
| Baseline | 0 or 1 | 0.001 | Base model without expert composition |
| SingleAdapter | 1 | 0.5 | One routed domain adapter |
| AdaptiveClamp | 2 | 0.5 | Two-adapter prompt-level mixture with bounded contribution |
| UnclampedMix | 2 | 999 | Two-adapter prompt-level mixture without effective bounding |

### 3.3 v1 Results: Negative Core Result

The v1 real benchmark covered 100 questions and 400 real MLX inferences. The aggregate results were:

| Method | Avg Semantic Similarity | Avg PPL | Avg Latency |
|---|---:|---:|---:|
| Baseline | 0.6196 | 64.5 | 2.803s |
| SingleAdapter | 0.6223 | 60.9 | 2.695s |
| AdaptiveClamp | 0.6106 | 58.0 | 2.672s |
| UnclampedMix | 0.5573 | 51.2 | 2.511s |

The preregistered headline hypothesis failed:

\[
\Delta_{\text{SIM}}(\text{AdaptiveClamp} - \text{SingleAdapter}) = -0.011
\]

The v1 result is therefore negative in the most important sense: bounded two-adapter composition did not beat the simpler routed single-adapter baseline.

### 3.4 v2 Results: Real Multi-Domain Benchmark

The later v2 benchmark was stronger because it separated single-domain and genuine multi-domain queries.

Aggregate v2 results:

#### Single-Domain Split (100 questions)

| Method | Avg Sim | Avg PPL | Avg Latency |
|---|---:|---:|---:|
| Baseline | 0.6090 | 64.5 | 3.700s |
| SingleAdapter | 0.6064 | 60.9 | 3.571s |
| AdaptiveClamp-v2 | 0.6058 | 57.9 | 3.657s |
| UnclampedMix-v2 | 0.6041 | 52.3 | 3.623s |

#### Multi-Domain Split (40 questions)

| Method | Avg Sim | Avg PPL | Avg Latency |
|---|---:|---:|---:|
| Baseline | 0.6473 | 12.7 | 4.059s |
| SingleAdapter | 0.6334 | 12.7 | 4.057s |
| AdaptiveClamp-v2 | 0.6505 | 12.6 | 4.090s |
| UnclampedMix-v2 | 0.6505 | 12.6 | 4.100s |

The main v2 interpretation is:

- composition became directionally positive on the multi-domain split,
- but the gain stayed below the preregistered threshold,
- and the old clamp implementation could not truly isolate clamp necessity because the chosen routing weights never activated the cap differently.

The key delta was:

\[
\Delta_{\text{SIM}}(\text{AdaptiveClamp-v2} - \text{SingleAdapter}) = +0.0171
\]

This is scientifically interesting but not decisive.

### 3.5 v2c Results: Routing Gap

The v2c study explicitly measured how much of the remaining multi-domain gain depended on routing quality.

| Method | Routing | Avg Sim | Avg PPL | Avg Latency | Avg K |
|---|---|---:|---:|---:|---:|
| SingleAdapter | CoT K=1 | 0.6296 | 12.8 | 4.178s | 1.00 |
| AC-v2-Norm-RealRouter | CoT Top-2 | 0.6350 | 12.7 | 4.167s | 1.75 |
| AC-v2-Norm-Oracle | Oracle K=2 | 0.6502 | 12.6 | 4.211s | 2.00 |

Derived quantities:

- Oracle headroom: `+0.0206`
- Realized gain: `+0.0054`
- Routing gap: `-0.0152`

The important conclusion is subtle: routing was a bottleneck, but even perfect routing only revealed modest total headroom. This suggested that prompt-level composition on a 1.5B model was not mainly failing because of the router. The model and the composition mechanism itself were likely the larger bottlenecks.

## 4. Motivation for the Architectural Pivot

The LoRI-MoE pivot arises from the weaknesses exposed above.

The older Synapta line encountered four problems:

1. prompt-level routing was too coarse for genuinely mixed reasoning,
2. semantic similarity was a weak measure of factual or deductive correctness,
3. the geometry of merged low-rank updates remained interference-prone,
4. even oracle routing yielded only modest gains.

The LoRI-MoE branch therefore adopts a new thesis:

- solve interference structurally rather than with scalar clamping,
- use routing for specialization rather than rescue,
- move evaluation toward exact-match and pass@1,
- target a stronger CUDA workflow with more memory and bandwidth.

## 5. Phase II: LoRI-MoE

### 5.1 Architectural Idea

LoRI-MoE uses a shared frozen projection and domain-specific sparse low-rank matrices.

At a high level:

\[
\Delta W_k = A_k B
\]

where:

- \(B\) is frozen and shared across experts,
- \(A_k\) is trainable and sparse for domain \(k\),
- the hope is that the shared random projection induces approximately orthogonal expert subspaces.

The repository frames this as a Johnson-Lindenstrauss style approximation to orthogonality rather than exact manifold optimization.

### 5.2 Claimed Advantages Over Synapta

Compared to prompt-level bounded mixing, the LoRI-MoE design aims to provide:

- structural interference reduction,
- routing at finer granularity,
- lower conceptual dependence on clamping hyperparameters,
- better fit for multi-domain composition on a single consumer GPU.

### 5.3 Implemented Components

The code base contains substantial LoRI-MoE implementation work:

| Component | Status |
|---|---|
| Shared random projection | Implemented |
| Sparse domain expert representation | Implemented |
| Token-router module classes | Implemented |
| Full LoRI-MoE model wrapper | Implemented |
| Domain adapter training pipeline | Implemented |
| Router training pipeline | Implemented |
| Autonomous multi-model pipeline | Implemented |
| Orthogonality check script | Implemented |
| End-to-end benchmark harness | Partial |
| Full token-level routed inference benchmark | Not convincingly completed |

Key files:

- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/shared_projection.py`
- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/lori_adapter.py`
- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/model/lori_moe_linear.py`
- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/model/lori_moe_model.py`
- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/model/router.py`
- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/training/train_lori_adapter.py`
- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/training/train_router.py`
- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/eval/run_benchmarks.py`
- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/inference/compose.py`

## 6. Executed LoRI-MoE Evidence

### 6.1 Completed Training Artifacts

The repository contains completed checkpoints for five Qwen2.5-1.5B domain experts:

- math
- code
- science
- legal
- medical

These are recorded as complete in:

- `/Users/uditjain/Desktop/mewtwo/checkpoints/lori_moe/pipeline_state.json`

The total recorded adapter training time across the five domains is approximately:

\[
115.52 \text{ minutes}
\]

Per-domain training summaries:

| Domain | Steps | Best Loss | Time |
|---|---:|---:|---:|
| math | 468 | 0.1287 | 31.98 min |
| code | 468 | 0.4242 | 17.43 min |
| science | 468 | 1.3592 | 22.59 min |
| legal | 468 | 0.0000129 | 17.54 min |
| medical | 468 | 0.1170 | 25.99 min |

### 6.2 Dataset Footprint

The saved LoRI training dataset statistics are:

| Domain | Count | Avg Length |
|---|---:|---:|
| math | 49,999 | 927.1 |
| code | 20,016 | 504.2 |
| science | 11,679 | 742.2 |
| legal | 109 | 810.3 |
| medical | 10,178 | 1112.8 |

This imbalance matters. The legal dataset is dramatically undersized relative to the others and weakens any broad legal-generalization claim.

### 6.3 Router Checkpoint

A trained router checkpoint exists at:

- `/Users/uditjain/Desktop/mewtwo/checkpoints/lori_moe/qwen2.5_1.5b/router/best/router.pt`

Its saved metadata records:

- hidden dimension `1536`
- `5` domains
- saved accuracy `100.0`

### 6.4 Orthogonality Check

I recomputed the cross-domain cosine matrix directly from the saved `dare_sparsified` adapter weights. The result matches the repository’s markdown claim:

| Domain Pair Statistic | Value |
|---|---:|
| Avg absolute cross-domain similarity | 0.00685 |
| Largest observed off-diagonal cosine | 0.0143 |

Cosine matrix:

|  | math | code | science | legal | medical |
|---|---:|---:|---:|---:|---:|
| math | 1.0000 | 0.0120 | 0.0062 | 0.0043 | 0.0143 |
| code | 0.0120 | 1.0000 | 0.0086 | 0.0008 | 0.0047 |
| science | 0.0062 | 0.0086 | 1.0000 | 0.0045 | 0.0035 |
| legal | 0.0043 | 0.0008 | 0.0045 | 1.0000 | 0.0095 |
| medical | 0.0143 | 0.0047 | 0.0035 | 0.0095 | 1.0000 |

This is the strongest executed LoRI-MoE result in the repository.

## 7. Critical Audit of the LoRI-MoE Claims

The repository contains real progress, but several key claims outrun the executed evidence.

### 7.1 Router Training Does Not Match the Mixed-Domain Story

The repository includes a routing-data generator designed to produce mixed-domain transitions:

- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/data/generate_routing_data.py`

That generator explicitly argues that training only on pure-domain prompts would collapse the router into prompt-level heuristics.

However, the actual trainer:

- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/training/train_router.py`

does not load `routing_mixed_train.jsonl`. Instead, it:

- reads pure-domain files like `math_train.jsonl` and `code_train.jsonl`,
- assigns one label per example,
- average-pools the final hidden state across the whole prompt,
- trains a single-label classifier.

So the saved `100%` accuracy is not proof of successful token-level multi-domain routing. It is evidence that the model can classify pooled single-domain prompts in its own training regime.

### 7.2 The Saved Router Accuracy Is Training Accuracy, Not Validation Accuracy

The markdown narrative says the router reached very high validation accuracy. The current training script does not implement a held-out validation split. The “best” checkpoint is chosen by epoch-level training accuracy. This is a significant difference.

### 7.3 The Main Inference Path Is Still Prompt-Level Top-1 Routing

The inference entry point:

- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/inference/compose.py`

does not run the full token-level routed multi-expert system in practice. Instead, it:

1. computes one pooled prompt representation,
2. predicts one domain distribution,
3. chooses the maximum-probability domain,
4. loads a single PEFT adapter,
5. generates with that single adapter.

This is operationally closer to prompt-level domain selection than to full token-level MoE composition.

### 7.4 The Benchmark Harness Is Partial

The evaluation harness:

- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/eval/run_benchmarks.py`

contains real benchmark logic for base and single-adapter evaluation, and it correctly emphasizes exact-match and pass@1. However, it does not presently demonstrate a completed full-benchmark path for the routed LoRI-MoE model itself.

The companion file:

- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/eval/routing_analysis.py`

is only a stub.

### 7.5 Autonomous Qwen3.5-0.8B Run Was Interrupted

The autonomous training log:

- `/Users/uditjain/Desktop/mewtwo/logs/lori_moe/pipeline.log.bak`

shows an attempted `Qwen3.5-0.8B` pipeline. The logged math training run was killed mid-execution. So the strongest completed LoRI artifacts are on `Qwen2.5-1.5B`, not on the later attempted branch.

## 8. Interpretation

The `mewtwo` repository should not be read as a finished “breakthrough” paper. It is better understood as a rigorous transition point between two regimes of thought.

### 8.1 What the Repository Successfully Demonstrates

1. Prompt-level bounded adapter composition on a 1.5B model is mostly safe but only modestly useful.
2. Real multi-domain data matters; stronger evaluation changed the interpretation of the older Synapta claims.
3. LoRI-style factorization with a shared frozen random projection can produce extremely low cross-domain parameter overlap in practice.
4. A five-domain expert training pipeline on CUDA hardware was successfully executed for Qwen2.5-1.5B.

### 8.2 What the Repository Does Not Yet Demonstrate

1. That token-level routed LoRI-MoE beats single-adapter or prompt-routed baselines on exact-match tasks.
2. That the saved router generalizes on held-out multi-domain routing trajectories.
3. That the full routed system works end to end on MATH500, GSM8K, MMLU, BBH, or HumanEval.
4. That LoRI-MoE is already superior to Mistral or other stronger baselines in judged answer quality.

## 9. Main Contributions of `mewtwo`

The repo still contains several meaningful contributions.

### 9.1 Empirical Contribution

It preserves a clear negative-to-mixed result on prompt-level composition rather than burying it. This is valuable because it narrows where future progress must come from.

### 9.2 Systems Contribution

It introduces a practical architecture for training multiple low-rank domain experts with shared structure on a single consumer GPU, and it does so with real checkpoints and logs rather than only conceptual diagrams.

### 9.3 Methodological Contribution

It explicitly pivots away from overreliance on semantic similarity and toward stronger benchmark concepts such as exact-match and pass@1, even if the full execution of that benchmark plan remains unfinished.

### 9.4 Geometric Contribution

It provides a concrete checkpoint-backed demonstration that a shared random projection plus sparse trainable expert factors can yield near-orthogonal expert weight geometry across domains.

## 10. Limitations

This reconstruction highlights several limitations that should be made explicit in any research manuscript built from `mewtwo`.

1. The strongest LoRI system claims are only partially validated.
2. Router evidence is weaker than the repository prose suggests.
3. The legal domain data is too small for strong general claims.
4. The main runnable inference path still behaves like prompt-level top-1 selection.
5. The end-to-end benchmark path for the full routed architecture is incomplete.

## 11. Conclusion

The `mewtwo` repository captures a real and intellectually honest research progression.

The older Synapta line showed that prompt-level multi-adapter composition on a small base model is not enough. Even with careful bounding and later routing refinements, the gains were small and often below preregistered thresholds.

The newer LoRI-MoE line contains a stronger architectural idea and real partial evidence in its favor. Five domain experts were trained, a router checkpoint was produced, and the saved experts display extremely low cross-domain overlap. These results support the thesis that interference can be reduced structurally rather than controlled only with scalar clamping.

But the repository does not yet finish the story. The full token-level routed LoRI-MoE system has not been benchmarked to the standard required for a final claim of reasoning superiority. The most accurate summary is therefore:

> `mewtwo` is not a completed proof of multi-expert reasoning dominance. It is a well-instrumented transition from a modestly positive prompt-level composition paradigm toward a more principled orthogonal-expert architecture whose geometric promise is real, but whose end-to-end reasoning advantage remains to be proven.

## Appendix A: Key Artifact Index

### Older Synapta Line

- `/Users/uditjain/Desktop/mewtwo/README.md`
- `/Users/uditjain/Desktop/mewtwo/results/decision_summary.md`
- `/Users/uditjain/Desktop/mewtwo/results/v2_decision_summary.md`
- `/Users/uditjain/Desktop/mewtwo/results/v2_routing_gap_summary.md`
- `/Users/uditjain/Desktop/mewtwo/results/real_benchmark_results.json`

### LoRI-MoE Line

- `/Users/uditjain/Desktop/mewtwo/research_results.md`
- `/Users/uditjain/Desktop/mewtwo/implementation_plan.md`
- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/model/lori_moe_model.py`
- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/model/lori_moe_linear.py`
- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/model/router.py`
- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/training/train_lori_adapter.py`
- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/training/train_router.py`
- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/inference/compose.py`
- `/Users/uditjain/Desktop/mewtwo/src/lori_moe/eval/run_benchmarks.py`
- `/Users/uditjain/Desktop/mewtwo/checkpoints/lori_moe/pipeline_state.json`
- `/Users/uditjain/Desktop/mewtwo/checkpoints/lori_moe/qwen2.5_1.5b/router/best/router.pt`
- `/Users/uditjain/Desktop/mewtwo/logs/lori_moe/pipeline.log.bak`
- `/Users/uditjain/Desktop/mewtwo/data/lori_moe/dataset_stats.json`

