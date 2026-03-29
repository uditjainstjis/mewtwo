# Composition Without Collapse: Pre-Registered Evidence for Safe but Modest Prompt-Level Multi-Adapter LoRA Composition on Apple Silicon

Udit Jain  
Independent Researcher  
hello@uditjain.in

## Abstract

This paper studies whether multiple domain-specific LoRA adapters can be composed at prompt time on a small edge-deployed language model without retraining. The repository evaluates a 4-bit `Qwen2.5-1.5B-Instruct` base model in MLX on Apple Silicon Unified Memory Architecture (UMA), together with 20 rank-8 domain adapters that are dynamically loaded into the live inference path. The central question is simple: when a query spans multiple domains, can routing to two adapters outperform routing to the single best adapter?

Across the core study and follow-up ablations, the repo contains 1,200 real MLX inferences. Phase 1 evaluates 100 single-domain questions with four methods. On this benchmark, the proposed multi-adapter method (`AdaptiveClamp`) does not beat the single-adapter baseline on semantic similarity: `0.6106` vs `0.6223`, a paired delta of `-0.0117` with a 95% confidence interval of `[-0.034, 0.011]`. After inspecting the benchmark itself, however, the negative result becomes easier to interpret: the questions collapse to 18 normalized prompt templates, the reference answers collapse to 4 normalized answer templates, and 98 of 100 items are effectively domain-swapped template recall problems rather than compositional reasoning tasks.

Phase 2 corrects this by introducing 40 genuinely multi-domain questions covering all 20 domains and 39 unique unordered domain pairs. Under oracle routing, multi-adapter composition becomes directionally positive: `0.6505` vs `0.6334` for `AdaptiveClamp-v2` over `SingleAdapter`, a paired delta of `+0.0171` with 95% confidence interval `[-0.0019, 0.0361]`. This sign reversal is the central finding of the repository. Yet the effect still misses the pre-registered practical threshold of `+0.03`, and the absolute headroom above the no-adapter baseline remains small.

Follow-up ablations clarify why. A true per-layer norm-ratio clamp is effectively identical to the simpler runtime weight cap (`delta = -0.0003` on the multi-domain split), while a real top-2 router recovers only about 26% of oracle headroom (`+0.0054` realized gain vs `+0.0206` oracle headroom). The most defensible conclusion is therefore narrow but useful: prompt-level multi-adapter composition is not fundamentally broken, but at 1.5B scale it appears safe, occasionally helpful, and still materially limited by adapter geometry and base-model capacity rather than by clamping mechanics alone.

## 1. Introduction

LoRA adapters make domain specialization cheap. They also create an obvious systems question: if one adapter can specialize a base model for law, medicine, or mathematics, can two adapters be composed at inference time when a prompt genuinely spans two such domains? This is especially attractive on Apple Silicon, where Unified Memory Architecture allows multiple adapter tensors to coexist without the explicit host-to-device transfers that complicate traditional GPU serving.

The repository investigates this question under a stricter constraint than most mixture-of-experts work: prompt-level routing, not token-level gating. The choice is practical. Token-level mixture-of-LoRA methods are more expressive, but they also demand more routing machinery and more frequent expert recomputation. Prompt-level routing is cruder, but it is far easier to deploy on a small on-device system.

The repo's experimental path matters as much as its headline metrics. Early drafts framed the project as a clean negative result on synthetic single-domain questions. Later experiments added genuinely multi-domain data, clamp ablations, and routing-gap analysis. This manuscript supersedes the earlier narrative by grounding every major claim in the authoritative artifacts present in the repository:

- `results_db.jsonl`
- `results/real_benchmark_results.json`
- `results/v2_both_raw.jsonl`
- `results/v2_md_clamp_ablation.jsonl`
- `results/v2_md_routing_ablation.jsonl`
- `results/v2_decision_summary.md`
- `results/v2_clamp_ablation_summary.md`
- `results/v2_routing_gap_summary.md`
- `results/v2_setup_log.txt`

The result is a more useful story than either a simple "success" or "failure" framing:

1. On a benchmark that does not truly require composition, multi-adapter routing underperforms the simpler single-adapter baseline.
2. On a benchmark that does require composition, the sign flips: two correctly chosen adapters help on average.
3. The improvement is real but modest, below the pre-registered threshold for a practically meaningful win.
4. Better routing helps, but only up to a point. The limiting factor appears to be representational compatibility among adapters inside a small 1.5B base model.

## 2. What Is Actually Implemented

The repo contains multiple conceptual descriptions of the system. The live inference path is narrower and more concrete than some of the older drafts imply. This section describes the implementation that actually produced the reported numbers.

### 2.1 Base model and hardware

- Base model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- Runtime: MLX / `mlx_lm`
- Hardware target: Apple Silicon, reported main runs on M3 Max UMA
- Core paper artifacts: 1,200 real inferences across v1, v2, v2b, and v2c

### 2.2 Expert adapters

- 20 domain-specific adapters listed in `backend/expert_registry.json`
- Each adapter is about `20.15 MB`
- Adapter configs in `backend/expert_adapters/*/adapter_config.json` report rank `8`
- The live runtime loads all adapters and updates routing weights per query

The domains are:

`LEGAL_ANALYSIS`, `MEDICAL_DIAGNOSIS`, `PYTHON_LOGIC`, `MATHEMATICS`, `MLX_KERNELS`, `LATEX_FORMATTING`, `SANSKRIT_LINGUISTICS`, `ARCHAIC_ENGLISH`, `QUANTUM_CHEMISTRY`, `ORGANIC_SYNTHESIS`, `ASTROPHYSICS`, `MARITIME_LAW`, `RENAISSANCE_ART`, `CRYPTOGRAPHY`, `ANCIENT_HISTORY`, `MUSIC_THEORY`, `ROBOTICS`, `CLIMATE_SCIENCE`, `PHILOSOPHY`, and `BEHAVIORAL_ECONOMICS`.

### 2.3 Injection path

The live runtime in `backend/dynamic_mlx_inference.py` does **not** inject adapters into every linear layer. It wraps only transformer submodules whose names contain `q_proj` or `v_proj`. This matters because it narrows the interpretation of "multi-adapter composition" in this repo: the method is not a full-model additive LoRA composition scheme, but a routed composition applied to a subset of attention projections.

### 2.4 Routing

The default `Orchestrator` is a chain-of-thought heuristic router implemented in `backend/orchestrator.py`. In its primary path it returns a one-hot top-1 routing vector, not a soft distribution. That implementation detail forced an important design change in Phase 2: the multi-domain evaluation uses oracle routing for the K=2 methods so that the experiment measures compositional value rather than top-2 discovery quality.

### 2.5 Clamping

The repo contains two clamp formulations:

1. **Live v1/v2 runtime weight cap**

$$
\hat{m} = \sum_i \min(w_i, c) \cdot s_i(x)
$$

where `w_i` is the routing weight and `s_i(x)` is the adapter contribution.

2. **Follow-up v2b norm-ratio clamp**

$$
m_l = \sum_i w_i \cdot s_{i,l}(x), \qquad
\gamma_l = \min\left(1,\; c \cdot \frac{\|z_l\|_2}{\|m_l\|_2 + \epsilon}\right), \qquad
\hat{z}_l = z_l + \gamma_l m_l
$$

The reference implementation in `src/adapters/adaptive_multi_lora_linear.py` matches the second formulation, but `results/v2_setup_log.txt` is explicit that this was **not** the live path for the original v1/v2 runs. The paper therefore treats the norm-ratio clamp as a follow-up ablation, not as the mechanism used throughout.

## 3. Experimental Design

### 3.1 Metrics

The repo reports three primary metrics:

- Semantic similarity: cosine similarity between generated text and the reference answer using `sentence-transformers/all-MiniLM-L6-v2`
- Perplexity: log-likelihood of the reference answer under the current routing configuration
- Latency: wall-clock generation time

The semantic similarity metric is useful for within-repo comparisons, but it is still an embedding-based proxy. This becomes particularly important on templated synthetic benchmarks, where lexical drift can dominate the score.

### 3.2 Phase 1: single-domain benchmark

Phase 1 uses 100 synthetic "hard questions" from `backend/ablation_benchmark.py`: 20 domains with 5 questions each, evaluated under four methods:

| Method | K | Clamp | Routing |
|---|---:|---:|---|
| Baseline | 0 or near-zero adapter effect | 0.001 | None |
| SingleAdapter | 1 | 0.5 | CoT top-1 |
| UnclampedMix | 2 | 999 | CoT top-2 |
| AdaptiveClamp | 2 | 0.5 | CoT top-2 |

This phase contributes 400 real inferences.

### 3.3 Phase 2: mixed single-domain and multi-domain evaluation

Phase 2 evaluates:

- the same 100 single-domain questions as a sanity split
- 40 new multi-domain questions from `data/multidomain_eval_v2.json`

The multi-domain benchmark has several useful properties:

- 40 questions
- all 20 domains appear at least once
- 39 unique unordered domain pairs
- one pair is repeated: `PYTHON_LOGIC x ROBOTICS`

Phase 2 contributes 560 real inferences:

| Method | K | Clamp | Routing |
|---|---:|---:|---|
| Baseline | 0 | 0.001 | None |
| SingleAdapter | 1 | 0.5 | CoT top-1 |
| AdaptiveClamp-v2 | 2 | 0.5 | Oracle |
| UnclampedMix-v2 | 2 | 999 | Oracle |

### 3.4 Follow-up ablations

Two additional follow-up studies operate on the 40-question multi-domain split:

- **v2b clamp ablation**: weight-cap vs norm-ratio clamp
- **v2c routing-gap ablation**: oracle K=2 vs real top-2 router

Each contributes 120 real inferences, bringing the core artifact total to 1,200.

### 3.5 Pre-registered thresholds

| Phase | Hypothesis | Threshold |
|---|---|---:|
| v1 | `Delta_SIM(AdaptiveClamp - SingleAdapter)` | `> +0.05` |
| v1 | `PPL(AdaptiveClamp) <= PPL(SingleAdapter)` | pass/fail |
| v1 | latency overhead | `<= 10%` |
| v2 SD | non-inferiority vs `SingleAdapter` | `>= -0.005` |
| v2 MD | `Delta_SIM(AdaptiveClamp-v2 - SingleAdapter)` | `> +0.03` |
| v2 | perplexity preservation | pass/fail |
| v2 | latency overhead | `<= 15%` |
| v2 MD | clamped better than unclamped | `> 0` |

## 4. A Benchmark Diagnosis Was Necessary

The strongest improvement in the repo is not numerical but methodological: the project eventually learned that its original benchmark was badly mismatched to its research question.

After normalizing domain names, the 100-question single-domain benchmark collapses to:

- 18 normalized question templates
- 4 normalized answer templates
- 58 instances of one "fundamental theorem of {domain}" answer pattern
- 40 instances of one "orthogonal projections in {domain}" answer pattern
- 2 genuinely non-template answers (`res ipsa loquitur` and a BFS snippet)

This means the Phase 1 benchmark is primarily a test of synthetic domain-token recall, not a test of compositional reasoning. That does not invalidate the results, but it sharply narrows what they mean: if the correct answer depends on one domain and a second adapter is almost always redundant, then failure of K=2 composition is the expected null.

This diagnosis is what makes the Phase 2 sign reversal interpretable rather than contradictory.

## 5. Results

### 5.1 Phase 1: multi-adapter composition loses on single-domain recall

| Method | Avg Sim | Avg PPL | Avg Latency (s) |
|---|---:|---:|---:|
| Baseline | 0.6196 | 64.46 | 2.803 |
| SingleAdapter | **0.6223** | 60.89 | 2.695 |
| AdaptiveClamp | 0.6106 | 58.03 | 2.673 |
| UnclampedMix | 0.5573 | **51.21** | **2.511** |

The pre-registered primary comparison fails:

- `Delta_SIM(AdaptiveClamp - SingleAdapter) = -0.0117`
- 95% CI: `[-0.034, 0.011]`
- Prompt-level wins/losses/ties: `45 / 52 / 3`

Two patterns matter.

First, the simpler single-adapter baseline is best on average. Adaptive composition does win in some domains, especially `MATHEMATICS` (`+0.0443`) and `MEDICAL_DIAGNOSIS` (`+0.0295`), but it loses badly in others, most notably `MARITIME_LAW` (`-0.1449`).

Second, unclamped mixing is clearly unsafe even when its average perplexity looks superficially good. On the 100 v1 prompts, `UnclampedMix` yields semantic similarity below `0.1` on 3 prompts and below `0.2` on 7 prompts, with a minimum of `0.0312`. The model can still assign high local likelihood to text while producing outputs that are semantically far from the target.

The v1 result therefore supports three claims:

1. Naive additive mixing is dangerous.
2. Bounded mixing is much safer.
3. Safe mixing still does not beat single-adapter routing when the benchmark does not actually require composition.

### 5.2 Phase 2 sanity split: single-domain performance is preserved

On the 100 single-domain questions re-evaluated inside the v2 harness:

| Method | Avg Sim | Avg PPL | Avg Latency (s) |
|---|---:|---:|---:|
| Baseline | 0.6090 | 64.5 | 3.700 |
| SingleAdapter | **0.6064** | 60.9 | **3.571** |
| AdaptiveClamp-v2 | 0.6058 | 57.9 | 3.657 |
| UnclampedMix-v2 | 0.6041 | **52.3** | 3.623 |

The non-inferiority hypothesis passes:

- `Delta_SIM(AdaptiveClamp-v2 - SingleAdapter) = -0.0006`
- 95% CI: `[-0.0076, 0.0064]`
- Prompt-level wins/losses/ties: `29 / 29 / 42`

This is exactly what the repo should hope to see once routing is constrained appropriately: composition no longer imposes a meaningful penalty on single-domain questions.

### 5.3 Phase 2 main result: composition helps on genuine multi-domain prompts, but modestly

| Method | Avg Sim | Avg PPL | Avg Latency (s) |
|---|---:|---:|---:|
| Baseline | 0.6473 | 12.7 | 4.059 |
| SingleAdapter | 0.6334 | 12.7 | **4.057** |
| AdaptiveClamp-v2 | **0.6505** | **12.6** | 4.090 |
| UnclampedMix-v2 | **0.6505** | **12.6** | 4.100 |

This is the repository's most important empirical result:

- `Delta_SIM(AdaptiveClamp-v2 - SingleAdapter) = +0.0171`
- 95% CI: `[-0.0019, 0.0361]`
- Prompt-level wins/losses/ties: `28 / 11 / 1`

The sign has flipped relative to Phase 1. Once the benchmark actually requires two domains and the K=2 method is routed to the correct pair, composition is beneficial on average. But the win is still modest:

- it misses the pre-registered `+0.03` threshold
- it is only `+0.0032` above the no-adapter baseline
- unclamped and clamped weight-cap variants are identical under equal-weight oracle routing

The last point foreshadows the clamp ablation.

### 5.4 Gains are highly pair-dependent

The multi-domain average hides large heterogeneity:

| Item | Domain pair | Delta vs SingleAdapter |
|---|---|---:|
| `md_32` | `MEDICAL_DIAGNOSIS x MATHEMATICS` | `+0.3029` |
| `md_19` | `LATEX_FORMATTING x MATHEMATICS` | `+0.1096` |
| `md_20` | `LEGAL_ANALYSIS x CRYPTOGRAPHY` | `+0.0824` |
| `md_02` | `MARITIME_LAW x BEHAVIORAL_ECONOMICS` | `+0.0671` |
| `md_34` | `SANSKRIT_LINGUISTICS x ANCIENT_HISTORY` | `+0.0644` |
| `md_09` | `PYTHON_LOGIC x ROBOTICS` | `-0.1252` |
| `md_25` | `CLIMATE_SCIENCE x ORGANIC_SYNTHESIS` | `-0.0610` |
| `md_38` | `ARCHAIC_ENGLISH x LEGAL_ANALYSIS` | `-0.0293` |
| `md_21` | `ASTROPHYSICS x MATHEMATICS` | `-0.0262` |
| `md_24` | `ROBOTICS x PYTHON_LOGIC` | `-0.0207` |

This pattern is consistent with an adapter-geometry hypothesis: some domain pairs occupy sufficiently different subspaces to compose productively, while others appear to interfere because they demand overlapping representational machinery. The repeated failure on `PYTHON_LOGIC x ROBOTICS` is the clearest example.

### 5.5 Clamp ablation: the mathematically cleaner clamp does not move the needle

The v2b follow-up isolates clamp formulation on the multi-domain split:

| Method | Clamp mode | Avg Sim | Avg PPL | Avg Latency (s) |
|---|---|---:|---:|---:|
| SingleAdapter | `weight_cap` | 0.6334 | 12.7 | 4.008 |
| AC-v2-WeightCap | `weight_cap` | 0.6505 | 12.6 | 4.055 |
| AC-v2-NormRatio | `norm_ratio` | 0.6502 | 12.6 | 4.221 |

Key deltas:

- `Delta_SIM(WeightCap - SingleAdapter) = +0.0171`
- `Delta_SIM(NormRatio - SingleAdapter) = +0.0168`
- `Delta_SIM(NormRatio - WeightCap) = -0.0003`

This is an important negative result inside the larger mixed story. The theoretically cleaner per-layer norm-ratio clamp is almost exactly equivalent to the simpler runtime weight cap on this model family and dataset. The repo's own explanation is plausible: the raw adapter contribution is already small enough relative to the base activation that `gamma_l` is near `1.0` almost everywhere.

### 5.6 Routing-gap ablation: better routing helps, but the oracle ceiling is already low

The v2c follow-up evaluates the best-case value of routing improvements:

| Method | Routing | Avg Sim | Avg PPL | Avg Latency (s) | Avg K |
|---|---|---:|---:|---:|---:|
| SingleAdapter | CoT top-1 | 0.6296 | 12.8 | 4.178 | 1.00 |
| AC-v2-Norm-RealRouter | CoT top-2 | 0.6350 | 12.7 | **4.167** | 1.75 |
| AC-v2-Norm-Oracle | Oracle top-2 | **0.6502** | **12.6** | 4.211 | 2.00 |

From this:

- oracle headroom over `SingleAdapter`: `+0.0206`
- realized gain with a real top-2 router: `+0.0054`
- fraction of oracle headroom recovered: about `26%`

This is the repo's clearest bottleneck ranking:

1. Composition capacity is limited even under oracle routing.
2. Routing quality matters, but improving it alone cannot create a large effect that is not already there.
3. Clamp formulation is the smallest factor among the three.

## 6. Interpreting the Whole Repository

The strongest defensible interpretation is not "composition failed" and not "composition worked." It is the following:

### 6.1 The negative result in Phase 1 is real, but narrow

Phase 1 is a genuine negative result on single-domain synthetic recall. On that task, the extra adapter mostly acts as noise, and the repo is correct to report that outcome honestly.

### 6.2 The sign reversal in Phase 2 is also real

Once the dataset requires two domains and the adapters are correctly chosen, the mean effect changes sign from `-0.0117` to `+0.0171`. That is too large a reversal to dismiss as wording or cherry-picking. The repository learned something important: the original "composition does not help" conclusion was partly an evaluation-design conclusion.

### 6.3 But the ceiling is low at 1.5B

The oracle K=2 condition only reaches `0.6505` on the multi-domain split, and even that is barely above the base-model baseline. If perfect routing only yields about two similarity points over the single-adapter baseline, then the main limitation is unlikely to be router engineering alone. The repo's hypothesis that adapter orthogonality and base-model capacity are the deeper constraints is reasonable.

### 6.4 Perplexity repeatedly improves without corresponding semantic gains

This pattern recurs throughout the repository: composition often lowers perplexity even when it does not improve semantic similarity. The simplest interpretation is that multi-adapter injection can make the model locally more confident in the style or token distribution of the reference answers without reliably improving answer selection. On templated synthetic data, that discrepancy becomes even easier to produce.

## 7. Limitations

Several limitations should remain in the main paper rather than being buried in footnotes.

### 7.1 Synthetic training and small evaluation sets

The single-domain benchmark is heavily templated, and the multi-domain benchmark contains only 40 questions. The v2 benchmark is much better matched to the research question, but it is still small.

### 7.2 Metric sensitivity

The main quality metric is embedding cosine similarity to a single reference answer. That is a pragmatic repo choice, but it is not equivalent to factual correctness or usefulness.

### 7.3 Partial-model injection

The live composition path only modifies `q_proj` and `v_proj`. This is an important engineering detail, and it narrows the scope of any general claim about "multi-adapter composition."

### 7.4 Router mismatch across experiments

The repo contains several routing experiments: one-hot CoT routing, a real top-2 extension, embedding and classifier routers, and a gated router. These are informative, but they are not all directly comparable, and only some are tied tightly to the pre-registered v1/v2 framing.

### 7.5 Exploratory artifacts should stay secondary

Files such as `FINAL_EXPERIMENT_REPORT.md`, `results/gated_routing_embedding_results.json`, and `results/mistral_vs_synapta_verified.md` contain interesting later-stage claims, including autonomous routing and Mistral comparisons. Those artifacts are useful engineering evidence, but they should not be the backbone of the paper because they sit outside the cleanest preregistered comparisons and combine different routing regimes with different comparators. A rigorous paper should mention them, at most, as exploratory follow-up rather than as the primary proof.

## 8. Practical Takeaways

For practitioners and researchers working from this repository, the paper supports the following operational guidance:

1. Do not use naive unclamped multi-adapter mixing in production.
2. Expect prompt-level K=2 composition to underperform on single-domain queries unless routing can suppress the second adapter.
3. If the query truly spans two domains, there is measurable headroom for composition, but on this 1.5B setup it is small.
4. If you improve this system, prioritize adapter compatibility and higher-capacity bases before spending large effort on more elaborate clamp formulas.
5. Treat routing as a meaningful but secondary bottleneck: important, but not the main source of the low oracle ceiling.

## 9. Conclusion

This repository does not show that prompt-level multi-adapter LoRA composition is a dead end. It shows something more specific and more useful.

On a synthetic single-domain benchmark, K=2 composition fails to beat a single routed adapter. That negative result is real. But once the evaluation is redesigned around genuinely multi-domain questions, the effect reverses sign: correctly routed composition is beneficial on average. The catch is that the benefit is modest and remains below the repo's pre-registered threshold for a practically meaningful win.

The most defensible conclusion is therefore mixed:

- composition can be made safe
- composition can help when the task truly requires it
- composition is not yet strong enough, at 1.5B scale, to justify broad claims of decisive advantage over single-expert routing

That is a worthwhile result. It narrows the problem, identifies better next steps, and turns an initially confusing collection of drafts into a coherent empirical story.

## Appendix A: Reproducibility Pointers

Core scripts used by the repo:

```bash
# v1 single-domain benchmark
python3 src/eval/real_benchmark.py

# v2 mixed SD + MD benchmark
python3 src/eval/run_eval_v2.py --real --split both

# v2b clamp ablation
PYTHONUNBUFFERED=1 python3 src/eval/run_eval_v2b.py --phase clamp --real

# v2c routing-gap ablation
PYTHONUNBUFFERED=1 python3 src/eval/run_eval_v2b.py --phase routing --real
```

Primary raw artifacts:

- `results_db.jsonl`
- `results/real_benchmark_results.json`
- `results/v2_both_raw.jsonl`
- `results/v2_md_clamp_ablation.jsonl`
- `results/v2_md_routing_ablation.jsonl`

## References

Apple Inc. MLX: An array framework for Apple Silicon.  
Buehler, M. J. (2024). X-LoRA: Mixture of low-rank adapter experts.  
Chen, L. et al. (2023). Punica: Multi-tenant LoRA serving. arXiv:2310.18547.  
Henderson, P. et al. (2018). Deep reinforcement learning that matters. AAAI.  
Hu, E. J. et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR.  
Sheng, Y. et al. (2023). S-LoRA: Serving thousands of concurrent LoRA adapters. arXiv:2311.03285.  
Turner, A. et al. (2023). Activation addition: Steering language models without optimization. arXiv:2308.10248.  
Zou, A. et al. (2023). Representation engineering: A top-down approach to AI transparency. arXiv:2310.01405.  
