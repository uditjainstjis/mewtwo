# Research Extraction from Experiment Artifacts

This document consolidates the measurable results, benchmark characteristics, and cross-file observations present in this repository. It is written to support paper drafting from the actual experiment artifacts, not from the more promotional narrative drafts.

## 1. Authoritative Artifacts

### Raw or near-raw result files

| File | What it contains | Scope |
|---|---|---|
| `results_db.jsonl` | Real v1 per-run log | 400 entries = 100 prompts x 4 methods |
| `results/real_benchmark_results.json` | Real v1 generated outputs grouped by domain/method | 20 domains x 4 methods x 5 questions |
| `results/v2_both_raw.jsonl` | Real v2 SD+MD log | 560 entries = 140 prompts x 4 methods |
| `results/v2_md_clamp_ablation.jsonl` | Real v2b clamp-ablation log | 120 entries = 40 MD prompts x 3 methods |
| `results/v2_md_routing_ablation.jsonl` | Real v2c routing-gap log | 120 entries = 40 MD prompts x 3 methods |

### Summary or interpretation files

| File | Role |
|---|---|
| `README.md` | Final repo summary for the main negative-result framing |
| `results/decision_summary.md` | v1 preregistered decision summary |
| `results/real_benchmark_table.md` | v1 domain table + aggregate table |
| `results/v2_decision_summary.md` | v2 SD/MD decision summary |
| `results/v2_clamp_ablation_summary.md` | v2b norm-ratio vs weight-cap summary |
| `results/v2_routing_gap_summary.md` | v2c real-router vs oracle summary |
| `results/v2_setup_log.txt` | Critical implementation notes for v2 |
| `table2_ablation.md` | Earlier clamp ablation on the 100-question synthetic benchmark |
| `results_summary.txt` | Synthetic sweep across mixed/pure/generalization datasets |
| `backend/mistral_vs_synapta.md` | Mistral-7B vs Synapta comparison summary |

## 2. Highest-Confidence Story from the Repo

1. On the **v1 single-domain benchmark**, prompt-level multi-adapter composition does **not** beat the simpler single-adapter baseline on semantic similarity.
   - `AdaptiveClamp` vs `SingleAdapter`: **0.6106 vs 0.6223**, delta **-0.0117**.
2. On the **v2 multi-domain benchmark**, oracle-routed composition is **directionally positive** versus single-adapter, but still misses the preregistered threshold.
   - `AdaptiveClamp-v2` vs `SingleAdapter` on MD: **0.6505 vs 0.6334**, delta **+0.0171**.
   - Threshold was **> +0.03**, so this still **fails** the preregistered compositional-gain test.
3. The **router matters**, but the router is not the whole story.
   - Real router gain over single-adapter on MD: **+0.0054**.
   - Oracle headroom over single-adapter on MD: **+0.0206**.
   - Real router recovers about **26%** of the available oracle headroom.
4. **Perplexity improves consistently** even when semantic similarity does not.
   - This is a repeated pattern across v1, v2 SD, v2 MD, and the earlier clamp ablation.
5. The **v1 single-domain benchmark is heavily templated**, so its negative result is informative but should be framed as a negative result on **single-domain synthetic recall**, not a universal statement about composition.

## 3. Benchmark and Data Observations

### v1 single-domain benchmark (`backend/ablation_benchmark.py`)

- 100 questions across 20 domains, exactly 5 questions per domain.
- After normalizing domain names out of the answers, the 100 reference answers collapse into only **4 normalized answer forms**:
  - **58** instances of the "fundamental theorem of {DOMAIN}" template.
  - **40** instances of the "orthogonal projections in {DOMAIN}" template.
  - **1** `res ipsa loquitur` answer.
  - **1** BFS code answer.
- After normalizing domain names out of the questions, the 100 prompts collapse into **18 normalized question templates**.
- In practice, **98/100** answers are template-derived and domain-swapped rather than independently authored.

### v2 multi-domain benchmark (`data/multidomain_eval_v2.json`)

- 40 questions.
- All 20 domains appear at least once.
- There are **39 unique domain pairs**; only one pair is repeated:
  - `PYTHON_LOGIC x ROBOTICS` appears twice.
- Domain frequency is uneven:
  - `MATHEMATICS`: 10 appearances
  - `PHILOSOPHY`: 7
  - `CLIMATE_SCIENCE`: 6
  - many others: 2 to 5

### Implementation observations from `results/v2_setup_log.txt`

- The real backend path uses a **per-adapter weight cap**, not the theorized per-layer norm-ratio clamp.
- `src/adapters/adaptive_multi_lora_linear.py` is described as a **reference implementation**, not the live inference path.
- The orchestrator returns **top-1 one-hot routing**, not a soft distribution.
- Because of that, the v2 oracle setup is necessary to answer the clean question:
  - "Does K=2 help when both adapters are correct?"

## 4. Exploratory Pilot: `benchmark_results.md`

This is a small 20-question exploratory routing benchmark and should be treated as pilot evidence, not as the main study.

- Average base recall: **0.222**
- Average orchestrated recall: **0.213**
- Average base latency: **3.91 s**
- Average orchestrated latency: **4.55 s**
- Average routing overhead: **0.7675 s**
- Average base tokens: **99.3**
- Average orchestrated tokens: **96.7**
- Routing accuracy from the displayed one-hot routing decisions: **17/20 = 85%**
- Recall improved on **3/20** questions, worsened on **2/20**, unchanged on **15/20**
- Mean recall delta: **-0.009**

Observed misroutes in this pilot:

- `SANSKRIT_LINGUISTICS` routed to `LEGAL_ANALYSIS` on two questions.
- `QUANTUM_CHEMISTRY` routed to `ASTROPHYSICS` on one question.

Interpretation:

- The pilot already showed the later pattern: routing was good but not clean enough to create broad quality gains, and overhead was non-trivial.

## 5. Early Clamp Ablation: `table2_ablation.md`

This looks like an earlier 100-question ablation on the synthetic domain benchmark, before the later v1/v2 framing stabilized.

### Aggregate results

| Config | Avg Semantic Sim | Avg PPL | Avg Latency |
|---|---:|---:|---:|
| Baseline (`c=0.0`) | 0.620 | 64.5 | 3.37 s |
| Synapta-Aggressive (`c=1.0`) | 0.618 | 52.7 | 3.27 s |
| Synapta-Balanced (`c=0.5`) | 0.620 | 58.2 | 3.32 s |

### Domain-level observations

- `Synapta-Balanced` improved semantic similarity in **7/20** domains, worsened it in **12/20**, and tied in **1/20**.
- Largest semantic gains for `Synapta-Balanced` vs baseline:
  - `PHILOSOPHY`: **+0.147**
  - `ARCHAIC_ENGLISH`: **+0.082**
  - `RENAISSANCE_ART`: **+0.062**
  - `LATEX_FORMATTING`: **+0.034**
  - `CLIMATE_SCIENCE`: **+0.015**
- Largest semantic losses:
  - `MARITIME_LAW`: **-0.083**
  - `ANCIENT_HISTORY`: **-0.054**
  - `ASTROPHYSICS`: **-0.035**
  - `QUANTUM_CHEMISTRY`: **-0.035**
  - `MLX_KERNELS`: **-0.030**

### Perplexity observations

- `Synapta-Balanced` improved perplexity in **19/20** domains.
- Only `LEGAL_ANALYSIS` worsened slightly: **72.4 -> 73.6**.
- Largest PPL reductions:
  - `MLX_KERNELS`: **-13.0**
  - `PHILOSOPHY`: **-11.6**
  - `MARITIME_LAW`: **-11.1**
  - `ARCHAIC_ENGLISH`: **-10.5**
  - `ROBOTICS`: **-9.3**

Interpretation:

- This earlier ablation already showed the now-recurring pattern:
  - semantic gains are selective and unstable,
  - but perplexity often improves broadly.

## 6. Synthetic Sweep: `results_summary.txt`

This file contains **31 aggregated records** over mixed, pure, and generalization datasets. It is useful, but it is less paper-safe than the v1/v2 real logs because several referenced datasets are not present in `data/`, and the latency numbers for `unclamped_mix` are suspiciously close to zero.

### Dataset availability check

Referenced datasets missing from `data/`:

- `gen_mmlu.json`
- `gen_wikitext.jsonl`
- `mixed_codephilo.jsonl`
- `mixed_mathlegal.jsonl`

Present in `data/`:

- `mixed_fincode.jsonl`
- `pure_code.json`
- `pure_math.json`

### Best config by dataset

| Dataset | Best config | Best score | Gain vs unclamped |
|---|---|---:|---:|
| `mixed_codephilo.jsonl` | `adaptive_clamp`, `K=2`, `c=0.5` | 0.6277 | +0.0771 |
| `mixed_fincode.jsonl` | `adaptive_clamp`, `K=2`, `c=0.5` | 0.6331 | +0.0578 |
| `mixed_mathlegal.jsonl` | `adaptive_clamp`, `K=2`, `c=0.5` | 0.6233 | +0.0458 |
| `pure_code.json` | `adaptive_clamp`, `K=3`, `c=0.5` | 0.8969 | +0.0659 |
| `pure_math.json` | `adaptive_clamp`, `K=2`, `c=0.5` | 0.8980 | +0.0468 |
| `gen_mmlu.json` | `adaptive_clamp`, `K=3`, `c=0.5` | 0.8970 | +0.0641 |
| `gen_wikitext.jsonl` | `adaptive_clamp`, `K=3`, `c=0.5` | 0.9050 | +0.0836 |

### Cross-dataset averages

- Mixed datasets:
  - `adaptive_clamp`, `K=2`, `c=0.5`: **0.6280**
  - `adaptive_clamp`, `K=3`, `c=0.5`: **0.6234**
  - `unclamped_mix`, `K=2`, `c=1.0`: **0.5678**
- Pure datasets:
  - `adaptive_clamp`, `K=2`, `c=0.5`: **0.8972**
  - `adaptive_clamp`, `K=3`, `c=0.5`: **0.8950**
  - `unclamped_mix`, `K=2`, `c=1.0`: **0.8411**
- Generalization datasets:
  - `adaptive_clamp`, `K=2`, `c=0.5`: **0.8943**
  - `adaptive_clamp`, `K=3`, `c=0.5`: **0.9010**
  - `unclamped_mix`, `K=2`, `c=1.0`: **0.8272**

Interpretation:

- Across this sweep, **`c=0.5` is consistently the best clamp value**.
- `K=3` helps on pure/generalization sets, but **not** on the mixed sets.
- The extremely small `unclamped_mix` latencies in this file look like **instrumentation artifacts or non-real timings**, so use these scores more confidently than these latencies.

## 7. Main v1 Real Benchmark: `results_db.jsonl` and `results/real_benchmark_results.json`

### Aggregate results

| Method | Avg Sim | Sim Std | Avg PPL | Avg Latency | Sim < 0.1 | Sim < 0.2 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.6196 | 0.1448 | 64.46 | 2.803 s | 1 | 2 |
| SingleAdapter | **0.6223** | 0.1261 | 60.89 | 2.695 s | 0 | 1 |
| AdaptiveClamp | 0.6106 | 0.1323 | 58.03 | 2.673 s | 1 | 1 |
| UnclampedMix | 0.5573 | 0.1696 | **51.21** | **2.511 s** | 3 | 7 |

### Pairwise semantic deltas

| Comparison | Mean | Median | Std | Positive | Negative |
|---|---:|---:|---:|---:|---:|
| `AdaptiveClamp - SingleAdapter` | **-0.0117** | -0.0014 | 0.1131 | 45 | 52 |
| `AdaptiveClamp - Baseline` | **-0.0089** | -0.0053 | 0.1450 | 43 | 57 |
| `UnclampedMix - SingleAdapter` | **-0.0649** | -0.0276 | 0.1616 | 31 | 69 |
| `SingleAdapter - Baseline` | **+0.0027** | -0.0009 | 0.1062 | 46 | 50 |

### What actually won in v1

- Prompt winner counts:
  - `Baseline`: **35**
  - `SingleAdapter`: **25**
  - `AdaptiveClamp`: **25**
  - `UnclampedMix`: **15**
- Domain winner counts:
  - `Baseline`: **9**
  - `SingleAdapter`: **6**
  - `AdaptiveClamp`: **5**

Important nuance:

- `results/decision_summary.md` lists only 4 domains where `AdaptiveClamp` won, but the raw/table values show `AdaptiveClamp > SingleAdapter` in **8/20 domains**:
  - `LATEX_FORMATTING`: **+0.0055**
  - `CLIMATE_SCIENCE`: **+0.0109**
  - `PHILOSOPHY`: **+0.0208**
  - `ORGANIC_SYNTHESIS`: **+0.0225**
  - `QUANTUM_CHEMISTRY`: **+0.0226**
  - `MLX_KERNELS`: **+0.0248**
  - `MEDICAL_DIAGNOSIS`: **+0.0295**
  - `MATHEMATICS`: **+0.0443**

The authored summary is therefore **non-exhaustive**, not wrong, but the paper should avoid implying that there were only 4 positive domains.

### Largest v1 domain losses for `AdaptiveClamp` vs `SingleAdapter`

- `MARITIME_LAW`: **-0.1449**
- `ANCIENT_HISTORY`: **-0.0389**
- `SANSKRIT_LINGUISTICS`: **-0.0352**
- `CRYPTOGRAPHY`: **-0.0330**
- `ARCHAIC_ENGLISH`: **-0.0322**

### Largest v1 prompt-level swings for `AdaptiveClamp - SingleAdapter`

Worst cases:

- `MARITIME_LAW_Q4`: **-0.7210** (`0.0541` vs `0.7751`)
- `ANCIENT_HISTORY_Q1`: **-0.2572**
- `RENAISSANCE_ART_Q1`: **-0.2518**
- `PYTHON_LOGIC_Q5`: **-0.1973**
- `MARITIME_LAW_Q5`: **-0.1573**

Best cases:

- `MATHEMATICS_Q1`: **+0.4046** (`0.5749` vs `0.1703`)
- `MLX_KERNELS_Q3`: **+0.3986**
- `RENAISSANCE_ART_Q4`: **+0.1496**
- `MARITIME_LAW_Q2`: **+0.1453**
- `PHILOSOPHY_Q3`: **+0.0938**

### Collapse observations

Prompts with semantic similarity < 0.1:

- `Baseline`: `PHILOSOPHY_Q4` (`-0.0133`)
- `AdaptiveClamp`: `MARITIME_LAW_Q4` (`0.0541`)
- `UnclampedMix`: `MARITIME_LAW_Q4` (`0.0312`), `ANCIENT_HISTORY_Q4` (`0.0549`), `MUSIC_THEORY_Q4` (`0.0564`)
- `SingleAdapter`: none

Interpretation:

- `SingleAdapter` is the safest v1 choice for semantic similarity.
- `UnclampedMix` gets the lowest perplexity and fastest latency, but the worst semantic behavior.
- `AdaptiveClamp` reduces perplexity without producing the hoped-for accuracy gain.

## 8. v2 SD/MD Benchmark: `results/v2_both_raw.jsonl`

### Single-domain split (SD, 100 questions)

| Method | Avg Sim | Sim Std | Avg PPL | Avg Latency |
|---|---:|---:|---:|---:|
| Baseline | **0.6090** | 0.1456 | 64.46 | 3.700 s |
| SingleAdapter | 0.6064 | 0.1305 | 60.89 | **3.571 s** |
| AdaptiveClamp-v2 | 0.6058 | 0.1317 | 57.85 | 3.657 s |
| UnclampedMix-v2 | 0.6041 | 0.1271 | **52.34** | 3.623 s |

Key SD deltas:

- `AdaptiveClamp-v2 - SingleAdapter`: **-0.0006**
  - positive on **29**
  - negative on **29**
  - exact ties on **42**
- `SingleAdapter - Baseline`: **-0.0026**
- `AdaptiveClamp-v2 - Baseline`: **-0.0032**

Interpretation:

- SD results support **non-inferiority**, not improvement.
- On this split, the **baseline is still the highest-similarity method**.

### Multi-domain split (MD, 40 questions)

| Method | Avg Sim | Sim Std | Avg PPL | Avg Latency |
|---|---:|---:|---:|---:|
| Baseline | 0.6473 | 0.1074 | 12.70 | 4.059 s |
| SingleAdapter | 0.6334 | 0.1044 | 12.75 | **4.057 s** |
| AdaptiveClamp-v2 | **0.6505** | 0.0913 | **12.64** | 4.090 s |
| UnclampedMix-v2 | **0.6505** | 0.0913 | **12.64** | 4.100 s |

Key MD deltas:

| Comparison | Mean | Median | Std | Positive | Negative |
|---|---:|---:|---:|---:|---:|
| `AdaptiveClamp-v2 - SingleAdapter` | **+0.0171** | +0.0091 | 0.0586 | 28 | 11 |
| `AdaptiveClamp-v2 - Baseline` | **+0.0031** | -0.0027 | 0.0577 | 17 | 21 |
| `SingleAdapter - Baseline` | **-0.0140** | -0.0088 | 0.0472 | 14 | 22 |

Largest MD gains for `AdaptiveClamp-v2 - SingleAdapter`:

- `md_32` (`MEDICAL_DIAGNOSIS x MATHEMATICS`): **+0.3029**
- `md_19` (`LATEX_FORMATTING x MATHEMATICS`): **+0.1096**
- `md_20` (`LEGAL_ANALYSIS x CRYPTOGRAPHY`): **+0.0824**
- `md_02` (`MARITIME_LAW x BEHAVIORAL_ECONOMICS`): **+0.0671**
- `md_34` (`SANSKRIT_LINGUISTICS x ANCIENT_HISTORY`): **+0.0644**

Largest MD losses:

- `md_09` (`PYTHON_LOGIC x ROBOTICS`): **-0.1252**
- `md_25` (`CLIMATE_SCIENCE x ORGANIC_SYNTHESIS`): **-0.0610**
- `md_38` (`ARCHAIC_ENGLISH x LEGAL_ANALYSIS`): **-0.0293**
- `md_21` (`ASTROPHYSICS x MATHEMATICS`): **-0.0262**
- `md_24` (`ROBOTICS x PYTHON_LOGIC`): **-0.0207**

Interpretation:

- v2 MD is the strongest pro-composition evidence in the repo.
- But the gain is **modest, heterogeneous, and only barely above the no-adapter baseline**.
- This is a better paper story than "composition failed completely":
  - composition helps on true multi-domain questions,
  - but not enough to clear the preregistered bar.

## 9. v2b Clamp Ablation: `results/v2_md_clamp_ablation.jsonl`

### Aggregate results

| Method | Avg Sim | Avg PPL | Avg Latency | Avg K |
|---|---:|---:|---:|---:|
| SingleAdapter | 0.6334 | 12.75 | 4.008 s | 1.00 |
| AC-v2-WeightCap | **0.6505** | 12.64 | **4.055 s** | 2.00 |
| AC-v2-NormRatio | 0.6502 | 12.64 | 4.221 s | 2.00 |

### Direct comparison: `NormRatio - WeightCap`

- Mean semantic delta: **-0.00029**
- Median semantic delta: **0.00000**
- Std: **0.00273**
- Min: **-0.0139**
- Max: **+0.0086**
- Exact same semantic score on **36/40** items
- Within **0.001** on **37/40** items

Interpretation:

- The true norm-ratio clamp does **not** materially change the current MD benchmark.
- It adds latency without measurable quality gain in this run.
- This supports the claim that the clamp mechanism is mostly inactive under the present adapter magnitudes and routing weights.

## 10. v2c Routing Gap: `results/v2_md_routing_ablation.jsonl`

### Aggregate results

| Method | Avg Sim | Avg PPL | Avg Latency | Avg K |
|---|---:|---:|---:|---:|
| SingleAdapter | 0.6296 | 12.78 | 4.178 s | 1.00 |
| AC-v2-Norm-RealRouter | 0.6350 | 12.72 | **4.167 s** | 1.75 |
| AC-v2-Norm-Oracle | **0.6502** | **12.64** | 4.211 s | 2.00 |

### Routing-headroom numbers

- Oracle headroom over single-adapter: **+0.0206**
- Realized gain over single-adapter: **+0.0054**
- Routing gap (`Oracle - RealRouter`): **+0.0152**
- Real router used `K=2` on **30/40** questions = **75%**

### Distribution facts

`Oracle - RealRouter`:

- positive on **20**
- negative on **12**
- nonnegative on **28**

`RealRouter - SingleAdapter`:

- positive on **16**
- negative on **13**
- nonnegative on **27**

Largest `RealRouter - SingleAdapter` gains:

- `md_11` (`PYTHON_LOGIC x CRYPTOGRAPHY`): **+0.1059**
- `md_21` (`ASTROPHYSICS x MATHEMATICS`): **+0.0976**
- `md_35` (`ROBOTICS x CLIMATE_SCIENCE`): **+0.0905**
- `md_12` (`MLX_KERNELS x MATHEMATICS`): **+0.0866**
- `md_34` (`SANSKRIT_LINGUISTICS x ANCIENT_HISTORY`): **+0.0602**

Largest losses:

- `md_24` (`ROBOTICS x PYTHON_LOGIC`): **-0.1378**
- `md_13` (`RENAISSANCE_ART x ANCIENT_HISTORY`): **-0.0899**
- `md_17` (`ROBOTICS x PHILOSOPHY`): **-0.0846**
- `md_15` (`ARCHAIC_ENGLISH x MUSIC_THEORY`): **-0.0437**
- `md_25` (`CLIMATE_SCIENCE x ORGANIC_SYNTHESIS`): **-0.0333**

Interpretation:

- Routing is a bottleneck, but the oracle ceiling is also low.
- Even perfect routing only buys about **+0.0206** over single-adapter on this 40-question MD set.
- That means the paper should avoid blaming the entire failure on the router.

## 11. Mistral-7B Comparison: `backend/mistral_vs_synapta.md`

This comparison is present only as a summary report. I did not find a raw per-question Mistral result file in the repo.

Reported numbers:

| Metric | Mistral-7B | Synapta-Balanced | Claimed improvement |
|---|---:|---:|---:|
| Avg Semantic Similarity | 0.579 | 0.620 | +7.2% relative |
| VRAM usage | ~4400 MB | ~1100 MB | 75% reduction |

Paper-safe caution:

- Use this only if you can either recover the raw comparison log or reproduce it.
- Right now the repo gives the summary, but not the underlying run artifact.

## 12. Cross-File Tensions to Clean Up Before Paper Submission

1. The repo contains both:
   - later preregistered negative/partial-positive summaries, and
   - older stronger-claim drafts (`synapta_iclr_final_draft.md`) that read as if the method already won decisively.
2. `results/decision_summary.md` is directionally right, but the "domains where AdaptiveClamp won" section is **not exhaustive**.
3. `results_summary.txt` references datasets that are **not present** in the repo.
4. The v2 setup log explicitly says the live backend does **not** implement the theorized clamp path, which must be stated honestly in the method section.

## 13. Paper-Defensible Claims

These claims are supported by the artifacts as they currently stand:

- **Single-domain negative result:** on a 100-question synthetic single-domain benchmark, prompt-level multi-adapter composition does not outperform single-adapter routing.
- **Perplexity benefit without answer-quality gain:** adapter mixing often lowers perplexity even when semantic similarity stays flat or drops.
- **Multi-domain partial positive:** on a 40-question multi-domain benchmark with oracle routing, composition improves over single-adapter by **+0.0171**, but misses the preregistered **+0.03** threshold.
- **Routing bottleneck is real but secondary:** the real router recovers only about **26%** of oracle headroom, but oracle headroom itself is small.
- **Clamp necessity is benchmark- and implementation-dependent:** unclamped mixing is clearly worse in v1, but under v2 oracle equal-weight routing the clamped and unclamped variants are effectively identical because the live implementation saturates at the same effective weight.

## 14. Claims That Need Careful Wording or More Evidence

- "World-class multi-adapter composition beats simpler baselines" is **not** supported.
- "The norm-ratio clamp is the mechanism behind the gains" is **not** supported by the current live backend.
- "The router is the main reason composition failed" is too strong; the oracle ceiling is too low for that.
- "The method decisively beats Mistral-7B" needs either raw logs or reruns.

## 15. Recommended Paper Spine from the Existing Evidence

If the paper is meant to be rigorous and submission-grade, the cleanest spine available in this repo is:

1. A **pre-registered negative result on single-domain synthetic recall**.
2. A **follow-up multi-domain evaluation** showing a real but small composition signal under oracle routing.
3. A **routing-gap analysis** showing that router quality matters, but does not fully explain the ceiling.
4. A **clamp-mechanism clarification** showing that the live backend implements weight capping, and that a true norm-ratio variant is nearly indistinguishable on the present benchmark.

That is the strongest honest research story presently recoverable from the files.
