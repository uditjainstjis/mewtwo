# Synapta Final Experiment Report (April 2026)

**Author:** Udit Jain  
**Repository:** https://github.com/uditjainstjis/mewtwo  
**Hardware:** Apple M3 Max  
**Base routed model:** Qwen2.5-1.5B-Instruct-4bit via MLX  
**Comparison model:** Mistral-7B via Ollama  

---

## 1. What This Report Is

This is the corrected final experiment ledger for Synapta.

It combines:

- the early single-domain and multi-domain adapter-composition work
- the 9-technique injection ablation over Qwen
- the later external 100-item benchmark
- the final blind pairwise judging against Mistral

This report is meant to replace any simplistic reading of the older internal-only story.

---

## 2. Executive Bottom Line

### What held up

- Multi-adapter composition is not pure noise. It can help on genuinely multi-domain tasks.
- Apple Silicon routing plus LoRA composition gives a strong latency and footprint advantage.
- Evaluation methodology matters a lot. Soft metrics and blind judging can change the ranking of methods.

### What did not hold up

- We do **not** currently have evidence that Synapta beats Mistral on externally judged answer quality.
- The earlier internal `+5.7%` “Mistral-killer” style claim does **not** survive the new external benchmark and blind judging.
- The flashy routing method `sequential_reverse` looked best on soft metrics, but it was not the best Qwen method under blind judging.

### Final claim boundary

- **Research-safe:** Synapta shows a real speed-quality tradeoff and a useful negative result about metric sensitivity in multi-adapter evaluation.
- **Startup-safe:** Synapta has an efficiency and controllability edge, not a proved final-answer quality edge.

---

## 3. Complete Experimental Timeline

| Phase | Purpose | Main Question | Result |
| --- | --- | --- | --- |
| v1 | Single-domain composition | Does adding a second adapter help on 1-domain queries? | No. It hurts. |
| v2 | Oracle multi-domain composition | Does K=2 help on true 2-domain queries? | Yes, slightly. |
| v2b | Clamp ablation | Does norm-ratio clamp help more than weight-cap? | No meaningful difference. |
| v2c | Routing gap | How much oracle gain survives real routing? | Only part of it. |
| Router ablation | Autonomous routing | Which router works best? | Embedding/classifier beat generative CoT. |
| Internal injection ablation | 9 Qwen methods | Which layer/time routing strategy looks strongest internally? | `early_third_only` or `sequential_reverse` on soft metrics. |
| External benchmark | Realer comparison | Do those wins hold on new externally authored MD data? | No clean quality win. |
| Blind judging | Correctness-first comparison | Does Mistral still win when answers are judged blindly? | Yes, clearly. |

---

## 4. Early Core Findings (Before External Re-Evaluation)

### 4.1 v1 Single-Domain Benchmark

| Method | Avg Sim | Avg PPL | Avg Latency |
| --- | ---: | ---: | ---: |
| Baseline | 0.620 | 64.5 | 2.80s |
| SingleAdapter | 0.622 | 60.9 | 2.69s |
| AdaptiveClamp | 0.611 | 58.0 | 2.67s |
| UnclampedMix | 0.557 | 51.2 | 2.51s |

Inference:

- On single-domain questions, a second adapter mostly adds noise.
- This was a useful negative result, but not a verdict on multi-domain composition itself.

### 4.2 v2 Oracle Multi-Domain Benchmark

| Method | Avg Sim | Avg PPL | Avg Latency | Delta vs SingleAdapter |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.6473 | 12.7 | 4.059s | +0.0139 |
| SingleAdapter | 0.6334 | 12.7 | 4.057s | — |
| AdaptiveClamp-v2 | 0.6505 | 12.6 | 4.090s | +0.0171 |
| UnclampedMix-v2 | 0.6505 | 12.6 | 4.100s | +0.0171 |

Inference:

- Multi-domain composition can help when the question really needs two domains.
- The effect size was real but modest.

### 4.3 Clamp and Routing Findings

| Experiment | Key Number | Interpretation |
| --- | ---: | --- |
| Norm-ratio vs weight-cap clamp | `-0.0003` similarity delta | No meaningful difference |
| Real router vs oracle | `+0.0054` vs `+0.0206` | Real routing recovers only part of oracle headroom |
| Real router headroom recovered | ~26% | Routing is a bottleneck, but not the only one |

---

## 5. The 9 Qwen Injection Techniques

All 9 techniques can be written as a change in the adapter routing coefficient:

\[
h^{(l)} = W^{(l)}x^{(l)} + \sum_d \gamma_{l,t,d}\,\Delta W_d^{(l)}x^{(l)}
\]

The experiments differ only in how `gamma(l,t,d)` changes across:

- layer depth `l`
- token position `t`
- domain `d`

### 5.1 What Each Method Did

| Method | Mathematical/operational idea |
| --- | --- |
| `weighted_merge` | Constant 50/50 two-adapter merge across all layers and tokens |
| `late_layer_injection` | Merge only in the upper half of layers |
| `late_last_quarter` | Merge only in the final quarter of layers |
| `early_third_only` | Merge only in the first third of layers |
| `sequential_token_segments` | Use domain 1 for early generated tokens, then domain 2 |
| `sequential_reverse` | Reverse the token-time order of the two domains |
| `oracle_single_d1` | Only one expert active everywhere |
| `oracle_single_d2` | Only the other expert active everywhere |
| `merge_high_clamp` | Same merge as baseline, but with higher cap / stronger adapter influence |

### 5.2 Internal 40-Item MD Track A Results

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

### 5.3 What We Thought Internally

- Depth-aware and time-aware routing looked better than naive merging.
- `sequential_reverse` looked like the strongest stable non-oracle method.
- Raising adapter strength alone did not help.

That internal story was directionally interesting, but it was not robust enough to survive external re-evaluation.

---

## 6. External 100-Item MD Benchmark

### 6.1 Why This Benchmark Matters

The older internal benchmark had clear risk:

- templated or closely coupled data
- similarity-heavy evaluation
- possibility of model/data leakage bias

So we generated a new 100-item external-style multi-domain dataset organized by workflow sections and tested again.

### 6.2 100-Item External Soft Metrics

| System | Semantic Sim | Token F1 | Latency | Rubric Coverage |
| --- | ---: | ---: | ---: | ---: |
| weighted_merge | 0.6592 | 0.2719 | 4.263s | 0.1261 |
| late_layer_injection | 0.6594 | 0.2715 | 3.890s | 0.1230 |
| sequential_reverse | 0.6623 | 0.2734 | 4.605s | 0.1338 |
| mistral | 0.6907 | 0.2917 | 10.654s | 0.1683 |

### 6.3 Immediate External Read

- `sequential_reverse` still looked best among Qwen methods on soft metrics.
- `late_layer_injection` was the fastest Qwen.
- `mistral` led quality metrics while being about `2.3x` to `2.7x` slower.

This was already weaker than the old internal “Mistral-killer” story.

---

## 7. Blind Pairwise Judging vs Mistral

We then ran blind pairwise judging on a 30-item stratified subset, using a strict external judge and comparing full answers rather than only embedding similarity.

### 7.1 Blind Results

| Qwen Method | Qwen Wins | Mistral Wins | Ties | Avg Qwen Score | Avg Mistral Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| weighted_merge | 6 | 23 | 1 | 3.767 | 5.300 |
| late_layer_injection | 4 | 26 | 0 | 3.500 | 5.333 |
| sequential_reverse | 4 | 25 | 1 | 3.533 | 5.300 |

### 7.2 Most Important Reversal

The blind judge changed the Qwen method ranking:

| Method | Looked best on soft metrics? | Best under blind judge? |
| --- | --- | --- |
| sequential_reverse | Yes | No |
| weighted_merge | No | Yes, least-bad Qwen |

This is probably the strongest research contribution of the final phase:

- soft metrics overstated progress
- externally authored data changed the story
- blind correctness-focused judging changed the method ranking

---

## 8. Final Novelty Assessment

### Real novelty we can honestly claim

| Area | Novelty level | Why |
| --- | --- | --- |
| Multi-adapter Apple Silicon serving | Real | Clean systems implementation with many experts in memory |
| Depth/time adapter scheduling ablation | Real | Shows routing structure matters internally |
| Evaluation methodology result | Strong | Blind judging overturned the soft-metric ranking |
| “Small routed model beats Mistral” | Not supported | External blind evidence does not support it |

### The strongest final research contribution

Not “we beat the bigger model.”

Instead:

1. Multi-adapter methods can look promising under internal semantic metrics.
2. Those wins may collapse or reorder under external correctness-focused evaluation.
3. Therefore, routed-LoRA research needs blind judging or stronger task-grounded scoring, not only embedding similarity.

That is a serious and publishable lesson if framed honestly.

---

## 9. Final Startup Assessment

### What a startup can still claim

| Claim | Supported? | Notes |
| --- | --- | --- |
| Faster than Mistral on this MD workload | Yes | ~2.3x to 2.7x faster |
| Much smaller modular expert system | Yes | Architectural and deployment advantage |
| Controllable domain composition | Yes | Real product/control story |
| Better final answer quality than Mistral | No | Blind judging says no |
| Stronger quality-per-latency tradeoff | Partly | Depends on buyer tolerance for quality gap |

### Startup pitch that remains defensible

“Synapta is a modular, fast expert-routing system for edge hardware. It does not yet outperform larger baselines on final answer quality, but it delivers materially lower latency and controllable domain specialization.”

---

## 10. Final Numbers That Matter Most

| Metric | Best Qwen | Mistral | Read |
| --- | ---: | ---: | --- |
| External semantic similarity | 0.6623 | 0.6907 | Mistral better |
| External token F1 | 0.2734 | 0.2917 | Mistral better |
| External rubric coverage | 0.1338 | 0.1683 | Mistral better |
| External latency | 3.890s to 4.605s | 10.654s | Qwen much faster |
| Blind wins on 30-item subset | 6 max | 23 min | Mistral clearly better |

---

## 11. Final Conclusion

Synapta did produce real novelty, but not the naive breakthrough we first hoped for.

The project succeeded in showing:

- multi-adapter composition is real and sometimes useful
- routing strategy matters
- edge deployment with many tiny experts is practical
- benchmark design and metric choice can completely change the scientific conclusion

The project did **not** yet show:

- decisive quality superiority over a larger general model
- robust external wins for fancy routing over simpler merging

So the final honest conclusion is:

**Synapta is a strong systems-and-evaluation contribution with a real efficiency edge, but not yet a model-quality breakthrough.**

---

## 12. Primary Artifacts

- `FULL_PROJECT_SUMMARY.md`
- `FINAL_EXPERIMENT_REPORT.md`
- `results/md_external_v2_blind_report.md`
- `results/md_external_v2_comparison_summary.json`
- `results/md_external_v2_soft_vs_blind_summary.json`
- `results/md_pairwise_merge_vs_mistral_v2_strat30_summary.json`
- `results/md_pairwise_latelayer_vs_mistral_v2_strat30_summary.json`
- `results/md_pairwise_seqrev_vs_mistral_v2_strat30_summary.json`
