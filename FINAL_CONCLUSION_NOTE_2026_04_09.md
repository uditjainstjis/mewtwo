# Final Conclusion Note

**Project:** Synapta  
**Date:** 2026-04-09  
**Hardware:** Apple M3 Max / Apple Silicon UMA  
**Base stack:** Qwen2.5-1.5B-Instruct-4bit via MLX + routed LoRA experts

---

## 1. Core Hypothesis

The original thesis was:

1. Multiple small domain LoRA adapters can be composed at inference time on Apple Silicon because UMA makes multi-adapter residency cheap.
2. This composition could solve cross-domain questions better than choosing a single expert.
3. Because the base system is much smaller than Mistral, it might recover quality while keeping a large speed advantage.

Over time that thesis split into three narrower hypotheses:

1. **Parameter-space composition hypothesis**
   Weighted merging, layer gating, and token scheduling can make multiple adapters cooperate in one forward pass.
2. **Router hypothesis**
   If collaborative reasoning works in principle, then router quality is the limiting factor.
3. **Inference-time scaling hypothesis**
   If weight composition breaks reasoning coherence, then letting experts answer independently and selecting or synthesizing afterward may recover System 2 behavior.

---

## 2. Working Assumptions

These were the main assumptions we tested, corrected, or discarded:

| Assumption | Status | What happened |
| --- | --- | --- |
| Soft semantic similarity is enough to rank methods | Rejected | Blind evaluation changed the ranking materially |
| Internal MD benchmark was enough for paper claims | Rejected | External benchmark was necessary |
| Fancy routing methods would clearly beat simple merge | Mostly rejected | Improvements existed, but did not hold strongly under blind judging |
| Better router training would unlock collaborative reasoning | Partly true | SFT helped a lot; DPO did not |
| Generative refiner is necessary | Partly rejected | It helps quality, but costs too much latency |
| Verifier-only selection can keep quality while killing latency | Rejected in current form | Speed win was real, F1 drop was too large |
| Test-time search over many sampled routes will recover F1 cheaply | Rejected on this hardware | DES exploded latency |

---

## 3. Experimental Arc

### Phase A: Multi-adapter weight composition

We tested 9 Qwen composition strategies plus Mistral:

- `weighted_merge`
- `late_layer_injection`
- `late_last_quarter`
- `early_third_only`
- `sequential_token_segments`
- `sequential_reverse`
- `oracle_single_d1`
- `oracle_single_d2`
- `merge_high_clamp`
- `mistral`

### Phase B: External evaluation correction

We replaced the old internal story with:

- an externally authored 100-item benchmark
- full 100-item Qwen and Mistral runs
- a 30-item stratified blind pairwise evaluation

### Phase C: TCAR collaborative inference

We moved from weight blending to:

1. router selects experts
2. experts answer independently
3. final stage either refines or verifies

### Phase D: Router training

We generated synthetic routing traces and trained a dedicated router:

- SFT on 5,000 routing examples
- DPO attempt on top of SFT
- later replacement of DPO direction with GRPO scaffold, but not full execution due wall-clock cost

### Phase E: Low-latency verifier mode and DES

We removed the generative refiner, used a discriminative verifier, enforced short expert answers, and then attempted Dynamic Experts Search with sampled unique routes.

---

## 4. Final Results by Phase

### 4.1 Best static Qwen methods vs Mistral on the 100-item external benchmark

| System | Semantic Sim | Token F1 | Mean Latency |
| --- | ---: | ---: | ---: |
| `late_layer_injection` | 0.6594 | 0.2715 | 3.890s |
| `weighted_merge` | 0.6592 | 0.2719 | 4.263s |
| `sequential_reverse` | 0.6623 | 0.2734 | 4.605s |
| `mistral` | 0.6907 | 0.2917 | 10.654s |

**Interpretation:**  
Static Qwen methods remained much faster, but none beat Mistral on the external 100-item benchmark.

### 4.2 Blind judge correction

30-item stratified blind comparison vs Mistral:

| Qwen Method | Qwen Wins | Mistral Wins | Ties |
| --- | ---: | ---: | ---: |
| `weighted_merge` | 6 | 23 | 1 |
| `late_layer_injection` | 4 | 26 | 0 |
| `sequential_reverse` | 4 | 25 | 1 |

**Interpretation:**  
The external blind judge did not support a “Qwen beats Mistral” claim.

### 4.3 Router training

| Router | Exact Match | Partial Overlap | Mean Overlap F1 | Mean Latency |
| --- | ---: | ---: | ---: | ---: |
| SFT router | 0.85 | 1.00 | 0.9450 | 1.079s |
| DPO router | 0.42 | 0.75 | 0.6333 | 1.697s |

**Interpretation:**  
SFT was a real success. DPO regressed routing quality badly.

### 4.4 TCAR with generative collaborative reasoning

10-item SFT-router pilot:

| System | Semantic Sim | Token F1 | Mean Latency |
| --- | ---: | ---: | ---: |
| `tcar_collaborative` + SFT router | 0.6902 | 0.2874 | 16.845s |
| `tcar_oracle_collaborative` | 0.7098 | 0.2774 | 15.175s |

**Interpretation:**  
This was the strongest signal that collaborative inference had real upside. It pushed small-Qwen semantic quality much closer to Mistral, but latency became worse than Mistral.

### 4.5 TCAR + DPO on the full 100-item benchmark

| System | Semantic Sim | Token F1 | Mean Latency |
| --- | ---: | ---: | ---: |
| `tcar_collaborative` + DPO router | 0.6900 | 0.2712 | 24.198s |
| `mistral` | 0.6907 | 0.2917 | 10.654s |

Latency breakdown for final TCAR + DPO:

| Component | Mean | Median | P95 |
| --- | ---: | ---: | ---: |
| Router | 3.784s | 1.695s | 4.634s |
| Shared-prefill branches | 11.149s | 7.539s | 33.687s |
| Refiner | 9.259s | 6.260s | 25.943s |
| Total | 24.198s | 16.246s | 85.909s |

**Interpretation:**  
The semantic ceiling was real, but the system was too slow and too unstable in the tail.

### 4.6 Verifier-only TCAR

10-item SFT-router verifier pilot:

| System | Semantic Sim | Token F1 | Mean Latency |
| --- | ---: | ---: | ---: |
| `tcar_verifier` + SFT router | 0.6492 | 0.2459 | 4.4242s |
| `mistral` | 0.7067 | 0.2971 | 10.718s |

Latency breakdown:

| Component | Mean |
| --- | ---: |
| Router | 1.067s |
| Branches | 2.852s |
| Verifier | 0.506s |
| Total | 4.424s |

**Interpretation:**  
This was a major structural speed win. It beat Mistral comfortably on latency, but it lost too much answer quality.

### 4.7 DES / sampled multi-route inference

Partial 4-item sampled-route pilot before early stop:

| Metric | Value |
| --- | ---: |
| Mean Latency | 78.853s |
| Mean Token F1 | 0.2776 |
| Mean Semantic Sim | 0.6651 |

Per-item latencies:

- `92.781s`
- `78.982s`
- `79.610s`
- `64.038s`

**Interpretation:**  
Inference-time search restored some diversity but destroyed latency. It failed the deployment criterion immediately.

---

## 5. What Was Actually Useful

### Useful for research

1. **Metric sensitivity was a real finding**
   Soft metrics overstated progress. Blind judging changed the story.
2. **Router training matters**
   SFT dramatically improved routing accuracy.
3. **Collaborative inference has a higher semantic ceiling than static weight blending**
   TCAR got much closer to Mistral than the static merge family.
4. **DPO can hurt routing**
   Preference optimization on router text did not translate to better routing decisions.
5. **Latency decomposition is now explicit**
   We know exactly where collaborative systems pay their cost.

### Useful for startup positioning

1. **Apple Silicon multi-expert serving works**
   The platform supports real expert composition on-device.
2. **Fast modular mode exists**
   Verifier-only TCAR is fast enough to beat Mistral latency by a wide margin.
3. **Controllable specialization is real**
   The architecture offers routing and expert composition knobs that monolithic baselines do not.

---

## 6. What Was Not Good Enough

1. **No final-answer breakthrough over Mistral**
   We never got a full external benchmark result that beats Mistral on quality.
2. **DPO was not a success**
   It damaged router accuracy and did not produce a net system win.
3. **Generative collaborative TCAR is too slow**
   It recovers quality, but the latency tail is not deployment-grade.
4. **Verifier-only TCAR is too lossy**
   It solves speed, but not enough quality.
5. **DES is too expensive on this hardware**
   Full sampled-route search is not a practical inference strategy here.

---

## 7. Final Bottom Line

The strongest defensible conclusion is:

> Small-model multi-expert inference on Apple Silicon is real, fast, and architecturally promising, but parameter-space merging alone does not solve cross-domain reasoning, and the current collaborative alternatives do not yet beat Mistral on externally evaluated answer quality.

The best concise summary of where we landed:

- **Static Qwen blends:** fast, but weaker than Mistral
- **Router SFT:** strong success
- **Collaborative TCAR with synthesis:** quality goes up, latency becomes too high
- **Verifier-only TCAR:** latency becomes excellent, quality drops
- **DES:** too slow to be practical

So the project is **useful and novel**, but the novelty is:

- systems design
- evaluation honesty
- routing/collaboration insight

not a final “we beat Mistral” capability claim.

---

## 8. Recommended Positioning

### For a paper

Position it as:

- a rigorous study of multi-adapter composition limits
- a demonstration that routing and evaluation methodology dominate conclusions
- a negative-plus-positive result:
  parameter merging is insufficient, but collaborative inference offers a real ceiling at a real cost

### For a startup

Position it as:

- an edge-native multi-expert inference platform
- fast controllable specialization on Apple Silicon
- a system with a strong modular foundation and clear next optimization targets

Do **not** position it as:

- beating Mistral overall
- solved multi-domain reasoning
- production-grade collaborative inference today

---

## 9. Canonical Supporting Files

- [FINAL_EXPERIMENT_REPORT_2026_04.md](/Users/uditjain/Desktop/adapter/FINAL_EXPERIMENT_REPORT_2026_04.md)
- [FULL_PROJECT_SUMMARY.md](/Users/uditjain/Desktop/adapter/FULL_PROJECT_SUMMARY.md)
- [md_external_v2_blind_report.md](/Users/uditjain/Desktop/adapter/results/md_external_v2_blind_report.md)
- [tcar_dpo_final_100_report_2026_04_09.md](/Users/uditjain/Desktop/adapter/results/tcar_dpo_final_100_report_2026_04_09.md)
- [tcar_verifier_pilot10_2026_04_09.md](/Users/uditjain/Desktop/adapter/results/tcar_verifier_pilot10_2026_04_09.md)

This file is the single best “where we ended up and what it means” summary.
