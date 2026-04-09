# TCAR DPO Final 100-Item Report

**Date:** 2026-04-09  
**Hardware:** Apple M3 Max / Apple Silicon UMA  
**Dataset:** [multidomain_eval_claude_external_v2_100.json](/Users/uditjain/Desktop/adapter/data/multidomain_eval_claude_external_v2_100.json)

## Scope

This report closes the router-upgrade phase:

1. MPS-only router SFT with assistant-only loss masking
2. MPS-only router DPO with response-only masked log-probs
3. Final 100-item TCAR collaborative benchmark with explicit latency breakdown

Relevant artifacts:

- SFT holdout: [router_accuracy_sft_5000_valid_holdout_mpsfix.json](/Users/uditjain/Desktop/adapter/results/router_accuracy_sft_5000_valid_holdout_mpsfix.json)
- DPO holdout: [router_accuracy_dpo_5000_valid_holdout_mpsfix.json](/Users/uditjain/Desktop/adapter/results/router_accuracy_dpo_5000_valid_holdout_mpsfix.json)
- Final TCAR 100-run: [tcar_collaborative_dpo5000_mpsfix_100.jsonl](/Users/uditjain/Desktop/adapter/results/tcar_collaborative_dpo5000_mpsfix_100.jsonl)
- Live log: [tcar_collaborative_dpo5000_mpsfix_100.live.log](/Users/uditjain/Desktop/adapter/results/tcar_collaborative_dpo5000_mpsfix_100.live.log)
- Mistral baseline: [md_head_to_head_v2_mistral_only_100.jsonl](/Users/uditjain/Desktop/adapter/results/md_head_to_head_v2_mistral_only_100.jsonl)

## Router Accuracy

| Router | Exact Match | Partial Overlap | Mean Overlap F1 | Mean Router Latency |
| --- | ---: | ---: | ---: | ---: |
| SFT | 0.85 | 1.00 | 0.9450 | 1.079s |
| DPO | 0.42 | 0.75 | 0.6333 | 1.697s |

### Interpretation

- DPO **regressed routing quality sharply** on the synthetic routing holdout.
- Exact-match routing fell from `85%` to `42%`.
- So DPO did **not** improve the router as a routing classifier, even though it successfully optimized the pairwise preference objective.

## 10-Item Downstream Sanity Check

| System | Semantic Sim | Token F1 | Latency |
| --- | ---: | ---: | ---: |
| TCAR + SFT router | 0.6902 | 0.2874 | 16.845s |
| TCAR + DPO router | 0.7032 | 0.3046 | 19.642s |

### Interpretation

- Despite worse routing accuracy, the DPO router improved the 10-item downstream pilot.
- That justified running the full 100-item benchmark instead of rejecting DPO on the holdout alone.

## Final 100-Item Comparison

| System | Semantic Sim | Token F1 | Exact Match | Mean Latency |
| --- | ---: | ---: | ---: | ---: |
| TCAR + DPO router | 0.6900 | 0.2712 | 0.0000 | 24.198s |
| Mistral-7B baseline | 0.6907 | 0.2917 | 0.0000 | 10.654s |
| Best prior Qwen similarity (`sequential_reverse`) | 0.6623 | 0.2734 | 0.0000 | 4.605s |
| Best prior Qwen speed (`late_layer_injection`) | 0.6594 | 0.2715 | 0.0000 | 3.890s |

### Interpretation

- TCAR + DPO **nearly matched** Mistral on semantic similarity: `0.6900` vs `0.6907`.
- TCAR + DPO did **not** beat Mistral on token F1: `0.2712` vs `0.2917`.
- TCAR + DPO did **not** beat Mistral on latency: `24.198s` vs `10.654s`.
- TCAR + DPO substantially improved over the old Qwen blend family on semantic similarity, but not on token F1.

## TCAR Latency Breakdown

| Component | Mean | Median | P95 | Max |
| --- | ---: | ---: | ---: | ---: |
| Router | 3.784s | 1.695s | 4.634s | 90.691s |
| Shared-prefill branches | 11.149s | 7.539s | 33.687s | 90.610s |
| Refiner | 9.259s | 6.260s | 25.943s | 101.685s |
| Total | 24.198s | 16.246s | 85.909s | 121.561s |

### Tail Behavior

| Threshold | Count |
| --- | ---: |
| Total latency > 20s | 36 |
| Total latency > 30s | 16 |
| Total latency > 60s | 8 |
| Total latency > 90s | 4 |

The worst outliers were dominated by:

- router stalls on `QUANTUM_CHEMISTRY` combinations
- branch stalls on `SANSKRIT_LINGUISTICS` and some chemistry/physics combinations
- refiner stalls on `RENAISSANCE_ART` and history/philosophy combinations

Worst cases from the final run:

| Item | Domains | Total | Router | Branches | Refiner |
| --- | --- | ---: | ---: | ---: | ---: |
| `ext_md_082` | `QUANTUM_CHEMISTRY + ASTROPHYSICS` | 121.561s | 80.747s | 34.702s | 6.106s |
| `ext_md_091` | `QUANTUM_CHEMISTRY + CLIMATE_SCIENCE` | 115.558s | 90.691s | 7.029s | 17.832s |
| `ext_md_089` | `SANSKRIT_LINGUISTICS + ANCIENT_HISTORY` | 108.130s | 0.969s | 5.466s | 101.685s |
| `ext_md_099` | `SANSKRIT_LINGUISTICS + PHILOSOPHY` | 96.217s | 2.600s | 90.610s | 3.001s |

## Honest Conclusion

- The router SFT phase was a real success.
- The DPO phase did **not** improve router accuracy.
- The DPO collaborative pipeline looked promising on a 10-item pilot, but that did **not** survive cleanly on the full 100-item benchmark.
- Final TCAR + DPO gets to **Mistral-level semantic similarity**, but still trails on token-level accuracy and is much slower because of severe latency tails.

## What Is Actually Novel Now

What holds up:

- Apple Silicon multi-adapter collaborative inference is real and measurable.
- Shared-prefill TCAR raises semantic quality far above static Qwen adapter blending.
- Latency can now be decomposed precisely into router, branch, and refiner phases.

What does not hold yet:

- DPO as currently constructed is not a net win for routing.
- The final TCAR system does not yet beat Mistral on the 100-item external benchmark.
- The latency distribution is not deployment-grade because of extreme tail behavior.

## Next Technical Moves

1. Add strict output-length caps and stop-token enforcement to router and refiner to kill tail latency.
2. Revisit DPO data construction. Current rejected traces likely teach stylistic preferences more than routing correctness.
3. Try best-of-N / self-consistency only on the router, not on full TCAR branches.
4. Keep SFT as the routing backbone unless a new preference stage improves holdout exact-match routing.
