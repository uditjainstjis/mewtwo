# Verifier TCAR Pilot 10

**Date:** 2026-04-09  
**Dataset:** [multidomain_eval_claude_external_v2_10_stratified.json](/Users/uditjain/Desktop/adapter/data/multidomain_eval_claude_external_v2_10_stratified.json)

## Architectural Change

- Removed the generative refiner.
- Enforced short expert answers (`< 50` words target, `72` token hard cap).
- Added a discriminative verifier that scores branch answers with the base model and returns the highest-confidence branch.
- Kept the SFT router checkpoint fixed.

## New Pilot Result

| System | Semantic Sim | Token F1 | Latency |
| --- | ---: | ---: | ---: |
| TCAR verifier + SFT router | 0.6492 | 0.2459 | 4.4242s |
| Mistral baseline | 0.7067 | 0.2971 | 10.7177s |
| Old TCAR + SFT router + generative refiner | 0.6902 | 0.2874 | 16.8450s |

## Latency Breakdown

| Component | Mean | Median |
| --- | ---: | ---: |
| Router | 1.0667s | 0.9595s |
| Shared-prefill branches | 2.8518s | 2.4205s |
| Verifier | 0.5055s | 0.4850s |
| Total | 4.4242s | 4.0625s |

## Conclusion

- The verifier pivot **succeeded on speed**. Mean latency fell well below Mistral and far below the old generative-TCAR path.
- The verifier pivot **failed on quality retention**. Token F1 dropped from `0.2874` to `0.2459`.
- So branch selection without synthesis is too lossy for cross-domain questions, even though it fixes the refiner overhead.

## Practical Read

This pivot is useful as a latency-controlled fallback mode, not yet as the main flagship architecture. The next GRPO phase should be evaluated against this verifier path, but the current verifier-only TCAR should not replace the generative TCAR in claims about answer quality.
