# Decision Summary — REAL Benchmark Results

> Generated from 400 real MLX inferences (100 questions × 4 methods)
> Model: Qwen2.5-1.5B-Instruct-4bit | 20 domains | sentence-transformers similarity

## Pre-Registered Thresholds vs Measured Values

| Criterion | Metric | Threshold | Measured | Verdict |
|-----------|--------|-----------|----------|---------|
| Compositional Gain | Δ_SIM(AC − SA) | > +0.05 | **-0.011** | **FAIL** |
| Compositional Gain | Δ_SIM(AC − Baseline) | > +0.05 | **-0.009** | **FAIL** |
| PPL Improvement | PPL(AC) vs PPL(SA) | AC ≤ SA | **58.0 < 60.9** | **PASS** |
| PPL Improvement | PPL(AC) vs PPL(Base) | AC ≤ Base | **58.0 < 64.5** | **PASS** |
| Latency Overhead | Δ_LAT(AC − SA) / SA | ≤ 10% | **(2.67−2.69)/2.69 = −0.7%** | **PASS** |

## Method Rankings (Avg Semantic Similarity)

1. **SingleAdapter: 0.622** (best)
2. **Baseline: 0.620**
3. **AdaptiveClamp: 0.611**
4. **UnclampedMix: 0.557** (worst — catastrophic collapses observed)

## Key Finding

The Adaptive Clamp does **NOT** outperform the SingleAdapter or Baseline on semantic similarity. It achieves lower perplexity (58.0 vs 64.5) and negligible latency overhead, but the core hypothesis — that multi-adapter composition improves compositional accuracy — is **not supported** by this experiment.

The UnclampedMix catastrophically degrades, confirming that norm bounding is necessary, but the bounded version still underperforms the simpler single-adapter approach.

## Domains Where AdaptiveClamp Won

- MEDICAL_DIAGNOSIS: 0.713 vs 0.683 (SA) — **+0.030**
- MATHEMATICS: 0.587 vs 0.543 (SA) — **+0.044**
- QUANTUM_CHEMISTRY: 0.634 vs 0.611 (SA) — **+0.023**
- PHILOSOPHY: 0.617 vs 0.596 (SA) — **+0.021**

## Domains Where AdaptiveClamp Lost

- MARITIME_LAW: 0.464 vs 0.609 (SA) — **-0.145**
- SANSKRIT_LINGUISTICS: 0.703 vs 0.738 (SA) — **-0.035**
- ASTROPHYSICS: 0.558 vs 0.566 (SA) — **-0.008**
- CRYPTOGRAPHY: 0.559 vs 0.592 (SA) — **-0.033**
