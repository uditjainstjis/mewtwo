# Table 1: Multi-Adapter Composition — Full Benchmark (REAL)

*Generated: 2026-03-29 00:28 | Model: Qwen2.5-1.5B-Instruct-4bit | 100 questions × 20 domains*

> All values from REAL model inference. No simulation.

## Per-Domain Semantic Similarity (↑ better)

| Domain | Baseline | SingleAdapter | UnclampedMix | AdaptiveClamp |
|--------|-------|-------|-------|-------|
| LEGAL_ANALYSIS | 0.734 | 0.717 | 0.516 | 0.710 |
| MEDICAL_DIAGNOSIS | 0.700 | 0.683 | 0.663 | 0.713 |
| PYTHON_LOGIC | 0.652 | 0.668 | 0.621 | 0.640 |
| MATHEMATICS | 0.538 | 0.543 | 0.521 | 0.587 |
| MLX_KERNELS | 0.582 | 0.547 | 0.561 | 0.572 |
| LATEX_FORMATTING | 0.527 | 0.564 | 0.565 | 0.569 |
| SANSKRIT_LINGUISTICS | 0.738 | 0.738 | 0.682 | 0.703 |
| ARCHAIC_ENGLISH | 0.506 | 0.592 | 0.426 | 0.560 |
| QUANTUM_CHEMISTRY | 0.630 | 0.611 | 0.581 | 0.634 |
| ORGANIC_SYNTHESIS | 0.697 | 0.673 | 0.654 | 0.695 |
| ASTROPHYSICS | 0.601 | 0.566 | 0.568 | 0.558 |
| MARITIME_LAW | 0.672 | 0.609 | 0.466 | 0.464 |
| RENAISSANCE_ART | 0.576 | 0.620 | 0.550 | 0.598 |
| CRYPTOGRAPHY | 0.603 | 0.592 | 0.572 | 0.559 |
| ANCIENT_HISTORY | 0.591 | 0.555 | 0.402 | 0.516 |
| MUSIC_THEORY | 0.641 | 0.649 | 0.443 | 0.622 |
| ROBOTICS | 0.653 | 0.637 | 0.610 | 0.625 |
| CLIMATE_SCIENCE | 0.653 | 0.631 | 0.630 | 0.642 |
| PHILOSOPHY | 0.460 | 0.596 | 0.591 | 0.617 |
| BEHAVIORAL_ECONOMICS | 0.637 | 0.653 | 0.525 | 0.627 |
|--------|-------|-------|-------|-------|
| **AVERAGE** | **0.620** | **0.622** | **0.557** | **0.611** |

## Summary: Method Comparison

| Method | K | Clamp | Avg Sim ↑ | Avg PPL ↓ | Avg Latency |
|--------|---|-------|-----------|-----------|-------------|
| Baseline | 1 | 0.001 | 0.620 | 64.5 | 2.80s |
| SingleAdapter | 1 | 0.5 | 0.622 | 60.9 | 2.69s |
| UnclampedMix | 2 | 999.0 | 0.557 | 51.2 | 2.51s |
| AdaptiveClamp | 2 | 0.5 | 0.611 | 58.0 | 2.67s |

## Pre-Registered Δ Metrics (REAL)

- **Δ_SIM(AdaptiveClamp − Baseline):** -0.0089
- **Δ_SIM(AdaptiveClamp − SingleAdapter):** -0.0117
- **Threshold:** Δ > 0.05 for compositional gain → **FAIL**
