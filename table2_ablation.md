# Table 2: Domain-Specific Expertise Gain

*Generated: 2026-03-27 03:40 | Model: Qwen2.5-1.5B-Instruct-4bit | n=5 per domain*

## Semantic Similarity (↑ is better)

| Domain | Baseline | Synapta-Agg (c=1.0) | Synapta-Bal (c=0.5) | Δ Improvement |  Lat Overhead |
|--------|----------|---------------------|---------------------|---------------|---------------|
| LEGAL_ANALYSIS | 0.734 | 0.740 | 0.710 | -3.2% | -0.48s |
| MEDICAL_DIAGNOSIS | 0.700 | 0.695 | 0.698 | -0.3% | +0.23s |
| PYTHON_LOGIC | 0.652 | 0.597 | 0.652 | -0.1% | -0.13s |
| MATHEMATICS | 0.538 | 0.518 | 0.543 | +0.8% | -0.01s |
| MLX_KERNELS | 0.582 | 0.534 | 0.552 | -5.2% | +0.00s |
| LATEX_FORMATTING | 0.527 | 0.630 | 0.561 | +6.6% | -0.04s |
| SANSKRIT_LINGUISTICS | 0.738 | 0.711 | 0.734 | -0.5% | +0.09s |
| ARCHAIC_ENGLISH | 0.506 | 0.545 | 0.588 | +16.0% | +0.14s |
| QUANTUM_CHEMISTRY | 0.630 | 0.616 | 0.595 | -5.6% | +0.01s |
| ORGANIC_SYNTHESIS | 0.697 | 0.690 | 0.679 | -2.5% | +0.03s |
| ASTROPHYSICS | 0.601 | 0.574 | 0.566 | -5.8% | -0.20s |
| MARITIME_LAW | 0.672 | 0.609 | 0.589 | -12.4% | -0.17s |
| RENAISSANCE_ART | 0.576 | 0.605 | 0.638 | +10.7% | +0.04s |
| CRYPTOGRAPHY | 0.603 | 0.587 | 0.583 | -3.3% | +0.09s |
| ANCIENT_HISTORY | 0.591 | 0.553 | 0.537 | -9.1% | +0.21s |
| MUSIC_THEORY | 0.641 | 0.660 | 0.650 | +1.5% | -0.00s |
| ROBOTICS | 0.653 | 0.627 | 0.637 | -2.4% | -0.25s |
| CLIMATE_SCIENCE | 0.653 | 0.638 | 0.668 | +2.2% | -0.18s |
| PHILOSOPHY | 0.460 | 0.599 | 0.607 | +32.0% | -0.12s |
| BEHAVIORAL_ECONOMICS | 0.637 | 0.631 | 0.619 | -3.0% | -0.28s |
|--------|----------|---------------------|---------------------|---------------|---------------|
| **AVERAGE** | **0.620** | **0.618** | **0.620** | **+0.1%** | **-0.05s** |

## Perplexity Delta (↓ is better)

*Lower perplexity = model assigns higher probability to ground truth = deeper internalization*

| Domain | PPL Baseline | PPL Synapta-Bal | Δ PPL | PPL Reduction % |
|--------|-------------|-----------------|-------|-----------------|
| LEGAL_ANALYSIS | 72.4 | 73.6 | -1.2 | -1.6% |
| MEDICAL_DIAGNOSIS | 58.9 | 52.1 | +6.8 | +11.5% |
| PYTHON_LOGIC | 62.7 | 56.9 | +5.8 | +9.2% |
| MATHEMATICS | 63.8 | 58.0 | +5.8 | +9.1% |
| MLX_KERNELS | 76.6 | 63.6 | +13.0 | +16.9% |
| LATEX_FORMATTING | 76.0 | 74.2 | +1.8 | +2.3% |
| SANSKRIT_LINGUISTICS | 47.0 | 44.8 | +2.2 | +4.8% |
| ARCHAIC_ENGLISH | 68.1 | 57.6 | +10.5 | +15.4% |
| QUANTUM_CHEMISTRY | 51.2 | 48.0 | +3.2 | +6.3% |
| ORGANIC_SYNTHESIS | 70.1 | 65.8 | +4.2 | +6.1% |
| ASTROPHYSICS | 46.7 | 42.1 | +4.6 | +9.8% |
| MARITIME_LAW | 74.9 | 63.8 | +11.2 | +14.9% |
| RENAISSANCE_ART | 54.8 | 47.9 | +6.8 | +12.5% |
| CRYPTOGRAPHY | 78.1 | 72.5 | +5.6 | +7.2% |
| ANCIENT_HISTORY | 59.8 | 51.9 | +7.9 | +13.2% |
| MUSIC_THEORY | 48.6 | 43.9 | +4.7 | +9.7% |
| ROBOTICS | 69.6 | 60.3 | +9.3 | +13.3% |
| CLIMATE_SCIENCE | 55.9 | 50.4 | +5.4 | +9.8% |
| PHILOSOPHY | 81.0 | 69.4 | +11.7 | +14.4% |
| BEHAVIORAL_ECONOMICS | 73.0 | 66.7 | +6.3 | +8.6% |
|--------|-------------|-----------------|-------|-----------------|
| **AVERAGE** | **64.5** | **58.2** | **+6.3** | **+9.7%** |

## Ablation: Clamping Effect

| Config | Avg Semantic Sim | Avg PPL | Avg Latency |
|--------|-----------------|---------|-------------|
| Baseline (c=0.0) | 0.620 | 64.5 | 3.37s |
| Synapta-Aggressive (c=1.0) | 0.618 | 52.7 | 3.27s |
| Synapta-Balanced (c=0.5) | 0.620 | 58.2 | 3.32s |
