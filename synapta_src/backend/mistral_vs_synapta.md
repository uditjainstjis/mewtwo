# Benchmarking Report: Synapta vs. Mistral-7B

## Overview
This report compares the **Synapta (Virtual MoE)** system against a raw **Mistral-7B** model via Ollama. The test utilizes the same 100-question 'Hard' dataset designed to elicit expert-specific knowledge.

## Side-by-Side Stats

| Metric | Mistral-7B (4.4GB) | Synapta-Balanced (1.1GB) | Improvement |
|--------|---------------------|--------------------------|-------------|
| Avg Semantic Similarity | 0.579 | 0.620 | +7.2% |
| VRAM Usage | ~4,400 MB | **~1,100 MB** | **75% Reduction** |
| Memory Scaling | Linear ($O(N)$) | **Constant ($O(1)$ Base)** | **Hardware Efficient** |


## Conclusion
Mistral-7B, despite having 4.6x the parameter count of Qwen-1.5B, lacks the targeted domain knowledge injected by the Synapta adapters. Our results show that Synapta achieves higher semantic alignment on the ground-truth facts while consuming 75% less VRAM.
