# Prompt-Level Multi-Adapter Composition with Norm-Proportional Clamping

> **Pre-registered negative result.** Multi-adapter composition with prompt-level routing and norm clamping does NOT outperform single-adapter routing on domain-specific queries, but does reduce perplexity and introduces zero latency overhead.

## Quick Results

| Method | K | Clamp | Avg Semantic Sim ↑ | Avg PPL ↓ | Avg Latency |
|--------|---|-------|--------------------|-----------|-------------|
| Baseline (no adapters) | 1 | 0.001 | 0.620 | 64.5 | 2.80s |
| SingleAdapter | 1 | 0.5 | **0.622** | 60.9 | 2.69s |
| UnclampedMix | 2 | 999 | 0.557 | 51.2 | 2.51s |
| **AdaptiveClamp** | 2 | 0.5 | 0.611 | **58.0** | 2.67s |

**Δ_SIM(AdaptiveClamp − SingleAdapter) = −0.011** → FAIL (threshold was > +0.05)

## Model & Adapters

- **Base model:** `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (auto-downloaded by mlx_lm)
- **Adapters:** 20 domain-specific LoRA experts in `backend/expert_adapters/`, each ~20 MB safetensor
- **Hardware:** Apple Silicon with Unified Memory Architecture (tested on M3 Max)

## Setup

```bash
# Clone and install
git clone <this-repo>
cd adapter
pip install -r requirements.txt
```

## Reproduce the Full 400-Inference Benchmark

```bash
# This runs 100 hard domain questions × 4 methods with real MLX inference
# Takes ~20 minutes on M3 Max
cd /path/to/adapter
export PYTHONPATH=$(pwd)
rm -f results_db.jsonl
python3 src/eval/real_benchmark.py
```

**Outputs:**
- `results/real_benchmark_table.md` — Per-domain results table
- `results/real_benchmark_results.json` — Raw per-question data (400 entries)
- `results_db.jsonl` — Log entries with `real_mode: true`

## Reproduce Single-Method Runs

```bash
# Simulation mode (pipeline testing, no real model)
python3 src/eval/run_eval.py --config_id debug_single_real

# Real mode (requires MLX + adapters)
python3 src/eval/run_eval.py --config_id debug_single_real --real
python3 src/eval/run_eval.py --config_id debug_adaptive_real --real
```

## Project Structure

```
adapter/
├── backend/
│   ├── dynamic_mlx_inference.py    # DynamicEngine with RoutedLoRALinear
│   ├── orchestrator.py             # CoT-based domain router
│   ├── expert_adapters/            # 20 trained LoRA adapters (.safetensors)
│   ├── expert_registry.json        # Adapter paths and metadata
│   └── ablation_benchmark.py       # Original 100 hard questions
├── src/
│   ├── eval/
│   │   ├── run_eval.py             # Eval harness (--real for live inference)
│   │   ├── real_benchmark.py       # Full 4-method benchmark
│   │   └── metrics.py              # Aggregation utilities
│   ├── adapters/
│   │   ├── adaptive_multi_lora_linear.py
│   │   └── registry.py
│   └── routers/
│       └── cot_router.py
├── configs/
│   └── uma_experiments.yaml
├── results/
│   ├── decision_summary.md         # PASS/FAIL verdicts
│   ├── real_benchmark_table.md     # Paper-ready table
│   └── real_benchmark_results.json # Raw data
├── paper.md                        # Full ICLR-style paper (negative result)
├── results_db.jsonl                # 400 real inference logs
└── README.md                       # This file
```

## Paper

See [`paper.md`](paper.md) for the full writeup. Title:

> **Prompt-Level Multi-Adapter Composition with Norm-Proportional Clamping: A Pre-Registered Negative Result**

## Key Findings

1. **AdaptiveClamp does NOT beat SingleAdapter** on semantic similarity (0.611 vs 0.622)
2. **AdaptiveClamp does reduce perplexity** (58.0 vs 60.9) — the model assigns higher probability to correct answers
3. **UnclampedMix is catastrophic** — 8/100 prompts collapse to near-random output (sim < 0.1)
4. **Router accuracy is a bottleneck** — CoT router fails exact domain matching on ~40% of queries
5. **Latency overhead is negligible** (−0.7%) on UMA hardware

## License

Research use only.
