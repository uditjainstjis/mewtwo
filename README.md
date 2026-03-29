# Prompt-Level Multi-Adapter Composition with Norm-Proportional Clamping

> The repo's final story is mixed: multi-adapter composition is negative on the original single-domain synthetic benchmark, but directionally positive on the later genuine multi-domain benchmark.

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

The current repo-grounded manuscript is [`Main_Paper_Composition_Updated.md`](Main_Paper_Composition_Updated.md). It supersedes the older draft in `paper.md` by incorporating the later v2, v2b, and v2c results and aligning the text with the live implementation.

Primary title:

> **Composition Without Collapse: Pre-Registered Evidence for Safe but Modest Prompt-Level Multi-Adapter LoRA Composition on Apple Silicon**

## Key Findings

1. **AdaptiveClamp does NOT beat SingleAdapter on the v1 single-domain benchmark** (0.611 vs 0.622).
2. **AdaptiveClamp-v2 is directionally positive on the v2 multi-domain benchmark** (0.6505 vs 0.6334), but still below the pre-registered `+0.03` threshold.
3. **UnclampedMix is unsafe on v1** — 3/100 prompts fall below `0.1` similarity and 7/100 fall below `0.2`.
4. **Router accuracy is a bottleneck, but not the only one** — the real top-2 router recovers about 26% of oracle headroom on the MD split.
5. **Clamp formulation is not the main constraint at this scale** — the norm-ratio clamp and weight cap differ by only `-0.0003` on the MD split.

## License

Research use only.
