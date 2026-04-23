# MEWTWO 20-Hour Research Sprint — Live Task Tracker
# Auto-updated by pipeline. Last updated: 2026-04-21 09:29

## Phase 1: Clean Single-Adapter Evals (Running)
- [x] 1.1 — ARC fixed for all 5 configs ARC-Challenge (fixed) — all 5 configs × 100 samples
- [x] 1.2 — HumanEval fixed for all 5 configs HumanEval (fixed) — all 5 configs × 100 samples
- [x] 1.3 — MATH-500 clean benchmark complete MATH-500 — all 5 configs × 200 samples (uncontaminated)
- [x] 1.4 — MBPP clean benchmark complete MBPP — all 5 configs × 100 samples (uncontaminated)

## Phase 2: TRUE Multi-Adapter Composition (Novel IP)
- [x] 2.1 — LayerBlendRouter module created Create 50 mixed-domain evaluation queries
- [x] 2.2 — 24 modules across 3 domains Implement 5 composition strategies via PEFT
- [x] 2.3 — All 8 configs evaluated on 50 queries Run mixed-domain experiment (8 configs × 50 queries)
- [x] 2.4 — composition Δ=+0.000 → FAIL Compute composite scores and comparative analysis

## Phase 3: Standardized Benchmarks (lm-eval-harness)
- [x] 3.1 — gsm8k complete for all configs — No LayerBlend adapter found GSM8K 8-shot CoT (base + adapters + composed)
- [x] 3.2 — arc_challenge complete for all configs ARC-Challenge 25-shot (base + adapters + composed)
- [x] 3.3 — MMLU-Pro complete MMLU-Pro 5-shot (base + adapters + composed)

## Phase 4: Startup Demo
- [ ] 4.1 Build Gradio interface
- [ ] 4.2 Wire to model inference
- [x] 4.3 — Final analysis complete Record demo video

## Phase 5: Paper Deliverables
- [x] 5.1 — Summary generated Generate comparison tables and figures
- [ ] 5.2 Compute bootstrap confidence intervals
- [x] 5.3 — Final results saved Final results JSON with all numbers
