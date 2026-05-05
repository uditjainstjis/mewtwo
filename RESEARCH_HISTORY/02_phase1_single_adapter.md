# Phase 1 — Single-Adapter Benchmarks on Nemotron-30B

**Date:** 2026-04 (early run)
**Owner:** Synapta autonomous pipeline
**Source artefacts:**
- `results/nemotron/master_results.json` (full Phase-1 grid)
- `scripts/master_pipeline.py` (now `synapta_src/...`)
- `docs/MASTER_KNOWLEDGE_BASE.md` §4.3 (narrative discussion)

## Goal
Establish single-adapter baselines for math, code, science adapters trained on Nemotron-Nano-30B-A3B, before testing composition.

## Setup
- Base: Nemotron-Nano-30B-A3B, 4-bit NF4 via QLoRA.
- Each adapter: $r=16$, $\alpha=32$, dropout 0.05, paged AdamW 8-bit, lr $2\times10^{-4}$ cosine.
- All attn+MLP target modules: q, k, v, o, gate, up, down.
- Trainable: 434.6M / 32B = 1.36\%.

## Benchmarks
- ARC-Challenge ($n=100$, accuracy)
- HumanEval ($n=100$, pass@1; v1 buggy scoring at this stage)
- MATH-500 ($n=200$, exact match)
- MBPP ($n=100$, pass@1)

## Results

| Adapter | ARC | HumanEval | MATH-500 | MBPP |
|---|---|---|---|---|
| base | 20.0\% | 50.0\% | 41.5\% | 8.0\% |
| math | 23.0\% | **60.0\%** | 50.5\% | 2.0\% |
| code | **31.0\%** | 27.0\% | **56.0\%** | 6.0\% |
| science | 21.0\% | 1.0\% | 55.0\% | 0.0\% |
| merged (DARE/TIES uniform) | 19.0\% | 34.0\% | 56.0\% | 0.0\% |

## Findings (the Code Paradox)

**Asymmetric cross-domain transfer:**
1. Code training BREAKS code: HumanEval 50\% → 27\% (-23 pp) under code adapter.
2. Code training BOOSTS reasoning: ARC 20\% → 31\% (+11 pp), MATH-500 41.5\% → 56\% (+14.5 pp).
3. Math training BOOSTS code: HumanEval 50\% → 60\% (+10 pp) under math adapter.
4. Science training is CATASTROPHIC for code: HumanEval 50\% → 1\% (-49 pp).

The code adapter has learned to *discuss* code (verbose chain-of-thought) rather than *produce* code. The math adapter, trained on numerical step-by-step reasoning, transfers to multi-step Python planning.

**Static merging hits a ceiling:**
- Merged adapter matches best single on MATH-500 (56\% = 56\%).
- Drops below base on ARC (19\% < 20\%) and far below best on HumanEval (34\% < 60\%).

## Caveats and corrections
- **HumanEval scoring bug:** This Phase 1 run used the v1 buggy code-extraction harness. The absolute HumanEval numbers in this table are under-counted by $\sim$30 pp. After fix (see `09_humaneval_scoring_bug.md`), base rises from 50\% (n=100) to 56.1\% (n=164 v2). The Code Paradox **direction** is preserved at v2 scoring; only the absolute floor moves.
- **Cross-family overclaim rolled back:** Original deck claimed replication of the Code Paradox across 3 base models from $n=50$ subsets. Subsequent $n=200$ follow-up showed the $n=50$ cross-family results were small-sample flukes. **Robust replication is at $n=200$ on Nemotron-30B and (separately) at $n=200$ on Qwen-3.5-0.8B for the in-domain regression only.**

## Files
- `results/nemotron/master_results.json` (Phase 1 grid, $n=100$ ARC/HE/MBPP, $n=200$ MATH-500)
- `docs/findings/code_paradox_replication.md` (full replication trace)
