# LoRI MoE — Composite Routing Experiment

**Date:** 2026-04-12
**Source artefacts:**
- `results/lori_moe/all_results.json`
- `results/lori_moe/phase1_baselines.json`
- `results/lori_moe/phase2_single_adapter.json`
- `results/lori_moe/phase3_composite.json`
- `adapters/lori_moe/lori-qwen2.5-1.5b-{code,legal,math,medical,science}/`

## Goal
Test whether 5-domain LoRI-style sparse adapters on a Qwen-2.5-1.5B base, with prompt-routed composite top-1 selection, beat the best single adapter on standard benchmarks.

## Setup
- Base: Qwen-2.5-1.5B.
- 5 specialised adapters: code, legal, math, medical, science.
- Three routing strategies: zero-shot prompt (base only), single-adapter routed, composite top-1 routed.
- Benchmarks: GSM8K (200), ARC-Challenge (200), MMLU (200), small subsets where noted.

## Phase 1: Base zero-shot baselines

| Benchmark | Score | $n$ |
|---|---|---|
| GSM8K | 26.0\% | 200 |
| ARC-Challenge | **76.5\%** | 200 |
| MMLU | 56.5\% | 200 |

## Phase 2: Single-adapter scores

| Adapter | GSM8K | ARC | (n) |
|---|---|---|---|
| math | **53.0\%** | --- | 200 |
| code | 3.5\% (regressed) | --- | 200 |
| science | 1.0\% (regressed) | 65.0\% | 200 / 200 |
| legal | 15.0\% | **77.5\%** | 100 / 200 |
| medical | 1.0\% (regressed) | 71.5\% | 100 / 200 |

The math adapter dominates GSM8K (+27 pp over base). The legal adapter slightly improves ARC (+1 pp), but every other adapter destroys GSM8K capability (regressions of $-22$ to $-25$ pp).

## Phase 3: Composite top-1 routed

| Benchmark | Composite | Best single | Δ |
|---|---|---|---|
| GSM8K | **4.0\%** | 53.0\% | **-49 pp catastrophic** |
| ARC | 72.0\% | 77.5\% | -5.5 pp |
| MMLU | 53.0\% | --- | --- |

## Verdict
**Composite top-1 routing on this base + benchmark mix did not exceed the single best expert.** On GSM8K the composite policy regressed catastrophically (presumably routed to non-math adapters on math-domain queries). On ARC the composite came within 5.5 pp of the legal-adapter peak.

This is the **same composition-fails-to-emerge** finding as static weight-merging on Nemotron-30B (`03_static_composition_failure.md`), now replicated at smaller base + sparser adapters + dynamic top-1 routing.

## Caveats
- Phase 3 routing strategy was prompt-keyword classifier (early prototype), not the token-level Format Guard.
- Phase 2 single-adapter regressions on out-of-domain tasks (medical adapter destroys GSM8K) match the Code Paradox in spirit: domain specialisation costs out-of-domain capability.

## Files
- `results/lori_moe/*.json`
- `adapters/lori_moe/lori-qwen2.5-1.5b-*/` (5 adapters)
