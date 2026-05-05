# Static Adapter Composition — Why It Fails at 30B

**Date:** 2026-04 (Phase 2)
**Source artefacts:**
- `results/nemotron/sprint_results.json` ($n=45$ mixed-domain probe)
- `results/nemotron/master_results.json` (4-benchmark grid)

## Goal
Test whether weight-merging the four task-specialised adapters (math, code, science, BFSI) yields emergent composition gain over the best single expert.

## Methods tested
1. **DARE** (Drop And REscale)~\citep{yu2024dare}: random weight pruning at $\rho=0.5$, then magnitude rescaling.
2. **TIES** (TrIm Elect Sign)~\citep{yadav2023ties}: top-$K$ magnitude trim + elect-sign + disjoint-sum.
3. **Linear weighted average**: uniform $w_i = 1/N$.

All operate on the LoRA $A$ and $B$ matrices independently, then re-attach via PEFT's `add_weighted_adapter` API.

## Results

### Phase-2 mixed-domain probe ($n=45$)

| Method | Score | Δ vs best single |
|---|---|---|
| Base | 51.1\% | --- |
| Best single (routed) | 60.0\% | --- |
| DARE merged | 60.0\% | **+0.0** |
| TIES merged | 60.0\% | **+0.0** |
| Linear merged | 60.0\% | **+0.0** |

### Phase-1 4-benchmark grid (full results)

| Method | ARC | HumanEval | MATH-500 | MBPP |
|---|---|---|---|---|
| Base | 20\% | 50\% | 41.5\% | 8\% |
| Best single | 31\% (code) | 60\% (math) | 56\% (code) | 6\% (code) |
| Static merge | 19\% | 34\% | 56\% | 0\% |

**Verdict:** weight-merging hits a ceiling at the best single expert (MATH-500: matches; HumanEval, ARC, MBPP: degrades). No emergent composition gain at 30B.

## Why does merging fail?

Hypothesis: at $r=16$, the four adapters' updates are concentrated in similar weight directions but with conflicting signs. DARE/TIES sparsify by sign-pruning small-magnitude updates, but at low rank the useful signal is dispersed across many small-magnitude entries. Both sparsifiers prune signal alongside noise. Linear averaging compresses 4× distinct skills into a single low-rank update with destructive interference.

This motivated the move to **token-level routing** (Format Guard, see `04_format_guard.md`), where each adapter remains intact and is selected per-token.

## Caveats
- Tested with **uniform weights only** ($w_i = 1/N$). Learned weighted composition (LoRAHub-style optimization over a held-out probe) is **not** tested at the 30B scale and may show different results.
- Tested at $r=16$ only. Higher-rank adapters may compose more cleanly under the same merging schemes; the rank-scaling zoo (`08_rank_scaling_zoo.md`) does not include 30B-scale merging ablations.

## Files
- `results/nemotron/sprint_results.json`
- `results/nemotron/master_results.json` (last row: merged_*)
- `synapta_src/scripts/research_sprint.py` (Phase 2 driver)
