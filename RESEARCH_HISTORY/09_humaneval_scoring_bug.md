# HumanEval Scoring Bug — Discovery and Correction

**Date:** 2026-05-02
**Source artefacts:**
- `synapta_src/overnight_scripts/run_humaneval_n164.py` (corrected harness)
- `synapta_src/overnight_scripts/rescore_humaneval.py` (rescore from saved JSONL)
- `results/overnight/qa_pairs/humaneval_full_base.jsonl` (raw outputs n=164)
- `results/overnight/qa_pairs/humaneval_full_format_guard.jsonl` (raw outputs n=164)
- `results/overnight/qa_pairs/humaneval_full_base_rescored.jsonl` (v2 scored)
- `results/overnight/qa_pairs/humaneval_full_format_guard_rescored.jsonl` (v2 scored)
- `docs/findings/humaneval_n164.md`
- `docs/findings/humaneval_n164_critical.md`

## The two bugs

### Bug 1: import-stripping
The original `extract_code` function read fenced code blocks but discarded the prompt's pre-def header (e.g., `from typing import List`). When the model returned a complete fenced solution, the header was thrown away and the resulting Python file failed at runtime with `NameError: name 'List' is not defined`.

### Bug 2: indent-stripping
When the model returned a code block containing only the function body (e.g., `    return [...]`), the harness called `body.strip()` which removed the leading 4-space indent. The body then parsed as a top-level statement when prepended to the prompt's `def f():`, producing an `IndentationError`.

## Effect on absolute pass@1 numbers

Reference: `results/overnight/qa_pairs/humaneval_rescored_summary.json`

| Mode | v1 (buggy) | v2 (fixed) | Change |
|---|---|---|---|
| Base Nemotron-30B | 22.0\% | **56.1\%** | +34 pp absolute |
| Format Guard | 70.7\% | **73.2\%** | +2.5 pp absolute |
| Delta (FG − base) | +48.7 (inflated) | **+17.1 (defensible)** | -31.6 pp |

## Why does the delta shrink?
The buggy scorer disproportionately penalised base outputs:
- Base outputs more verbose chain-of-thought before code, hitting both bugs.
- Format Guard outputs are constrained to inside fenced code blocks via the code-block-parity router check, so they were already in cleaner form even under the buggy scorer.

After the fix, base "catches up" but Format Guard remains meaningfully better. **The +17.1 pp delta is the robust gap** — what's actually attributable to the routing strategy, not extraction artefacts.

## What we kept and what we corrected

### Kept
- "Format Guard improves HumanEval over base" — yes, by **+17.1 points at n=164**.
- McNemar paired test: $\chi^2 = 15.68$, $p < 10^{-3}$ on the corrected outcomes.

### Corrected in deck and pitch
- Replaced n=25 deck numbers with n=164 numbers.
- Softened the "+24 points" headline to "+17 points," emphasising 6.5× sample size at publication-credible $n$.
- Updated absolute numbers: 56\% → 73\% (not 24\% → 48\%).

## Methodology lesson

A 30-pp absolute floor movement from extraction-bug fixes alone is itself a methodology finding worth disclosure. **Many published HumanEval numbers for $\sim$30B-class models are evaluated with greedy chain-of-thought generation + body-only completion extraction**, both of which are vulnerable to the same bugs. Practitioners replicating from this codebase should use the v2 corrected extraction.

## Files
- `synapta_src/overnight_scripts/run_humaneval_n164.py`
- `synapta_src/overnight_scripts/rescore_humaneval.py`
- `results/overnight/qa_pairs/humaneval_full_*_rescored.jsonl`
- `docs/findings/humaneval_n164.md`
- `docs/findings/humaneval_n164_critical.md`
- `docs/findings/humaneval_statistical_analysis.md`
