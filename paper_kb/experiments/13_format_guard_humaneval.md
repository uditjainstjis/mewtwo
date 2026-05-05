# Experiment Card 13 — Format Guard on HumanEval ($n=164$ paired)

PKB wrapper around `RESEARCH_HISTORY/04_format_guard.md` and `RESEARCH_HISTORY/09_humaneval_scoring_bug.md`. Open those for full narrative.

## 1. Research question
Does a token-level adapter-routing logits-processor (Format Guard) outperform the base model on the full HumanEval test set, and does it exceed the best single adapter?

## 2. Dataset
- HumanEval, full test set $n=164$, paired evaluation (same seeds).

## 3. Model
- Base Nemotron-Nano-30B-A3B 4-bit.
- 4 adapters loaded simultaneously into VRAM (math, code, science, bfsi_extract).
- Format Guard `LogitsProcessor` swaps adapter every $K=10$ generated tokens via regex over the decoded suffix.
- Paradox-aware router: code-like text → math adapter; math notation → code adapter; BFSI terms → bfsi_extract; default → code (generic reasoner).
- All swaps via `model.set_adapter(target)` ($O(1)$ pointer flip).

## 4. Evaluation
- Pass@1 with **v2 corrected** code-extraction harness (import-stripping + indent-stripping bugs fixed; see `17_humaneval_scoring_bug.md`).
- Wilson 95\% CIs on the binary outcome.
- Paired McNemar exact-binomial $\chi^2$ on the 4-cell contingency.

## 5. Results

| Mode | Pass@1 | Wilson 95\% CI | (v1 inflated, do not cite) |
|---|---:|---|---|
| Base Nemotron-30B | **56.1\%** (92/164) | [48.4, 63.5] | (22.0\%) |
| Format Guard | **73.2\%** (120/164) | [65.9, 79.4] | (70.7\%) |

**$\Delta = +17.1$ pp.** Wilson CIs non-overlapping.

Paired McNemar contingency:
| | FG passes | FG fails |
|---|---:|---:|
| Base passes | 81 | 11 |
| Base fails | 39 | 33 |

McNemar $\chi^2 = (39-11)^2 / (39+11) = 15.68$, $p < 10^{-3}$.

Per-category: math_arith $+33$ pp, strings $+15$ pp, list_ops $-7$ pp.

Multi-benchmark grid (FG vs base): ARC $+11$ pp (matches best single), MATH-500 $+14.5$ pp (matches best single), HumanEval **$+17.1$ pp** (exceeds every single expert), MBPP $-3$ pp (format-rigid regression).

## 6. Negatives + caveats
- Single 30B base for headline. Multi-base FG replication is `missing_artifacts.md` item 6.
- Heuristic regex router. Learned router is natural follow-up.
- MBPP $-3$ pp regression on format-rigid tasks; debounce or learned router would likely close it (untried; see `98_KNOWN_LIMITATIONS_AND_BUGS.md` item H).

## 7. Artifact map
PRIMARY:
- `results/overnight/qa_pairs/humaneval_full_format_guard.jsonl` (raw)
- `results/overnight/qa_pairs/humaneval_full_format_guard_rescored.jsonl` (v2 scored)
- `results/overnight/qa_pairs/humaneval_full_base_rescored.jsonl`
- `results/overnight/qa_pairs/humaneval_rescored_summary.json`
- `synapta_src/data_pipeline/08_eval_bfsi_extract.py` (FG implementation)
- `synapta_src/overnight_scripts/run_humaneval_n164.py`, `rescore_humaneval.py`
- `docs/findings/humaneval_n164.md`, `humaneval_statistical_analysis.md`

SECONDARY:
- `RESEARCH_HISTORY/04_format_guard.md`
- `RESEARCH_HISTORY/09_humaneval_scoring_bug.md`
