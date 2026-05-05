# Experiment Card 17 — HumanEval Scoring Bug Discovery and Correction

PKB wrapper around `RESEARCH_HISTORY/09_humaneval_scoring_bug.md`.

## 1. Research question
Diagnose why the v1 HumanEval pass@1 numbers (n=25 deck claim, n=164 v1 run) are anomalously low; correct extraction to make pass@1 paper-safe.

## 2. Dataset
- HumanEval $n=164$ (full test set), saved JSONL outputs from prior runs.
- Re-scored from raw saved completions; no new generation needed.

## 3. Bugs identified

### Bug 1: import-stripping
`extract_code` returned only the contents of fenced code blocks, dropping the prompt's pre-def header (e.g., `from typing import List`). Caused `NameError` at runtime.

### Bug 2: indent-stripping
When the model returned a body-only fenced block (e.g., `    return [...]`), `body.strip()` removed the leading 4-space indent. The body then parsed as a top-level statement when prepended to `def f():`, producing `IndentationError`.

## 4. Correction
Both bugs patched. Re-ran the SAME saved completions through corrected extraction; produced `*_rescored.jsonl`.

## 5. Results

| Mode | v1 (buggy) | v2 (fixed) | Change |
|---|---:|---:|---|
| Base Nemotron-30B | 22.0\% | **56.1\%** | $+34$ pp absolute |
| Format Guard | 70.7\% | **73.2\%** | $+2.5$ pp absolute |
| $\Delta$ (FG − base) | $+48.7$ (inflated) | **$+17.1$ (defensible)** | $-31.6$ pp |

The shrinking delta after fix reflects that v1 disproportionately under-counted base outputs (more verbose chain-of-thought, hit both bugs more often). After fix, base catches up but FG remains meaningfully better.

## 6. Negatives + caveats
- **All v1 HumanEval numbers anywhere in the codebase are stale.** Use v2 only.
- **Methodology-finding worth flagging:** many published HumanEval numbers for $\sim$30B-class models likely have similar extraction issues. A 30 pp absolute floor movement from extraction fixes alone is a publication-worthy methodology note.

## 7. Artifact map
PRIMARY:
- `synapta_src/overnight_scripts/run_humaneval_n164.py`
- `synapta_src/overnight_scripts/rescore_humaneval.py`
- `results/overnight/qa_pairs/humaneval_full_*_rescored.jsonl`
- `results/overnight/qa_pairs/humaneval_rescored_summary.json`

SECONDARY:
- `docs/findings/humaneval_n164.md`
- `docs/findings/humaneval_n164_critical.md`
- `docs/findings/humaneval_statistical_analysis.md`
- `RESEARCH_HISTORY/09_humaneval_scoring_bug.md`
