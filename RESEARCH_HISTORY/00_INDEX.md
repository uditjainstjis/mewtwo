# Synapta — Research History Index

A chronological + topical map of every research thread, with pointers to canonical artifacts.
This directory is the single source of truth for "what did we run, when, with what numbers."

## How to navigate
- **Chronological** entry points: `01_TIMELINE.md`
- **By topic**: `02_*..14_*` (one file per research thread)
- **Canonical numbers**: `99_HEADLINE_NUMBERS.md` (the table reviewers will scan)
- **Negative results / honest gaps**: `98_KNOWN_LIMITATIONS_AND_BUGS.md`

## Research threads (in roughly the order they were run)

| # | Topic | File | Status | Headline finding |
|---|---|---|---|---|
| 1 | Single-adapter Phase-1 benchmarks (Nemotron-30B) | `02_phase1_single_adapter.md` | done | Code Paradox baseline |
| 2 | Static composition (DARE/TIES/linear) | `03_static_composition_failure.md` | done | $+0.0$ pp on $n=45$ probe |
| 3 | Token-level routing / Format Guard | `04_format_guard.md` | done | $+17.1$ pp HumanEval $p<0.001$ |
| 4 | Cold-swap latency profiling | `05_cold_swap_latency.md` | done | 315.9 ms NVMe; $O(1)$ warm |
| 5 | LoRI MoE (composite routing) | `06_lori_moe.md` | done | composite < single best on GSM8K |
| 6 | Code Paradox replication ($n=200$) | `07_code_paradox_replication.md` | done | robust at 30B; $n=50$ flukes rolled back |
| 7 | Multi-rank ablation zoo | `08_rank_scaling_zoo.md` | done | 30 adapters Nemotron-Mini-4B + 37 Qwen-3.5-0.8B |
| 8 | HumanEval scoring-bug discovery + rescore | `09_humaneval_scoring_bug.md` | done | $+30$ pp absolute floor moved |
| 9 | BFSI corpus + 3-tier QA pipeline | `10_bfsi_pipeline.md` | done | 2931 train + 664 eval, doc-disjoint |
| 10 | BFSI extract adapter (n=664 paired) | `11_bfsi_extract_eval.md` | done | $+30.9$ pp McNemar $p=1.66\times10^{-48}$ |
| 11 | BFSI recall adapter (n=214 paired F1) | `12_bfsi_recall_eval.md` | done | $+38.4\%$ rel. Wilcoxon $p=1.5\times10^{-16}$ |
| 12 | IndiaFinBench OOD probe (n=324) | `13_indiafinbench_ood.md` | done | 32.1\% [27.3, 37.4] vs Gemini Flash 89.7\% |
| 13 | Synapta Benchmark v1 (n=60 paired, 3 modes) | `14_benchmark_v1_release.md` | done | $+10$ pp adapter, FG identical to direct |
| 14 | Frontier comparison (Synapta vs Claude, n=15) | `15_frontier_comparison.md` | done | Synapta substring 87\%, Claude 7-27\% |

## Cross-cutting reference
- `99_HEADLINE_NUMBERS.md` — every quotable number with primary source
- `98_KNOWN_LIMITATIONS_AND_BUGS.md` — bug history, methodology corrections, retracted claims
- `97_REPRODUCIBILITY.md` — exact seeds, software versions, hardware, scripts
- `96_ADAPTERS_RELEASED.md` — every adapter with rank, base, training data, where the weights live

## Paper(s)
- `/paper/neurips_2026/synapta_neurips2026.tex` — NeurIPS 2026 Main Track / Use-Inspired submission
- `/paper/neurips_2026/SUBMISSION_GUIDE.md` — deadline checklist + compile instructions
