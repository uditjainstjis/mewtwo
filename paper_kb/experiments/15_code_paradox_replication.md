# Experiment Card 15 — Code Paradox Replication and Robustness

PKB wrapper around `RESEARCH_HISTORY/07_code_paradox_replication.md`.

## 1. Research question
Does the Code Paradox (training on code degrades in-domain code performance, asymmetric positive cross-domain transfer) replicate across base models and adapter ranks?

## 2. Dataset
- Same 4 reasoning benchmarks as Phase 1 (ARC, HumanEval, MATH-500, MBPP).
- Replication on Qwen-3.5-0.8B at $n=50$ (NOT robust) and $n=200$ (in-domain regression robust).
- Rank-scaling on Nemotron-Mini-4B at $n=50$ across ranks {8, 128, 1024}.

## 3. Model
- Base 1: Nemotron-30B (Phase 1, single-adapter benchmarks).
- Base 2: Qwen-3.5-0.8B.
- Base 3: Nemotron-Mini-4B-Instruct (rank ablation).

## 4. Evaluation
- Pass@1 (HumanEval-style) as primary; per-task breakdown.

## 5. Results

### $n=50$ cross-family — NOT robust
| Base | Code adapter | Math adapter | base |
|---|---:|---:|---:|
| Qwen-3.5-0.8B | 16\% | 10\% | 8\% |
| Nemotron-Mini-4B | 10\% | 8\% | 12\% |

### $n=200$ Qwen-3.5-0.8B — robust in-domain regression
| Mode | Acc |
|---|---:|
| base | 15.0\% (30/200) |
| math adapter | 16.0\% (32/200) |
| code adapter | **12.0\%** (24/200) — code-on-code regression replicated |

### Rank-scaling Nemotron-Mini-4B ($n=50$ HumanEval)
| Rank | math | code |
|---|---:|---:|
| 8 | 2\% | 4\% |
| 128 | 8\% | 8\% |
| 1024 | 4\% | 10\% |

Even at $r=1024$, the code adapter does not exceed the base on a code task.

## 6. Negatives + caveats
- **Cross-family overclaim rolled back:** the original "Code Paradox replicates across 3 bases at $n=50$" claim is retracted. Robust claim is in-domain regression only at $n=200$ Qwen.
- **No $n=200+$ cross-family positive transfer:** only the Nemotron-30B Phase 1 result supports the asymmetric positive transfer (code → math, math → code). See `missing_artifacts.md` item 4.

## 7. Artifact map
PRIMARY:
- `results/overnight/qa_pairs/code_paradox_qwen_n200_summary.json`
- `results/overnight/qa_pairs/code_paradox_summary.json` (n=50)
- `results/bfsi_swarm_extras/code_paradox_rank_scaling.json`
- `synapta_src/overnight_scripts/run_code_paradox_*.py`

SECONDARY:
- `RESEARCH_HISTORY/07_code_paradox_replication.md`
- `docs/findings/code_paradox_replication.md`
