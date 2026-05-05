# Code Paradox — Replication and Robustness

**Date:** 2026-04-22 to 2026-04-30
**Source artefacts:**
- `results/overnight/qa_pairs/code_paradox_summary.json` (n=50 cross-base, NOT robust)
- `results/overnight/qa_pairs/code_paradox_qwen_n200_summary.json` (n=200 Qwen-3.5-0.8B, ROBUST)
- `results/bfsi_swarm_extras/code_paradox_rank_scaling.json` (Nemotron-Mini-4B rank ablation)
- `docs/findings/code_paradox_replication.md`
- `synapta_src/overnight_scripts/run_code_paradox_*.py`

## Goal
Test whether the Code Paradox (Nemotron-30B Phase 1) replicates across base models and rank choices, to distinguish "Nemotron quirk" from "genuine PEFT phenomenon."

## n=50 cross-family — NOT robust

Initial smoke runs at $n=50$ on Qwen-3.5-0.8B and Nemotron-Mini-4B-Instruct showed mixed signals:

| Base | $n$ | base | math adapter | code adapter |
|---|---|---|---|---|
| Qwen-3.5-0.8B | 50 | 8\% | 10\% | 16\% |
| Nemotron-Mini-4B | 50 | 12\% | 8\% | 10\% |

The Qwen result at $n=50$ initially looked like Code Paradox replication ($+8$ pp from code adapter), but the small sample size made any direction unreliable. **This deck claim was rolled back** (see `98_KNOWN_LIMITATIONS_AND_BUGS.md`).

## n=200 Qwen-3.5-0.8B — ROBUST in-domain regression

Followed up with $n=200$ on Qwen-3.5-0.8B:

| Mode | Acc | Source |
|---|---|---|
| base | 15.0\% (30/200) | `code_paradox_qwen_n200_summary.json` |
| math adapter | 16.0\% (32/200) | same |
| code adapter | 12.0\% (24/200) | same |

**Pattern preserved:** code adapter scores **below** base on a Python code benchmark; math adapter slightly exceeds base. The "code-on-code regression" replicates at smaller scale and on a different base architecture.

We do **not** claim cross-base replication of the asymmetric *positive* transfer (code → math, math → code) at this n.

## Rank-scaling ablation (Nemotron-Mini-4B-Instruct)

| Rank | math acc | code acc |
|---|---|---|
| 8 | 2\% | 4\% |
| 128 | 8\% | 8\% |
| 1024 | 4\% | 10\% |

Rank does not rescue the code adapter on a code benchmark. Even at rank 1024 the code adapter scores only 10\% on the same task it was trained for, on a 4B base where the $n=50$ baseline was 8\%.

## Verdict
**Robust claim:** training on code degrades in-domain code performance, replicated on Nemotron-30B ($n=164$ HumanEval), Qwen-3.5-0.8B ($n=200$), and Nemotron-Mini-4B (rank ablation).

**Not-yet-supported claim:** the asymmetric *positive* cross-domain transfer (code → math, math → code) replicates across bases. Only the Nemotron-30B Phase 1 result supports this at n=100/200.

## Files
- `results/overnight/qa_pairs/code_paradox_qwen_n200_summary.json`
- `results/bfsi_swarm_extras/code_paradox_rank_scaling.json`
- `synapta_src/overnight_scripts/run_code_paradox_replication.py`
- `synapta_src/overnight_scripts/run_code_paradox_rank_scaling.py`
- `docs/findings/code_paradox_replication.md`
