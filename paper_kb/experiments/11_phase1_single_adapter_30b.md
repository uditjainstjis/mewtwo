# Experiment Card 11 — Phase-1 Single-Adapter Benchmarks on Nemotron-30B

This card is a PKB-format wrapper around `RESEARCH_HISTORY/02_phase1_single_adapter.md` (parent repo). Open that file for full narrative + tables.

## 1. Research question
Establish single-adapter baselines for math, code, science adapters trained on Nemotron-Nano-30B-A3B (4-bit), before any composition / routing.

## 2. Dataset
- ARC-Challenge ($n=100$), HumanEval ($n=100$, v1 buggy scoring), MATH-500 ($n=200$), MBPP ($n=100$).

## 3. Model
- Base Nemotron-Nano-30B-A3B 4-bit NF4. LoRA $r=16$, $\alpha=32$, 1.36\% trainable. paged AdamW 8-bit, lr 2e-4 cosine, 1 epoch. RTX 5090 32 GB.

## 4. Evaluation
- ARC accuracy, HumanEval pass@1 (v1), MATH-500 exact-match, MBPP pass@1.

## 5. Results

| Adapter | ARC | HumanEval | MATH-500 | MBPP |
|---|---:|---:|---:|---:|
| base | 20.0 | 50.0 | 41.5 | 8.0 |
| math | 23.0 | **60.0** | 50.5 | 2.0 |
| code | **31.0** | 27.0 | **56.0** | 6.0 |
| science | 21.0 | 1.0 | 55.0 | 0.0 |
| merged (DARE/TIES) | 19.0 | 34.0 | 56.0 | 0.0 |

## 6. Negatives + caveats
- HumanEval scoring bug at v1 — see `17_humaneval_scoring_bug.md`.
- Cross-family Code Paradox $n=50$ overclaim rolled back — see `15_code_paradox_replication.md`.

## 7. Artifact map
PRIMARY: `results/nemotron/master_results.json`
SECONDARY: `RESEARCH_HISTORY/02_phase1_single_adapter.md`, `docs/MASTER_KNOWLEDGE_BASE.md` §4.3
