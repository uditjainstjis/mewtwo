# Experiment Card 16 — Rank-Scaling Adapter Zoo (67 adapters released)

PKB wrapper around `RESEARCH_HISTORY/08_rank_scaling_zoo.md` and `RESEARCH_HISTORY/96_ADAPTERS_RELEASED.md`.

## 1. Research question
Validate the Code Paradox and rank-effectiveness across a 67-adapter zoo on smaller bases.

## 2. Dataset
- HumanEval ($n=50$ rank-scaling).
- 4-benchmark grid spot-check on the strongest/weakest adapters.

## 3. Model

### Nemotron-Mini-4B-Instruct (30 adapters)
| Domain | Method | Ranks |
|---|---|---|
| code | SFT | {1, 2, 8, 128, 1024, 3072} |
| math | SFT | {1, 2, 8, 128, 1024, 3072} |
| math | DPO | {1, 2, 8, 128, 1024, 3072} |
| science | SFT | {1, 2, 8, 128, 1024, 3072} |
| merged | DARE-sparsified | {1, 2, 8, 128, 1024, 3072} |

### Qwen family (37 adapters)
- Bases: Qwen2.5-{0.5B, 1.5B, 7B, 14B, 14B-instruct}, Qwen3.5-{0.8B, 2B, 4B, 9B, 27B}
- Multi-rank ablations on each, plus router and shared-projection-B prototypes.

## 4. Evaluation
- See `15_code_paradox_replication.md` for the rank-scaling spot-check.

## 5. Results
- Rank does not rescue code-on-code regression even at $r=1024$ (see card 15).
- Per-rank full benchmark grid: **NOT consolidated** — only spot-check at $n=50$ HumanEval.

## 6. Negatives + caveats
- Per-rank evaluation grid is incomplete. Only `code_paradox_rank_scaling.json` provides comparable numbers across ranks.
- DARE-sparsified merges are released but not evaluated head-to-head against single-best.

## 7. Artifact map
PRIMARY:
- `adapters/small_models_zoo/from_hf_kaggle/nemotron_4b_*` (30 adapters)
- `adapters/small_models_zoo/from_hf_kaggle/qwen{2.5,3.5}_*` (37 adapters)
- `adapters/small_models_zoo/from_hf_kaggle/family_summary.json`
- `adapters/published/{math,code,science,merged}_README.md`
- `synapta_src/scripts/publish_adapters.py`, `optimize_kaggle.py`, `optimize_independent_kaggle.py`

SECONDARY:
- `RESEARCH_HISTORY/08_rank_scaling_zoo.md`
- `RESEARCH_HISTORY/96_ADAPTERS_RELEASED.md`
