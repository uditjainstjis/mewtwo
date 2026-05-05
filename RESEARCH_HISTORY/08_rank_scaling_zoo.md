# Rank-Scaling Adapter Zoo (67 adapters)

**Source artefacts:**
- `adapters/small_models_zoo/from_hf_kaggle/nemotron_4b_*/` (30 adapters)
- `adapters/small_models_zoo/from_hf_kaggle/qwen3.5_*/` (37 adapters)
- `synapta_src/scripts/publish_adapters.py`

## Inventory

### Nemotron-Mini-4B-Instruct (30 adapters)
| Domain | Methods | Ranks |
|---|---|---|
| code | SFT | 1, 2, 8, 128, 1024, 3072 |
| math | SFT, DPO | 1, 2, 8, 128, 1024, 3072 (each) |
| science | SFT | 1, 2, 8, 128, 1024, 3072 |
| merged | DARE-sparsified | 1, 2, 8, 128, 1024, 3072 |

Total: 6 ranks × (1 code + 2 math + 1 science + 1 merged) = 30 adapters.

### Qwen-3.5-* family (37 adapters)
- Qwen2.5-0.5B / 1.5B / 7B / 14B / 14B-instruct
- Qwen3.5-0.8B / 2B / 4B / 9B / 27B
- Multi-rank ablations on each, plus router and shared-projection-B prototypes.

## Why this zoo exists

Three motivations:
1. **Cross-base validation of the Code Paradox** (`07_code_paradox_replication.md`).
2. **Rank-effectiveness ablation**: does $r=1024$ recover the code-on-code regression? Answer: no (Table in `07_*`).
3. **Composition substrate**: a rich set of adapters across ranks, methods (SFT/DPO/DARE), and domains for future routing experiments.

## Released

Files are organized for HuggingFace and Kaggle release:
- `adapters/published/{math,code,science}_README.md` — public model cards.
- `synapta_src/scripts/publish_adapters.py` — push utility.
- `synapta_src/scripts/optimize_kaggle.py` — packaging.

## Not yet released
- Fine-grained training logs per rank (only the `family_summary.json` is consolidated).
- Per-rank evaluation grid (only $n=50$ rank-scaling sample is reported in `code_paradox_rank_scaling.json`).

## Files
- `adapters/small_models_zoo/from_hf_kaggle/family_summary.json`
- `adapters/published/*_README.md`
