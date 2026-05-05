# Adapters Released — Complete Inventory

| # | Adapter | Base | Rank | Method | Domain | Path |
|---|---|---|---|---|---|---|
| 1 | math (30B) | Nemotron-Nano-30B-A3B | 16 | LoRA SFT | math | `adapters/nemotron_30b/math/best/` |
| 2 | code (30B) | Nemotron-Nano-30B-A3B | 16 | LoRA SFT | code | `adapters/nemotron_30b/code/best/` |
| 3 | science (30B) | Nemotron-Nano-30B-A3B | 16 | LoRA SFT | science | `adapters/nemotron_30b/science/best/` |
| 4 | bfsi_extract | Nemotron-Nano-30B-A3B | 16 | LoRA SFT | RBI/SEBI extractive QA | `adapters/nemotron_30b/bfsi_extract/best/` |
| 5 | bfsi_recall | Nemotron-Nano-30B-A3B | 16 | LoRA SFT | RBI/SEBI no-context recall | `adapters/nemotron_30b/bfsi_recall/best/` |
| 6 | gc_router | Nemotron-Nano-30B-A3B | 16 | LoRA SFT | gating/routing | `adapters/nemotron_30b/gc_router/` |

## Nemotron-Mini-4B-Instruct zoo (rank-scaling ablation, 30 adapters)

| Domain | Method | Ranks |
|---|---|---|
| code | SFT | 1, 2, 8, 128, 1024, 3072 |
| math | SFT | 1, 2, 8, 128, 1024, 3072 |
| math | DPO | 1, 2, 8, 128, 1024, 3072 |
| science | SFT | 1, 2, 8, 128, 1024, 3072 |
| merged | DARE-sparsified | 1, 2, 8, 128, 1024, 3072 |

Path: `adapters/small_models_zoo/from_hf_kaggle/nemotron_4b_<domain>_<method>_rank<rank>/`.

## Qwen-3.5-* family (37 adapters)

| Base | Adapters per base |
|---|---|
| Qwen2.5-0.5B | math, code, science, legal, medical (multi-rank ablations) |
| Qwen2.5-1.5B | same |
| Qwen2.5-7B | same |
| Qwen2.5-14B / 14B-instruct | same |
| Qwen3.5-0.8B | math, code, science (multi-rank) |
| Qwen3.5-2B / 4B / 9B / 27B | partial domain coverage |

Plus router and `shared_projection_B.pt` prototypes.

Path: `adapters/small_models_zoo/from_hf_kaggle/qwen3.5_<size>/` and `qwen2.5_<size>/`.

## LoRI MoE (5 adapters on Qwen-2.5-1.5B)

| Adapter | Path |
|---|---|
| lori-qwen2.5-1.5b-code | `adapters/lori_moe/lori-qwen2.5-1.5b-code/` |
| lori-qwen2.5-1.5b-legal | `adapters/lori_moe/lori-qwen2.5-1.5b-legal/` |
| lori-qwen2.5-1.5b-math | `adapters/lori_moe/lori-qwen2.5-1.5b-math/` |
| lori-qwen2.5-1.5b-medical | `adapters/lori_moe/lori-qwen2.5-1.5b-medical/` |
| lori-qwen2.5-1.5b-science | `adapters/lori_moe/lori-qwen2.5-1.5b-science/` |

## Total

**72+ adapters released** spanning 8+ base models and 6+ ranks.

## Public release tooling
- `synapta_src/scripts/publish_adapters.py` — push to HuggingFace
- `synapta_src/scripts/optimize_kaggle.py` — Kaggle packaging
- `synapta_src/scripts/optimize_independent_kaggle.py` — independent variant
- `adapters/published/{math,code,science,merged}_README.md` — public model cards
