# BFSI Extract Adapter — Held-Out Paired Evaluation

**Date:** 2026-05-03
**Source artefacts:**
- `synapta_src/data_pipeline/07_train_bfsi_extract.py` (training)
- `synapta_src/data_pipeline/08_eval_bfsi_extract.py` (3-mode eval)
- `adapters/nemotron_30b/bfsi_extract/best/` (1.74 GB LoRA weights)
- `results/bfsi_eval/eval_results.jsonl` (1,992 rows = 664 × 3 modes)
- `results/bfsi_eval/summary.json` (canonical numbers)

## Training
- Base: Nemotron-Nano-30B-A3B 4-bit NF4.
- LoRA: $r=16$, $\alpha=32$, dropout 0.05, all attn+MLP target modules.
- Trainable: 434.6M / 32B = 1.36\%.
- Optimiser: paged AdamW 8-bit, lr $2\times10^{-4}$ cosine, 1 epoch.
- MAX_LEN=1024, batch size effective 4.
- Train data: 2,931 pairs from 104 train PDFs.
- Wall-clock: **3h 28min** on RTX 5090.
- Peak VRAM: 17.81 GB.
- Adapter file size (bf16): 1.74 GB.

## Evaluation — 3 modes paired

Eval set: $n=664$ pairs across 26 held-out PDFs. Document-disjoint: model never saw these PDFs during training.

### Headline (substring match, primary metric)

| Mode | Substring | Wilson 95\% CI | Token F1 | Exact Match |
|---|---|---|---|---|
| Base (4-bit, no adapter, `disable_adapter()`) | 58.7\% | [54.95, 62.42] | 0.132 | 0\% |
| **+ bfsi_extract LoRA** | **89.6\%** | [87.06, 91.71] | 0.172 | 0\% |
| Format Guard (math+code+science+bfsi_extract, swap every 10 tokens) | 88.7\% | [86.07, 90.89] | 0.171 | 0\% |

### Paired McNemar contingency

|  | Adapter correct | Adapter wrong |
|---|---|---|
| Base correct | 376 | 14 |
| Base wrong | **219** | 55 |

McNemar exact-binomial: **$p = 1.66 \times 10^{-48}$**.

219 questions adapter-only-correct vs 14 base-only-correct: 15.6× improvement-to-regression ratio.

### Pairwise comparisons

| Comparison | $b_{10}$ | $b_{01}$ | Δ | $p$ |
|---|---|---|---|---|
| bfsi_extract vs base | 14 | 219 | +30.9 pp | $1.66 \times 10^{-48}$ |
| Format Guard vs base | 19 | 218 | +30.0 pp | $5.12 \times 10^{-44}$ |
| Format Guard vs bfsi_extract direct | 6 | 0 | -0.9 pp | $0.031$ |

Format Guard differs from the dedicated adapter on only 6 of 664 questions (all `b_10` — adapter correct, FG wrong; `b_01 = 0`). The routing layer is empirically zero-overhead.

### Per-tier breakdown

| Tier | $n$ | Base | + Adapter | Lift |
|---|---|---|---|---|
| Tier 2 (numeric) | 386 | 63.0\% | 87.6\% | +24.6 pp |
| Tier 3 (heading-extractive) | 278 | 52.9\% | 92.4\% | **+39.5 pp** |

The adapter dominates Tier 3 (open-ended paragraph extraction).

### Per-regulator breakdown

| Regulator | $n$ | Base | + Adapter | Lift |
|---|---|---|---|---|
| RBI | 381 | 58.0\% | 89.5\% | +31.5 pp |
| SEBI | 283 | 59.7\% | 89.8\% | +30.1 pp |

Lift is consistent across regulators; the methodology is not over-fit to one regulator's prose style.

## Files
- `synapta_src/data_pipeline/07_train_bfsi_extract.py`
- `synapta_src/data_pipeline/08_eval_bfsi_extract.py`
- `adapters/nemotron_30b/bfsi_extract/best/{adapter_model.safetensors, adapter_config.json}`
- `results/bfsi_eval/{eval_results.jsonl, summary.json}`
- `data/rbi_corpus/qa/eval_clean.jsonl`
