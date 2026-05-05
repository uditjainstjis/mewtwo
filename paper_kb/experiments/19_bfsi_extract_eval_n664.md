# Experiment Card 19 — BFSI Extract Adapter, Held-Out Paired Eval ($n=664$)

PKB wrapper around `RESEARCH_HISTORY/11_bfsi_extract_eval.md`.

## 1. Research question
Does a single LoRA adapter trained on the BFSI 3-tier deterministic QA set lift Nemotron-30B substring-match performance on a document-disjoint held-out set?

## 2. Dataset
- 664 paired questions across 26 held-out PDFs (document-disjoint).

## 3. Model
- Base Nemotron-Nano-30B-A3B 4-bit NF4. LoRA $r=16$, $\alpha=32$, all attn+MLP target modules.
- Trainable: 434.6M / 32B = 1.36\%.
- 1 epoch, paged AdamW 8-bit, lr 2e-4 cosine, MAX_LEN=1024.
- Wall-clock: 3h 28min on RTX 5090. Peak VRAM 17.81 GB. Adapter file size 1.74 GB.

## 4. Evaluation
- 3 modes paired: base (no adapter), bfsi_extract direct, Format Guard with 4 adapters (math+code+science+bfsi_extract).
- Substring match (case-insensitive) primary metric. Wilson 95\% CIs. Paired McNemar exact-binomial.

## 5. Results

| Mode | Substring | Wilson 95\% CI | Token F1 |
|---|---:|---|---:|
| Base | 58.7\% | [54.95, 62.42] | 0.132 |
| + bfsi_extract | **89.6\%** | [87.06, 91.71] | 0.172 |
| Format Guard | 88.7\% | [86.07, 90.89] | 0.171 |

Paired McNemar contingency (adapter vs base): $b_{11}=376, b_{10}=14, b_{01}=219, b_{00}=55$. **$p = 1.66 \times 10^{-48}$.**

| Comparison | $b_{10}$ | $b_{01}$ | $\Delta$ | $p$ |
|---|---:|---:|---:|---:|
| adapter vs base | 14 | 219 | $+30.9$ pp | $\mathbf{1.66 \times 10^{-48}}$ |
| FG vs base | 19 | 218 | $+30.0$ pp | $5.12 \times 10^{-44}$ |
| FG vs adapter direct | 6 | 0 | $-0.9$ pp | $0.031$ |

Per-tier: Tier 2 numeric $+24.6$ pp ($n=386$), Tier 3 heading-extractive **$+39.5$ pp** ($n=278$).
Per-regulator: RBI $+31.5$ pp, SEBI $+30.1$ pp.

## 6. Negatives + caveats
- Format Guard differs from dedicated adapter on 6 of 664 questions, all in $b_{10}$ (FG never improves over direct adapter on BFSI).
- See `98_KNOWN_LIMITATIONS_AND_BUGS.md` item H for the BFSI router false-firing on `Section X(Y)(z)` patterns.
- Earlier mid-run snapshot used $n=595$, $p = 6.26 \times 10^{-44}$; the canonical numbers above are the full $n=664$ sweep.

## 7. Artifact map
PRIMARY:
- `synapta_src/data_pipeline/{07_train_bfsi_extract, 08_eval_bfsi_extract}.py`
- `adapters/nemotron_30b/bfsi_extract/best/{adapter_model.safetensors, adapter_config.json}`
- `results/bfsi_eval/{eval_results.jsonl, summary.json}`
- `data/rbi_corpus/qa/eval_clean.jsonl`

SECONDARY: `RESEARCH_HISTORY/11_bfsi_extract_eval.md`
