# Experiment Card 21 — Synapta on IndiaFinBench OOD ($n=324$)

PKB wrapper around `RESEARCH_HISTORY/13_indiafinbench_ood.md`.

## 1. Research question
External, contemporary public Indian-BFSI benchmark probe: how does the customer-corpus-trained `bfsi_extract` adapter score on a third-party benchmark released after our training was finalised?

## 2. Dataset
- IndiaFinBench~\citep{pall2026indiafinbench}: 406 expert-annotated Q&A from 192 SEBI/RBI documents, 4 task types (regulatory_interpretation, numerical_reasoning, contradiction_detection, temporal_reasoning).
- Test split $n=324$.
- Released April 2026; published top score (Gemini 2.5 Flash zero-shot) 89.7\%.

## 3. Model
- Same Nemotron-30B 4-bit + bfsi_extract from card 19. Identical decoding settings.

## 4. Evaluation
- Three metrics: substring (case-insensitive), normalised match (case+punctuation insensitive), token F1.
- Wilson 95\% CIs.

## 5. Results

### Overall ($n=324$)
| Metric | Score | Wilson 95\% CI |
|---|---:|---|
| Substring | **32.1\%** | [27.3, 37.4] |
| Normalised match | 32.7\% | [27.8, 38.0] |
| Token F1 mean | 0.288 | --- |

### Per task type (Synapta vs published Gemini Flash zero-shot)
| Task type | $n$ | Synapta sub\% | Synapta F1 | Gemini Flash | Gap |
|---|---:|---:|---:|---:|---:|
| regulatory_interpretation | 139 | 34.5\% | 0.347 | 93.1\% | $-58.6$ pp |
| temporal_reasoning | 62 | 16.1\% | 0.366 | 88.5\% | $-72.4$ pp |
| numerical_reasoning | 73 | 9.6\% | 0.297 | 84.8\% | $-75.2$ pp |
| contradiction_detection | 50 | 78.0\%$^*$ | 0.015 | 88.7\% | $-10.7$ pp |
| **Overall** | **324** | **32.1\%** | **0.288** | **89.7\%** | $-57.6$ pp |

$^*$ contradiction_detection 78\% substring is an artefact: gold = "yes"/"no" coincidentally appears as substring in verbose outputs. F1 = 0.015 confirms the model is not actually performing the task.

By regulator: SEBI 33.0\% ($n=273$), RBI 27.5\% ($n=51$).

Wall-clock: 73 min on RTX 5090. Sample contexts ranged 200–8000 tokens (longer than internal training contexts).

## 6. Negatives + caveats
- **The 57.6 pp OOD gap is the predicted failure mode** of fine-tuned adapters under distribution shift (`liang2023helm`, `magar2022data`).
- **The interpretation in the paper:** this gap is the central evidence FOR per-customer training, not a refutation of the methodology. A single domain-fine-tuned model cannot transfer to differently-styled questions in the same domain.
- **Comparison rules:** Synapta 89.6\% on its own held-out and Gemini 89.7\% on IndiaFinBench are NOT directly comparable; the apples-to-apples number for Synapta on IndiaFinBench is 32.1\%.

## 7. Artifact map
PRIMARY:
- `synapta_src/data_pipeline/16_eval_indiafinbench.py`
- `external_benchmarks/IndiaFinBench/data/test-00000-of-00001.parquet`
- `results/indiafinbench_eval/{predictions.jsonl, summary.json}`

SECONDARY:
- `RESEARCH_HISTORY/13_indiafinbench_ood.md`
- `docs/recent/perplexity_research/indiafinbench_finding.md`
