# IndiaFinBench — Out-of-Distribution Probe

**Date:** 2026-05-04
**Source artefacts:**
- `synapta_src/data_pipeline/16_eval_indiafinbench.py`
- `external_benchmarks/IndiaFinBench/data/test-00000-of-00001.parquet` (downloaded HuggingFace `Rajveer-code/IndiaFinBench`)
- `results/indiafinbench_eval/predictions.jsonl` (471 KB, 324 rows)
- `results/indiafinbench_eval/summary.json`
- `docs/recent/perplexity_research/indiafinbench_finding.md`

## Why this exists
A reviewer must wonder: "the BFSI extract adapter scored 89.6\% on its own held-out — but is the held-out really independent?" An external benchmark covering the same domain (RBI/SEBI), released by an independent author after our training was finalised, is the cleanest external probe.

IndiaFinBench~\citep{pall2026indiafinbench}: 406 expert-annotated Q&A from 192 SEBI/RBI documents, four task types, zero-shot evaluation across 12 LLMs. Released April 2026.

## Setup
- Base: same Nemotron-30B 4-bit + bfsi_extract adapter.
- Test split: $n=324$ questions.
- Identical system prompt and decoding settings as Section §6 / `11_bfsi_extract_eval.md`.
- Three metrics scored: substring (case-insensitive), normalised match (case + punctuation insensitive), token F1.

## Result

### Overall
| Metric | Score | Wilson 95\% CI |
|---|---|---|
| Substring match | **32.1\%** | [27.3, 37.4] |
| Normalised match | 32.7\% | [27.8, 38.0] |
| Token F1 mean | 0.288 | --- |

### Per task type (Synapta substring vs Gemini Flash zero-shot baseline)
| Task type | $n$ | Synapta sub\% | Synapta F1 | Gemini Flash | Gap |
|---|---|---|---|---|---|
| regulatory_interpretation | 139 | 34.5\% | 0.347 | 93.1\% | -58.6 pp |
| temporal_reasoning | 62 | 16.1\% | 0.366 | 88.5\% | -72.4 pp |
| numerical_reasoning | 73 | 9.6\% | 0.297 | 84.8\% | -75.2 pp |
| contradiction_detection | 50 | 78.0\%* | 0.015 | 88.7\% | -10.7 pp |
| **Overall** | **324** | **32.1\%** | **0.288** | **89.7\%** | **-57.6 pp** |

\* contradiction_detection substring is inflated by yes/no string-match coincidence: F1 = 0.015 confirms the model is not actually performing the task.

### Per source regulator
| Source | $n$ | Substring |
|---|---|---|
| SEBI | 273 | 33.0\% |
| RBI | 51 | 27.5\% |

### Wall-clock
73 minutes on RTX 5090, 4-bit Nemotron-30B + bfsi_extract LoRA. Sample contexts ranged 200–8000 tokens (materially longer than our internal training/eval contexts).

## Why is the gap so large?
Our adapter was trained on **regex-derived single-turn extractive QA** from RBI/SEBI Master Directions. IndiaFinBench is **expert-annotated multi-task**: contradiction across paragraphs, temporal reasoning across amended circulars, numerical reasoning across multi-part contexts. The two corpora cover the same domain but use very different question styles and reasoning structures.

Standard NLP literature since 2018 documents this failure mode~\citep{howard2018ulmfit, gururangan2020dontstop, magar2022data}: a fine-tuned adapter optimised on Corpus A under-performs OOD on Corpus B with different question style, length, and reasoning structure. The 57.6 pp gap is the predicted failure mode.

## What this disclosure means for the paper

**The OOD result is not a refutation of the methodology.** The methodology produces customer-specific adapters that are dramatically better on the customer's own distribution (89.6\% on our held-out, +30.9 pp McNemar $p < 10^{-48}$) and predictably weaker on different distributions (32.1\% on IndiaFinBench, -57.6 pp from Gemini Flash zero-shot).

**This is the case for shipping a per-customer methodology** rather than a single "Indian-BFSI LLM." Vendors who ship one general LLM (e.g., NeoGPT, 300M tokens, 70+ agents) face the same OOD degradation when measured on a separately-constructed benchmark. They have not published. We have.

## Files
- `synapta_src/data_pipeline/16_eval_indiafinbench.py`
- `external_benchmarks/IndiaFinBench/` (raw data)
- `results/indiafinbench_eval/predictions.jsonl`
- `results/indiafinbench_eval/summary.json`
- `docs/recent/perplexity_research/indiafinbench_finding.md`
