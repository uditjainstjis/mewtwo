---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: task_type
    dtype: string
  - name: difficulty
    dtype: string
  - name: source
    dtype: string
  - name: context
    dtype: string
  - name: question
    dtype: string
  - name: reference_answer
    dtype: string
  - name: source_document
    dtype: string
  splits:
  - name: test
    num_bytes: 296135
    num_examples: 324
  - name: dev
    num_bytes: 73010
    num_examples: 82
  download_size: 165242
  dataset_size: 369145
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
  - split: dev
    path: data/dev-*
license: cc-by-4.0
task_categories:
- question-answering
- text-classification
language:
- en
tags:
- finance
- legal
- regulatory
- india
- benchmark
- llm-evaluation
- sebi
- rbi
pretty_name: IndiaFinBench
size_categories:
- n<1K
---

# IndiaFinBench

### An Evaluation Benchmark for Large Language Model Performance on Indian Financial Regulatory Text

**Rajveer Singh Pall** · Gyan Ganga Institute of Technology and Sciences, Jabalpur, India

[![arXiv](https://img.shields.io/badge/arXiv-2604.19298-b31b1b.svg)](https://arxiv.org/abs/2604.19298)
[![GitHub](https://img.shields.io/badge/GitHub-IndiaFinBench-181717?logo=github)](https://github.com/rajveerpall/IndiaFinBench)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/Rajveer-code/IndiaFinBench)

---

| 406 QA Pairs | 192 Source Documents | 4 Task Types | 12 Models Evaluated |
|:---:|:---:|:---:|:---:|
| Expert-annotated | SEBI · RBI · 1992–2026 | REG · NUM · CON · TMP | Zero-shot, full benchmark |

---

## Overview

IndiaFinBench is, to our knowledge, the **first publicly available evaluation benchmark** for assessing large language model performance on Indian financial regulatory text. Existing financial NLP benchmarks — FinQA, ConvFinQA, FinanceBench, FLUE — draw exclusively from Western corpora (SEC filings, US earnings reports, English-language financial news), leaving regulatory reasoning outside the Western context unmeasured.

IndiaFinBench fills this gap with **406 expert-annotated question-answer pairs** drawn from **192 primary source documents** from the Securities and Exchange Board of India (SEBI) and the Reserve Bank of India (RBI), spanning 1992 to 2026. The benchmark is designed to make the specific reasoning challenges of Indian regulatory text directly measurable:

- **Dense numerical thresholds** embedded in regulatory prose (capital adequacy ratios, margin requirements, dividend payout limits)
- **Amendment chains** where later circulars supersede earlier ones, requiring temporal reasoning to untangle
- **Jurisdiction-specific terminology** (LODR, PMLA, SFB, AIF, FEMA) that models trained predominantly on Western corpora may not reliably interpret

Overall accuracy across twelve evaluated models ranges from **70.4% to 89.7%**. All models outperform the non-specialist human baseline of **69.0%** (n = 100; 95% Wilson CI: [59.4%, 77.2%]). Crucially, performance is **not predicted by model size or general-domain capability rank**: a 17B model statistically matches a 70B model; a 120B model offers no measurable gain over 20B; and a reasoning-specialist architecture ranks second-to-last.

---

## Task Types

| Task | Code | Items | What It Tests |
|------|:----:|:-----:|---------------|
| Regulatory Interpretation | REG | 174 | Extract correct rules, compliance thresholds, or scope from a regulatory passage |
| Numerical Reasoning | NUM | 92 | Perform multi-step arithmetic over numerical figures embedded in regulatory text |
| Contradiction Detection | CON | 62 | Determine whether two passages from different regulatory instruments contradict each other |
| Temporal Reasoning | TMP | 78 | Establish chronological order of regulatory events; identify which version of a rule was operative at a given date |
| **Total** | | **406** | |

---

## Leaderboard

Twelve models evaluated under identical **zero-shot, context-only** conditions on the full 406-item benchmark. Scores are accuracy (%). 95% Wilson score confidence intervals are shown for overall accuracy. Primary scores use the conservative automated pipeline; see [Scoring Pipeline](#evaluation-protocol) for corrected figures.

| Model | REG | NUM | CON | TMP | **Overall** | 95% CI |
|-------|:---:|:---:|:---:|:---:|:-----------:|:------:|
| Gemini 2.5 Flash | 93.1 | 84.8 | 88.7 | 88.5 | **89.7** | [86.3, 92.3] |
| Qwen3-32B | 85.1 | 77.2 | 90.3 | 92.3 | **85.5** | [81.7, 88.6] |
| LLaMA-3.3-70B | 86.2 | 75.0 | 95.2 | 79.5 | **83.7** | [79.8, 87.0] |
| Llama 4 Scout 17B | 86.2 | 66.3 | 98.4 | 84.6 | **83.3** | [79.3, 86.6] |
| Kimi K2 | 89.1 | 65.2 | 91.9 | 75.6 | **81.5** | [77.5, 85.0] |
| LLaMA-3-8B | 79.9 | 64.1 | 93.5 | 78.2 | **78.1** | [73.8, 81.8] |
| GPT-OSS 120B | 79.9 | 59.8 | 95.2 | 76.9 | **77.1** | [72.8, 80.9] |
| GPT-OSS 20B | 79.9 | 58.7 | 95.2 | 76.9 | **76.8** | [72.5, 80.7] |
| Gemini 2.5 Pro † | 89.7 | 48.9 | 93.5 | 64.1 | **76.1** | [71.7, 80.0] |
| Mistral-7B | 79.9 | 66.3 | 80.6 | 74.4 | **75.9** | [71.5, 79.8] |
| DeepSeek R1 70B | 72.4 | 69.6 | 96.8 | 70.5 | **75.1** | [70.7, 79.1] |
| Gemma 4 E4B | 83.9 | 50.0 | 72.6 | 62.8 | **70.4** | [65.8, 74.7] |
| **Human Baseline** (non-specialist, n = 100) | — | — | — | — | **69.0** | [59.4, 77.2] |

> **† Gemini 2.5 Pro:** The low NUM (48.9%) and TMP (64.1%) scores are a **scoring artifact** of its verbose output style under string-matching evaluation — not a true capability deficit. A secondary LLM-as-judge validation raises its corrected overall accuracy to **84.5%**, placing it between Tier 1 and Tier 2. See Appendix D of the paper.

> **CON majority-class note:** The contradiction detection label distribution is 85.5% "No", making the trivial majority-class baseline **85.5%** — not 50%. Gemma 4 E4B (72.6%) and Mistral-7B (80.6%) fall *below* this baseline. See Appendix C of the paper for per-model balanced accuracy.

> **Primary scores are conservative lower bounds.** Judge-corrected CSVs are in `evaluation/results_judged/` in the repository. Tier rankings are unchanged under corrected scoring.

---

## Performance Tiers

Paired bootstrap significance testing (10,000 resamples) across all 66 model pairs identifies three statistically distinct performance tiers. All tier boundaries remain significant after Bonferroni correction (α = 0.05 / 66 ≈ 0.00076).

| Tier | Models | Overall Accuracy Range |
|:----:|--------|:---------------------:|
| **Tier 1** — Strong | Gemini 2.5 Flash · Qwen3-32B · LLaMA-3.3-70B · Llama 4 Scout 17B · Kimi K2 | 81.5% – 89.7% |
| **Tier 2** — Mid | LLaMA-3-8B · GPT-OSS 120B · GPT-OSS 20B · Gemini 2.5 Pro† · Mistral-7B · DeepSeek R1 70B | 75.1% – 78.1% |
| **Tier 3** — Weakest | Gemma 4 E4B | 70.4% |

---

## Key Findings

> **Efficiency paradox.** Llama 4 Scout 17B (Tier 1, 83.3%) is statistically indistinguishable from LLaMA-3.3-70B (83.7%, p = 0.790) — matching a 70B model with one-quarter the parameters. Instruction-tuning alignment, not raw parameter count, determines performance in the 17B–70B range on this benchmark.

> **Scaling provides no benefit on this domain.** GPT-OSS 120B (77.1%) and GPT-OSS 20B (76.8%) differ by just 0.3 pp (p = 0.910). Model capacity is not the binding constraint.

> **The reasoning-specialist paradox.** DeepSeek R1 70B ranks 11th of 12 despite its chain-of-thought architecture. Its error profile reveals why: 49% of its failures are Temporal Reasoning Failures — the highest of any model. Explicit reasoning chains do not reliably assist in tracking regulatory amendment timelines across multiple dated documents.

> **Task difficulty hierarchy.** Numerical reasoning is the most discriminative task (35.9 pp spread). Temporal reasoning failure is the dominant error mode for top-tier models; smaller models fail primarily at the domain-knowledge level.

> **All models exceed the human baseline.** The non-specialist human baseline (69.0%, n = 100) is surpassed by all twelve models. All Tier 1 models are significantly above the human upper confidence bound of 77.2% (p < 0.01).

---

## Few-Shot Results (3-shot, Top 4 Models)

Zero-shot evaluation is a **conservative lower bound**. 3-shot prompting with fixed in-context examples improves numerical reasoning by 2.1–16.3 pp across all tested models. The overall tier ordering is preserved, confirming that the observed tier structure reflects genuine capability differences rather than prompting strategy sensitivity.

| Model | Δ REG | Δ NUM | Δ CON | Δ TMP | Δ Overall |
|-------|------:|------:|------:|------:|----------:|
| Gemini 2.5 Flash | −1.1 | +2.2 | +3.2 | +0.0 | +0.4 |
| Qwen3-32B | +1.7 | +2.1 | 0.0 | 0.0 | +1.2 |
| LLaMA-3.3-70B | +1.2 | +2.2 | +3.2 | +7.7 | +3.0 |
| Llama 4 Scout 17B | +4.0 | +16.3 | −12.9 | −7.7 | +1.9 |

Full 3-shot prediction files are available in `evaluation/results_fewshot/` in the repository.

---

## Models Evaluated

| Model | Provider / Access | Parameters |
|-------|-------------------|:----------:|
| Gemini 2.5 Flash | Google AI Studio API | — |
| Gemini 2.5 Pro | Google Cloud Vertex AI | — |
| Qwen3-32B | Alibaba / Groq API | 32B |
| LLaMA-3.3-70B | Meta / Groq API | 70B |
| Llama 4 Scout 17B | Meta / Groq API | 17B |
| Kimi K2 | Moonshot AI / Groq API | 1T total · 32B active (MoE) |
| LLaMA-3-8B | Meta / Ollama (local) | 8B |
| GPT-OSS 120B ‡ | OpenAI / Groq API | 120B |
| GPT-OSS 20B ‡ | OpenAI / Groq API | 20B |
| Mistral-7B | Mistral AI / Ollama (local) | 7B |
| DeepSeek R1 70B | DeepSeek / Groq API | 70B |
| Gemma 4 E4B | Google / Ollama (local) | 4B |

Local models (LLaMA-3-8B, Mistral-7B, Gemma 4 E4B) were run on an Intel i7-13650HX + NVIDIA RTX 4060 (8 GB VRAM) workstation via Ollama. All models evaluated at temperature = 0.0 with no fine-tuning or prompt adaptation.

> **‡ GPT-OSS note:** Exact checkpoint identifiers (model strings) are documented in the evaluation code at the GitHub repository.

---

## Dataset Details

### Source Documents

| Source | Documents | Document Types |
|--------|:---------:|----------------|
| SEBI (sebi.gov.in) | 92 | Circulars, master circulars, regulations, orders |
| RBI (rbi.org.in) | 100 | Circulars, monetary policy statements, master directions |
| **Total** | **192** | Spanning 1992 – 2026 |

### Difficulty Distribution

| Difficulty | Items | Share | Description |
|------------|:-----:|:-----:|-------------|
| Easy | 160 | 39.4% | Single-step extraction from context |
| Medium | 182 | 44.8% | Multi-clause reasoning or calculation |
| Hard | 64 | 15.8% | Multi-instrument tracking or complex arithmetic |

### Splits

| Split | Items | Share |
|-------|:-----:|:-----:|
| test | 324 | 79.8% |
| dev | 82 | 20.2% |
| **Total** | **406** | |

---

## Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique item identifier, e.g. `REG_001`, `NUM_042` |
| `task_type` | string | `regulatory_interpretation` · `numerical_reasoning` · `contradiction_detection` · `temporal_reasoning` |
| `difficulty` | string | `easy` · `medium` · `hard` |
| `source` | string | `SEBI` or `RBI` |
| `context` | string | Regulatory passage(s) provided to the model (80–500 words). For CON items, contains Passage A and Passage B separated by a delimiter |
| `question` | string | The question to be answered from the provided context |
| `reference_answer` | string | Gold-standard reference answer |
| `source_document` | string | Filename of the source regulatory document |

---

## Annotation and Validation

All 406 QA pairs were authored by the primary annotator, who has prior experience with Indian financial regulatory documents. Every item was individually reviewed to ensure: (1) the answer is unambiguously derivable from the provided context; (2) the question has exactly one correct answer; and (3) the context is sufficient without external knowledge.

### Model-Based Secondary Validation

A secondary validation pass over 150 items using LLaMA-3.3-70B-Versatile as a context-only quality-checker confirmed item tractability (90.7% overall agreement). This follows benchmark construction practice established in FinanceBench (Islam et al., 2023) and CUAD (Hendrycks et al., 2021).

| Task Type | Items | Agreement | Cohen's κ |
|-----------|:-----:|:---------:|:---------:|
| Regulatory Interpretation | 53 | 100.0% | ~1.00 |
| Numerical Reasoning | 32 | 84.4% | — |
| Contradiction Detection | 30 | 96.7% | 0.918 |
| Temporal Reasoning | 35 | 77.1% | — |
| **Overall** | **150** | **90.7%** | — |

*κ is reported only for the binary CON task; dashes indicate κ is not defined for extractive tasks under string-matching agreement.*

### Human Inter-Annotator Agreement

A second human annotator independently answered 120 randomly selected items across all four task types. Agreement was computed using the same four-stage scoring procedure applied to model predictions.

| Task Type | Agreement | Cohen's κ |
|-----------|:---------:|:---------:|
| Regulatory Interpretation | 100.0% | — |
| Temporal Reasoning | 87.5% | — |
| Contradiction Detection | 82.4% | **0.611** |
| Numerical Reasoning | 43.8% * | — |
| **Overall (initial 60-item sample)** | **76.7%** | — |
| **Stability confirmed across full 120-item sample** | stable | **κ = 0.611** |

κ = 0.611 corresponds to *substantial agreement* (Landis & Koch, 1977), comparable to human agreement rates for binary contradiction detection in legal NLP.

> **\* Numerical reasoning (43.8%):** This figure reflects formatting convention differences between annotators (intermediate steps vs. final value only; currency symbol style; comma placement) — not substantive disagreement about the computed answer. Post-hoc review confirmed that the underlying computed values were equivalent in every discordant numerical item.

---

## Evaluation Protocol

Models are evaluated under **zero-shot, context-only** conditions. The system prompt establishes the context-only constraint. Task-specific user prompts provide formatting instructions appropriate to each task type. All models were evaluated at temperature = 0.0.

The scoring pipeline applies four stages in sequence, stopping at the first match:

1. **Exact match** — after case-normalisation and punctuation stripping
2. **Fuzzy token match** — RapidFuzz `token_set_ratio ≥ 0.72` (threshold calibrated by manual inspection of 20 borderline cases; validated at adjacent thresholds 0.65 and 0.80)
3. **Numerical extraction match** — correct when extracted number sets from reference and prediction agree (handling currency symbols, comma separators, and units)
4. **Yes/No match** — for CON items, leading-word comparison only

### Scoring Pipeline Validation (LLM-as-Judge)

The automated pipeline systematically penalises semantically correct predictions that differ from the reference in format or verbosity. To quantify this false-negative rate (FNR), Gemini 2.5 Flash was applied as a semantic judge over all 874 items marked incorrect on NUM, REG, and TMP tasks across the twelve models. CON items use exact Yes/No matching and were not judged.

Judge reliability was confirmed on a stratified 28-item manual audit: **89.3% accuracy**. The three judge errors all involved multi-step numerical problems where the model showed correct intermediate reasoning but an incorrect final value.

| Task | Items Reviewed | Reclassified Correct | FNR (corrected) |
|------|:--------------:|:--------------------:|:---------------:|
| REG | 327 | 261 (79.8%) | 0.52 |
| NUM | 348 | 272 (78.2%) | 0.56 |
| TMP | 199 | 152 (76.4%) | 0.32 |
| CON | — | exact match; not evaluated | 0.00 |
| **Overall** | **874** | **685 (78.4%)** | — |

Three root causes account for the majority of false negatives: (1) numeric format differences ('50%' vs. 'fifty per cent'); (2) verbose answers that prefix the correct response with calculation steps; (3) abbreviated-but-correct answers (e.g., ref: 'Six months after completion of the open offer'; pred: 'Six months').

Primary leaderboard scores are **conservative lower bounds** on true model accuracy. Judge-corrected scores are released at `evaluation/results_judged/` in the repository. Model tier rankings are unchanged under corrected scoring.

---

## Error Taxonomy

Model failures are classified into four interpretable categories:

| Code | Error Type | Description |
|:----:|------------|-------------|
| DKF | Domain Knowledge Failure | Incorrect answer due to unfamiliarity with Indian regulatory concepts, terminology, or thresholds |
| NRF | Numerical Reasoning Failure | Arithmetic error, wrong unit conversion, or misapplied formula |
| TRF | Temporal Reasoning Failure | Incorrect ordering of regulatory events, wrong operative circular, or miscalculated elapsed time |
| CGF | Context Grounding Failure | Use of external knowledge instead of the provided passage; failure to locate the correct span |

**Key pattern:** Temporal Reasoning Failure dominates for top-tier models (Gemini 2.5 Flash: 40%; LLaMA-3.3-70B: 41%; DeepSeek R1 70B: **49%**). Domain Knowledge Failure is the primary failure mode for smaller models (Gemma 4 E4B: 43%). Context Grounding Failure is rare across all models (1–3%).

---

## Limitations

- The automated scoring pipeline penalises semantically correct responses with format differences; leaderboard figures are conservative lower bounds — judge-corrected scores are available in the repository
- All primary evaluation is zero-shot; 3-shot results for four top models (Appendix E of the paper) confirm this is a lower bound, particularly on NUM and TMP
- The context-injection setup provides the relevant passage directly — this tests reading comprehension over a provided excerpt, not full-document retrieval or information search
- Primary annotation was conducted by a single annotator; multi-annotator IAA covers 120 of 406 items (29.6% coverage, meeting standard reporting thresholds)
- All questions and gold answers are in English; Hindi-language versions of the same circulars are not included
- The benchmark does not currently cover Hindi–English code-switched regulatory text
- Coverage is limited to SEBI and RBI; extension to IRDAI, PFRDA, and commodity-segment regulation is planned
- The benchmark evaluates short extractive responses, not longer-form generation (regulatory summaries, compliance gap analysis, or legally coherent reasoning chains)
- The dataset is a snapshot of documents as of early 2026; regulatory frameworks evolve continuously and the dataset will require periodic refresh

---

## Citation

```bibtex
@article{pall2026indiafinbench,
  title        = {IndiaFinBench: An Evaluation Benchmark for Large Language Model
                  Performance on Indian Financial Regulatory Text},
  author       = {Pall, Rajveer Singh},
  journal      = {arXiv preprint arXiv:2604.19298},
  year         = {2026},
  url          = {https://arxiv.org/abs/2604.19298}
}
```

---

## License

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All source documents are publicly available from sebi.gov.in and rbi.org.in and carry no copyright restrictions on research use. No personally identifiable information is present in any source document or derived annotation.

---

## Contact

**Rajveer Singh Pall**  
rajveer.singhpall.cb23@ggits.net  
Gyan Ganga Institute of Technology and Sciences, Jabalpur, India