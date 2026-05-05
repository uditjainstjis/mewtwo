---
license: cc-by-sa-4.0
language:
- en
pretty_name: Synapta Indian BFSI Benchmark v1
size_categories:
- n<1K
task_categories:
- question-answering
task_ids:
- extractive-qa
- closed-domain-qa
tags:
- finance
- legal
- regulation
- india
- rbi
- sebi
- bfsi
- extractive-qa
- benchmark
- evaluation
configs:
- config_name: default
  data_files:
  - split: test
    path: questions.jsonl
  default: true
---

# Synapta Indian BFSI Benchmark v1

A 60-question, hand-curated, openly licensed evaluation set for
**extractive question answering over Indian financial regulation** -
specifically Reserve Bank of India (RBI) Master Directions and
Securities and Exchange Board of India (SEBI) Master Circulars.

- 60 questions, 30 RBI / 30 SEBI
- 30 numeric / named-fact extraction (tier 2) + 30 heading-bound passage
  questions (tier 3)
- 22 distinct source PDFs from a document-disjoint held-out split of
  the Synapta RBI/SEBI corpus
- Hand-validated scoring methods, alternative-answer lists, difficulty
  and topic tags
- Three deterministic scoring metrics (exact match, substring,
  token-F1) with Wilson 95% CI and paired McNemar significance helpers

## Why this benchmark exists

At the time of v1 release there is **no published extractive-QA
benchmark focused on Indian financial regulation.** General LegalBench
and FinanceBench do not include RBI/SEBI corpora; the FinQA / TAT-QA
tradition targets US 10-K tables; SQuAD-style benchmarks have neither
the regulatory genre nor the Indian-English drafting conventions
(`lakh`/`crore`, `Rs.`, regulation referencing, paragraph-numbered
Master Directions).

We need this benchmark internally to track regression on Synapta's
domain-adapted models. Releasing it openly lets other groups working on
Indian BFSI NLP - banks, insurers, fintech, academic NLP labs,
auditors - evaluate their systems against the same instrument and
report comparable numbers.

## Methodology

1. **Source pool.** All 60 questions are drawn from the held-out split
   of an internally curated RBI/SEBI Master Direction + Master Circular
   corpus. The split is **document-disjoint**: no source PDF appearing
   in any training partition appears in the eval partition (asserted
   in `build_benchmark.py`).

2. **Deterministic templates, not LLM generation.** All candidate
   questions and gold answers were produced by deterministic
   pattern-matching templates over headings, numeric spans and
   regulation references; **no language model authored a question or
   an answer.** This matters because LLM-generated benchmarks routinely
   smuggle in the distributional fingerprint of the generator model and
   reward systems that share that fingerprint.

3. **Hand-validated quality gate.** Each candidate was filtered against
   the rules in `build_benchmark.py::is_high_value_question`:
   - context length 200-4000 chars,
   - no "the passage above" anaphora,
   - tier 2 answers must contain a digit, a currency / percentage
     marker, or a named regulation/section reference, and be <=80 chars,
   - tier 3 answers must be <=250 chars and not start with a stopword
     (rejects mid-sentence chunk artifacts),
   - questions whose trailing topic phrase exceeds 80 chars (a sign the
     question only makes sense after seeing the passage) are dropped.

4. **Stratified greedy selection.** From the filtered pool, 60
   questions were picked under three constraints simultaneously:
   regulator balance (30/30), tier balance (30/30), per-PDF cap (5 for
   RBI PDFs, 8 for SEBI PDFs because the SEBI side has only 5 distinct
   documents). Within each stratum the next pick is the candidate from
   the least-touched PDF and the least-touched topic, breaking ties by
   sorted `qa_id` for full reproducibility.

5. **Hand annotation.** Each retained record carries:
   - `difficulty` in `{easy, medium, hard}` derived from the answer
     shape (numeric short answer = easy, regulation / section reference
     = medium, multi-sentence heading body = hard).
   - `topic_tag` in
     `{kyc, aml, fraud, capital, derivatives, mutual_fund, insurance,
       payments, banking_ops, foreign_exchange, governance, reporting}`.
   - `alternative_answers`: 0-3 deterministic variants
     (`Rs. 21` / `Rs.21` / `INR 21` / `1 lakh` for `0.1 million`, etc.).
   - `scoring_method` in
     `{substring, token_f1_threshold_0.5, exact_match}`.

## Schema

Each line of `questions.jsonl` is one JSON object with the following
fields:

| field | type | description |
|---|---|---|
| `benchmark_id` | string | stable identifier of the form `sib1-NNN-<hash>` |
| `regulator` | string | `"RBI"` or `"SEBI"` |
| `tier` | int | `2` (numeric / short-span) or `3` (heading-body) |
| `source_pdf` | string | filename of the source Master Direction / Circular |
| `source_section` | string | heading or anchoring sentence in the source |
| `context` | string | the passage the answer must be extracted from |
| `question` | string | the question to answer |
| `gold_answer` | string | reference answer |
| `alternative_answers` | list[string] | accepted normalisation variants |
| `scoring_method` | string | one of `substring`, `token_f1_threshold_0.5`, `exact_match` |
| `difficulty` | string | `easy`, `medium`, `hard` |
| `topic_tag` | string | see list above |
| `_provenance` | object | `{qa_id, split}` for traceability |

### Example

```json
{
  "benchmark_id": "sib1-002-6c5e1f6b89",
  "regulator": "RBI",
  "tier": 2,
  "source_pdf": "MD28A4C421E7F7724C07B38E3C6207F3548E.PDF",
  "source_section": "5. Closure of Fraud Cases",
  "context": "5. Closure of Fraud Cases --- ... details of fraud cases of ₹0.1 million and above closed ...",
  "question": "How much is specified for Closure of Fraud Cases under section 5?",
  "gold_answer": "₹0.1 million",
  "alternative_answers": ["Rs. 0.1 million", "Rs.0.1 million", "INR 0.1 million", "0.1 million"],
  "scoring_method": "substring",
  "difficulty": "easy",
  "topic_tag": "fraud"
}
```

## How to use

### Install

```bash
pip install datasets huggingface_hub
```

### Load the benchmark

```python
from datasets import load_dataset

ds = load_dataset("synapta/indian-bfsi-bench-v1", split="test")
print(ds[0])
print(f"{len(ds)} questions across {len(set(r['source_pdf'] for r in ds))} source PDFs")
```

`synapta/indian-bfsi-bench-v1` is the placeholder repo id - replace
with the actual organisation namespace once you know where the dataset
is hosted.

### Run your model

For each record, give your model the `(context, question)` pair and
ask it to extract the **shortest faithful answer**. No retrieval is
expected; this is a pure reading-comprehension benchmark.

Write predictions to a JSONL file with one record per line:

```json
{"benchmark_id": "sib1-001-8e7097b9fd", "prediction": "<your model's answer>"}
```

### Score

```bash
# single-system evaluation
python scoring.py \
  --predictions preds.jsonl \
  --benchmark questions.jsonl

# paired comparison between two systems (prints McNemar p-value)
python scoring.py \
  --predictions preds_systemA.jsonl \
  --predictions preds_systemB.jsonl \
  --benchmark questions.jsonl
```

`scoring.py` reports primary score (per-question `scoring_method`),
exact-match rate, substring rate, mean token-F1, and a Wilson 95%
confidence interval. With two `--predictions` arguments it also runs a
paired McNemar test (statsmodels -> scipy.binomtest -> pure-python
fallback, whichever is importable).

### The three scoring metrics

| metric | when to use |
|---|---|
| `exact_match(gold, pred)` | strictest; useful only for short, normalised numeric answers |
| `substring_match(gold, pred, alts)` | use when the gold span is short and a model that mentions it should be credited |
| `token_f1(gold, pred, stopwords)` | use for longer extractive spans; thresholded at 0.5 for the `correct` boolean |

Each question fixes its `scoring_method`, so the comparison across
systems is well-defined.

## Baselines

Synapta seed baselines below; external API rows remain open for
community contribution. Please open a PR or issue with your numbers
(and your evaluation harness) once you run them.

| System | Primary score | 95% CI (Wilson) | Notes |
|---|---|---|---|
| **Synapta-Nemotron-30B + bfsi_extract LoRA** | **50.0%** (30/60) | [37.7, 62.3] | Run 2026-05-04, see below |
| Synapta Format Guard (math+code+science+bfsi_extract, swap-every-10-tokens) | 50.0% (30/60) | [37.7, 62.3] | Run 2026-05-04, +0.0 pp vs direct, mean 0.1 swaps/Q — routing overhead is empirically free |
| Synapta-Nemotron-30B base (no adapter) | 40.0% (24/60) | [28.6, 52.6] | Run 2026-05-04, paired ablation |
| Anthropic Claude (Opus / Sonnet) | _pending_ | _pending_ | external API |
| OpenAI GPT-4o | _pending_ | _pending_ | external API |
| Google Gemini | _pending_ | _pending_ | external API |

**Lift +10.0 pp · McNemar p = 0.0313 (scipy.binomtest, just barely
significant at α=0.05 on n=60)**

Per scoring-method breakdown:

| Scoring method | n | Base | Adapter | Lift |
|---|---|---|---|---|
| `substring` | 30 | 80.0% | **100.0%** | +20.0 pp clean win, every adapter answer matches |
| `token_f1_threshold_0.5` | 30 | 0.0% | 0.0% | both fail F1 cutoff (mean F1 0.12 base, 0.16 adapter) |

**The F1 threshold finding is honest disclosure.** On Tier 3
heading-extractive questions the model's verbose paragraph-style
answers have low per-token precision against the short gold span,
even when the right paragraph is quoted. Mean F1 (~0.12 / 0.16) sits
well below the 0.5 cutoff. Future scoring revisions for this
benchmark may want F1>=0.3 or a sentence-overlap variant for Tier 3
to better separate model behaviors that are semantically right but
verbose.

Run details:
- Date: 2026-05-04
- Hardware: single RTX 5090 (32 GB), 4-bit NF4 base
- Prompt: system message "senior Indian banking and financial
  regulation expert ... quote directly from the regulation when
  possible"; greedy decoding; max_new_tokens=200; max_input_tokens=1536
- Eval harness: `synapta_src/data_pipeline/17_eval_benchmark_v1.py`
  (calls into this repo's `scoring.py`)
- Wall-clock: 22 min for paired base + adapter generation across 60 Qs
- Raw outputs: `results/benchmark_v1_eval/predictions_{base,bfsi_extract}.jsonl`
- Summary: `results/benchmark_v1_eval/summary.json`

## Composition snapshot

- Regulator: RBI 30, SEBI 30
- Tier: 2 (numeric/short-span) 30, 3 (heading-body) 30
- Topic tags: `governance` 22, `mutual_fund` 12, `foreign_exchange` 9,
  `banking_ops` 6, `reporting` 4, `fraud` 3, `derivatives` 3,
  `insurance` 1.
- Difficulty: `medium` 36, `hard` 14, `easy` 10.
- Scoring methods: `substring` 30, `token_f1_threshold_0.5` 30.
- Source-PDF coverage: 22 distinct PDFs.

## Coverage limitations (honest)

v1 reflects what the underlying held-out document set covers. The
distribution skews **governance-heavy**: KYC, AML / PMLA, capital
adequacy ratios, ATM/UPI, NBFC supervision, outsourcing, related-party
transactions and stress-testing topics are **under-represented or
absent** in v1, because no Master Direction on those subjects landed
in the held-out split.

This is an artefact of the held-out partitioning, not of the
benchmark's intent. **v2** will broaden the document set so that the
topic histogram matches the actual regulatory perimeter more closely.

If your application is sensitive to those gaps, treat v1 as a
**floor**, not a ceiling: a model that does badly on v1 will likely
do badly on the omitted topics, but the converse does not hold.

## Contamination notice

This is **v1**, frozen on release so that scores remain comparable.
Once a public benchmark is indexed by web crawlers, it will eventually
end up in pretraining corpora; we accept that trade-off in exchange
for broad community use.

When reporting scores, please indicate:

- which **version** of the benchmark you evaluated against,
- whether your model's training-data cutoff predates this release
  (2026-05) or includes web crawls after it,
- the **exact prompt** used to elicit answers (zero-shot / few-shot,
  with or without the `source_section` field, etc.).

Subsequent versions (v2, v3, ...) will refresh the question set from
documents that did not appear in v1, so contamination of v1 does not
permanently degrade the family.

## How to contribute

We welcome contributions, especially:

- **Baseline numbers** for any frontier or open-weights model. Open a
  PR adding a row to the baselines table with your prompt and harness.
- **Errata** - if a gold answer is wrong, ambiguous, or has been
  superseded by a regulator amendment, open an issue with the
  `benchmark_id`, the source PDF, and the corrected text.
- **New evaluation rows for v2** - if you have a Master Direction or
  Master Circular our v1 corpus did not cover, please send the source
  PDF (or a stable RBI/SEBI URL) plus 1-3 candidate
  `(context, question, gold_answer)` triples in the schema above.

To contribute:

1. Fork <https://github.com/synapta/indian-bfsi-bench> (placeholder).
2. Add or edit JSONL rows / baseline entries.
3. Run `python scoring.py --predictions ... --benchmark questions.jsonl`
   to make sure the scorer still parses every row.
4. Open a pull request describing the change and citing the source
   document.

## Files

- `questions.jsonl` - 60 questions, one per line
- `scoring.py` - reference scorer + CLI
- `build_benchmark.py` - deterministic builder (documentation of
  construction; relies on internal corpus paths and is not runnable
  from the published bundle)
- `LICENSE.md` - CC-BY-SA-4.0 terms and citation
- `README.md` - this file

## License

Released under the
**Creative Commons Attribution-ShareAlike 4.0 International License
(CC-BY-SA-4.0)**. See `LICENSE.md` for the full notice and the
suggested BibTeX citation.

We chose CC-BY-SA-4.0 over Apache-2.0 because the benchmark is data,
not code, and CC-BY-SA's ShareAlike clause keeps community-built
extensions and translations open as well.

## Citation

```bibtex
@misc{synapta2026bfsi,
  title        = {Synapta Indian BFSI Benchmark v1: an extractive
                  question-answering evaluation set for Indian
                  financial regulation},
  author       = {Synapta},
  year         = {2026},
  howpublished = {\url{https://github.com/synapta/indian-bfsi-bench}},
  note         = {Released under CC-BY-SA-4.0.}
}
```
